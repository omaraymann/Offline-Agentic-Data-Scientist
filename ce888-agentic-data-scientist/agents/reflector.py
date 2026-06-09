from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ttest_rel


GOOD_SCORE = 0.80
MODEST_SCORE = 0.65
POOR_SCORE = 0.55

# Return the best-metric block regardless of evaluation payload shape
def extract_best_metrics(evaluation):
    if "best_metrics" in evaluation:
        return evaluation["best_metrics"]
    return evaluation

# Return the classification report dictionary when available
def extract_class_report(evaluation):
    return evaluation.get("classification_report_dict", {})

# Safely convert a value to float
def safe_float(value, default: float = 0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# Summarise how spread out the model performances are
def model_spread(all_metrics):
    scores = [safe_float(metric.get("balanced_accuracy"), 0.0) for metric in all_metrics]
    if not scores:
        return {"best": 0.0, "median": 0.0, "std": 0.0, "range": 0.0}
    return {
        "best": float(max(scores)),
        "median": float(median(scores)),
        "std": float(np.std(scores)),
        "range": float(max(scores) - min(scores)),
    }

# Compare the top two models using paired CV scores when available
def paired_significance(best_metrics, runner_up):
    best_folds = best_metrics.get("cv_balanced_accuracy_scores") or []
    runner_folds = [] if runner_up is None else (runner_up.get("cv_balanced_accuracy_scores") or [])

    if len(best_folds) >= 3 and len(best_folds) == len(runner_folds):
        try:
            stat = ttest_rel(best_folds, runner_folds)
            return {
                "available": True,
                "p_value": None if np.isnan(stat.pvalue) else float(stat.pvalue),
                "mean_gap": float(np.mean(best_folds) - np.mean(runner_folds)),
            }
        except Exception:
            pass

    return {
        "available": False,
        "p_value": None,
        "mean_gap": None,
    }

# Reflect on the current run and produce actionable diagnostics
def reflect(dataset_profile, evaluation, all_metrics):
    best_metrics = extract_best_metrics(evaluation)
    best_model = str(best_metrics.get("model", "unknown"))
    best_bal_acc = safe_float(best_metrics.get("balanced_accuracy"), 0.0)
    best_f1 = safe_float(best_metrics.get("f1_macro"), 0.0)
    best_acc = safe_float(best_metrics.get("accuracy"), 0.0)
    train_bal_acc = safe_float(best_metrics.get("train_balanced_accuracy"), 0.0)
    cv_mean = safe_float(best_metrics.get("cv_balanced_accuracy_mean"), default=np.nan)
    cv_std = safe_float(best_metrics.get("cv_balanced_accuracy_std"), default=np.nan)

    imbalance_ratio = safe_float(dataset_profile.get("imbalance_ratio"), 1.0)
    rows = int(dataset_profile.get("shape", {}).get("rows", 0))
    high_cardinality = bool(dataset_profile.get("high_cardinality_features"))
    overall_missing = safe_float(dataset_profile.get("missingness_summary", {}).get("overall_pct"))
    missing_severity = str(dataset_profile.get("missingness_summary", {}).get("severity", "low"))
    weak_signal_risk = bool(dataset_profile.get("weak_signal_risk", False))
    num_classes = int(dataset_profile.get("num_classes", 0) or 0)

    spread = model_spread(all_metrics)
    ordered = sorted(
        all_metrics,
        key=lambda metric: (safe_float(metric.get("balanced_accuracy"), 0.0), safe_float(metric.get("f1_macro"), 0.0)),
        reverse=True,
    )
    dummy = next((metric for metric in ordered if str(metric.get("model", "")).startswith("Dummy")), None)
    runner_up = ordered[1] if len(ordered) > 1 else None
    significance = paired_significance(best_metrics, runner_up)

    class_report = extract_class_report(evaluation)
    minority_recalls: List[float] = []
    for key, value in class_report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(value, dict):
            minority_recalls.append(safe_float(value.get("recall"), 0.0))
    worst_class_recall = min(minority_recalls) if minority_recalls else None

    baseline_gain = None
    f1_gain = None
    if dummy is not None:
        baseline_gain = best_bal_acc - safe_float(dummy.get("balanced_accuracy"), 0.0)
        f1_gain = best_f1 - safe_float(dummy.get("f1_macro"), 0.0)

    ranking_gap = None if runner_up is None else best_bal_acc - safe_float(runner_up.get("balanced_accuracy"), 0.0)
    overfit_gap = train_bal_acc - best_bal_acc

    issues: List[str] = []
    root_causes: List[str] = []
    suggestions: List[str] = []
    confidence = "high"

    if baseline_gain is not None and baseline_gain < 0.03:
        issues.append(f"Best model improves balanced accuracy over the dummy baseline by only {baseline_gain:.3f}.")
        root_causes.append("weak_features")
        suggestions.append("Review feature usefulness, confirm the target column, and prioritise stronger non-linear models.")

    if best_bal_acc < POOR_SCORE:
        issues.append(f"Balanced accuracy is weak at {best_bal_acc:.3f}.")
    elif best_bal_acc < MODEST_SCORE:
        issues.append(f"Balanced accuracy is only moderate at {best_bal_acc:.3f}.")

    if best_f1 < POOR_SCORE:
        if "imbalance_handling" not in root_causes and imbalance_ratio >= 3.0:
            root_causes.append("imbalance_handling")
        issues.append(f"Macro F1 is weak at {best_f1:.3f}, suggesting uneven class performance.")

    if worst_class_recall is not None and worst_class_recall < 0.45:
        issues.append(f"At least one class has very low recall ({worst_class_recall:.3f}).")
        if imbalance_ratio >= 2.5:
            root_causes.append("imbalance_handling")
            suggestions.append("Keep balanced metrics as the main ranking signal and enable stronger imbalance handling.")
        else:
            root_causes.append("weak_features")

    if imbalance_ratio >= 4.0 and (best_bal_acc - best_acc) < -0.08:
        if "imbalance_handling" not in root_causes:
            root_causes.append("imbalance_handling")
        suggestions.append("Use class weighting and threshold tuning instead of trusting raw accuracy.")

    if high_cardinality and best_bal_acc < GOOD_SCORE:
        root_causes.append("preprocessing_mismatch")
        suggestions.append("Group rare categories before one-hot encoding to reduce noisy sparse features.")

    if missing_severity in {"high", "severe"} or overall_missing >= 12.0:
        root_causes.append("data_quality")
        suggestions.append("Drop or isolate very high-missing columns and compare results against the default imputation pipeline.")

    if rows < 500 and (np.isnan(cv_std) or cv_std > 0.06):
        root_causes.append("too_little_data")
        confidence = "medium"
        suggestions.append("Use cross-validation-driven ranking because holdout estimates are unstable on very small datasets.")

    if overfit_gap > 0.12:
        root_causes.append("unstable_model_rankings")
        suggestions.append("Prefer simpler or more regularised models because the best model appears to overfit the training split.")

    if ranking_gap is not None and ranking_gap < 0.01:
        root_causes.append("unstable_model_rankings")
        if np.isnan(cv_std) or cv_std > 0.03:
            confidence = "medium"
        suggestions.append("Top models are nearly tied; rank candidates using cross-validation stability as a tie-breaker.")

    if spread["std"] < 0.025 and best_bal_acc < MODEST_SCORE:
        root_causes.append("overly_simple_models")
        suggestions.append("Expand the model pool slightly because all candidates behave too similarly on this dataset.")

    if weak_signal_risk and best_bal_acc < GOOD_SCORE:
        if "weak_features" not in root_causes:
            root_causes.append("weak_features")
        suggestions.append("Treat this as a weak-signal problem and prioritise robust ensembles over extensive tuning.")

    if significance["available"] and significance["p_value"] is not None and significance["p_value"] > 0.10:
        suggestions.append("The top two models are not clearly separated in CV; prefer the simpler one if results are similar.")

    root_causes = list(dict.fromkeys(root_causes))
    suggestions = list(dict.fromkeys(suggestions))

    status = "ok"
    if issues:
        status = "needs_attention"

    diagnostics = {
        "best_balanced_accuracy": best_bal_acc,
        "best_macro_f1": best_f1,
        "best_accuracy": best_acc,
        "dummy_gain_balanced_accuracy": baseline_gain,
        "dummy_gain_macro_f1": f1_gain,
        "ranking_gap_to_runner_up": ranking_gap,
        "train_test_gap_balanced_accuracy": overfit_gap,
        "cv_balanced_accuracy_mean": None if np.isnan(cv_mean) else cv_mean,
        "cv_balanced_accuracy_std": None if np.isnan(cv_std) else cv_std,
        "model_spread": spread,
        "significance_vs_runner_up": significance,
        "worst_class_recall": worst_class_recall,
    }

    replan_recommended = False
    if status == "needs_attention":
        likely_benefit = 0
        if baseline_gain is not None and baseline_gain < 0.08:
            likely_benefit += 1
        if "imbalance_handling" in root_causes:
            likely_benefit += 1
        if "preprocessing_mismatch" in root_causes:
            likely_benefit += 1
        if "overly_simple_models" in root_causes:
            likely_benefit += 1
        if "unstable_model_rankings" in root_causes:
            likely_benefit += 1
        if best_bal_acc < GOOD_SCORE and likely_benefit >= 1:
            replan_recommended = True

    return {
        "status": status,
        "best_model": best_model,
        "issues": issues,
        "suggestions": suggestions,
        "priority_actions": suggestions[:3],
        "root_causes": root_causes,
        "confidence": confidence,
        "diagnostics": diagnostics,
        "replan_recommended": replan_recommended,
    }

# Decide whether a replan is worthwhile
def should_replan(reflection, current_replans: int = 0, max_replans = 1, history = None,):
    if current_replans >= max_replans:
        return False

    if not reflection.get("replan_recommended", False):
        return False

    diagnostics = reflection.get("diagnostics", {})
    best_bal_acc = safe_float(diagnostics.get("best_balanced_accuracy"))
    baseline_gain = diagnostics.get("dummy_gain_balanced_accuracy")
    ranking_gap = diagnostics.get("ranking_gap_to_runner_up")
    cv_std = diagnostics.get("cv_balanced_accuracy_std")
    root_causes = set(reflection.get("root_causes", []))

    if history and len(history) >= 2:
        last = history[-1]
        previous = history[-2]
        last_score = safe_float(last.get("balanced_accuracy"))
        prev_score = safe_float(previous.get("balanced_accuracy"))
        if (last_score - prev_score) < 0.01 and current_replans >= 1:
            return False

    if best_bal_acc >= GOOD_SCORE and baseline_gain is not None and baseline_gain >= 0.10:
        return False

    potential_gain = 0
    if baseline_gain is None or baseline_gain < 0.08:
        potential_gain += 1
    if ranking_gap is not None and ranking_gap < 0.02:
        potential_gain += 1
    if cv_std is None or cv_std > 0.04:
        potential_gain += 1
    if {"imbalance_handling", "preprocessing_mismatch", "overly_simple_models", "unstable_model_rankings"} & root_causes:
        potential_gain += 1

    return potential_gain >= 2

# Apply meaningful strategy changes rather than appending a note
def apply_replan_strategy(plan, dataset_profile, reflection,):
    new_plan = list(plan)
    new_profile = deepcopy(dataset_profile)
    flags = deepcopy(new_profile.get("strategy_flags", {}))
    flags["replan_count"] = int(flags.get("replan_count", 0)) + 1

    root_causes = set(reflection.get("root_causes", []))
    num_classes = int(new_profile.get("num_classes", 0) or 0)

    if "imbalance_handling" in root_causes:
        flags["use_class_weight"] = True
        flags["enable_cv"] = True
        flags["compare_on_cv"] = True
        if num_classes == 2:
            flags["tune_decision_threshold"] = True

    if "preprocessing_mismatch" in root_causes:
        flags["group_rare_categories"] = True
        flags["drop_high_missing_features"] = True

    if "too_little_data" in root_causes:
        flags["prefer_simple_models"] = True
        flags["enable_cv"] = True
        flags["compare_on_cv"] = True

    if "overly_simple_models" in root_causes or "weak_features" in root_causes:
        flags["expand_model_diversity"] = True
        flags["include_extra_trees"] = True
        flags["include_boosting"] = True
        flags["include_gradient_boosting"] = True
        flags["include_kernel_model"] = True

    if "unstable_model_rankings" in root_causes:
        flags["enable_cv"] = True
        flags["compare_on_cv"] = True
        flags["prefer_simple_models"] = True

    new_profile["strategy_flags"] = flags
    new_notes = list(new_profile.get("notes", []))
    new_notes.append(
        "Replan applied: strategy flags updated based on reflection root causes: "
        + ", ".join(sorted(root_causes))
    )
    new_profile["notes"] = new_notes

    extra_steps = [
        "apply_replanned_strategy",
        "rebuild_preprocessor",
        "reselect_models",
    ]

    if flags.get("group_rare_categories"):
        extra_steps.append("group_rare_categories")
    if flags.get("drop_high_missing_features"):
        extra_steps.append("drop_high_missing_columns")
    if flags.get("enable_cv"):
        extra_steps.append("validate_with_cross_validation")
    if flags.get("compare_on_cv"):
        extra_steps.append("rank_with_cv_stability")
    if flags.get("expand_model_diversity"):
        extra_steps.append("expand_model_pool")
    if flags.get("tune_decision_threshold"):
        extra_steps.append("tune_decision_thresholds")

    new_plan = extra_steps + new_plan
    deduplicated: List[str] = []
    seen = set()
    for step in new_plan:
        if step not in seen:
            deduplicated.append(step)
            seen.add(step)

    return deduplicated, new_profile