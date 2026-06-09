from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional

# Return a list without duplicates while preserving first appearance order
def deduplicate_preserve_order(steps):
    seen = set()
    deduplicated = []
    for step in steps:
        if step not in seen:
            deduplicated.append(step)
            seen.add(step)
    return deduplicated

# Infer a dataset size bucket when the profiler did not provide one
def infer_size_bucket(rows):
    if rows < 300:
        return "tiny"
    if rows < 3000:
        return "small"
    if rows < 30000:
        return "medium"
    return "large"

# Merge strategy flags with safe defaults expected by the planner
def get_strategy_flags(dataset_profile):
    flags = deepcopy(dataset_profile.get("strategy_flags", {}))
    flags.setdefault("use_class_weight", False)
    flags.setdefault("group_rare_categories", False)
    flags.setdefault("enable_cv", False)
    flags.setdefault("compare_on_cv", False)
    flags.setdefault("prefer_simple_models", False)
    flags.setdefault("expand_model_diversity", False)
    flags.setdefault("include_kernel_model", False)
    flags.setdefault("include_boosting", True)
    flags.setdefault("include_extra_trees", True)
    flags.setdefault("include_gradient_boosting", True)
    flags.setdefault("tune_decision_threshold", False)
    flags.setdefault("drop_high_missing_features", False)
    flags.setdefault("replan_count", 0)
    return flags

# Extract model names worth prioritising from memory guidance
def memory_preference_models(memory_hint):
    if not memory_hint:
        return []

    preferred = []
    for model_name, score in memory_hint.get("preferred_models", []):
        if float(score) >= 0.0:
            preferred.append(str(model_name))

    if not preferred and memory_hint.get("best_model"):
        preferred.append(str(memory_hint["best_model"]))

    return preferred

# Return the core end to end workflow shared by all runs
def base_plan():
    return [
        "profile_dataset",
        "validate_target",
        "assess_risks",
        "build_preprocessor",
        "select_models",
        "train_baseline",
        "train_candidate_models",
        "compare_candidates",
        "evaluate",
        "reflect",
        "write_report",
    ]

# Insert a step before an anchor step if it is not already present
def insert_before(plan, anchor, new_step):
    if new_step in plan:
        return
    if anchor in plan:
        plan.insert(plan.index(anchor), new_step)
    else:
        plan.append(new_step)

# Insert a step after an anchor step if it is not already present
def insert_after(plan, anchor, new_step):
    if new_step in plan:
        return
    if anchor in plan:
        plan.insert(plan.index(anchor) + 1, new_step)
    else:
        plan.append(new_step)

# Normalize planner inputs into a compact feature dictionary
def extract_features(profile):
    rows = int(profile.get("shape", {}).get("rows", 0))
    cols = int(profile.get("shape", {}).get("cols", 0))
    feature_types = profile.get("feature_types", {})
    num_classes = int(profile.get("num_classes", 0) or 0)

    features = {
        "rows": rows,
        "cols": cols,
        "feature_count": max(cols - 1, 1),
        "imbalance_ratio": float(profile.get("imbalance_ratio") or 1.0),
        "overall_missing": float(profile.get("missingness_summary", {}).get("overall_pct", 0.0)),
        "max_missing": float(profile.get("missingness_summary", {}).get("max_pct", 0.0)),
        "missing_severity": str(profile.get("missingness_summary", {}).get("severity", "low")),
        "high_cardinality": bool(profile.get("high_cardinality_features")),
        "high_dimensional": bool(profile.get("high_dimensional", False)),
        "numeric_dominance": str(profile.get("dominant_feature_type", "mixed")) == "numeric",
        "num_classes": num_classes,
        "target_uncertain": str(profile.get("target_inference", {}).get("confidence", "high")) == "low",
        "weak_signal_risk": bool(profile.get("weak_signal_risk", False)),
        "size_bucket": str(profile.get("size_bucket") or infer_size_bucket(rows)),
        "is_classification": bool(profile.get("is_classification", True)),
        "has_numeric_features": len(feature_types.get("numeric", [])) > 0,
        "has_categorical_features": len(feature_types.get("categorical", [])) > 0,
        "binary_target": num_classes == 2,
    }
    return features

# Create a data aware execution plan for the offline agent
def create_plan(dataset_profile, memory_hint = None):
    profile = deepcopy(dataset_profile)
    flags = get_strategy_flags(profile)
    features = extract_features(profile)

    tiny_dataset = features["size_bucket"] == "tiny"
    small_dataset = features["size_bucket"] == "small"
    large_dataset = features["size_bucket"] == "large"
    extremely_wide = features["feature_count"] > max(features["rows"], 1)
    threshold_tuning_needed = (
        features["is_classification"]
        and features["binary_target"]
        and (
            features["imbalance_ratio"] >= 4.0
            or bool(flags.get("tune_decision_threshold"))
        )
    )

    plan = base_plan()

    if features["is_classification"]:
        insert_before(plan, "select_models", "set_problem_type::classification")
    else:
        insert_before(plan, "select_models", "set_problem_type::regression")

    if features["target_uncertain"]:
        insert_after(plan, "validate_target", "confirm_target_assumption")

    insert_after(plan, "validate_target", "check_for_data_leakage")
    insert_after(plan, "check_for_data_leakage", "remove_leaky_features")

    if features["missing_severity"] in {"medium", "high", "severe"} or features["overall_missing"] >= 5.0:
        insert_before(plan, "build_preprocessor", "audit_missingness")
        if features["has_numeric_features"]:
            insert_before(plan, "build_preprocessor", "impute_numeric_features")
        if features["has_categorical_features"]:
            insert_before(plan, "build_preprocessor", "impute_categorical_features")
        insert_before(plan, "build_preprocessor", "add_missing_indicators")
        if features["max_missing"] >= 35.0 or flags.get("drop_high_missing_features"):
            insert_before(plan, "build_preprocessor", "drop_high_missing_columns")

    if features["high_cardinality"]:
        insert_before(plan, "build_preprocessor", "control_high_cardinality")

    if features["is_classification"] and features["imbalance_ratio"] >= 2.5:
        insert_before(plan, "select_models", "activate_imbalance_strategy")
        if flags.get("use_class_weight") or features["imbalance_ratio"] >= 3.0:
            insert_before(plan, "select_models", "enable_class_weighted_models")
        if threshold_tuning_needed:
            insert_after(plan, "train_candidate_models", "tune_decision_thresholds")

    if tiny_dataset:
        insert_before(plan, "select_models", "guard_against_tiny_sample")
        insert_before(plan, "train_candidate_models", "prefer_simple_models")
        insert_before(plan, "train_candidate_models", "limit_model_complexity")
        insert_after(plan, "compare_candidates", "validate_with_cross_validation")
    elif small_dataset:
        insert_after(plan, "train_candidate_models", "validate_with_cross_validation")
    elif large_dataset:
        insert_before(plan, "train_candidate_models", "use_runtime_aware_model_pool")
        insert_after(plan, "compare_candidates", "check_scalability_tradeoffs")

    if features["high_dimensional"] or extremely_wide:
        insert_before(plan, "build_preprocessor", "apply_feature_selection")
        insert_before(plan, "select_models", "reduce_model_complexity_risk")

    if features["numeric_dominance"] and not features["high_cardinality"] and not extremely_wide:
        insert_before(plan, "select_models", "prioritise_linear_and_distance_models")
    else:
        insert_before(plan, "select_models", "prioritise_tree_ensembles_for_mixed_data")

    if flags.get("include_kernel_model"):
        insert_before(plan, "train_candidate_models", "include_kernel_models")
    if flags.get("include_boosting") is False:
        insert_before(plan, "train_candidate_models", "exclude_boosting_models")
    if flags.get("include_extra_trees") is False:
        insert_before(plan, "train_candidate_models", "exclude_extra_trees_models")
    if flags.get("include_gradient_boosting") is False:
        insert_before(plan, "train_candidate_models", "exclude_gradient_boosting_models")

    if features["weak_signal_risk"]:
        insert_before(plan, "select_models", "prefer_regularised_models")
        insert_before(plan, "train_candidate_models", "compare_with_dummy_baseline")
        insert_before(plan, "reflect", "prepare_fallback_strategy")

    if flags.get("enable_cv"):
        insert_after(plan, "train_candidate_models", "validate_with_cross_validation")
    if flags.get("compare_on_cv"):
        insert_after(plan, "compare_candidates", "rank_with_cv_stability")
    if flags.get("expand_model_diversity"):
        insert_before(plan, "train_candidate_models", "expand_model_pool")
    if flags.get("prefer_simple_models"):
        insert_before(plan, "train_candidate_models", "prefer_simple_models")
    if flags.get("group_rare_categories"):
        insert_before(plan, "build_preprocessor", "group_rare_categories")

    if features["is_classification"]:
        if features["imbalance_ratio"] >= 2.5:
            insert_before(plan, "evaluate", "use_balanced_accuracy")
            insert_before(plan, "evaluate", "use_macro_f1")
        else:
            insert_before(plan, "evaluate", "use_accuracy")
            insert_before(plan, "evaluate", "use_macro_f1")
    else:
        insert_before(plan, "evaluate", "use_rmse")
        insert_before(plan, "evaluate", "use_mae")

    preferred_models = memory_preference_models(memory_hint)
    if preferred_models:
        insert_before(plan, "select_models", "prioritise_models_from_memory")
        insert_before(plan, "train_candidate_models", "try_memory_models_first")
        for model_name in preferred_models[:3]:
            plan.append(f"memory_priority::{model_name}")

    if memory_hint and memory_hint.get("avoid_models"):
        insert_before(plan, "select_models", "avoid_previously_unstable_models")
        for model_name in memory_hint.get("avoid_models", [])[:3]:
            plan.append(f"memory_avoid::{model_name}")

    if memory_hint and memory_hint.get("model_performance"):
        insert_before(plan, "select_models", "rank_models_by_past_performance")

    if int(flags.get("replan_count", 0)) > 0:
        insert_before(plan, "build_preprocessor", "apply_replanned_strategy")
        insert_after(plan, "reflect", "assess_replan_impact")

    return deduplicate_preserve_order(plan)