from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Save an object as formatted JSON
def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)

# Plot and save a confusion matrix figure
def plot_confusion_matrix(cm, labels, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

# Produce a compact explanation of why the chosen model won
# This helps the report/demo show reasoning rather than just numbers
def _selection_explanation(training_payload, profile):
    all_metrics = training_payload.get("all_metrics", [])
    best = training_payload["best"]["metrics"]
    primary_metric = training_payload.get("primary_metric", "accuracy")
    compare_on_cv = bool((profile or {}).get("strategy_flags", {}).get("compare_on_cv", False))

    runner_up = None
    ordered = sorted(
        all_metrics,
        key=lambda item: (
            float(item.get(primary_metric, 0.0)),
            float(item.get("f1_macro", 0.0)),
        ),
        reverse=True,
    )
    if len(ordered) > 1:
        runner_up = ordered[1]

    dummy = next((metric for metric in all_metrics if str(metric.get("model", "")).startswith("Dummy")), None)

    explanation = {
        "selected_model": best.get("model"),
        "primary_metric": primary_metric,
        "compare_on_cv": compare_on_cv,
        "selection_policy": "cv_first_then_holdout" if compare_on_cv else "holdout_first_then_cv",
        "best_primary_metric": best.get(primary_metric),
        "best_cv_balanced_accuracy_mean": best.get("cv_balanced_accuracy_mean"),
        "best_cv_balanced_accuracy_std": best.get("cv_balanced_accuracy_std"),
        "runner_up_model": None if runner_up is None else runner_up.get("model"),
        "runner_up_primary_metric": None if runner_up is None else runner_up.get(primary_metric),
        "runner_up_cv_balanced_accuracy_mean": None if runner_up is None else runner_up.get("cv_balanced_accuracy_mean"),
        "dummy_balanced_accuracy": None if dummy is None else dummy.get("balanced_accuracy"),
        "dummy_gain_balanced_accuracy": (
            None
            if dummy is None
            else float(best.get("balanced_accuracy", 0.0)) - float(dummy.get("balanced_accuracy", 0.0))
        ),
        "human_summary": (
            "Selected using cross-validation-aware ranking to prioritise stability."
            if compare_on_cv
            else "Selected mainly from holdout performance, with CV used as supporting evidence."
        ),
    }
    return explanation

# Evaluate the selected best model and build a report-friendly payload
def evaluate_best(training_payload, output_dir, profile = None):
    best = training_payload["best"]
    all_metrics = training_payload["all_metrics"]

    y_test = np.asarray(best["y_test"])
    y_pred = np.asarray(best["y_pred"])
    labels = sorted({str(x) for x in np.concatenate([y_test, y_pred])})

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")

    cls_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cls_report_text = classification_report(y_test, y_pred, zero_division=0)

    selection_explanation = _selection_explanation(training_payload, profile)

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "primary_metric": training_payload.get("primary_metric"),
        "split_info": training_payload.get("split_info", {}),
        "failures": training_payload.get("failures", []),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report_text,
        "classification_report_dict": cls_report_dict,
        "selection_explanation": selection_explanation,
    }

# Write a markdown report that explains the full agent run clearly
def write_markdown_report( out_path, ctx, fingerprint, dataset_profile, plan, eval_payload, reflection, memory_hint = None, attempt_index = 0, selection_rationale = None):
    best = eval_payload["best_metrics"]
    numeric = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])
    flags = dataset_profile.get("strategy_flags", {})
    selection = selection_rationale or eval_payload.get("selection_explanation", {})

    memory_text = "None"
    if memory_hint:
        memory_text = json.dumps(memory_hint, indent=2)

    notes_text = "- None"
    if notes:
        notes_text = "\n".join([f"- {note}" for note in notes])

    plan_text = "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(plan)])

    issues_text = "- None"
    if reflection.get("issues"):
        issues_text = "\n".join([f"- {item}" for item in reflection.get("issues", [])])

    root_causes_text = "- None"
    if reflection.get("root_causes"):
        root_causes_text = "\n".join([f"- {item}" for item in reflection.get("root_causes", [])])

    actions_text = "- None"
    if reflection.get("priority_actions"):
        actions_text = "\n".join([f"- {item}" for item in reflection.get("priority_actions", [])])

    top_rows = []
    for metric in eval_payload.get("all_metrics", []):
        top_rows.append(
            "| {model} | {acc:.3f} | {bal:.3f} | {f1:.3f} | {cv_mean} |".format(
                model=metric.get("model"),
                acc=float(metric.get("accuracy", 0.0)),
                bal=float(metric.get("balanced_accuracy", 0.0)),
                f1=float(metric.get("f1_macro", 0.0)),
                cv_mean="None"
                if metric.get("cv_balanced_accuracy_mean") is None
                else f"{float(metric.get('cv_balanced_accuracy_mean')):.3f}",
            )
        )

    top_rows_text = "\n".join(top_rows) if top_rows else "| None | 0.000 | 0.000 | 0.000 | None |"

    lines = [
        "# Agentic Data Scientist Report",
        "",
        f"**Run ID:** `{ctx.run_id}`  ",
        f"**Started (UTC):** {ctx.started_at}  ",
        f"**Dataset:** `{ctx.data_path}`  ",
        f"**Target:** `{ctx.target}`  ",
        f"**Fingerprint:** `{fingerprint}`  ",
        f"**Attempt Index:** {attempt_index}  ",
        "",
        "## 1. Dataset Profile",
        f"- Rows: **{dataset_profile['shape']['rows']}**",
        f"- Columns: **{dataset_profile['shape']['cols']}**",
        f"- Size bucket: **{dataset_profile.get('size_bucket')}**",
        f"- Dominant feature type: **{dataset_profile.get('dominant_feature_type')}**",
        f"- Number of classes: **{dataset_profile.get('num_classes')}**",
        f"- Imbalance ratio: **{dataset_profile.get('imbalance_ratio')}**",
        f"- Missingness severity: **{dataset_profile.get('missingness_summary', {}).get('severity')}**",
        f"- Weak-signal risk: **{dataset_profile.get('weak_signal_risk')}**",
        "",
        "### Feature breakdown",
        f"- Numeric ({len(numeric)}): {', '.join(numeric[:12]) or '(none)'}",
        f"- Categorical ({len(categorical)}): {', '.join(categorical[:12]) or '(none)'}",
        f"- High-cardinality categorical columns: {', '.join(dataset_profile.get('high_cardinality_features', [])) or '(none)'}",
        "",
        "### Notes",
        notes_text,
        "",
        "## 2. Planning",
        "The planner generated the following dependency-aware steps:",
        plan_text,
        "",
        "### Active strategy flags",
        "```json",
        json.dumps(flags, indent=2),
        "```",
        "",
        "### Memory guidance",
        "```json",
        memory_text,
        "```",
        "",
        "## 3. Model Results",
        f"**Best model:** `{best.get('model')}`  ",
        f"**Primary ranking metric:** `{eval_payload.get('primary_metric')}`",
        "",
        f"- Accuracy: **{best.get('accuracy'):.3f}**",
        f"- Balanced accuracy: **{best.get('balanced_accuracy'):.3f}**",
        f"- Macro F1: **{best.get('f1_macro'):.3f}**",
        f"- Macro precision: **{best.get('precision_macro'):.3f}**",
        f"- Macro recall: **{best.get('recall_macro'):.3f}**",
        f"- Train balanced accuracy: **{best.get('train_balanced_accuracy'):.3f}**",
        f"- CV balanced accuracy mean: **{best.get('cv_balanced_accuracy_mean')}**",
        f"- CV balanced accuracy std: **{best.get('cv_balanced_accuracy_std')}**",
        f"- Decision threshold: **{best.get('decision_threshold')}**",
        "",
        "### Ranked candidate summary",
        "| Model | Accuracy | Balanced Acc. | Macro F1 | CV Balanced Acc. Mean |",
        "|---|---:|---:|---:|---:|",
        top_rows_text,
        "",
        "### Selection rationale",
        "```json",
        json.dumps(selection, indent=2),
        "```",
        "",
        "### Classification report",
        "```text",
        eval_payload.get("classification_report", ""),
        "```",
        "",
        "## 4. Reflection",
        f"**Status:** `{reflection.get('status')}`  ",
        f"**Confidence:** `{reflection.get('confidence')}`",
        "",
        "### Issues",
        issues_text,
        "",
        "### Root causes",
        root_causes_text,
        "",
        "### Prioritised suggestions",
        actions_text,
        "",
        "### Diagnostics",
        "```json",
        json.dumps(reflection.get("diagnostics", {}), indent=2),
        "```",
        "",
        "## 5. Artefacts",
        "- `eda_summary.json`",
        "- `plan.json`",
        "- `metrics.json`",
        "- `reflection.json`",
        "- `selection_rationale.json`",
        "- `attempt_history.json`",
        "- `execution_log.json`",
        "- `confusion_matrix.png`",
        "- `report.md`",
        "",
        "## 6. AI Assistance Disclosure",
        "This project implementation was developed with AI assistance for drafting, code review, and documentation support. Final selection, understanding, testing, and submission decisions remain the student's responsibility.",
        "",
    ]

    report_text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)