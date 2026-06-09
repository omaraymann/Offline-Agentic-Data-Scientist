from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

COMMON_TARGET_NAMES = [
    "target",
    "label",
    "class",
    "y",
    "outcome",
    "income",
    "subscribed",
    "deposit",
    "diagnosis",
]

# Infer the likely target column using simple, explainable heuristics.
# Priority:
#   1. Common target names
#   2. Low-cardinality / object-like columns
#   3. Slight preference for the last column
def infer_target_column(df, return_metadata = False):
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    metadata: Dict[str, Any] = {
        "method": None,
        "confidence": "low",
        "candidates": [],
    }

    for name in COMMON_TARGET_NAMES:
        if name in lower_map:
            target = lower_map[name]
            metadata.update(
                {
                    "method": "known_name",
                    "confidence": "high",
                    "candidates": [target],
                }
            )
            return (target, metadata) if return_metadata else target

    scored_candidates = []
    n_rows = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        unique = int(series.nunique(dropna=True))
        missing_pct = float(series.isna().mean() * 100.0)
        is_object_like = (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        )
        unique_ratio = unique / n_rows

        score = 0.0
        if is_object_like:
            score += 1.0
        if unique <= max(50, int(0.05 * n_rows)):
            score += 1.0
        if unique_ratio < 0.20:
            score += 0.5
        if missing_pct < 30.0:
            score += 0.25
        if str(column) == str(df.columns[-1]):
            score += 0.5

        scored_candidates.append((score, str(column)))

    scored_candidates.sort(reverse=True)
    metadata["candidates"] = [name for _, name in scored_candidates[:3]]
    best_score, best_column = scored_candidates[0] if scored_candidates else (0.0, None)

    if best_score >= 2.0:
        metadata.update({"method": "heuristic_score", "confidence": "medium"})
    elif best_score >= 1.25:
        metadata.update({"method": "heuristic_score", "confidence": "low"})
    else:
        best_column = None
        metadata.update({"method": "failed", "confidence": "low"})

    return (best_column, metadata) if return_metadata else best_column

# Heuristic check for whether a target looks classification like
def is_classification_target(series):
    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    ):
        return True

    unique = int(series.nunique(dropna=True))
    return unique <= max(20, int(0.05 * max(len(series), 1)))

# Create a stable lightweight fingerprint from shape, columns, dtypes and target
def dataset_fingerprint(df, target):
    base = "|".join(
        [
            f"shape={df.shape[0]}x{df.shape[1]}",
            f"target={target}",
            ",".join(df.columns.astype(str).tolist()),
            ",".join(df.dtypes.astype(str).tolist()),
        ]
    )
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"fp_{digest}"

# Bucket dataset size for planner/model-selection logic
def size_bucket(rows):
    if rows < 300:
        return "tiny"
    if rows < 3000:
        return "small"
    if rows < 30000:
        return "medium"
    return "large"

# Map missingness levels to a simple severity label
def missing_severity(overall_missing, max_missing):
    if overall_missing >= 25 or max_missing >= 60:
        return "severe"
    if overall_missing >= 10 or max_missing >= 35:
        return "high"
    if overall_missing >= 3 or max_missing >= 15:
        return "medium"
    return "low"

# Turn profiler signals into initial strategy flags.
def default_strategy_flags(profile):
    imbalance_ratio = float(profile.get("imbalance_ratio") or 1.0)
    binary_target = int(profile.get("num_classes", 0) or 0) == 2
    size = str(profile.get("size_bucket", "medium"))
    high_cardinality = bool(profile.get("high_cardinality_features"))
    dominant = str(profile.get("dominant_feature_type", "mixed"))
    high_dimensional = bool(profile.get("high_dimensional", False))
    max_missing = float(profile.get("missingness_summary", {}).get("max_pct", 0.0))
    smallest_class = int(profile.get("minority_class_count", 0) or 0)

    enable_cv = size in {"tiny", "small"} or smallest_class < 25
    compare_on_cv = size == "tiny" or smallest_class < 15
    include_kernel_model = size in {"tiny", "small"} and dominant != "categorical" and not high_cardinality
    include_boosting = dominant != "categorical" and size != "large"
    include_gradient_boosting = dominant != "categorical" and size != "large"
    include_extra_trees = True

    return {
        "use_class_weight": imbalance_ratio >= 3.0,
        "group_rare_categories": high_cardinality,
        "enable_cv": enable_cv,
        "compare_on_cv": compare_on_cv,
        "prefer_simple_models": size == "tiny",
        "expand_model_diversity": False,
        "include_kernel_model": include_kernel_model,
        "include_boosting": include_boosting,
        "include_extra_trees": include_extra_trees,
        "include_gradient_boosting": include_gradient_boosting,
        "tune_decision_threshold": binary_target and imbalance_ratio >= 4.0,
        "drop_high_missing_features": max_missing >= 40.0,
        "replan_count": 0,
    }

# Profile a dataset to produce explainable signals for planning
# The profile is intentionally descriptive rather than overly complex so it can be justified clearly in the assignment demo/report.
def profile_dataset(df, target, target_inference = None):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    X = df.drop(columns=[target])

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    categorical_cols = [str(c) for c in X.columns if str(c) not in numeric_cols]

    rows = int(df.shape[0])
    cols_including_target = int(df.shape[1])
    feature_count = max(cols_including_target - 1, 1)

    profile: Dict[str, Any] = {
        "profile_id": dataset_fingerprint(df, target),
        "shape": {"rows": rows, "cols": cols_including_target},
        "columns": df.columns.astype(str).tolist(),
        "target": str(target),
        "target_dtype": str(y.dtype),
        "is_classification": bool(is_classification_target(y)),
        "feature_types": {"numeric": numeric_cols, "categorical": categorical_cols},
        "n_unique_by_col": {str(c): int(df[c].nunique(dropna=True)) for c in df.columns},
        "target_inference": target_inference
        or {"method": "manual", "confidence": "high", "candidates": [target]},
    }

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing_pct.items()}

    overall_missing = float(df.isna().mean().mean() * 100.0)
    max_missing = float(max(profile["missing_pct"].values())) if profile["missing_pct"] else 0.0
    profile["missingness_summary"] = {
        "overall_pct": round(overall_missing, 3),
        "max_pct": round(max_missing, 3),
        "severity": missing_severity(overall_missing, max_missing),
    }

    numeric_ratio = len(numeric_cols) / feature_count
    categorical_ratio = len(categorical_cols) / feature_count

    if numeric_ratio > 0.7:
        dominant_feature_type = "numeric"
    elif categorical_ratio > 0.7:
        dominant_feature_type = "categorical"
    else:
        dominant_feature_type = "mixed"

    high_cardinality_features = [
        str(c)
        for c in categorical_cols
        if X[c].nunique(dropna=True) > min(50, max(20, int(0.1 * max(rows, 1))))
    ]

    feature_to_row_ratio = round(feature_count / max(rows, 1), 4)
    high_dimensional = bool(
        feature_count > max(80, int(np.sqrt(max(rows, 1)) * 6))
        or feature_count > rows
    )

    profile["feature_balance"] = {
        "numeric_ratio": round(numeric_ratio, 3),
        "categorical_ratio": round(categorical_ratio, 3),
    }
    profile["dominant_feature_type"] = dominant_feature_type
    profile["high_cardinality_features"] = high_cardinality_features
    profile["size_bucket"] = size_bucket(rows)
    profile["high_dimensional"] = high_dimensional
    profile["feature_to_row_ratio"] = feature_to_row_ratio
    profile["feature_count"] = feature_count

    notes = []

    if profile["size_bucket"] == "tiny":
        notes.append("Tiny dataset: prefer simpler models and cross-validation-aware decisions.")
    elif profile["size_bucket"] == "small":
        notes.append("Small dataset: use cross-validation to reduce single-split noise.")

    if high_dimensional:
        notes.append("High-dimensional feature space detected: reduce overfitting risk and prefer robust ranking.")

    if high_cardinality_features:
        notes.append("High-cardinality categorical features detected: group rare categories before encoding.")

    if profile["missingness_summary"]["severity"] in {"high", "severe"}:
        notes.append("Substantial missingness detected: compare default imputation against dropping the worst columns.")

    if target_inference and target_inference.get("confidence") == "low":
        notes.append("Automatic target inference is uncertain and should be justified in the demo.")

    if profile["is_classification"]:
        value_counts = y.value_counts(dropna=False)
        non_missing_counts = y.dropna().value_counts()

        profile["class_counts"] = {str(k): int(v) for k, v in value_counts.items()}
        profile["num_classes"] = int(y.nunique(dropna=True))

        if len(non_missing_counts) >= 2:
            imbalance_ratio = float(non_missing_counts.max() / max(non_missing_counts.min(), 1))
            minority_class_count = int(non_missing_counts.min())
            majority_class_count = int(non_missing_counts.max())
        else:
            imbalance_ratio = 1.0
            minority_class_count = int(non_missing_counts.iloc[0]) if len(non_missing_counts) == 1 else 0
            majority_class_count = minority_class_count

        profile["imbalance_ratio"] = round(imbalance_ratio, 3)
        profile["minority_class_count"] = minority_class_count
        profile["majority_class_count"] = majority_class_count
        profile["safe_for_stratified_cv"] = minority_class_count >= 3

        if imbalance_ratio >= 3.0:
            notes.append("Class imbalance detected: rank candidates using balanced accuracy and macro F1.")
        if minority_class_count < 20:
            notes.append("Minority class is small: prefer CV-aware selection and avoid over-trusting a single split.")
    else:
        profile["class_counts"] = None
        profile["num_classes"] = 0
        profile["imbalance_ratio"] = None
        profile["minority_class_count"] = 0
        profile["majority_class_count"] = 0
        profile["safe_for_stratified_cv"] = False
        notes.append("This template is classification-focused; non-classification targets may need extra work.")

    weak_signal_risk = False
    if rows < 500 and feature_count > max(rows // 5, 10):
        weak_signal_risk = True
    if profile["missingness_summary"]["severity"] in {"high", "severe"} and len(high_cardinality_features) > 0:
        weak_signal_risk = True
    if profile["target_inference"].get("confidence") == "low":
        weak_signal_risk = True

    profile["weak_signal_risk"] = weak_signal_risk
    if weak_signal_risk:
        notes.append("Potential weak-signal scenario: use cautious interpretation and stronger baselines.")

    profile["notes"] = notes
    profile["strategy_flags"] = default_strategy_flags(profile)
    return profile