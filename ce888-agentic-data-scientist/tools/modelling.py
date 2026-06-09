from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

# Build a basic mixed-type preprocessing pipeline guided by profile flags
def build_preprocessor(profile):
    num_cols = profile["feature_types"]["numeric"]
    cat_cols = profile["feature_types"]["categorical"]
    flags = profile.get("strategy_flags", {})
    rows = int(profile.get("shape", {}).get("rows", 0))
    max_missing = float(profile.get("missingness_summary", {}).get("max_pct", 0.0))

    numeric_imputer = "median" if max_missing >= 15 else "mean"
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if num_cols:
        numeric_steps.append(("scaler", StandardScaler(with_mean=True)))

    ohe_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}
    try:
        ohe_kwargs["sparse_output"] = False
    except Exception:
        ohe_kwargs["sparse"] = False

    if flags.get("group_rare_categories") or profile.get("high_cardinality_features"):
        ohe_kwargs["min_frequency"] = 0.01 if rows >= 1000 else 2
        ohe_kwargs["max_categories"] = 25

    try:
        encoder = OneHotEncoder(**ohe_kwargs)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), num_cols),
            ("cat", Pipeline(steps=categorical_steps), cat_cols),
        ],
        remainder="drop",
    )

# Move memory preferred models earlier and unstable models later
def reorder_by_memory(candidates, profile):
    guidance = profile.get("memory_guidance") or {}
    preferred = [name for name, _ in guidance.get("preferred_models", [])]
    if guidance.get("best_model") and guidance["best_model"] not in preferred:
        preferred.insert(0, guidance["best_model"])
    avoid = set(guidance.get("avoid_models", []))

    priority = {name: idx for idx, name in enumerate(preferred)}

    def sort_key(item):
        name = item[0]
        is_avoid = 1 if name in avoid else 0
        pref_rank = priority.get(name, len(priority) + 10)
        return (is_avoid, pref_rank, name)

    return sorted(candidates, key=sort_key)

# Select a diverse but still explainable model pool based on dataset signals
def select_models(profile, seed = 42):
    rows = int(profile["shape"]["rows"])
    cols = max(int(profile["shape"]["cols"]) - 1, 1)
    imbalance_ratio = float(profile.get("imbalance_ratio") or 1.0)
    dominant = profile.get("dominant_feature_type", "mixed")
    high_dimensional = bool(profile.get("high_dimensional", False))
    high_cardinality = bool(profile.get("high_cardinality_features"))
    num_classes = int(profile.get("num_classes", 0) or 0)
    size_bucket = str(profile.get("size_bucket", "medium"))
    flags = profile.get("strategy_flags", {})

    class_weight = "balanced" if flags.get("use_class_weight") or imbalance_ratio >= 3.0 else None
    prefer_simple = bool(flags.get("prefer_simple_models"))
    include_boosting = bool(flags.get("include_boosting", True)) and bool(flags.get("include_gradient_boosting", True))
    include_extra_trees = bool(flags.get("include_extra_trees", True))

    candidates = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=2500,
                class_weight=class_weight,
                solver="lbfgs",
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=250 if rows >= 10000 else 350,
                max_depth=8 if prefer_simple else None,
                min_samples_leaf=2 if rows < 1000 else 1,
                random_state=seed,
                n_jobs=-1,
                class_weight=class_weight,
            ),
        ),
    ]

    if include_extra_trees or high_dimensional:
        candidates.append(
            (
                "ExtraTrees",
                ExtraTreesClassifier(
                    n_estimators=300 if rows >= 5000 else 400,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight=class_weight,
                    max_depth=10 if prefer_simple else None,
                ),
            )
        )

    if include_boosting and rows >= 200 and dominant != "categorical":
        candidates.append(
            (
                "GradientBoosting",
                GradientBoostingClassifier(
                    learning_rate=0.05 if prefer_simple else 0.1,
                    n_estimators=100 if rows >= 10000 else 150,
                    max_depth=2 if prefer_simple else 3,
                    random_state=seed,
                ),
            )
        )

    if high_dimensional or dominant == "numeric" or size_bucket in {"tiny", "small"}:
        candidates.append(
            (
                "LinearSVC",
                LinearSVC(
                    C=0.5 if prefer_simple else 1.0,
                    class_weight=class_weight,
                    max_iter=5000,
                    dual="auto",
                ),
            )
        )

    if rows <= 6000 and cols <= 40 and dominant == "numeric" and not high_dimensional:
        candidates.append(("KNN", KNeighborsClassifier(n_neighbors=7 if rows > 1000 else 5)))

    if (
        flags.get("include_kernel_model")
        and rows <= 8000
        and cols <= 80
        and not high_cardinality
        and num_classes <= 5
    ):
        candidates.append(
            (
                "SVC_RBF",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight=class_weight,
                    gamma="scale",
                    C=1.0,
                ),
            )
        )

    if flags.get("expand_model_diversity") and dominant != "categorical" and rows <= 12000 and cols <= 100:
        candidates.append(
            (
                "SVC_LinearProb",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight=class_weight,
                    C=0.8,
                ),
            )
        )

    deduplicated: List[Tuple[str, Any]] = []
    seen = set()
    for name, model in candidates:
        if name not in seen:
            deduplicated.append((name, model))
            seen.add(name)

    return reorder_by_memory(deduplicated, profile)

# Return True when the wrapped estimator exposes scores for threshold tuning
def supports_probabilities(pipe):
    model = pipe.named_steps["model"]
    return hasattr(model, "predict_proba") or hasattr(model, "decision_function")

# Wrap a fitted pipeline and apply a custom binary decision threshold
class ThresholdedPipeline:
    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X):
        if hasattr(self.pipeline, "predict_proba"):
            scores = self.pipeline.predict_proba(X)[:, 1]
        else:
            raw = self.pipeline.decision_function(X)
            raw = np.asarray(raw).reshape(-1)
            scores = 1.0 / (1.0 + np.exp(-raw))
        return (scores >= self.threshold).astype(int)

# Choose a threshold that maximises balanced accuracy on a validation split
def binary_validation_threshold(pipe, X_valid, y_valid):
    if hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(X_valid)[:, 1]
    else:
        raw = pipe.decision_function(X_valid)
        raw = np.asarray(raw).reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-raw))

    thresholds = np.linspace(0.2, 0.8, 13)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_valid, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold

# Fit one candidate model, evaluate it and optionally compute CV metrics
def train_single_model( name, model, preprocessor, X_train, y_train, X_test, y_test, profile, seed):
    flags = profile.get("strategy_flags", {})
    binary_target = int(profile.get("num_classes", 0) or 0) == 2

    threshold = None
    train_X_for_fit = X_train
    train_y_for_fit = y_train

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clone(model))])

    if flags.get("tune_decision_threshold") and binary_target and supports_probabilities(pipe):
        stratify = y_train if y_train.value_counts().min() >= 2 else None
        X_sub, X_valid, y_sub, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )
        threshold_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clone(model))])
        threshold_pipe.fit(X_sub, y_sub)
        threshold = binary_validation_threshold(threshold_pipe, X_valid, y_valid)

    pipe.fit(train_X_for_fit, train_y_for_fit)

    if threshold is not None and binary_target and supports_probabilities(pipe):
        tuned_pipe = ThresholdedPipeline(pipe, threshold)
        y_pred = tuned_pipe.predict(X_test)
        y_train_pred = tuned_pipe.predict(X_train)
    else:
        y_pred = pipe.predict(X_test)
        y_train_pred = pipe.predict(X_train)

    metrics: Dict[str, Any] = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "train_f1_macro": float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)),
        "decision_threshold": threshold,
    }

    if flags.get("enable_cv") or int(profile["shape"]["rows"]) < 4000:
        n_splits = 3
        min_class_size = int(y_train.value_counts().min()) if y_train.nunique() > 1 else 0
        if min_class_size >= n_splits:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            cv_payload = cross_validate(
                Pipeline(steps=[("preprocess", preprocessor), ("model", clone(model))]),
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "balanced_accuracy": "balanced_accuracy",
                    "f1_macro": "f1_macro",
                    "accuracy": "accuracy",
                },
                n_jobs=None,
                error_score="raise",
            )
            metrics["cv_balanced_accuracy_mean"] = float(np.mean(cv_payload["test_balanced_accuracy"]))
            metrics["cv_balanced_accuracy_std"] = float(np.std(cv_payload["test_balanced_accuracy"]))
            metrics["cv_f1_macro_mean"] = float(np.mean(cv_payload["test_f1_macro"]))
            metrics["cv_accuracy_mean"] = float(np.mean(cv_payload["test_accuracy"]))
            metrics["cv_balanced_accuracy_scores"] = [float(x) for x in cv_payload["test_balanced_accuracy"].tolist()]
        else:
            metrics["cv_balanced_accuracy_mean"] = None
            metrics["cv_balanced_accuracy_std"] = None
            metrics["cv_f1_macro_mean"] = None
            metrics["cv_accuracy_mean"] = None
            metrics["cv_balanced_accuracy_scores"] = []
    else:
        metrics["cv_balanced_accuracy_mean"] = None
        metrics["cv_balanced_accuracy_std"] = None
        metrics["cv_f1_macro_mean"] = None
        metrics["cv_accuracy_mean"] = None
        metrics["cv_balanced_accuracy_scores"] = []

    return {
        "name": name,
        "pipeline": pipe,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": np.asarray(y_pred),
        "error": None,
    }

# Train all candidate models and rank them using the profile aware metric
def train_models( df, target, preprocessor, candidates, profile, seed, test_size, output_dir, verbose = True):
    del output_dir

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique(dropna=True) < 2:
        raise ValueError("The target must contain at least two classes after dropping missing values.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for name, model in candidates:
        try:
            if verbose:
                print(f"[Modelling] Training: {name}")
            result = train_single_model(
                name=name,
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                profile=profile,
                seed=seed,
            )
            if result is not None:
                results.append(result)
        except Exception as exc:
            failures.append({"model": name, "error": str(exc)})
            if verbose:
                print(f"[Modelling] Skipping {name} due to error: {exc}")

    if not results:
        raise RuntimeError(f"All models failed. Errors: {failures}")

    primary_metric = "balanced_accuracy" if float(profile.get("imbalance_ratio") or 1.0) >= 2.5 else "accuracy"
    compare_on_cv = bool(profile.get("strategy_flags", {}).get("compare_on_cv", False))

    def ranking_key(result: Dict[str, Any]) -> Tuple[float, float, float]:
        metrics = result["metrics"]
        cv_mean = metrics.get("cv_balanced_accuracy_mean")
        cv_value = -1.0 if cv_mean is None else float(cv_mean)
        primary = float(metrics.get(primary_metric, 0.0))
        f1_macro = float(metrics.get("f1_macro", 0.0))
        if compare_on_cv and cv_mean is not None:
            return (cv_value, primary, f1_macro)
        return (primary, f1_macro, cv_value)

    results.sort(key=ranking_key, reverse=True)

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [result["metrics"] for result in results],
        "failures": failures,
        "primary_metric": primary_metric,
        "split_info": {
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "stratified": stratify is not None,
        },
    }