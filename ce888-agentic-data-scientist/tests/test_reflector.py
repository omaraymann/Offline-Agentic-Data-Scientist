from agents.reflector import apply_replan_strategy, reflect, should_replan


def test_reflector_recommends_replan_for_weak_signal():
    dataset_profile = {
        "shape": {"rows": 2000, "cols": 15},
        "imbalance_ratio": 4.0,
        "high_cardinality_features": ["occupation"],
        "missingness_summary": {"overall_pct": 15.0, "severity": "high"},
        "weak_signal_risk": True,
        "num_classes": 2,
    }
    evaluation = {
        "best_metrics": {
            "model": "RandomForest",
            "balanced_accuracy": 0.56,
            "f1_macro": 0.49,
            "accuracy": 0.84,
            "train_balanced_accuracy": 0.88,
            "cv_balanced_accuracy_mean": 0.55,
            "cv_balanced_accuracy_std": 0.06,
            "cv_balanced_accuracy_scores": [0.54, 0.57, 0.54],
        },
        "classification_report_dict": {
            "0": {"recall": 0.95},
            "1": {"recall": 0.20},
            "accuracy": 0.84,
            "macro avg": {"recall": 0.575},
        },
    }
    all_metrics = [
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.50, "f1_macro": 0.45},
        {
            "model": "RandomForest",
            "balanced_accuracy": 0.56,
            "f1_macro": 0.49,
            "cv_balanced_accuracy_scores": [0.54, 0.57, 0.54],
        },
        {"model": "LogisticRegression", "balanced_accuracy": 0.55, "f1_macro": 0.48},
    ]
    reflection = reflect(dataset_profile, evaluation, all_metrics)
    assert reflection["status"] == "needs_attention"
    assert reflection["replan_recommended"] is True
    assert "imbalance_handling" in reflection["root_causes"]
    assert should_replan(reflection, current_replans=0, max_replans=1) is True


def test_apply_replan_strategy_changes_flags_and_plan():
    plan = ["profile_dataset", "select_models", "train_candidate_models", "reflect"]
    dataset_profile = {
        "num_classes": 2,
        "notes": [],
        "strategy_flags": {
            "use_class_weight": False,
            "group_rare_categories": False,
            "enable_cv": False,
            "compare_on_cv": False,
            "tune_decision_threshold": False,
        },
    }
    reflection = {"root_causes": ["imbalance_handling", "preprocessing_mismatch", "weak_features"]}
    new_plan, new_profile = apply_replan_strategy(plan, dataset_profile, reflection)
    flags = new_profile["strategy_flags"]
    assert flags["use_class_weight"] is True
    assert flags["group_rare_categories"] is True
    assert flags["enable_cv"] is True
    assert flags["tune_decision_threshold"] is True
    assert "apply_replanned_strategy" in new_plan
    assert "expand_model_pool" in new_plan