from agents.planner import create_plan


def test_planner_adds_imbalance_and_high_cardinality_steps():
    profile = {
        "shape": {"rows": 1200, "cols": 18},
        "imbalance_ratio": 5.2,
        "missingness_summary": {"overall_pct": 8.0, "max_pct": 20.0, "severity": "medium"},
        "high_cardinality_features": ["job_title"],
        "high_dimensional": False,
        "dominant_feature_type": "mixed",
        "num_classes": 2,
        "weak_signal_risk": False,
        "target_inference": {"confidence": "high"},
        "strategy_flags": {},
    }
    plan = create_plan(profile)
    assert "activate_imbalance_strategy" in plan
    assert "control_high_cardinality" in plan
    assert "audit_missingness" in plan
    assert "tune_decision_thresholds" in plan


def test_planner_adds_tiny_dataset_guards_and_memory_priority():
    profile = {
        "shape": {"rows": 120, "cols": 10},
        "imbalance_ratio": 1.4,
        "missingness_summary": {"overall_pct": 0.0, "max_pct": 0.0, "severity": "low"},
        "high_cardinality_features": [],
        "high_dimensional": False,
        "dominant_feature_type": "numeric",
        "num_classes": 2,
        "weak_signal_risk": True,
        "target_inference": {"confidence": "low"},
        "strategy_flags": {},
    }
    hint = {"best_model": "RandomForest", "preferred_models": [("RandomForest", 2.0)]}
    plan = create_plan(profile, memory_hint=hint)
    assert "guard_against_tiny_sample" in plan
    assert "validate_with_cross_validation" in plan
    assert "confirm_target_assumption" in plan
    assert "prioritise_models_from_memory" in plan
    assert "memory_priority::RandomForest" in plan