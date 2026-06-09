from agents.memory import JSONMemory


def test_memory_persistence_and_exact_lookup(tmp_path):
    memory_path = tmp_path / "memory.json"
    memory = JSONMemory(str(memory_path))
    memory.upsert_dataset_record(
        "fp_123",
        {
            "profile_signature": {
                "size_bucket": "small",
                "rows": 1000,
                "cols": 20,
                "numeric_ratio": 0.6,
                "categorical_ratio": 0.4,
                "missing_severity": "low",
                "overall_missing": 1.0,
                "imbalance_ratio": 2.0,
                "num_classes": 2,
                "high_dimensional": False,
                "high_cardinality_count": 0,
                "dominant_feature_type": "mixed",
            },
            "best_model": "RandomForest",
            "latest_run": {
                "best_model": "RandomForest",
                "best_score": 0.82,
                "strategy_flags": {"use_class_weight": True},
            },
        },
    )
    reloaded = JSONMemory(str(memory_path))
    record = reloaded.get_dataset_record("fp_123")
    assert record is not None
    assert record["best_model"] == "RandomForest"
    assert record["successful_models"]["RandomForest"] == 1


def test_memory_similarity_lookup(tmp_path):
    memory = JSONMemory(str(tmp_path / "memory.json"))
    memory.upsert_dataset_record(
        "fp_a",
        {
            "profile_signature": {
                "size_bucket": "medium",
                "rows": 12000,
                "cols": 18,
                "numeric_ratio": 0.4,
                "categorical_ratio": 0.6,
                "missing_severity": "medium",
                "overall_missing": 5.0,
                "imbalance_ratio": 4.0,
                "num_classes": 2,
                "high_dimensional": False,
                "high_cardinality_count": 2,
                "dominant_feature_type": "mixed",
            },
            "best_model": "RandomForest",
            "latest_run": {
                "best_model": "RandomForest",
                "best_score": 0.79,
                "strategy_flags": {"use_class_weight": True, "group_rare_categories": True},
            },
        },
    )
    hint = memory.get_similar_dataset_hint(
        {
            "size_bucket": "medium",
            "shape": {"rows": 10000, "cols": 16},
            "feature_balance": {"numeric_ratio": 0.5, "categorical_ratio": 0.5},
            "missingness_summary": {"severity": "medium", "overall_pct": 6.0},
            "imbalance_ratio": 3.8,
            "num_classes": 2,
            "high_dimensional": False,
            "high_cardinality_features": ["job", "education"],
            "dominant_feature_type": "mixed",
        }
    )
    assert hint is not None
    assert hint["best_model"] == "RandomForest"
    assert hint["preferred_models"][0][0] == "RandomForest"