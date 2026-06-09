import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from agentic_data_scientist import AgenticDataScientist


def _make_mixed_dataset(path):
    X, y = make_classification(
        n_samples=260,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(6)])
    df["cat_a"] = np.where(df["num_0"] > 0, "high", "low")
    df["cat_b"] = np.where(df["num_1"] > 0.5, "group1", "group2")
    df.loc[df.sample(frac=0.08, random_state=1).index, "cat_b"] = np.nan
    df["target"] = y
    df.to_csv(path, index=False)


def test_agent_runs_end_to_end_and_creates_outputs(tmp_path):
    data_path = tmp_path / "synthetic.csv"
    _make_mixed_dataset(data_path)

    memory_path = tmp_path / "memory.json"
    output_root = tmp_path / "outputs"
    agent = AgenticDataScientist(memory_path=str(memory_path), verbose=False)
    out_dir = agent.run(
        data_path=str(data_path),
        target="target",
        output_root=str(output_root),
        seed=42,
        test_size=0.2,
        max_replans=1,
    )

    expected_files = [
        "report.md",
        "eda_summary.json",
        "plan.json",
        "metrics.json",
        "reflection.json",
        "confusion_matrix.png",
        "execution_log.json",
    ]
    for filename in expected_files:
        assert os.path.exists(os.path.join(out_dir, filename)), filename

    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert "best_metrics" in metrics
    assert metrics["best_metrics"]["model"]

    with open(memory_path, "r", encoding="utf-8") as handle:
        memory = json.load(handle)
    assert memory["dataset_records"]