import os
import subprocess
import sys


def test_smoke_run_creates_outputs(tmp_path):
    out_root = tmp_path / "agt_outputs"
    cmd = [
        sys.executable,
        "run_agent.py",
        "--data",
        "data/demo.csv",
        "--target",
        "auto",
        "--output_root",
        str(out_root),
        "--quiet",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"run_agent failed: {res.stderr}"
    out_dir = res.stdout.strip().splitlines()[-1].strip()
    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "metrics.json"))