"""Smoke-test direct form-evaluation benchmark correctness and modes."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_form_direct_evaluation_benchmark():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_form_direct_evaluation.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--label",
            "smoke",
            "--workload",
            "double_form_arithmetic_22_3d",
            "--maxh",
            "1.0",
            "--points",
            "2",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["benchmark"] == "compact_form_direct_evaluation"
    assert report["points"] == 2
    result = report["results"]["double_form_arithmetic_22_3d"]
    assert set(result["timings"]) == set(report["variants"])
    assert max(result["relative_value_errors"].values()) < report["tolerance"]
