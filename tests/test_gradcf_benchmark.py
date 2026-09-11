import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("mode", "expected_simd"),
    [
        ("volume", True),
        ("element-boundary", True),
        ("surface-boundary", False),
    ],
)
def test_gradcf_benchmark_exercises_all_modes(mode, expected_simd):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_gradcf.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--mode",
            mode,
            "--maxh",
            "0.8",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["mode"] == mode
    assert report["results"]["gradcf_simd"]["simd_requested"]
    assert report["results"]["gradcf_simd"]["simd_active"] is expected_simd
    assert all(
        result["relative_value_error"] < 2e-8
        for result in report["results"].values()
    )
