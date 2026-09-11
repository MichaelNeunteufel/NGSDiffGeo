import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("simd", [False, True])
def test_covariant_inner_benchmark_smoke(simd):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_covariant_inner.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--depth",
            "1",
            "--maxh",
            "0.8",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--construction-iterations",
            "1",
            "--simd" if simd else "--no-simd",
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
    assert report["simd_requested"] is simd
    assert report["relative_value_error"] < 2e-8
    assert report["input_graph"]["unique_nodes"] > 0
    assert set(report["construction"]) == {"default", "graph"}
    assert set(report["assembly"]) == {"default", "graph"}
    assert report["assembly"]["default"]["simd_active"] is simd
    assert report["assembly"]["graph"]["simd_active"] is simd
    assert report["break_even_assemblies"] is None or report[
        "break_even_assemblies"
    ] >= 0
