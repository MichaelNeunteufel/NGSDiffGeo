"""Smoke-test the benchmark interface without imposing timing thresholds."""

import json
from pathlib import Path
import subprocess
import sys


def test_tensor_algebra_benchmark_smoke(tmp_path):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_tensor_algebra.py"
    )
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--label",
            "smoke",
            "--maxh",
            "0.8",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--construction-iterations",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["label"] == "smoke"
    assert {
        "raise_rank1",
        "raise_rank2_axis0",
        "raise_rank2_axis1",
        "lower_rank1",
        "lower_rank2_axis1",
        "inner_product_rank1",
        "trace_rank2",
        "raise_rank3_axis1",
        "lower_rank3_axis2",
        "inner_product_rank3",
        "trace_rank4_axes1_3",
        "kform_wedge_degree1",
        "kform_hodge_degree1",
        "kform_inner_product_degree2",
        "double_form_slot_inner_22",
    } == set(report["results"])
    for result in report["results"].values():
        assert result["construction"]["median_seconds"] >= 0
        assert result["assembly"]["median_seconds"] >= 0
        assert result["tree"]["unique_nodes"] > 0
        assert result["l2_error"] < 2e-10
    assert {"rank1", "rank2_axis0", "rank2_axis1"} == set(
        report["raise_alternatives"]
    )
    for comparison in report["raise_alternatives"].values():
        assert comparison["matrix_product"]["tree"]["unique_nodes"] > 0
        assert comparison["einsum"]["tree"]["unique_nodes"] > 0
        assert comparison["l2_error"] < 2e-10
