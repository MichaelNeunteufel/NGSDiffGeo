"""Smoke tests for the compressed-form dispatch attribution benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_form_dispatch_benchmark_all_operations():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_form_dispatch.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--label",
            "smoke",
            "--maxh",
            "1.0",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--profile-iterations",
            "1",
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["benchmark"] == "compressed_forms_dispatch_attribution"
    assert report["label"] == "smoke"
    assert report["threads"] == 1
    assert report["iterations"] == 1
    assert report["profile_iterations"] == 1
    assert len(report["ngsdiffgeo_extension"]["sha256"]) == 64
    assert len(report["ngsdiffgeo_wrapper"]["sha256"]) == 64
    assert len(report["benchmark_script"]["sha256"]) == 64
    assert len(report["repository"]["revision"]) == 40
    assert len(report["repository"]["tracked_diff_sha256"]) == 64
    assert set(report["results"]) == {
        "kform_wedge_1_1",
        "kform_wedge_2_1",
        "kform_exterior_derivative_2",
        "kform_scale_3",
        "kform_scale_numeric_3",
        "kform_add_3",
        "kform_full",
        "double_form_wedge_10_01",
        "double_form_wedge_11_10",
        "double_form_wedge_21_01",
        "double_form_scale_numeric_11",
        "double_form_chain_full",
    }

    for name, result in report["results"].items():
        assert result["public"]["median_seconds"] >= 0
        assert result["native"]["median_seconds"] >= 0
        assert result["public_over_native"] >= 0
        assert len(result["execution_order"]) == 1
        assert set(result["execution_order"][0]) == {"public", "native"}
        assert result["l2_error"] < report["tolerance"]
        assert result["semantic_metadata_equal"]
        assert result["public_tree"]["unique_nodes"] > 0
        assert result["native_tree"]["unique_nodes"] > 0
        if name in {"kform_full", "double_form_chain_full"}:
            assert result["profile"]["kind"] == "cprofile_diagnostic"
            assert result["profile"]["iterations"] == 1
            assert result["profile"]["functions"]
        else:
            assert result["profile"] is None


def test_form_dispatch_benchmark_operation_filter():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_form_dispatch.py"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--operation",
            "kform_wedge_1_1",
            "--maxh",
            "1.0",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--profile-iterations",
            "0",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["operations"] == ["kform_wedge_1_1"]
    assert set(report["results"]) == {"kform_wedge_1_1"}
