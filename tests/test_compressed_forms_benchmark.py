"""Smoke-test every compressed-forms benchmark mode without timing limits."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("mode", "expected_backends"),
    [
        ("construction", set()),
        ("scalar", {"scalar"}),
        ("simd", {"simd"}),
        ("all", {"scalar", "simd"}),
    ],
)
def test_compressed_forms_benchmark_modes(mode, expected_backends):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_compressed_forms.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--label",
            "smoke",
            "--mode",
            mode,
            "--maxh",
            "1.0",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--construction-iterations",
            "1",
            "--compile-iterations",
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
    assert report["benchmark"] == "compressed_forms_mvp"
    assert report["label"] == "smoke"
    assert report["mode"] == mode
    assert report["threads"] == 1
    assert len(report["ngsdiffgeo_extension"]["sha256"]) == 64
    assert len(report["repository"]["revision"]) == 40
    assert set(report["results"]) == {
        "kform_wedge_d_3d",
        "double_form_wedge_22_3d",
        "double_form_wedge_22_4d",
    }

    for result in report["results"].values():
        assert result["construction"]["median_seconds"] >= 0
        assert result["construction_with_full_output"]["median_seconds"] >= 0
        assert result["wrapper_tree"]["unique_nodes"] > 0
        assert result["full_output_tree"]["unique_nodes"] > 0
        assert result["full_components"] >= result["independent_components"]
        assert result["l2_error"] < report["tolerance"]

        if result["physical_evaluation"] and expected_backends:
            assert result["compile"]["median_seconds"] >= 0
            assert result["compiled_tree"]["unique_nodes"] > 0
            assert set(result["evaluation"]) == expected_backends
            for backend, variants in result["evaluation"].items():
                assert set(variants) == {"interpreted", "compiled"}
                assert variants["interpreted"]["median_seconds"] >= 0
                assert variants["compiled"]["median_seconds"] >= 0
                assert variants["interpreted"]["simd_requested"] is (backend == "simd")
                assert variants["interpreted"]["simd_active"] is (backend == "simd")
                assert variants["compiled"]["simd_active"] is (backend == "simd")
                assert (
                    variants["compiled"]["relative_value_error"] < report["tolerance"]
                )
            assert all(
                error < report["tolerance"]
                for error in result["scalar_simd_relative_value_error"].values()
            )
        else:
            assert result["compile"] is None
            assert result["evaluation"] == {}
