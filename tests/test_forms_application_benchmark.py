"""Smoke tests for the application-style compressed-form benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("backend", "retained_pipelines"),
    [("scalar", 0), ("simd", 2)],
)
def test_forms_application_benchmark(backend, retained_pipelines):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_forms_application.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--label",
            "smoke",
            "--backend",
            backend,
            "--maxh",
            "1.0",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--reuse-warmup",
            "0",
            "--reuse-iterations",
            "1",
            "--retained-pipelines",
            str(retained_pipelines),
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=120,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["benchmark"] == "compressed_forms_application"
    assert report["label"] == "smoke"
    assert report["backend"] == backend
    assert report["threads"] == 1
    assert report["retained_pipelines"] == retained_pipelines
    assert len(report["ngsdiffgeo_extension"]["sha256"]) == 64
    assert len(report["ngsdiffgeo_wrapper"]["sha256"]) == 64
    assert len(report["benchmark_script"]["sha256"]) == 64
    assert len(report["repository"]["revision"]) == 40
    assert len(report["repository"]["tracked_diff_sha256"]) == 64

    result = report["results"]["mixed_forms_energy_3d"]
    assert len(result["pipeline_execution_order"]) == 1
    assert len(result["reuse_execution_order"]) == 1
    assert set(result["pipeline_execution_order"][0]) == {
        "public",
        "native",
        "dense",
    }
    for name, variant in result["variants"].items():
        assert set(variant["phases"]) == {
            "construction",
            "compile",
            "form_setup",
            "assembly",
            "total",
        }
        assert all(
            phase["median_seconds"] >= 0
            for phase in variant["phases"].values()
        )
        assert variant["reuse_assembly"]["median_seconds"] >= 0
        assert variant["relative_value_error"] < report["tolerance"]
        assert variant["coefficient_tree"]["unique_nodes"] > 0
        assert variant["compiled_tree"]["unique_nodes"] > 0
        assert variant["simd_requested"] is (backend == "simd")
        assert variant["simd_active"] is (backend == "simd")
        if name in {"public", "native"}:
            assert set(result["compact_over_dense"][name]) == {
                "construction",
                "compile",
                "form_setup",
                "assembly",
                "total",
                "reuse_assembly",
            }

    memory = result["retained_pipeline_memory"]
    if retained_pipelines:
        assert set(memory) == {
            "public",
            "native",
            "dense",
            "dense_over_public_estimate",
            "dense_over_native_estimate",
        }
        for name in ("public", "native", "dense"):
            assert memory[name]["supported"]
            assert memory[name]["retained_pipelines"] == retained_pipelines
            assert memory[name]["peak_rss_delta_bytes"] >= 0
    else:
        assert memory is None
