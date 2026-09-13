"""Smoke tests for the revision-neutral form benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ("construction", "compile", "application", "all"))
def test_revision_forms_benchmark_modes(mode):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_revision_forms.py"
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
            "--workload",
            "kform_wedge_d_3d",
            "--backend",
            "simd",
            "--maxh",
            "1.0",
            "--construction-warmup",
            "0",
            "--construction-iterations",
            "1",
            "--compile-warmup",
            "0",
            "--compile-iterations",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--reuse-warmup",
            "0",
            "--reuse-iterations",
            "1",
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["benchmark"] == "revision_neutral_forms"
    assert report["mode"] == mode
    assert len(report["ngsdiffgeo_extension"]["sha256"]) == 64
    if mode in {"construction", "compile", "all"}:
        result = report["results"]["kform_wedge_d_3d"]
        assert result["construction"]["median_seconds"] >= 0
        assert result["value_norm"] >= 0
        assert (result["compile"] is not None) is (mode in {"compile", "all"})
    else:
        assert report["results"] == {}
    application = report["application"]["result"]
    assert (application is not None) is (mode in {"application", "all"})
    if application is not None:
        assert application["phases"]["total"]["median_seconds"] >= 0
        assert application["reuse_assembly"]["median_seconds"] >= 0
        assert application["simd_active"]


def test_revision_forms_memory_worker():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_revision_forms.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--memory-worker",
            "kform_wedge_4_4d",
            "--retained-graphs",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["supported"]
    assert report["retained_graphs"] == 2
    assert report["peak_rss_delta_bytes"] >= 0


def test_revision_forms_pipeline_memory_worker():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_revision_forms.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--pipeline-memory-worker",
            "--retained-pipelines",
            "2",
            "--backend",
            "scalar",
            "--maxh",
            "1.0",
            "--threads",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["supported"]
    assert report["retained_pipelines"] == 2
    assert report["peak_rss_delta_bytes"] >= 0


def test_revision_forms_realcompile_mode():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_revision_forms.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--label",
            "realcompile-smoke",
            "--mode",
            "compile",
            "--compile-kind",
            "real",
            "--workload",
            "kform_wedge_d_3d",
            "--maxh",
            "1.0",
            "--construction-warmup",
            "0",
            "--construction-iterations",
            "1",
            "--compile-warmup",
            "0",
            "--compile-iterations",
            "1",
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["compile_kind"] == "real"
    result = report["results"]["kform_wedge_d_3d"]["compile"]
    assert result["kind"] == "native_cpp"
    assert result["status"] == "completed"
    assert result["generated_source_bytes"]["median"] > 0
    assert result["generated_source_lines"]["median"] > 0
    assert result["compiler_cache"].startswith("disabled")
    assert result["maximum_relative_value_error"] <= report["tolerance"]
    sample = result["samples"][0]
    assert not sample["values"]["scalar"]["realcompiled_simd_active"]
    assert sample["values"]["simd"]["realcompiled_simd_active"]


def test_revision_forms_realcompile_timeout_is_censored():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_revision_forms.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--mode",
            "compile",
            "--compile-kind",
            "real",
            "--realcompile-timeout",
            "0.001",
            "--workload",
            "kform_wedge_d_3d",
            "--maxh",
            "1.0",
            "--construction-warmup",
            "0",
            "--construction-iterations",
            "1",
            "--compile-warmup",
            "0",
            "--compile-iterations",
            "1",
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    result = report["results"]["kform_wedge_d_3d"]
    assert result["compile"]["status"] == "censored_timeout"
    assert result["compile"]["timed_out_samples"] == 1
    assert result["compiled_tree"] is None
