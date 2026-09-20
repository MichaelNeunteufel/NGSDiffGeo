"""Smoke-test every compressed-forms benchmark mode without timing limits."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_HIGH_ORDER_WORKLOADS = (
    ("kform_wedge_4_4d", (4,), 4**4, 1),
    ("double_form_wedge_31_4d", (3, 1), 4**4, 16),
    ("double_form_wedge_33_4d", (3, 3), 4**6, 16),
    ("double_form_wedge_44_4d", (4, 4), 4**8, 1),
)


def test_compact_consumer_lookup_table_storage_is_reported():
    from ngsdiffgeo import ngsdiffgeo as cpp

    assert cpp._InducedFormMetricTableStorageBytesLowerBound(3, 2) > 0
    assert cpp._CompactHodgeMapTableStorageBytesLowerBound(3, 2) > 0
    assert cpp._CompactDoubleTraceTableStorageBytesLowerBound(3, 2, 2) > 0


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
            "--construction-warmup",
            "0",
            "--construction-iterations",
            "1",
            "--compile-iterations",
            "1",
            "--compile-warmup",
            "0",
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
    assert report["construction_path"] == "both"
    assert report["cache_state"] == "warm"
    assert report["threads"] == 1
    assert report["construction_warmup"] == 0
    assert report["compile_warmup"] == 0
    assert report["retained_graphs"] == 0
    assert report["retained_compact_graphs"] == 0
    assert report["retained_dense_graphs"] == 0
    assert report["cache_container_storage_bytes"] > 0
    cache_breakdown = report["cache_container_storage_breakdown_bytes"]
    assert set(cache_breakdown) == {
        "form_basis",
        "double_form_expansion",
        "double_trace",
        "wedge",
        "exterior_derivative",
        "induced_metric",
        "hodge_map",
    }
    assert all(value > 0 for value in cache_breakdown.values())
    assert sum(cache_breakdown.values()) == report["cache_container_storage_bytes"]
    assert set(report["measurement_boundaries"]) == {
        "construction",
        "compile",
        "evaluation",
        "retained_graph_memory",
        "cache_storage_lower_bound",
    }
    assert len(report["ngsdiffgeo_extension"]["sha256"]) == 64
    assert len(report["ngsdiffgeo_wrapper"]["sha256"]) == 64
    assert len(report["benchmark_script"]["sha256"]) == 64
    assert len(report["repository"]["revision"]) == 40
    assert len(report["repository"]["tracked_diff_sha256"]) == 64
    source_manifest = report["repository"]["source_manifest"]
    assert len(source_manifest["sha256"]) == 64
    assert source_manifest["file_count"] == len(source_manifest["files"])
    assert "src/kforms_internal.hpp" in source_manifest["files"]
    assert "src/kforms/kforms_compact_algebra.inc" in source_manifest["files"]
    assert "benchmarks/benchmark_common.py" in source_manifest["files"]
    assert isinstance(source_manifest["has_changes"], bool)
    assert set(report["results"]) == {
        "kform_wedge_d_3d",
        "double_form_wedge_22_3d",
        "double_form_arithmetic_22_3d",
        "double_form_wedge_chain_22_3d",
        "double_form_wedge_22_4d",
        "double_form_wedge_chain_22_4d",
    }

    for result in report["results"].values():
        assert result["construction"]["median_seconds"] >= 0
        assert result["dense_construction"]["median_seconds"] >= 0
        assert result["public_construction"]["median_seconds"] >= 0
        assert result["native_construction"]["median_seconds"] >= 0
        assert len(result["construction_execution_order"]) == 1
        assert set(result["construction_execution_order"][0]) == {
            "public",
            "native",
            "dense",
        }
        assert result["construction_speedup"] >= 0
        assert result["wrapper_tree"]["unique_nodes"] > 0
        assert result["dense_wrapper_tree"]["unique_nodes"] > 0
        assert result["mathematical_reference_tree"]["unique_nodes"] > 0
        cache_storage = result["cache_storage_lower_bound"]
        assert cache_storage["kind"] == "owned_storage_lower_bound"
        assert cache_storage["total_bytes"] > 0
        assert cache_storage["form_basis"]
        assert cache_storage["wedge_tables"]
        for statistics in (
            result["construction"],
            result["dense_construction"],
        ):
            assert statistics["median_absolute_deviation_seconds"] >= 0
            assert statistics["relative_median_absolute_deviation"] >= 0
        assert result["full_components"] >= result["independent_components"]
        assert result["l2_error"] < report["tolerance"]
        assert result["dense_l2_error"] < report["tolerance"]
        assert result["compact_dense_l2_error"] < report["tolerance"]
        assert result["retained_graph_memory"] is None

        if result["physical_evaluation"] and expected_backends:
            assert result["compile"]["median_seconds"] >= 0
            assert result["dense_compile"]["median_seconds"] >= 0
            assert len(result["compile_execution_order"]) == 1
            assert set(result["compile_execution_order"][0]) == {
                "compact",
                "dense",
            }
            assert result["compile_speedup"] >= 0
            assert result["compiled_tree"]["unique_nodes"] > 0
            assert result["dense_compiled_tree"]["unique_nodes"] > 0
            assert set(result["evaluation"]) == expected_backends
            for backend, variants in result["evaluation"].items():
                assert set(variants) == {
                    "interpreted",
                    "compiled",
                    "dense_interpreted",
                    "dense_compiled",
                }
                assert variants["interpreted"]["median_seconds"] >= 0
                assert variants["compiled"]["median_seconds"] >= 0
                for variant, statistics in variants.items():
                    assert statistics["median_seconds"] >= 0
                    assert statistics["simd_requested"] is (backend == "simd")
                    assert statistics["simd_active"] is (backend == "simd")
                    if variant != "interpreted":
                        assert (
                            statistics["relative_value_error"]
                            < report["tolerance"]
                        )
                assert set(result["compact_speedup_over_dense"][backend]) == {
                    "interpreted",
                    "compiled",
                }
                assert all(
                    speedup >= 0
                    for speedup in result["compact_speedup_over_dense"][
                        backend
                    ].values()
                )
            assert all(
                error < report["tolerance"]
                for error in result["scalar_simd_relative_value_error"].values()
            )
        else:
            assert result["compile"] is None
            assert result["dense_compile"] is None
            assert result["compile_execution_order"] == []
            assert result["evaluation"] == {}
            assert result["compact_speedup_over_dense"] == {}


@pytest.mark.parametrize(
    ("construction_path", "cache_state"),
    [("native", "warm"), ("public", "cold")],
)
def test_compressed_forms_filtered_construction_paths(
    construction_path, cache_state
):
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
            "filtered-smoke",
            "--mode",
            "construction",
            "--workload",
            "double_form_wedge_22_3d",
            "--construction-path",
            construction_path,
            "--cache-state",
            cache_state,
            "--maxh",
            "1.0",
            "--construction-warmup",
            "0",
            "--construction-iterations",
            "1",
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
    assert report["workloads"] == ["double_form_wedge_22_3d"]
    assert report["construction_path"] == construction_path
    assert report["cache_state"] == cache_state
    result = report["results"]["double_form_wedge_22_3d"]
    assert result["construction"]["median_seconds"] >= 0
    assert result["dense_construction"]["median_seconds"] >= 0
    selected = {construction_path, "dense"}
    assert set(result["construction_execution_order"][0]) == selected
    if construction_path == "public":
        assert result["public_construction"] is not None
        assert result["native_construction"] is None
    else:
        assert result["public_construction"] is None
        assert result["native_construction"] is not None


def test_compressed_forms_retained_graph_memory_worker():
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
            "--memory-worker",
            "kform_wedge_d_3d",
            "compact",
            "--retained-graphs",
            "2",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )

    assert process.returncode == 0, process.stderr
    sample = json.loads(process.stdout)
    assert sample["supported"]
    assert sample["retained_graphs"] == 2
    assert sample["peak_rss_after_bytes"] >= sample["peak_rss_before_bytes"]
    assert sample["peak_rss_delta_bytes"] >= 0
    assert sample["estimated_incremental_bytes_per_graph"] >= 0


@pytest.mark.parametrize(
    ("workload", "degrees", "full_components", "independent_components"),
    _HIGH_ORDER_WORKLOADS,
)
def test_compressed_forms_high_order_construction_smoke(
    workload, degrees, full_components, independent_components
):
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
            "high-order-smoke",
            "--mode",
            "construction",
            "--workload",
            workload,
            "--construction-path",
            "both",
            "--maxh",
            "1.0",
            "--construction-warmup",
            "0",
            "--construction-iterations",
            "1",
            "--threads",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=180,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["workloads"] == [workload]
    result = report["results"][workload]
    assert tuple(result["form_degrees"]) == degrees
    assert result["full_components"] == full_components
    assert result["independent_components"] == independent_components
    assert result["l2_error"] < report["tolerance"]
    assert result["dense_l2_error"] < report["tolerance"]
    assert result["compact_dense_l2_error"] < report["tolerance"]
