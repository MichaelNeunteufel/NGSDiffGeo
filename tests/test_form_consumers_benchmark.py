"""Smoke-test compact form-consumer attribution without timing thresholds."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_EXPECTED_PAIRS = {
    "core": {
        "k_component",
        "double_component",
        "k_generic_ip",
        "double_generic_ip",
        "combined_generic_ip",
        "k_metric_ip",
        "double_metric_ip",
        "combined_metric_ip",
        "slot_ip",
    },
    "differentiated": {
        "k_metric_ip_diff_input",
        "double_metric_ip_diff_input",
        "k_metric_ip_diff_metric",
        "double_metric_ip_diff_metric",
    },
    "geometry": {
        "trace_once",
        "trace_full",
        "curved_slot_ip",
        "k_hodge",
        "double_hodge",
    },
}


@pytest.mark.parametrize("backend", ("scalar", "simd"))
@pytest.mark.parametrize("group", tuple(_EXPECTED_PAIRS))
def test_form_consumers_benchmark(backend, group):
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_form_consumers.py"
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
            "--group",
            group,
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
    assert report["benchmark"] == "compact_form_consumer_attribution"
    assert report["backend"] == backend
    assert report["group"] == group
    cache_breakdown = report[
        "cache_container_storage_breakdown_bytes"
    ]
    assert sum(cache_breakdown.values()) == report[
        "cache_container_storage_bytes"
    ]
    cache_storage = report["cache_storage_lower_bound"]
    assert set(cache_storage) >= {
        "induced_metric_tables",
        "double_trace_tables",
        "hodge_map_tables",
    }
    assert cache_storage["total_bytes"] == sum(
        item["bytes"]
        for family in (
            cache_storage["induced_metric_tables"],
            cache_storage["double_trace_tables"],
            cache_storage["hodge_map_tables"],
        )
        for item in family
    )
    assert set(report["results"]["variants"]) == set(report["variants"])
    assert set(report["results"]["compact_over_dense"]) == _EXPECTED_PAIRS[group]
    for result in report["results"]["variants"].values():
        assert result["phases"]["total"]["median_seconds"] >= 0
        assert result["reuse_assembly"]["median_seconds"] >= 0
        assert result["simd_active"] is (backend == "simd")
    assert all(
        pair["relative_value_error"] < report["tolerance"]
        for pair in report["results"]["compact_over_dense"].values()
    )
    if group in {"differentiated", "geometry"}:
        for pair_name in _EXPECTED_PAIRS[group]:
            if pair_name == "k_hodge":
                expected_type = "CompactWedgeIndependentCoefficientFunction"
            elif pair_name == "double_hodge":
                expected_type = "InducedFormMetricCoefficientFunction"
            elif group == "geometry" and pair_name in {
                "trace_once",
                "trace_full",
                "curved_slot_ip",
            }:
                expected_type = (
                    "CompactDoubleTraceIndependentCoefficientFunction"
                )
            elif group == "differentiated" and pair_name.startswith(
                "double_"
            ):
                expected_type = "CompactWedgeIndependentCoefficientFunction"
            elif pair_name.startswith("k_"):
                expected_type = "CompactKFormOperationCoefficientFunction"
            else:
                expected_type = "CompactDoubleFormOperationCoefficientFunction"
            compact_types = report["results"]["variants"][
                f"compact_{pair_name}"
            ]["coefficient_tree"]["types"]
            dense_types = report["results"]["variants"][
                f"dense_{pair_name}"
            ]["coefficient_tree"]["types"]
            assert any(expected_type in name for name in compact_types)
            if not (
                group == "geometry"
                and pair_name in {
                    "trace_once",
                    "trace_full",
                    "curved_slot_ip",
                }
            ):
                assert not any(expected_type in name for name in dense_types)
            if group == "differentiated" and pair_name.endswith(
                "_diff_metric"
            ):
                compact_visits = report["results"]["variants"][
                    f"compact_{pair_name}"
                ]["coefficient_tree"]["visits"]
                dense_visits = report["results"]["variants"][
                    f"dense_{pair_name}"
                ]["coefficient_tree"]["visits"]
                assert compact_visits < dense_visits
            if group == "geometry" and pair_name in {
                "trace_once",
                "trace_full",
                "curved_slot_ip",
            }:
                compact_count = sum(
                    count
                    for name, count in compact_types.items()
                    if expected_type in name
                )
                dense_count = sum(
                    count
                    for name, count in dense_types.items()
                    if expected_type in name
                )
                assert compact_count > dense_count
    assert report["results"]["retained_pipeline_memory"] is None


def test_form_consumers_memory_worker():
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_form_consumers.py"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--memory-worker",
            "compact_double_metric_ip",
            "--retained-pipelines",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["variant"] == "compact_double_metric_ip"
    assert report["retained_pipelines"] == 1
    assert report["peak_rss_delta_bytes"] >= 0
