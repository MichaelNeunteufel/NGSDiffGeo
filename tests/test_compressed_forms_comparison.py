"""Smoke tests for the dependency-free benchmark comparison utility."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def _report(construction, evaluation):
    return {
        "benchmark": "compressed_forms_mvp",
        "repository": {"revision": "0" * 40},
        "ngsdiffgeo_extension": {"sha256": "1" * 64},
        "results": {
            "double_form_wedge_22_3d": {
                "construction": {"median_seconds": construction},
                "public_construction": {"median_seconds": construction},
                "native_construction": {
                    "median_seconds": 0.5 * construction
                },
                "dense_construction": {"median_seconds": 0.6},
                "compile": None,
                "dense_compile": None,
                "evaluation": {
                    "scalar": {
                        "interpreted": {"median_seconds": evaluation},
                        "dense_interpreted": {"median_seconds": 2.0},
                    }
                },
                "retained_graph_memory": None,
            }
        },
    }


def _dispatch_report(public, native):
    return {
        "benchmark": "compressed_forms_dispatch_attribution",
        "repository": {"revision": "0" * 40},
        "ngsdiffgeo_extension": {"sha256": "1" * 64},
        "results": {
            "kform_wedge_1_1": {
                "public": {"median_seconds": public},
                "native": {"median_seconds": native},
            }
        },
    }


def _application_report(public_total, dense_total):
    def variant(total):
        return {
            "phases": {
                "construction": {"median_seconds": 0.1 * total},
                "compile": {"median_seconds": 0.2 * total},
                "form_setup": {"median_seconds": 0.1 * total},
                "assembly": {"median_seconds": 0.6 * total},
                "total": {"median_seconds": total},
            },
            "reuse_assembly": {"median_seconds": 0.5 * total},
        }

    return {
        "benchmark": "compressed_forms_application",
        "repository": {"revision": "0" * 40},
        "ngsdiffgeo_extension": {"sha256": "1" * 64},
        "results": {
            "mixed_forms_energy_3d": {
                "variants": {
                    "public": variant(public_total),
                    "native": variant(0.9 * public_total),
                    "dense": variant(dense_total),
                },
                "retained_pipeline_memory": None,
            }
        },
    }


def test_compressed_forms_comparison_cli(tmp_path):
    utility = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "compare_compressed_forms.py"
    )
    baseline_paths = []
    candidate_paths = []
    for index, value in enumerate((1.0, 1.1, 0.9)):
        path = tmp_path / f"baseline-{index}.json"
        path.write_text(json.dumps(_report(value, 1.0)), encoding="utf-8")
        baseline_paths.append(path)
    for index, value in enumerate((0.8, 0.88, 0.72)):
        path = tmp_path / f"candidate-{index}.json"
        path.write_text(json.dumps(_report(value, 1.03)), encoding="utf-8")
        candidate_paths.append(path)

    command = [sys.executable, str(utility)]
    for path in baseline_paths:
        command.extend(("--baseline", str(path)))
    for path in candidate_paths:
        command.extend(("--candidate", str(path)))
    command.extend(
        (
            "--require-improvement",
            "double_form_wedge_22_3d.construction=15",
            "--limit-regression",
            "double_form_wedge_22_3d.evaluation.scalar.interpreted=5",
            "--json",
        )
    )
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    comparison = report["comparisons"]["double_form_wedge_22_3d"]
    assert comparison["construction"]["improvement_percent"] == pytest.approx(
        20.0
    )
    assert comparison["evaluation.scalar.interpreted"][
        "improvement_percent"
    ] == pytest.approx(-3.0)
    assert report["requirements_evaluated"] == 2
    assert report["requirements_passed"]
    assert comparison[
        "paired.evaluation.scalar.interpreted_compact_over_dense"
    ]["candidate_median"] == pytest.approx(0.515)


def test_compressed_forms_comparison_can_only_aggregate(tmp_path):
    utility = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "compare_compressed_forms.py"
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_report(0.8, 1.0)), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(utility),
            "--candidate",
            str(candidate),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["candidate"]["report_count"] == 1


def test_compressed_forms_comparison_rejects_unmatched_requirement(tmp_path):
    utility = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "compare_compressed_forms.py"
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report(1.0, 1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_report(0.8, 1.0)), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(utility),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--require-improvement",
            "missing.metric=10",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["requirements_evaluated"] == 1
    assert not report["requirements_passed"]
    assert report["unmatched_requirements"] == [
        {
            "kind": "minimum_improvement_percent",
            "limit": 10.0,
            "pattern": "missing.metric",
        }
    ]


def test_compressed_forms_comparison_supports_dispatch_reports(tmp_path):
    utility = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "compare_compressed_forms.py"
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(_dispatch_report(2.0, 1.0)), encoding="utf-8"
    )
    candidate.write_text(
        json.dumps(_dispatch_report(1.5, 1.0)), encoding="utf-8"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(utility),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--require-improvement",
            "kform_wedge_1_1.paired.*=20",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["requirements_passed"]
    assert report["baseline"]["benchmark"] == (
        "compressed_forms_dispatch_attribution"
    )
    comparison = report["comparisons"]["kform_wedge_1_1"]
    assert comparison["paired.construction.public_over_native"][
        "improvement_percent"
    ] == pytest.approx(25.0)


def test_compressed_forms_comparison_supports_application_reports(tmp_path):
    utility = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "compare_compressed_forms.py"
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(_application_report(0.8, 1.0)), encoding="utf-8"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(utility),
            "--candidate",
            str(candidate),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    metrics = report["candidate"]["workloads"]["mixed_forms_energy_3d"]
    assert metrics["paired.total.public_over_dense"]["median"] == (
        pytest.approx(0.8)
    )
    assert metrics["paired.total.native_over_dense"]["median"] == (
        pytest.approx(0.72)
    )
