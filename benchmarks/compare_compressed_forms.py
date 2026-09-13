"""Aggregate and compare compressed-form benchmark reports.

This utility intentionally has no NGSolve dependency. It consumes raw JSON
reports produced by ``benchmark_compressed_forms.py`` or
``benchmark_form_dispatch.py`` and keeps performance acceptance decisions
outside CI timing tests.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
from pathlib import Path


def _statistics(values):
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "median": median,
        "median_absolute_deviation": mad,
        "relative_median_absolute_deviation": mad / median if median else 0.0,
        "process_values": values,
    }


_SUPPORTED_BENCHMARKS = {
    "compressed_forms_mvp",
    "compressed_forms_dispatch_attribution",
    "compressed_forms_application",
}


def _result_metrics(result, benchmark):
    if benchmark == "compressed_forms_dispatch_attribution":
        public = result["public"]["median_seconds"]
        native = result["native"]["median_seconds"]
        return {
            "public": public,
            "native": native,
            "paired.construction.public_over_native": public / native,
        }
    if benchmark == "compressed_forms_application":
        metrics = {}
        variants = result["variants"]
        for variant_name, variant in variants.items():
            for phase_name, phase in variant["phases"].items():
                metrics[f"{variant_name}.{phase_name}"] = phase[
                    "median_seconds"
                ]
            metrics[f"{variant_name}.reuse_assembly"] = variant[
                "reuse_assembly"
            ]["median_seconds"]
        for compact_name in ("public", "native"):
            for phase_name in (
                "construction",
                "compile",
                "form_setup",
                "assembly",
                "total",
                "reuse_assembly",
            ):
                metrics[
                    f"paired.{phase_name}.{compact_name}_over_dense"
                ] = (
                    metrics[f"{compact_name}.{phase_name}"]
                    / metrics[f"dense.{phase_name}"]
                )

        memory = result.get("retained_pipeline_memory")
        if memory:
            for variant_name in ("public", "native", "dense"):
                sample = memory.get(variant_name)
                if sample and sample.get("supported"):
                    metrics[f"memory.{variant_name}.bytes_per_pipeline"] = (
                        sample["estimated_incremental_bytes_per_pipeline"]
                    )
            dense_bytes = metrics.get("memory.dense.bytes_per_pipeline")
            for compact_name in ("public", "native"):
                compact_bytes = metrics.get(
                    f"memory.{compact_name}.bytes_per_pipeline"
                )
                if dense_bytes is not None and compact_bytes:
                    metrics[
                        f"memory.{compact_name}_over_dense_ratio"
                    ] = compact_bytes / dense_bytes
        return metrics

    metrics = {}
    for name in (
        "construction",
        "public_construction",
        "native_construction",
        "dense_construction",
        "compile",
        "dense_compile",
    ):
        value = result.get(name)
        if value is not None:
            metrics[name] = value["median_seconds"]

    dense_construction = result.get("dense_construction")
    public_construction = result.get("public_construction")
    native_construction = result.get("native_construction")
    if dense_construction and public_construction:
        metrics["paired.construction.public_over_dense"] = (
            public_construction["median_seconds"]
            / dense_construction["median_seconds"]
        )
    if dense_construction and native_construction:
        metrics["paired.construction.native_over_dense"] = (
            native_construction["median_seconds"]
            / dense_construction["median_seconds"]
        )

    compact_compile = result.get("compile")
    dense_compile = result.get("dense_compile")
    if compact_compile and dense_compile:
        metrics["paired.compile.compact_over_dense"] = (
            compact_compile["median_seconds"]
            / dense_compile["median_seconds"]
        )

    for backend, variants in result.get("evaluation", {}).items():
        for variant, value in variants.items():
            metrics[f"evaluation.{backend}.{variant}"] = value[
                "median_seconds"
            ]
        for variant in ("interpreted", "compiled"):
            dense_variant = f"dense_{variant}"
            if variant in variants and dense_variant in variants:
                metrics[
                    f"paired.evaluation.{backend}.{variant}_compact_over_dense"
                ] = (
                    variants[variant]["median_seconds"]
                    / variants[dense_variant]["median_seconds"]
                )

    retained = result.get("retained_graph_memory")
    if retained:
        for variant in ("compact", "dense"):
            sample = retained.get(variant)
            if sample and sample.get("supported"):
                metrics[f"memory.{variant}.bytes_per_graph"] = sample[
                    "estimated_incremental_bytes_per_graph"
                ]
        ratio = retained.get("dense_over_compact_estimate")
        if ratio is not None:
            # This ratio is a score for which larger is better; store its
            # inverse as a cost so all comparison metrics retain the same
            # lower-is-better convention.
            metrics["memory.compact_over_dense_ratio"] = 1.0 / ratio
    return metrics


def aggregate(paths):
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("at least one report is required")
    benchmark = reports[0].get("benchmark")
    if benchmark not in _SUPPORTED_BENCHMARKS:
        raise ValueError("input is not a supported compressed-forms report")
    for report in reports:
        if report.get("benchmark") != benchmark:
            raise ValueError("all reports in one aggregate need the same benchmark")

    workload_names = set(reports[0]["results"])
    for report in reports[1:]:
        if set(report["results"]) != workload_names:
            raise ValueError("all reports in one aggregate need identical workloads")

    workloads = {}
    for workload_name in sorted(workload_names):
        process_metrics = [
            _result_metrics(report["results"][workload_name], benchmark)
            for report in reports
        ]
        metric_names = set(process_metrics[0])
        for metrics in process_metrics[1:]:
            if set(metrics) != metric_names:
                raise ValueError(
                    f"metric mismatch across reports for {workload_name}"
                )
        workloads[workload_name] = {
            name: _statistics([metrics[name] for metrics in process_metrics])
            for name in sorted(metric_names)
        }

    return {
        "benchmark": benchmark,
        "report_count": len(reports),
        "paths": [str(Path(path)) for path in paths],
        "revisions": sorted(
            {
                report.get("repository", {}).get("revision")
                for report in reports
            },
            key=lambda value: "" if value is None else value,
        ),
        "extension_sha256": sorted(
            {
                report.get("ngsdiffgeo_extension", {}).get("sha256")
                for report in reports
            },
            key=lambda value: "" if value is None else value,
        ),
        "workloads": workloads,
    }


def _parse_requirements(values):
    requirements = []
    for value in values:
        try:
            pattern, percent = value.rsplit("=", 1)
            percent = float(percent)
        except ValueError as error:
            raise ValueError(
                f"invalid requirement {value!r}; expected PATTERN=PERCENT"
            ) from error
        if not pattern:
            raise ValueError("requirement pattern must not be empty")
        requirements.append((pattern, percent))
    return requirements


def compare(baseline, candidate, requirements=(), regression_limits=()):
    if baseline["benchmark"] != candidate["benchmark"]:
        raise ValueError("baseline and candidate need the same benchmark")
    if set(baseline["workloads"]) != set(candidate["workloads"]):
        raise ValueError("baseline and candidate need identical workloads")

    comparisons = {}
    decisions = []
    requirement_matches = [0] * len(requirements)
    regression_limit_matches = [0] * len(regression_limits)
    for workload_name in sorted(baseline["workloads"]):
        baseline_metrics = baseline["workloads"][workload_name]
        candidate_metrics = candidate["workloads"][workload_name]
        common_metrics = sorted(set(baseline_metrics) & set(candidate_metrics))
        workload_comparison = {}
        for metric_name in common_metrics:
            before = baseline_metrics[metric_name]["median"]
            after = candidate_metrics[metric_name]["median"]
            speedup = before / after if after else None
            improvement_percent = (
                100.0 * (before - after) / before if before else None
            )
            metric = {
                "baseline_median": before,
                "candidate_median": after,
                "baseline_over_candidate": speedup,
                "improvement_percent": improvement_percent,
            }
            qualified_name = f"{workload_name}.{metric_name}"
            checks = []
            for requirement_index, (pattern, required) in enumerate(
                requirements
            ):
                if fnmatch.fnmatchcase(qualified_name, pattern):
                    requirement_matches[requirement_index] += 1
                    passed = (
                        improvement_percent is not None
                        and improvement_percent >= required
                    )
                    check = {
                        "kind": "minimum_improvement_percent",
                        "pattern": pattern,
                        "limit": required,
                        "passed": passed,
                    }
                    checks.append(check)
                    decisions.append(passed)
            for limit_index, (pattern, allowed) in enumerate(
                regression_limits
            ):
                if fnmatch.fnmatchcase(qualified_name, pattern):
                    regression_limit_matches[limit_index] += 1
                    passed = (
                        improvement_percent is not None
                        and improvement_percent >= -allowed
                    )
                    check = {
                        "kind": "maximum_regression_percent",
                        "pattern": pattern,
                        "limit": allowed,
                        "passed": passed,
                    }
                    checks.append(check)
                    decisions.append(passed)
            if checks:
                metric["checks"] = checks
            workload_comparison[metric_name] = metric
        comparisons[workload_name] = workload_comparison

    unmatched_requirements = [
        {
            "kind": "minimum_improvement_percent",
            "pattern": requirements[index][0],
            "limit": requirements[index][1],
        }
        for index, matches in enumerate(requirement_matches)
        if matches == 0
    ]
    unmatched_requirements.extend(
        {
            "kind": "maximum_regression_percent",
            "pattern": regression_limits[index][0],
            "limit": regression_limits[index][1],
        }
        for index, matches in enumerate(regression_limit_matches)
        if matches == 0
    )
    decisions.extend(False for _ in unmatched_requirements)

    return {
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": comparisons,
        "requirements_evaluated": len(decisions),
        "requirements_passed": all(decisions) if decisions else None,
        "unmatched_requirements": unmatched_requirements,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        default=[],
        help="baseline report; repeat for independent processes",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        required=True,
        help="candidate report; repeat for independent processes",
    )
    parser.add_argument(
        "--require-improvement",
        action="append",
        default=[],
        metavar="PATTERN=PERCENT",
    )
    parser.add_argument(
        "--limit-regression",
        action="append",
        default=[],
        metavar="PATTERN=PERCENT",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        candidate = aggregate(args.candidate)
        if args.baseline:
            report = compare(
                aggregate(args.baseline),
                candidate,
                _parse_requirements(args.require_improvement),
                _parse_requirements(args.limit_regression),
            )
        else:
            report = {"candidate": candidate}
    except ValueError as error:
        raise SystemExit(str(error)) from error

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(rendered)


if __name__ == "__main__":
    main()
