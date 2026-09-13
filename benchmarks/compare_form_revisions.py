"""Run and compare the public form benchmark against two Python package roots."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def _subprocess_json(command, environment):
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("revision benchmark produced no JSON output")
    return json.loads("\n".join(lines))


def _environment(package_root):
    environment = os.environ.copy()
    previous = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(package_root), previous) if part
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _median_metric(reports, getter):
    values = [getter(report) for report in reports]
    completed = [value for value in values if value is not None]
    return {
        "median": statistics.median(completed) if completed else None,
        "samples": values,
    }


def _comparison(baseline, candidate):
    ratio = (
        candidate["median"] / baseline["median"]
        if baseline["median"] not in (None, 0)
        and candidate["median"] is not None
        else None
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_over_baseline": ratio,
        "improvement_percent": None if ratio is None else (1.0 - ratio) * 100.0,
    }


def _aggregate(reports):
    baseline_reports = reports["baseline"]
    candidate_reports = reports["candidate"]
    comparison = {"workloads": {}, "application": None}
    for name in baseline_reports[0]["results"]:
        item = {}
        for metric in ("construction", "compile"):
            if baseline_reports[0]["results"][name][metric] is None:
                continue
            item[metric] = _comparison(
                _median_metric(
                    baseline_reports,
                    lambda report, n=name, m=metric: report["results"][n][m][
                        "median_seconds"
                    ],
                ),
                _median_metric(
                    candidate_reports,
                    lambda report, n=name, m=metric: report["results"][n][m][
                        "median_seconds"
                    ],
                ),
            )
        baseline_memory = baseline_reports[0]["results"][name][
            "retained_graph_memory"
        ]
        if baseline_memory is not None:
            item["retained_graph_memory"] = _comparison(
                _median_metric(
                    baseline_reports,
                    lambda report, n=name: report["results"][n][
                        "retained_graph_memory"
                    ]["estimated_incremental_bytes_per_graph"],
                ),
                _median_metric(
                    candidate_reports,
                    lambda report, n=name: report["results"][n][
                        "retained_graph_memory"
                    ]["estimated_incremental_bytes_per_graph"],
                ),
            )
        baseline_value = statistics.median(
            report["results"][name]["value_norm"] for report in baseline_reports
        )
        candidate_value = statistics.median(
            report["results"][name]["value_norm"] for report in candidate_reports
        )
        item["relative_value_error"] = abs(candidate_value - baseline_value) / max(
            1.0, abs(baseline_value)
        )
        comparison["workloads"][name] = item

    if baseline_reports[0]["application"]["result"] is not None:
        application = {"phases": {}}
        for phase in baseline_reports[0]["application"]["result"]["phases"]:
            application["phases"][phase] = _comparison(
                _median_metric(
                    baseline_reports,
                    lambda report, p=phase: report["application"]["result"][
                        "phases"
                    ][p]["median_seconds"],
                ),
                _median_metric(
                    candidate_reports,
                    lambda report, p=phase: report["application"]["result"][
                        "phases"
                    ][p]["median_seconds"],
                ),
            )
        application["reuse_assembly"] = _comparison(
            _median_metric(
                baseline_reports,
                lambda report: report["application"]["result"][
                    "reuse_assembly"
                ]["median_seconds"],
            ),
            _median_metric(
                candidate_reports,
                lambda report: report["application"]["result"][
                    "reuse_assembly"
                ]["median_seconds"],
            ),
        )
        baseline_value = statistics.median(
            report["application"]["result"]["value_norm"]
            for report in baseline_reports
        )
        candidate_value = statistics.median(
            report["application"]["result"]["value_norm"]
            for report in candidate_reports
        )
        application["relative_value_error"] = abs(
            candidate_value - baseline_value
        ) / max(1.0, abs(baseline_value))
        baseline_memory = baseline_reports[0]["application"]["result"].get(
            "retained_pipeline_memory"
        )
        if baseline_memory is not None:
            application["retained_pipeline_memory"] = _comparison(
                _median_metric(
                    baseline_reports,
                    lambda report: report["application"]["result"][
                        "retained_pipeline_memory"
                    ]["estimated_incremental_bytes_per_pipeline"],
                ),
                _median_metric(
                    candidate_reports,
                    lambda report: report["application"]["result"][
                        "retained_pipeline_memory"
                    ]["estimated_incremental_bytes_per_pipeline"],
                ),
            )
        comparison["application"] = application
    return comparison


def _worker_command(args, label):
    benchmark = Path(__file__).with_name("benchmark_revision_forms.py")
    command = [
        sys.executable,
        str(benchmark),
        "--label",
        label,
        "--mode",
        args.mode,
        "--compile-kind",
        args.compile_kind,
        "--backend",
        args.backend,
        "--maxh",
        str(args.maxh),
        "--construction-warmup",
        str(args.construction_warmup),
        "--construction-iterations",
        str(args.construction_iterations),
        "--compile-warmup",
        str(args.compile_warmup),
        "--compile-iterations",
        str(args.compile_iterations),
        "--realcompile-timeout",
        str(args.realcompile_timeout),
        "--realcompile-layout",
        args.realcompile_layout,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--reuse-warmup",
        str(args.reuse_warmup),
        "--reuse-iterations",
        str(args.reuse_iterations),
        "--retained-graphs",
        str(args.retained_graphs),
        "--retained-pipelines",
        str(args.retained_pipelines),
        "--threads",
        str(args.threads),
        "--tolerance",
        str(args.tolerance),
        "--json",
    ]
    for workload in args.workload or ():
        command.extend(("--workload", workload))
    return command


def run(args):
    roots = {
        "baseline": args.baseline_pythonpath.resolve(),
        "candidate": args.candidate_pythonpath.resolve(),
    }
    reports = {name: [] for name in roots}
    execution_order = []
    for process_index in range(args.processes):
        order = (
            ("baseline", "candidate")
            if process_index % 2 == 0
            else ("candidate", "baseline")
        )
        execution_order.append(list(order))
        for name in order:
            reports[name].append(
                _subprocess_json(
                    _worker_command(args, f"{name}-{process_index + 1}"),
                    _environment(roots[name]),
                )
            )
    comparison = _aggregate(reports)
    errors = [
        result["relative_value_error"]
        for result in comparison["workloads"].values()
    ]
    if comparison["application"] is not None:
        errors.append(comparison["application"]["relative_value_error"])
    if errors and max(errors) > args.tolerance:
        raise RuntimeError(
            "baseline/candidate value mismatch: "
            f"maximum relative error {max(errors):.3e} exceeds "
            f"{args.tolerance:.3e}"
        )
    return {
        "benchmark": "revision_neutral_forms_comparison",
        "baseline_pythonpath": str(roots["baseline"]),
        "candidate_pythonpath": str(roots["candidate"]),
        "baseline_revision": args.baseline_revision,
        "candidate_revision": args.candidate_revision,
        "compile_kind": args.compile_kind,
        "realcompile_timeout": args.realcompile_timeout or None,
        "realcompile_layout": args.realcompile_layout,
        "processes": args.processes,
        "tolerance": args.tolerance,
        "execution_order": execution_order,
        "reports": reports,
        "comparison": comparison,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pythonpath", type=Path, required=True)
    parser.add_argument("--candidate-pythonpath", type=Path, required=True)
    parser.add_argument("--baseline-revision")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--workload", action="append")
    parser.add_argument(
        "--mode",
        choices=("construction", "compile", "application", "all"),
        default="all",
    )
    parser.add_argument(
        "--compile-kind", choices=("graph", "real"), default="graph"
    )
    parser.add_argument("--backend", choices=("scalar", "simd"), default="simd")
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--construction-warmup", type=int, default=10)
    parser.add_argument("--construction-iterations", type=int, default=100)
    parser.add_argument("--compile-warmup", type=int, default=2)
    parser.add_argument("--compile-iterations", type=int, default=5)
    parser.add_argument("--realcompile-timeout", type=float, default=0)
    parser.add_argument(
        "--realcompile-layout",
        choices=("scalar_variables", "tensor_temporaries"),
        default="scalar_variables",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--reuse-warmup", type=int, default=3)
    parser.add_argument("--reuse-iterations", type=int, default=31)
    parser.add_argument("--retained-graphs", type=int, default=0)
    parser.add_argument("--retained-pipelines", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.realcompile_timeout < 0:
        parser.error("--realcompile-timeout must be nonnegative")
    for root_name in ("baseline_pythonpath", "candidate_pythonpath"):
        if not getattr(args, root_name).is_dir():
            parser.error(f"--{root_name.replace('_', '-')} must be a directory")
    return args


def main():
    args = parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(rendered)


if __name__ == "__main__":
    main()
