"""Benchmark batched direct evaluation of compact and dense form graphs.

Unlike the assembly benchmarks, this calls each coefficient function as the
root evaluator on one retained array of mapped points.  It therefore exercises
the direct ``T_Evaluate`` overloads, including their temporary storage.  Timing
thresholds remain outside CI; the test suite only smoke-tests execution and
the value/provenance contract.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import ngsolve
import numpy as np
from netgen.occ import unit_cube
from ngsolve import Mesh

from benchmark_common import (
    _alternating_order,
    _extension_identity,
    _file_identity,
    _repository_identity,
    _statistics,
    _wrapper_identity,
)
from benchmark_compressed_forms import _workloads


_WORKLOAD_NAMES = (
    "double_form_arithmetic_22_3d",
    "double_form_wedge_22_3d",
    "double_form_wedge_chain_22_3d",
    "kform_wedge_d_3d",
)
_VARIANT_NAMES = ("compact", "compiled", "dense", "dense_compiled")


def _mapped_points(mesh, count):
    indices = np.arange(count, dtype=float) + 0.5
    x_values = 0.1 + 0.8 * indices / count
    y_values = 0.1 + 0.8 * ((37 * indices) % count) / count
    z_values = 0.1 + 0.8 * ((101 * indices) % count) / count
    return mesh(x_values, y_values, z_values)


def _variants(workload):
    compact = workload["factory"]()
    dense = workload["dense_factory"]()
    return {
        "compact": compact,
        "compiled": compact.Compile(
            realcompile=False, wait=True, maxderiv=0
        ),
        "dense": dense,
        "dense_compiled": dense.Compile(
            realcompile=False, wait=True, maxderiv=0
        ),
    }


def _benchmark(values, points, warmup, iterations):
    names = tuple(values)
    for iteration in range(warmup):
        for name in _alternating_order(names, iteration):
            values[name](points)

    samples = {name: [] for name in names}
    outputs = {}
    execution_order = []
    for iteration in range(iterations):
        order = _alternating_order(names, iteration)
        execution_order.append(list(order))
        for name in order:
            started = time.perf_counter_ns()
            outputs[name] = np.asarray(values[name](points))
            samples[name].append((time.perf_counter_ns() - started) * 1e-9)
    return outputs, {
        name: _statistics(times) for name, times in samples.items()
    }, execution_order


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    points = _mapped_points(mesh, args.points)
    workloads = _workloads()
    selected = args.workload or list(_WORKLOAD_NAMES)
    results = {}
    for name in selected:
        variants = _variants(workloads[name])
        outputs, timings, order = _benchmark(
            variants, points, args.warmup, args.iterations
        )
        reference = outputs["compact"]
        errors = {
            variant: float(
                np.linalg.norm(value - reference)
                / max(1.0, np.linalg.norm(reference))
            )
            for variant, value in outputs.items()
        }
        if max(errors.values()) > args.tolerance:
            raise RuntimeError(
                f"{name} direct evaluation disagreement: {errors}"
            )
        results[name] = {
            "timings": timings,
            "relative_value_errors": errors,
            "execution_order": order,
            "result_shape": list(reference.shape),
            "result_components": int(variants["compact"].dim),
        }

    return {
        "benchmark": "compact_form_direct_evaluation",
        "label": args.label,
        "workloads": selected,
        "variants": list(_VARIANT_NAMES),
        "ngsolve_version": ngsolve.__version__,
        "ngsdiffgeo_extension": _extension_identity(),
        "ngsdiffgeo_wrapper": _wrapper_identity(),
        "benchmark_script": _file_identity(__file__),
        "repository": _repository_identity(),
        "platform": platform.platform(),
        "threads": args.threads,
        "maxh": args.maxh,
        "points": args.points,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tolerance": args.tolerance,
        "measurement_boundary": (
            "one batched root CoefficientFunction evaluation on retained "
            "mapped points; graph construction, compilation, point location, "
            "and output validation excluded"
        ),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument(
        "--workload", action="append", choices=_WORKLOAD_NAMES
    )
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.maxh <= 0 or args.tolerance <= 0:
        parser.error("--maxh and --tolerance must be positive")
    if args.points < 1 or args.iterations < 1 or args.threads < 1:
        parser.error("--points, --iterations, and --threads must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
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
