"""Baseline benchmark for the compressed-forms MVP.

The same public workloads can be recorded before and after compact form
dispatch is implemented.  Reports contain construction and full-output costs,
coefficient-graph sizes, graph-compilation time, and interpreted/compiled
assembly timings.  Timing thresholds deliberately do not belong in CI.

Dimension-four data is constant and construction-only.  Physical evaluation
uses nonzero, spatially varying coefficients on a three-dimensional mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
from netgen.occ import unit_cube
from ngsolve import (
    BilinearForm,
    CF,
    InnerProduct,
    Integrate,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
    x,
    y,
    z,
)


def _workloads():
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    beta = dg.OneForm(CF((2 - x, 1 + y * z, 1 + z)))
    gamma = dg.OneForm(CF((1 + y, 2 + z, 3 + x * y)))
    delta = dg.OneForm(CF((2 + x * y, 1 - z, 2 + y)))

    def kform_factory():
        pair = dg.Wedge(alpha, beta)
        return dg.Wedge(pair, gamma) + 0.25 * dg.d(pair)

    pair = dg.Wedge(alpha, beta)
    kform_reference = dg.Wedge(pair, gamma) + 0.25 * (
        dg.Wedge(dg.d(alpha), beta) - dg.Wedge(alpha, dg.d(beta))
    )

    left = dg.DoubleForm(dg.Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=3)
    right = dg.DoubleForm(dg.Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=3)

    def double_form_factory():
        return dg.Wedge(left, right)

    double_form_reference = dg.DoubleForm(
        dg.Einsum(
            "ij,kl->ijkl",
            dg.Wedge(alpha, gamma),
            dg.Wedge(beta, delta),
        ),
        p=2,
        q=2,
        dim=3,
    )

    alpha4 = dg.OneForm(CF((1, 2, 3, 4)))
    beta4 = dg.OneForm(CF((2, 3, 5, 7)))
    gamma4 = dg.OneForm(CF((3, 1, 4, 2)))
    delta4 = dg.OneForm(CF((5, 2, 1, 3)))
    left4 = dg.DoubleForm(dg.Einsum("i,j->ij", alpha4, beta4), p=1, q=1, dim=4)
    right4 = dg.DoubleForm(dg.Einsum("i,j->ij", gamma4, delta4), p=1, q=1, dim=4)

    def double_form_4d_factory():
        return dg.Wedge(left4, right4)

    double_form_4d_reference = dg.DoubleForm(
        dg.Einsum(
            "ij,kl->ijkl",
            dg.Wedge(alpha4, gamma4),
            dg.Wedge(beta4, delta4),
        ),
        p=2,
        q=2,
        dim=4,
    )

    return {
        "kform_wedge_d_3d": {
            "factory": kform_factory,
            "reference": kform_reference,
            "physical": True,
            "space_dimension": 3,
            "form_degrees": [3],
            "full_components": 27,
            "independent_components": 1,
        },
        "double_form_wedge_22_3d": {
            "factory": double_form_factory,
            "reference": double_form_reference,
            "physical": True,
            "space_dimension": 3,
            "form_degrees": [2, 2],
            "full_components": 81,
            "independent_components": 9,
        },
        "double_form_wedge_22_4d": {
            "factory": double_form_4d_factory,
            "reference": double_form_4d_reference,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [2, 2],
            "full_components": 256,
            "independent_components": 36,
        },
    }


def _statistics(samples):
    return {
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
    }


def _benchmark_factory(factory, iterations, materialize_full_output=False):
    samples = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = factory()
        if materialize_full_output:
            output = output.coef
        samples.append((time.perf_counter_ns() - started) * 1e-9)
    return output, _statistics(samples)


def _benchmark_compile(factory, iterations):
    samples = []
    compiled = None
    for _ in range(iterations):
        coefficient = factory().coef
        started = time.perf_counter()
        compiled = coefficient.Compile(
            realcompile=False,
            wait=True,
            maxderiv=0,
        )
        samples.append(time.perf_counter() - started)
    return compiled, {
        "kind": "ngsolve_graph",
        **_statistics(samples),
    }


def _l2_error(left, right, mesh):
    difference = left - right
    error_sq = (
        difference * difference
        if not difference.dims
        else InnerProduct(difference, difference)
    )
    return math.sqrt(max(0.0, Integrate(error_sq, mesh)))


def _make_form(coefficient, mesh, simd):
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    integrand = (
        coefficient * coefficient
        if not coefficient.dims
        else InnerProduct(coefficient, coefficient)
    )
    form = BilinearForm(space)
    integrator = SymbolicBFI(
        integrand * trial * test,
        simd_evaluate=simd,
    )
    form += integrator
    return form, integrator


def _benchmark_evaluation(output, compiled, mesh, backends, warmup, iterations):
    forms = {}
    for backend in backends:
        simd = backend == "simd"
        forms[(backend, "interpreted")] = _make_form(output, mesh, simd)
        forms[(backend, "compiled")] = _make_form(compiled, mesh, simd)

    timings = {key: [] for key in forms}
    keys = list(forms)
    with TaskManager():
        for _ in range(warmup):
            for form, _ in forms.values():
                form.Assemble()
        for iteration in range(iterations):
            order = keys if iteration % 2 == 0 else list(reversed(keys))
            for key in order:
                form, _ = forms[key]
                started = time.perf_counter()
                form.Assemble()
                timings[key].append(time.perf_counter() - started)

    report = {}
    for backend in backends:
        variants = {}
        for variant in ("interpreted", "compiled"):
            key = (backend, variant)
            form, integrator = forms[key]
            variants[variant] = {
                **_statistics(timings[key]),
                "value_norm": Norm(form.mat.AsVector()),
                "simd_requested": backend == "simd",
                "simd_active": bool(integrator.simd_evaluate),
            }
        reference = variants["interpreted"]["value_norm"]
        variants["compiled"]["relative_value_error"] = abs(
            variants["compiled"]["value_norm"] - reference
        ) / max(1.0, abs(reference))
        report[backend] = variants
    return report


def _backend_equivalence(evaluation):
    if set(evaluation) != {"scalar", "simd"}:
        return {}
    result = {}
    for variant in ("interpreted", "compiled"):
        scalar = evaluation["scalar"][variant]["value_norm"]
        simd = evaluation["simd"][variant]["value_norm"]
        result[variant] = abs(simd - scalar) / max(1.0, abs(scalar))
    return result


def _extension_identity():
    extension = importlib.import_module("ngsdiffgeo.ngsdiffgeo")
    path = Path(extension.__file__).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
    }


def _repository_identity():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "tracked_changes": None}
    return {
        "revision": revision,
        "tracked_changes": bool(tracked_status.strip()),
    }


def _selected_backends(mode):
    if mode == "construction":
        return ()
    if mode == "all":
        return ("scalar", "simd")
    return (mode,)


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    backends = _selected_backends(args.mode)
    results = {}

    for name, workload in _workloads().items():
        output, construction = _benchmark_factory(
            workload["factory"], args.construction_iterations
        )
        full_output, full_output_construction = _benchmark_factory(
            workload["factory"],
            args.construction_iterations,
            materialize_full_output=True,
        )
        l2_error = _l2_error(output, workload["reference"], mesh)
        if l2_error > args.tolerance:
            raise RuntimeError(
                f"{name} disagrees with its independent reference: " f"{l2_error:.3e}"
            )

        result = {
            "physical_evaluation": workload["physical"],
            "space_dimension": workload["space_dimension"],
            "form_degrees": workload["form_degrees"],
            "full_components": workload["full_components"],
            "independent_components": workload["independent_components"],
            "theoretical_compression_ratio": (
                workload["full_components"] / workload["independent_components"]
            ),
            "construction": construction,
            "construction_with_full_output": full_output_construction,
            "wrapper_tree": dg.CFStats(output),
            "full_output_tree": dg.CFStats(full_output),
            "l2_error": l2_error,
            "compile": None,
            "evaluation": {},
            "scalar_simd_relative_value_error": {},
        }
        if workload["physical"] and backends:
            compiled, compile_statistics = _benchmark_compile(
                workload["factory"], args.compile_iterations
            )
            result["compile"] = compile_statistics
            result["compiled_tree"] = dg.CFStats(compiled)
            result["evaluation"] = _benchmark_evaluation(
                output,
                compiled,
                mesh,
                backends,
                args.warmup,
                args.iterations,
            )
            result["scalar_simd_relative_value_error"] = _backend_equivalence(
                result["evaluation"]
            )
            for backend, variants in result["evaluation"].items():
                relative_error = variants["compiled"]["relative_value_error"]
                if relative_error > args.tolerance:
                    raise RuntimeError(
                        f"{name} compiled {backend} evaluation disagrees with "
                        f"interpreted evaluation: {relative_error:.3e}"
                    )
            for variant, relative_error in result[
                "scalar_simd_relative_value_error"
            ].items():
                if relative_error > args.tolerance:
                    raise RuntimeError(
                        f"{name} {variant} SIMD evaluation disagrees with "
                        f"scalar evaluation: {relative_error:.3e}"
                    )
        results[name] = result

    return {
        "benchmark": "compressed_forms_mvp",
        "label": args.label,
        "mode": args.mode,
        "ngsolve_version": ngsolve.__version__,
        "ngsdiffgeo_version": getattr(dg, "__version__", "unknown"),
        "ngsdiffgeo_extension": _extension_identity(),
        "repository": _repository_identity(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "cxx_compiler": getattr(ngsolve.config, "CMAKE_CXX_COMPILER", "unknown"),
        "logical_cpus": os.cpu_count(),
        "threads": args.threads,
        "clock": "time.perf_counter",
        "maxh": args.maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "construction_iterations": args.construction_iterations,
        "compile_iterations": args.compile_iterations,
        "tolerance": args.tolerance,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument(
        "--mode",
        choices=("construction", "scalar", "simd", "all"),
        default="all",
    )
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--construction-iterations", type=int, default=100)
    parser.add_argument("--compile-iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.maxh <= 0:
        parser.error("--maxh must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.construction_iterations < 1 or args.compile_iterations < 1:
        parser.error("construction and compile iteration counts must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
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
