"""Application-style compact/native/dense compressed-form benchmark.

The workload builds a mixed 3D energy from a K-form wedge/exterior-derivative
expression and double-form arithmetic, compiles it, creates a bilinear form,
and assembles it.  Mesh, finite-element space, and reusable input fields are
outside the timed pipeline.  Per-phase and end-to-end timings are recorded,
followed by an assembly-only reuse measurement.

Optional retained-pipeline memory is measured in fresh subprocesses.  Timing
thresholds deliberately remain outside CI; tests only smoke-test execution and
the correctness/provenance contract.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
from netgen.occ import unit_cube
from ngsolve import (
    BilinearForm,
    InnerProduct,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
)

from benchmark_common import (
    _alternating_order,
    _extension_identity,
    _file_identity,
    _repository_identity,
    _retained_memory_sample,
    _statistics,
    _subprocess_json,
    _wrapper_identity,
)
from benchmark_compressed_forms import _workloads


_APPLICATION_NAME = "mixed_forms_energy_3d"
_VARIANTS = ("public", "native", "dense")
_PHASES = ("construction", "compile", "form_setup", "assembly", "total")


def _application_factories():
    workloads = _workloads()
    kform = workloads["kform_wedge_d_3d"]
    double_form = workloads["double_form_arithmetic_22_3d"]

    def energy(kform_factory, double_form_factory):
        k_value = kform_factory()
        double_value = double_form_factory()
        return InnerProduct(k_value, k_value) + 0.125 * InnerProduct(
            double_value, double_value
        )

    return {
        "public": lambda: energy(kform["factory"], double_form["factory"]),
        "native": lambda: energy(
            kform["native_factory"], double_form["native_factory"]
        ),
        "dense": lambda: energy(
            kform["dense_factory"], double_form["dense_factory"]
        ),
    }


def _make_form(coefficient, space, trial, test, simd):
    form = BilinearForm(space)
    integrator = SymbolicBFI(
        coefficient * trial * test,
        simd_evaluate=simd,
    )
    form += integrator
    return form, integrator


def _run_pipeline(factory, space, trial, test, simd):
    started = time.perf_counter_ns()
    coefficient = factory()
    constructed = time.perf_counter_ns()
    compiled = coefficient.Compile(
        realcompile=False,
        wait=True,
        maxderiv=0,
    )
    compilation_finished = time.perf_counter_ns()
    form, integrator = _make_form(compiled, space, trial, test, simd)
    form_finished = time.perf_counter_ns()
    form.Assemble()
    assembled = time.perf_counter_ns()
    return (
        (coefficient, compiled, form, integrator),
        {
            "construction": (constructed - started) * 1e-9,
            "compile": (compilation_finished - constructed) * 1e-9,
            "form_setup": (form_finished - compilation_finished) * 1e-9,
            "assembly": (assembled - form_finished) * 1e-9,
            "total": (assembled - started) * 1e-9,
        },
    )


def _benchmark_pipelines(factories, space, trial, test, simd, warmup, iterations):
    bundles = {name: None for name in factories}
    phase_samples = {
        name: {phase: [] for phase in _PHASES} for name in factories
    }

    with TaskManager():
        for iteration in range(warmup):
            for name in _alternating_order(tuple(factories), iteration):
                bundles[name], _ = _run_pipeline(
                    factories[name], space, trial, test, simd
                )

        execution_order = []
        for iteration in range(iterations):
            order = _alternating_order(tuple(factories), iteration)
            execution_order.append(list(order))
            for name in order:
                # Destruction belongs outside all timed phases.
                bundles[name] = None
                bundles[name], sample = _run_pipeline(
                    factories[name], space, trial, test, simd
                )
                for phase in _PHASES:
                    phase_samples[name][phase].append(sample[phase])

    statistics = {
        name: {
            phase: _statistics(samples)
            for phase, samples in phases.items()
        }
        for name, phases in phase_samples.items()
    }
    return bundles, statistics, execution_order


def _benchmark_reuse(bundles, warmup, iterations):
    samples = {name: [] for name in bundles}
    with TaskManager():
        for iteration in range(warmup):
            for name in _alternating_order(tuple(bundles), iteration):
                bundles[name][2].Assemble()

        execution_order = []
        for iteration in range(iterations):
            order = _alternating_order(tuple(bundles), iteration)
            execution_order.append(list(order))
            for name in order:
                started = time.perf_counter()
                bundles[name][2].Assemble()
                samples[name].append(time.perf_counter() - started)
    return (
        {name: _statistics(values) for name, values in samples.items()},
        execution_order,
    )


def _pipeline_value(bundle):
    return float(Norm(bundle[2].mat.AsVector()))


def _relative_error(value, reference):
    return abs(value - reference) / max(1.0, abs(reference))


def _retained_pipeline_memory_sample(args, variant):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    factory = _application_factories()[variant]
    simd = args.backend == "simd"

    def build_pipeline():
        bundle, _ = _run_pipeline(factory, space, trial, test, simd)
        return bundle

    return _retained_memory_sample(
        build_pipeline,
        args.retained_pipelines,
        count_key="retained_pipelines",
        per_item_key="estimated_incremental_bytes_per_pipeline",
        metadata={"variant": variant},
        context_factory=TaskManager,
    )


def _retained_pipeline_memory(args):
    samples = {}
    for variant in _VARIANTS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--memory-worker",
            variant,
            "--retained-pipelines",
            str(args.retained_pipelines),
            "--backend",
            args.backend,
            "--maxh",
            str(args.maxh),
            "--threads",
            str(args.threads),
        ]
        samples[variant] = _subprocess_json(command)

    dense_bytes = samples["dense"].get(
        "estimated_incremental_bytes_per_pipeline"
    )
    for variant in ("public", "native"):
        compact_bytes = samples[variant].get(
            "estimated_incremental_bytes_per_pipeline"
        )
        samples[f"dense_over_{variant}_estimate"] = (
            dense_bytes / compact_bytes
            if dense_bytes is not None and compact_bytes
            else None
        )
    return samples


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    simd = args.backend == "simd"
    factories = _application_factories()

    bundles, phases, pipeline_order = _benchmark_pipelines(
        factories,
        space,
        trial,
        test,
        simd,
        args.warmup,
        args.iterations,
    )
    reuse, reuse_order = _benchmark_reuse(
        bundles, args.reuse_warmup, args.reuse_iterations
    )
    values = {name: _pipeline_value(bundle) for name, bundle in bundles.items()}
    reference = values["dense"]
    relative_errors = {
        name: _relative_error(value, reference)
        for name, value in values.items()
    }
    if max(relative_errors.values()) > args.tolerance:
        raise RuntimeError(
            "application compact/native/dense disagreement: "
            + ", ".join(
                f"{name}={error:.3e}"
                for name, error in relative_errors.items()
            )
        )

    variants = {}
    for name, bundle in bundles.items():
        variants[name] = {
            "phases": phases[name],
            "reuse_assembly": reuse[name],
            "value_norm": values[name],
            "relative_value_error": relative_errors[name],
            "coefficient_tree": dg.CFStats(bundle[0]),
            "compiled_tree": dg.CFStats(bundle[1]),
            "simd_requested": simd,
            "simd_active": bool(bundle[3].simd_evaluate),
        }

    ratios = {}
    for variant in ("public", "native"):
        ratios[variant] = {
            phase: (
                phases[variant][phase]["median_seconds"]
                / phases["dense"][phase]["median_seconds"]
            )
            for phase in _PHASES
        }
        ratios[variant]["reuse_assembly"] = (
            reuse[variant]["median_seconds"]
            / reuse["dense"]["median_seconds"]
        )

    memory = (
        _retained_pipeline_memory(args) if args.retained_pipelines else None
    )
    return {
        "benchmark": "compressed_forms_application",
        "label": args.label,
        "backend": args.backend,
        "ngsolve_version": ngsolve.__version__,
        "ngsdiffgeo_version": getattr(dg, "__version__", "unknown"),
        "ngsdiffgeo_extension": _extension_identity(),
        "ngsdiffgeo_wrapper": _wrapper_identity(),
        "benchmark_script": _file_identity(__file__),
        "repository": _repository_identity(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "cxx_compiler": getattr(ngsolve.config, "CMAKE_CXX_COMPILER", "unknown"),
        "logical_cpus": os.cpu_count(),
        "threads": args.threads,
        "clock": "time.perf_counter_ns for pipeline phases; "
        "time.perf_counter for reuse assembly",
        "maxh": args.maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "reuse_warmup": args.reuse_warmup,
        "reuse_iterations": args.reuse_iterations,
        "retained_pipelines": args.retained_pipelines,
        "measurement_boundaries": {
            "excluded_setup": (
                "mesh, NumberSpace, trial/test functions, and reusable input "
                "fields"
            ),
            "construction": "mixed scalar energy coefficient factory",
            "compile": "CoefficientFunction.Compile(realcompile=False)",
            "form_setup": "BilinearForm and SymbolicBFI construction",
            "assembly": "first BilinearForm.Assemble for the new pipeline",
            "total": "construction through first assembly",
            "reuse_assembly": "repeated Assemble on the retained pipeline",
            "memory": (
                "fresh subprocess peak-RSS increase while retaining compiled, "
                "assembled pipelines; imports, mesh, space, inputs, and one "
                "warm pipeline excluded"
            ),
        },
        "tolerance": args.tolerance,
        "results": {
            _APPLICATION_NAME: {
                "variants": variants,
                "compact_over_dense": ratios,
                "pipeline_execution_order": pipeline_order,
                "reuse_execution_order": reuse_order,
                "retained_pipeline_memory": memory,
            }
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument("--backend", choices=("scalar", "simd"), default="simd")
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--reuse-warmup", type=int, default=3)
    parser.add_argument("--reuse-iterations", type=int, default=31)
    parser.add_argument(
        "--retained-pipelines",
        type=int,
        default=0,
        help="measure retained prepared-pipeline RSS; 0 disables",
    )
    parser.add_argument(
        "--memory-worker",
        choices=_VARIANTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.maxh <= 0:
        parser.error("--maxh must be positive")
    for name in ("warmup", "reuse_warmup", "retained_pipelines"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.iterations < 1 or args.reuse_iterations < 1:
        parser.error("iteration counts must be positive")
    if args.memory_worker and args.retained_pipelines < 1:
        parser.error("memory worker requires --retained-pipelines > 0")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    return args


def main():
    args = parse_args()
    if args.memory_worker:
        print(
            json.dumps(
                _retained_pipeline_memory_sample(args, args.memory_worker),
                sort_keys=True,
            )
        )
        return
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(rendered)


if __name__ == "__main__":
    main()
