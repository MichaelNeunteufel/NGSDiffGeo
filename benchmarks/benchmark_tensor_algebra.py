"""Benchmark the low-rank metric tensor-algebra paths.

The report records correctness, construction time, assembly time, and
expression-tree size for ``Raise``, ``Lower``, ``InnerProduct``, and ``Trace``.
It deliberately has no timing assertions: use ``--output`` to retain reports
from two builds and compare them on the same machine.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
import importlib
from netgen.occ import unit_square
from ngsolve import (
    BilinearForm,
    CF,
    InnerProduct,
    Integrate,
    Inv,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
    x,
    y,
)


def _cases(manifold, metric):
    vector = dg.VectorField(CF((1 + x, 2 - y)))
    one_form = dg.OneForm(CF((2 + y, 1 - x)))
    covariant = dg.TensorField(
        CF((1 + x, x * y, x * y, 2 + y), dims=(2, 2)), "11"
    )
    contravariant = dg.TensorField(
        CF((2 - x, x + y, x + y, 3 - y), dims=(2, 2)), "00"
    )
    return {
        "raise_rank1": (
            lambda: manifold.Raise(one_form),
            lambda: dg.VectorField(Inv(metric) * one_form.coef),
        ),
        "raise_rank2_axis0": (
            lambda: manifold.Raise(covariant, 0),
            lambda: dg.TensorField(Inv(metric) * covariant.coef, "01"),
        ),
        "raise_rank2_axis1": (
            lambda: manifold.Raise(covariant, 1),
            lambda: dg.TensorField(covariant.coef * Inv(metric), "10"),
        ),
        "lower_rank1": (
            lambda: manifold.Lower(vector),
            lambda: dg.OneForm(metric * vector.coef),
        ),
        "inner_product_rank1": (
            lambda: manifold.InnerProduct(vector, vector),
            lambda: dg.ScalarField(InnerProduct(metric * vector.coef, vector.coef), dim=2),
        ),
        "trace_rank2": (
            lambda: manifold.Trace(covariant),
            lambda: dg.ScalarField(InnerProduct(Inv(metric), covariant.coef), dim=2),
        ),
        "lower_rank2_axis1": (
            lambda: manifold.Lower(contravariant, 1),
            lambda: dg.TensorField(contravariant.coef * metric, "01"),
        ),
    }


def _construction_statistics(factory, iterations):
    samples = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = factory()
        samples.append((time.perf_counter_ns() - started) * 1e-9)
    return output, {
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
    }


def _make_form(coefficient, mesh):
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    integrand = coefficient * coefficient if not coefficient.dims else InnerProduct(
        coefficient, coefficient
    )
    form = BilinearForm(space)
    form += SymbolicBFI(integrand * trial * test, simd_evaluate=True)
    return form


def _assembly_statistics(form, warmup, iterations):
    samples = []
    with TaskManager():
        for _ in range(warmup):
            form.Assemble()
        for _ in range(iterations):
            started = time.perf_counter()
            form.Assemble()
            samples.append(time.perf_counter() - started)
    return {
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "value_norm": Norm(form.mat.AsVector()),
    }


def _benchmark_raise_alternatives(manifold, metric, mesh, args):
    cpp = importlib.import_module("ngsdiffgeo.ngsdiffgeo")
    metric_inv = Inv(metric)
    one_form = dg.OneForm(CF((2 + y, 1 - x)))
    covariant = dg.TensorField(
        CF((1 + x, x * y, x * y, 2 + y), dims=(2, 2)), "11"
    )
    alternatives = {
        "rank1": (
            lambda: manifold.Raise(one_form),
            lambda: dg.VectorField(
                cpp._EinsumCoefficient("b,ab->a", [one_form, metric_inv])
            ),
        ),
        "rank2_axis0": (
            lambda: manifold.Raise(covariant, 0),
            lambda: dg.TensorField(
                cpp._EinsumCoefficient("cb,ac->ab", [covariant, metric_inv]),
                "01",
            ),
        ),
        "rank2_axis1": (
            lambda: manifold.Raise(covariant, 1),
            lambda: dg.TensorField(
                cpp._EinsumCoefficient("ac,bc->ab", [covariant, metric_inv]),
                "10",
            ),
        ),
    }
    report = {}
    for name, (specialized_factory, einsum_factory) in alternatives.items():
        construction_samples = {"matrix_product": [], "einsum": []}
        outputs = {}
        for iteration in range(args.construction_iterations):
            order = (
                (("matrix_product", specialized_factory), ("einsum", einsum_factory))
                if iteration % 2 == 0
                else (("einsum", einsum_factory), ("matrix_product", specialized_factory))
            )
            for variant, factory in order:
                started = time.perf_counter_ns()
                outputs[variant] = factory()
                construction_samples[variant].append(
                    (time.perf_counter_ns() - started) * 1e-9
                )

        difference = outputs["matrix_product"] - outputs["einsum"]
        difference_sq = (
            difference * difference
            if not difference.dims
            else InnerProduct(difference, difference)
        )
        l2_error = math.sqrt(max(0.0, Integrate(difference_sq, mesh)))
        if l2_error > 2e-10:
            raise RuntimeError(
                f"Raise {name} alternatives disagree: {l2_error:.3e}"
            )

        forms = {variant: _make_form(output, mesh) for variant, output in outputs.items()}
        assembly_samples = {"matrix_product": [], "einsum": []}
        with TaskManager():
            for _ in range(args.warmup):
                for form in forms.values():
                    form.Assemble()
            for iteration in range(args.iterations):
                order = (
                    ("matrix_product", "einsum")
                    if iteration % 2 == 0
                    else ("einsum", "matrix_product")
                )
                for variant in order:
                    started = time.perf_counter()
                    forms[variant].Assemble()
                    assembly_samples[variant].append(time.perf_counter() - started)

        variants = {}
        for variant in ("matrix_product", "einsum"):
            variants[variant] = {
                "construction": {
                    "minimum_seconds": min(construction_samples[variant]),
                    "median_seconds": statistics.median(construction_samples[variant]),
                },
                "assembly": {
                    "minimum_seconds": min(assembly_samples[variant]),
                    "median_seconds": statistics.median(assembly_samples[variant]),
                    "value_norm": Norm(forms[variant].mat.AsVector()),
                },
                "tree": dg.CFStats(outputs[variant]),
            }
        variants["construction_speedup"] = (
            variants["einsum"]["construction"]["median_seconds"]
            / variants["matrix_product"]["construction"]["median_seconds"]
        )
        variants["assembly_speedup"] = (
            variants["einsum"]["assembly"]["median_seconds"]
            / variants["matrix_product"]["assembly"]["median_seconds"]
        )
        variants["l2_error"] = l2_error
        report[name] = variants
    return report


def run(args):
    mesh = Mesh(unit_square.GenerateMesh(maxh=args.maxh))
    metric = CF((1 + x * x, 0.1 * x * y, 0.1 * x * y, 1 + y * y), dims=(2, 2))
    manifold = dg.RiemannianManifold(metric)
    results = {}

    for name, (factory, reference_factory) in _cases(manifold, metric).items():
        output, construction = _construction_statistics(
            factory, args.construction_iterations
        )
        reference = reference_factory()
        error = output - reference
        error_sq = error * error if not error.dims else InnerProduct(error, error)
        l2_error = math.sqrt(max(0.0, Integrate(error_sq, mesh)))
        if l2_error > 2e-10:
            raise RuntimeError(f"{name} disagrees with its reference: {l2_error:.3e}")
        form = _make_form(output, mesh)
        results[name] = {
            "construction": construction,
            "assembly": _assembly_statistics(form, args.warmup, args.iterations),
            "tree": dg.CFStats(output),
            "l2_error": l2_error,
        }

    return {
        "label": args.label,
        "ngsolve_version": ngsolve.__version__,
        "platform": platform.platform(),
        "maxh": args.maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "construction_iterations": args.construction_iterations,
        "results": results,
        "raise_alternatives": _benchmark_raise_alternatives(
            manifold, metric, mesh, args
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument("--maxh", type=float, default=0.12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--construction-iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.maxh <= 0:
        parser.error("--maxh must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.iterations < 1 or args.construction_iterations < 1:
        parser.error("iteration counts must be positive")
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
