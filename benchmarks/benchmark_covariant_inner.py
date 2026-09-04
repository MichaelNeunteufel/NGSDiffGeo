"""Compare default and graph-compiled covariant-derivative evaluation.

The benchmark separates one-time operator construction from repeated assembly.
It validates numerical equivalence but deliberately enforces no timing limit.
The workload isolates the volume ``CovDeriv`` core shared by the higher-level
covariant exterior and codifferential operations.
Run it from the repository root after installing the current extension.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import ngsolve
import ngsdiffgeo as dg
from netgen.occ import unit_square
from ngsolve import (
    BilinearForm,
    CF,
    InnerProduct,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
    cos,
    sin,
    x,
    y,
)


def make_input(depth):
    value = x + 2 * y + x * y
    for level in range(depth):
        scale = 0.05 / (level + 1)
        value = (
            sin(value + scale * x)
            + cos((level + 1) * x * y)
            + scale * value**2
        )
    return dg.TensorField(
        CF(
            (
                value,
                value + x,
                value + y,
                value + x * y,
            ),
            dims=(2, 2),
        ),
        "10",
    )


def make_manifold():
    scale = 1 + x**2 + y**2
    metric = CF((scale, 0, 0, scale), dims=(2, 2))
    manifold = dg.RiemannianManifold(metric)
    manifold.Christoffel(second_kind=True)
    return manifold


def benchmark_construction(manifold, tensor, iterations):
    timings = {"default": [], "graph": []}
    outputs = {}
    for iteration in range(iterations):
        order = (
            ("default", "graph")
            if iteration % 2 == 0
            else ("graph", "default")
        )
        for name in order:
            started = time.perf_counter()
            outputs[name] = manifold.CovDeriv(
                tensor,
                compile_inner="graph" if name == "graph" else False,
            )
            timings[name].append(time.perf_counter() - started)
    return outputs, {
        name: {
            "minimum_seconds": min(samples),
            "median_seconds": statistics.median(samples),
        }
        for name, samples in timings.items()
    }


def make_form(coefficient, mesh, *, simd, bonus_intorder):
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    form = BilinearForm(space)
    integrator = SymbolicBFI(
        InnerProduct(coefficient, coefficient) * trial * test,
        simd_evaluate=simd,
        bonus_intorder=bonus_intorder,
    )
    form += integrator
    return form, integrator


def benchmark_assembly(forms, *, warmup, iterations):
    timings = {name: [] for name in forms}
    with TaskManager():
        for _ in range(warmup):
            for form, _ in forms.values():
                form.Assemble()
        for iteration in range(iterations):
            order = (
                ("default", "graph")
                if iteration % 2 == 0
                else ("graph", "default")
            )
            for name in order:
                form, _ = forms[name]
                started = time.perf_counter()
                form.Assemble()
                timings[name].append(time.perf_counter() - started)

    return {
        name: {
            "minimum_seconds": min(samples),
            "median_seconds": statistics.median(samples),
            "value_norm": Norm(forms[name][0].mat.AsVector()),
            "simd_active": bool(forms[name][1].simd_evaluate),
        }
        for name, samples in timings.items()
    }


def run(args):
    mesh = Mesh(unit_square.GenerateMesh(maxh=args.maxh))
    manifold = make_manifold()
    tensor = make_input(args.depth)
    outputs, construction = benchmark_construction(
        manifold,
        tensor,
        args.construction_iterations,
    )
    forms = {
        name: make_form(
            output,
            mesh,
            simd=args.simd,
            bonus_intorder=args.bonus_intorder,
        )
        for name, output in outputs.items()
    }
    assembly = benchmark_assembly(
        forms,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    reference = assembly["default"]["value_norm"]
    relative_error = abs(assembly["graph"]["value_norm"] - reference) / max(
        1.0, abs(reference)
    )
    if relative_error > 2e-8:
        raise RuntimeError(
            "graph-compiled covariant derivative disagrees with the default: "
            f"relative error {relative_error:.3e}"
        )

    default_assembly = assembly["default"]["median_seconds"]
    graph_assembly = assembly["graph"]["median_seconds"]
    construction_overhead = max(
        0.0,
        construction["graph"]["median_seconds"]
        - construction["default"]["median_seconds"],
    )
    savings_per_assembly = default_assembly - graph_assembly
    if savings_per_assembly > 0.0:
        break_even_assemblies = (
            math.ceil(construction_overhead / savings_per_assembly)
            if construction_overhead > 0.0
            else 0
        )
    else:
        break_even_assemblies = None

    return {
        "ngsolve_version": ngsolve.__version__,
        "depth": args.depth,
        "maxh": args.maxh,
        "elements": mesh.ne,
        "simd_requested": args.simd,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "construction_iterations": args.construction_iterations,
        "input_graph": dg.CFStats(tensor.coef),
        "output_graphs": {
            name: dg.CFStats(output.coef) for name, output in outputs.items()
        },
        "construction": construction,
        "assembly": assembly,
        "relative_value_error": relative_error,
        "assembly_speedup": default_assembly / graph_assembly,
        "break_even_assemblies": break_even_assemblies,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--maxh", type=float, default=0.12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--construction-iterations", type=int, default=5)
    parser.add_argument("--bonus-intorder", type=int, default=4)
    parser.add_argument(
        "--simd",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.depth < 0:
        parser.error("--depth must be nonnegative")
    if args.maxh <= 0:
        parser.error("--maxh must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.construction_iterations < 1:
        parser.error("--construction-iterations must be positive")
    return args


def main():
    args = parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(
        f"NGSolve {report['ngsolve_version']}; depth={report['depth']}; "
        f"elements={report['elements']}; simd={report['simd_requested']}"
    )
    for stage in ("construction", "assembly"):
        print(stage)
        for name, result in report[stage].items():
            print(
                f"  {name:8} median={result['median_seconds']:.6f}s "
                f"min={result['minimum_seconds']:.6f}s"
            )
    print(f"assembly speedup: {report['assembly_speedup']:.3f}x")
    print(f"relative value error: {report['relative_value_error']:.3e}")
    print(f"break-even assemblies: {report['break_even_assemblies']}")


if __name__ == "__main__":
    main()
