"""Compare GradCF scalar/SIMD assembly with an analytic-gradient baseline.

This is intentionally a standalone benchmark rather than a correctness test.
Run it from the repository root after installing the current extension.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import ngsolve
import ngsdiffgeo as dg
from netgen.occ import unit_cube, unit_square
from ngsolve import (
    BND,
    BilinearForm,
    CF,
    InnerProduct,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
    specialcf,
    x,
    y,
    z,
)


def make_form(coefficient, mesh, *, mode, simd, bonus_intorder):
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    form = BilinearForm(space)
    kwargs = {
        "simd_evaluate": simd,
        "bonus_intorder": bonus_intorder,
    }
    if mode == "element-boundary":
        kwargs["element_vb"] = BND
    if mode == "surface-boundary":
        integrator = SymbolicBFI(coefficient * trial * test, BND, **kwargs)
    else:
        integrator = SymbolicBFI(coefficient * trial * test, **kwargs)
    form += integrator
    return form, integrator


def benchmark_form(form, *, warmup, iterations):
    timings = []
    with TaskManager():
        for _ in range(warmup):
            form.Assemble()
        for _ in range(iterations):
            started = time.perf_counter()
            form.Assemble()
            timings.append(time.perf_counter() - started)
    return {
        "minimum_seconds": min(timings),
        "median_seconds": statistics.median(timings),
        "value_norm": Norm(form.mat.AsVector()),
    }


def run(args):
    geometry = unit_square if args.dim == 2 else unit_cube
    maxh = args.maxh if args.maxh is not None else (0.08 if args.dim == 2 else 0.2)
    mesh = Mesh(geometry.GenerateMesh(maxh=maxh))
    coordinates = (x, y, z)[: args.dim]
    coefficient = sum(
        (direction + 1) * coordinate**3
        + (direction + 2) * coordinate**2
        for direction, coordinate in enumerate(coordinates)
    )
    analytic_gradient = CF(
        tuple(coefficient.Diff(coordinate) for coordinate in coordinates),
        dims=(args.dim,),
    )
    surface = args.mode == "surface-boundary"
    if surface:
        normal = specialcf.normal(args.dim)
        analytic_gradient = analytic_gradient - normal * InnerProduct(
            normal, analytic_gradient
        )
    compiled_coefficient = coefficient.Compile(
        realcompile=False,
        wait=True,
        maxderiv=0,
    )
    numerical_gradient = dg.GradCF(
        compiled_coefficient,
        args.dim,
        surface=surface,
    )

    cases = {
        "analytic_scalar": (InnerProduct(analytic_gradient, analytic_gradient), False),
        "analytic_simd": (InnerProduct(analytic_gradient, analytic_gradient), True),
        "gradcf_scalar": (InnerProduct(numerical_gradient, numerical_gradient), False),
        "gradcf_simd": (InnerProduct(numerical_gradient, numerical_gradient), True),
    }
    results = {}
    for name, (integrand, simd) in cases.items():
        form, integrator = make_form(
            integrand,
            mesh,
            mode=args.mode,
            simd=simd,
            bonus_intorder=args.bonus_intorder,
        )
        result = benchmark_form(
            form,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result["simd_requested"] = simd
        result["simd_active"] = bool(integrator.simd_evaluate)
        results[name] = result

    reference = results["analytic_scalar"]["value_norm"]
    for name, result in results.items():
        result["relative_value_error"] = abs(result["value_norm"] - reference) / max(
            1.0, abs(reference)
        )
        if result["relative_value_error"] > 2e-8:
            raise RuntimeError(
                f"{name} disagrees with the analytic baseline: "
                f"relative error {result['relative_value_error']:.3e}"
            )

    return {
        "ngsolve_version": ngsolve.__version__,
        "mode": args.mode,
        "dimension": args.dim,
        "maxh": maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--mode",
        choices=("volume", "element-boundary", "surface-boundary"),
        default="volume",
    )
    parser.add_argument("--maxh", type=float)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--bonus-intorder", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.maxh is not None and args.maxh <= 0:
        parser.error("--maxh must be positive")
    return args


def main():
    args = parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(
        f"NGSolve {report['ngsolve_version']}; mode={report['mode']}; "
        f"dim={report['dimension']}; "
        f"elements={report['elements']}; maxh={report['maxh']}"
    )
    for name, result in report["results"].items():
        print(
            f"{name:16} median={result['median_seconds']:.6f}s "
            f"min={result['minimum_seconds']:.6f}s "
            f"relerr={result['relative_value_error']:.3e} "
            f"simd={result['simd_active']}"
        )


if __name__ == "__main__":
    main()
