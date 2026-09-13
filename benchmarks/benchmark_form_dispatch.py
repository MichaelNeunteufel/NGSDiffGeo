"""Attribute public/native construction overhead for compressed forms.

This diagnostic benchmark isolates individual public operations from the
corresponding direct pybind calls.  It complements, but does not replace, the
dense/compact performance benchmark: there is no dense candidate here because
the question is where the remaining Python API overhead is spent.

Every timed candidate operates on pre-built inputs.  Public/native values and
semantic metadata are checked before results are reported.  Optional cProfile
data records calls per complete construction; its timings are diagnostic only.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import platform
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
from ngsdiffgeo import ngsdiffgeo as cpp
from netgen.occ import unit_cube
from ngsolve import CF, Mesh, x, y, z

from benchmark_common import (
    _extension_identity,
    _file_identity,
    _repository_identity,
    _wrapper_identity,
)
from benchmark_compressed_forms import _benchmark_factory_pair, _l2_error


_OPERATION_NAMES = (
    "kform_wedge_1_1",
    "kform_wedge_2_1",
    "kform_exterior_derivative_2",
    "kform_scale_3",
    "kform_scale_numeric_3",
    "kform_add_3",
    "kform_full",
    "double_form_wedge_10_01",
    "double_form_wedge_11_10",
    "double_form_wedge_21_01",
    "double_form_scale_numeric_11",
    "double_form_chain_full",
)

_PROFILED_OPERATION_NAMES = (
    "kform_full",
    "double_form_chain_full",
)

_PROFILE_FUNCTIONS = {
    "Wedge",
    "_wedge_typed_doubleforms",
    "_common_form_dimension",
    "_infer_dim",
    "_call_if_callable",
    "as_kform",
    "d",
    "_wrap",
    "_add",
    "_scale_form",
    "__add__",
    "__mul__",
    "__rmul__",
}


def _operations():
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    beta = dg.OneForm(CF((2 - x, 1 + y * z, 1 + z)))
    gamma = dg.OneForm(CF((1 + y, 2 + z, 3 + x * y)))
    delta = dg.OneForm(CF((2 + x * y, 1 - z, 2 + y)))

    pair_public = dg.Wedge(alpha, beta)
    pair_native = cpp.Wedge(alpha, beta)
    leading_public = dg.Wedge(pair_public, gamma)
    leading_native = cpp.Wedge(pair_native, gamma)
    exterior_public = dg.d(pair_public)
    exterior_native = cpp.d(pair_native)
    quarter = CF(0.25)
    scaled_public = exterior_public * quarter
    scaled_native = cpp._ScaleKForm(exterior_native, quarter)

    def kform_full_public():
        pair = dg.Wedge(alpha, beta)
        return dg.Wedge(pair, gamma) + 0.25 * dg.d(pair)

    def kform_full_native():
        pair = cpp.Wedge(alpha, beta)
        return cpp._AddKForms(
            cpp.Wedge(pair, gamma),
            cpp._ScaleKFormConstant(cpp.d(pair), 0.25),
        )

    alpha_left = dg.DoubleForm(alpha.coef, p=1, q=0, dim=3)
    beta_right = dg.DoubleForm(beta.coef, p=0, q=1, dim=3)
    gamma_left = dg.DoubleForm(gamma.coef, p=1, q=0, dim=3)
    delta_right = dg.DoubleForm(delta.coef, p=0, q=1, dim=3)
    double_first_public = dg.Wedge(alpha_left, beta_right)
    double_first_native = cpp.Wedge(alpha_left, beta_right)
    double_second_public = dg.Wedge(double_first_public, gamma_left)
    double_second_native = cpp.Wedge(double_first_native, gamma_left)

    def double_chain_public():
        first = dg.Wedge(alpha_left, beta_right)
        second = dg.Wedge(first, gamma_left)
        return dg.Wedge(second, delta_right)

    def double_chain_native():
        first = cpp.Wedge(alpha_left, beta_right)
        second = cpp.Wedge(first, gamma_left)
        return cpp.Wedge(second, delta_right)

    return {
        "kform_wedge_1_1": {
            "public": lambda: dg.Wedge(alpha, beta),
            "native": lambda: cpp.Wedge(alpha, beta),
        },
        "kform_wedge_2_1": {
            "public": lambda: dg.Wedge(pair_public, gamma),
            "native": lambda: cpp.Wedge(pair_native, gamma),
        },
        "kform_exterior_derivative_2": {
            "public": lambda: dg.d(pair_public),
            "native": lambda: cpp.d(pair_native),
        },
        "kform_scale_3": {
            "public": lambda: exterior_public * quarter,
            "native": lambda: cpp._ScaleKForm(exterior_native, quarter),
        },
        "kform_scale_numeric_3": {
            "public": lambda: 0.25 * exterior_public,
            "native": lambda: cpp._ScaleKFormConstant(
                exterior_native, 0.25
            ),
        },
        "kform_add_3": {
            "public": lambda: leading_public + scaled_public,
            "native": lambda: cpp._AddKForms(leading_native, scaled_native),
        },
        "kform_full": {
            "public": kform_full_public,
            "native": kform_full_native,
        },
        "double_form_wedge_10_01": {
            "public": lambda: dg.Wedge(alpha_left, beta_right),
            "native": lambda: cpp.Wedge(alpha_left, beta_right),
        },
        "double_form_wedge_11_10": {
            "public": lambda: dg.Wedge(double_first_public, gamma_left),
            "native": lambda: cpp.Wedge(double_first_native, gamma_left),
        },
        "double_form_wedge_21_01": {
            "public": lambda: dg.Wedge(double_second_public, delta_right),
            "native": lambda: cpp.Wedge(double_second_native, delta_right),
        },
        "double_form_scale_numeric_11": {
            "public": lambda: 0.5 * double_first_public,
            "native": lambda: cpp._ScaleDoubleFormConstant(
                double_first_native, 0.5
            ),
        },
        "double_form_chain_full": {
            "public": double_chain_public,
            "native": double_chain_native,
        },
    }


def _semantic_metadata(value):
    metadata = {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "dims": list(value.dims),
        "dim_space": int(value.dim_space),
    }
    if isinstance(value, cpp.DoubleForm):
        metadata.update(
            degree_left=int(value.degree_left),
            degree_right=int(value.degree_right),
        )
    else:
        metadata["degree"] = int(value.degree)
    return metadata


def _profile_calls(factory, iterations):
    profiler = cProfile.Profile()
    profiler.enable()
    candidate = None
    for _ in range(iterations):
        candidate = factory()
    profiler.disable()
    assert candidate is not None

    calls = []
    for entry in profiler.getstats():
        code = entry.code
        if not hasattr(code, "co_name"):
            continue
        if code.co_name not in _PROFILE_FUNCTIONS:
            continue
        if not code.co_filename.endswith("ngsdiffgeo/wrappers.py"):
            continue
        calls.append(
            {
                "function": code.co_name,
                "line": int(code.co_firstlineno),
                "calls": int(entry.callcount),
                "calls_per_construction": entry.callcount / iterations,
                "inline_seconds_per_construction": entry.inlinetime / iterations,
                "cumulative_seconds_per_construction": entry.totaltime / iterations,
            }
        )
    calls.sort(key=lambda item: (item["line"], item["function"]))
    return {
        "kind": "cprofile_diagnostic",
        "iterations": iterations,
        "functions": calls,
    }


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    operations = _operations()
    selected = args.operation or list(_OPERATION_NAMES)
    results = {}

    for name in selected:
        factories = operations[name]
        outputs, statistics, execution_order = _benchmark_factory_pair(
            factories,
            args.warmup,
            args.iterations,
        )
        error = _l2_error(outputs["public"], outputs["native"], mesh)
        public_metadata = _semantic_metadata(outputs["public"])
        native_metadata = _semantic_metadata(outputs["native"])
        semantic_metadata_equal = {
            key: value
            for key, value in public_metadata.items()
            if key != "python_type"
        } == {
            key: value
            for key, value in native_metadata.items()
            if key != "python_type"
        }
        if error > args.tolerance or not semantic_metadata_equal:
            raise RuntimeError(
                f"{name} public/native disagreement: error={error:.3e}, "
                f"public={public_metadata}, native={native_metadata}"
            )

        public_median = statistics["public"]["median_seconds"]
        native_median = statistics["native"]["median_seconds"]
        result = {
            "public": statistics["public"],
            "native": statistics["native"],
            "public_over_native": (
                public_median / native_median if native_median else None
            ),
            "public_minus_native_seconds": public_median - native_median,
            "execution_order": execution_order,
            "l2_error": error,
            "semantic_metadata_equal": semantic_metadata_equal,
            "public_metadata": public_metadata,
            "native_metadata": native_metadata,
            "public_tree": dg.CFStats(outputs["public"]),
            "native_tree": dg.CFStats(outputs["native"]),
            "profile": None,
        }
        if args.profile_iterations and name in _PROFILED_OPERATION_NAMES:
            result["profile"] = _profile_calls(
                factories["public"], args.profile_iterations
            )
        results[name] = result

    return {
        "benchmark": "compressed_forms_dispatch_attribution",
        "label": args.label,
        "operations": selected,
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
        "clock": "time.perf_counter_ns",
        "maxh": args.maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "measurement_boundary": (
            "counterbalanced calls on pre-built inputs; previous result "
            "destruction excluded; cProfile timings are diagnostic only"
        ),
        "tolerance": args.tolerance,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument(
        "--operation",
        action="append",
        choices=_OPERATION_NAMES,
        help="run only this operation; repeat to select multiple operations",
    )
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--profile-iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.maxh <= 0:
        parser.error("--maxh must be positive")
    if args.warmup < 0 or args.profile_iterations < 0:
        parser.error("warmup/profile iteration counts must be nonnegative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
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
