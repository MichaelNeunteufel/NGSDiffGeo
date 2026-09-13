"""Attribution benchmark for compact-form downstream consumers.

The benchmark compares compact public expressions with their exact dense
controls.  It separates one-component expansion, generic componentwise inner
products, Riemannian form inner products and their derivatives, trace/slot
contractions, and Hodge stars.  Tensor-valued consumer results are reduced by
a generic squared norm so every variant can use the same scalar assembly
pipeline. Timing limits remain outside CI; smoke tests validate execution and
values.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
from netgen.occ import unit_cube
from ngsdiffgeo import ngsdiffgeo as cpp
from ngsolve import CF, Id, InnerProduct, Mesh, NumberSpace, Parameter, x, y, z

from benchmark_common import (
    _extension_identity,
    _file_identity,
    _repository_identity,
    _retained_memory_sample,
    _subprocess_json,
    _wrapper_identity,
)
from benchmark_compressed_forms import _workloads
from benchmark_forms_application import (
    _benchmark_pipelines,
    _benchmark_reuse,
    _pipeline_value,
    _relative_error,
    _run_pipeline,
)


_CORE_PAIR_NAMES = (
    "k_component",
    "double_component",
    "k_generic_ip",
    "double_generic_ip",
    "combined_generic_ip",
    "k_metric_ip",
    "double_metric_ip",
    "combined_metric_ip",
    "slot_ip",
)
_DIFFERENTIATED_PAIR_NAMES = (
    "k_metric_ip_diff_input",
    "double_metric_ip_diff_input",
    "k_metric_ip_diff_metric",
    "double_metric_ip_diff_metric",
)
_GEOMETRY_PAIR_NAMES = (
    "trace_once",
    "trace_full",
    "curved_slot_ip",
    "k_hodge",
    "double_hodge",
)
_PAIR_GROUPS = {
    "core": _CORE_PAIR_NAMES,
    "differentiated": _DIFFERENTIATED_PAIR_NAMES,
    "geometry": _GEOMETRY_PAIR_NAMES,
    "all": (
        _CORE_PAIR_NAMES
        + _DIFFERENTIATED_PAIR_NAMES
        + _GEOMETRY_PAIR_NAMES
    ),
}
_VARIANT_NAMES = ("constant",) + tuple(
    name
    for pair_name in _PAIR_GROUPS["all"]
    for name in (f"compact_{pair_name}", f"dense_{pair_name}")
)


def _compact_consumer_cache_storage_lower_bound():
    induced_metric = [
        {
            "dim": 3,
            "degree": degree,
            "bytes": cpp._InducedFormMetricTableStorageBytesLowerBound(
                3, degree
            ),
        }
        for degree in (2, 3)
    ]
    double_trace = [
        {
            "dim": 3,
            "degrees": list(degrees),
            "bytes": cpp._CompactDoubleTraceTableStorageBytesLowerBound(
                3, *degrees
            ),
        }
        for degrees in ((2, 2), (1, 1))
    ]
    hodge_map = [
        {
            "dim": 3,
            "degree": 2,
            "bytes": cpp._CompactHodgeMapTableStorageBytesLowerBound(3, 2),
        }
    ]
    families = (induced_metric, double_trace, hodge_map)
    return {
        "kind": "owned_storage_lower_bound",
        "induced_metric_tables": induced_metric,
        "double_trace_tables": double_trace,
        "hodge_map_tables": hodge_map,
        "total_bytes": sum(
            item["bytes"] for family in families for item in family
        ),
        "excluded": "allocator bookkeeping and shared_ptr control blocks",
    }


def _factories():
    workloads = _workloads()
    kform = workloads["kform_wedge_d_3d"]
    double_form = workloads["double_form_arithmetic_22_3d"]
    manifold = dg.RiemannianManifold(Id(3))
    curved_metric = CF(
        (
            2 + x,
            0.1,
            0.05,
            0.1,
            3 + y,
            0.15,
            0.05,
            0.15,
            4 + z,
        ),
        dims=(3, 3),
    )
    curved_manifold = dg.RiemannianManifold(curved_metric)
    input_parameter = Parameter(1.25)
    input_parameter_form = dg.ScalarField(input_parameter, dim=3)
    metric_parameter = Parameter(0.75)
    differentiated_metric = curved_metric + metric_parameter * Id(3)
    differentiated_manifold = dg.RiemannianManifold(differentiated_metric)

    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    beta = dg.OneForm(CF((2 - x, 1 + y * z, 1 + z)))

    def compact_k2():
        return dg.Wedge(alpha, beta)

    def dense_k2():
        return cpp._WedgeDenseKForms(alpha, beta)

    def component(factory, index):
        def build():
            value = factory().coef[index]
            return value * value

        return build

    def generic(factory):
        def build():
            value = factory()
            return InnerProduct(value, value)

        return build

    def metric(factory):
        def build():
            value = factory()
            return manifold.InnerProduct(value, value, forms=True)

        return build

    def combined(first, second):
        return lambda: first() + 0.125 * second()

    def slot(factory):
        return lambda: manifold.SlotInnerProduct(factory(), forms=True)

    def differentiated_input(factory):
        def build():
            left = input_parameter_form * factory()
            right = factory()
            value = curved_manifold.InnerProduct(
                left, right, forms=True
            )
            return value.Diff(input_parameter, CF(1))

        return build

    def differentiated_metric_inner_product(factory):
        def build():
            value = factory()
            inner_product = differentiated_manifold.InnerProduct(
                value, value, forms=True
            )
            return inner_product.Diff(metric_parameter, CF(1))

        return build

    def squared_norm(value):
        return value * value if not value.dims else InnerProduct(value, value)

    def trace(factory, contractions):
        def build():
            value = curved_manifold.Trace(factory(), l=contractions)
            return squared_norm(value)

        return build

    def curved_slot(factory):
        return lambda: curved_manifold.SlotInnerProduct(
            factory(), forms=True
        )

    def hodge(factory):
        def build():
            value = dg.star(factory(), curved_manifold)
            return squared_norm(value)

        return build

    compact_k_generic = generic(kform["factory"])
    dense_k_generic = generic(kform["dense_factory"])
    compact_double_generic = generic(double_form["factory"])
    dense_double_generic = generic(double_form["dense_factory"])
    compact_k_metric = metric(kform["factory"])
    dense_k_metric = metric(kform["dense_factory"])
    compact_double_metric = metric(double_form["factory"])
    dense_double_metric = metric(double_form["dense_factory"])

    return {
        "constant": lambda: CF(1.0),
        "compact_k_component": component(kform["factory"], (0, 1, 2)),
        "dense_k_component": component(kform["dense_factory"], (0, 1, 2)),
        "compact_double_component": component(
            double_form["factory"], (0, 1, 0, 1)
        ),
        "dense_double_component": component(
            double_form["dense_factory"], (0, 1, 0, 1)
        ),
        "compact_k_generic_ip": compact_k_generic,
        "dense_k_generic_ip": dense_k_generic,
        "compact_double_generic_ip": compact_double_generic,
        "dense_double_generic_ip": dense_double_generic,
        "compact_combined_generic_ip": combined(
            compact_k_generic, compact_double_generic
        ),
        "dense_combined_generic_ip": combined(
            dense_k_generic, dense_double_generic
        ),
        "compact_k_metric_ip": compact_k_metric,
        "dense_k_metric_ip": dense_k_metric,
        "compact_double_metric_ip": compact_double_metric,
        "dense_double_metric_ip": dense_double_metric,
        "compact_combined_metric_ip": combined(
            compact_k_metric, compact_double_metric
        ),
        "dense_combined_metric_ip": combined(
            dense_k_metric, dense_double_metric
        ),
        "compact_slot_ip": slot(double_form["factory"]),
        "dense_slot_ip": slot(double_form["dense_factory"]),
        "compact_k_metric_ip_diff_input": differentiated_input(
            kform["factory"]
        ),
        "dense_k_metric_ip_diff_input": differentiated_input(
            kform["dense_factory"]
        ),
        "compact_double_metric_ip_diff_input": differentiated_input(
            double_form["factory"]
        ),
        "dense_double_metric_ip_diff_input": differentiated_input(
            double_form["dense_factory"]
        ),
        "compact_k_metric_ip_diff_metric": (
            differentiated_metric_inner_product(kform["factory"])
        ),
        "dense_k_metric_ip_diff_metric": (
            differentiated_metric_inner_product(kform["dense_factory"])
        ),
        "compact_double_metric_ip_diff_metric": (
            differentiated_metric_inner_product(double_form["factory"])
        ),
        "dense_double_metric_ip_diff_metric": (
            differentiated_metric_inner_product(double_form["dense_factory"])
        ),
        "compact_trace_once": trace(double_form["factory"], 1),
        "dense_trace_once": trace(double_form["dense_factory"], 1),
        "compact_trace_full": trace(double_form["factory"], 2),
        "dense_trace_full": trace(double_form["dense_factory"], 2),
        "compact_curved_slot_ip": curved_slot(double_form["factory"]),
        "dense_curved_slot_ip": curved_slot(double_form["dense_factory"]),
        "compact_k_hodge": hodge(compact_k2),
        "dense_k_hodge": hodge(dense_k2),
        "compact_double_hodge": hodge(double_form["factory"]),
        "dense_double_hodge": hodge(double_form["dense_factory"]),
    }


def _paired_results(variants, pair_names):
    pairs = {}
    for consumer in pair_names:
        compact = variants[f"compact_{consumer}"]
        dense = variants[f"dense_{consumer}"]
        pairs[consumer] = {
            "phases": {
                phase: (
                    compact["phases"][phase]["median_seconds"]
                    / dense["phases"][phase]["median_seconds"]
                )
                for phase in compact["phases"]
            },
            "reuse_assembly": (
                compact["reuse_assembly"]["median_seconds"]
                / dense["reuse_assembly"]["median_seconds"]
            ),
            "relative_value_error": _relative_error(
                compact["value_norm"], dense["value_norm"]
            ),
        }
    return pairs


def _retained_pipeline_memory_sample(args, variant):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    factory = _factories()[variant]
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
        context_factory=ngsolve.TaskManager,
    )


def _retained_pipeline_memory(args, variants):
    samples = {}
    for variant in variants:
        samples[variant] = _subprocess_json(
            [
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
        )
    return samples


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    simd = args.backend == "simd"
    pair_names = _PAIR_GROUPS[args.group]
    selected_names = args.variant or (
        (["constant"] if args.group in {"core", "all"} else [])
        + [
            name
            for pair_name in pair_names
            for name in (f"compact_{pair_name}", f"dense_{pair_name}")
        ]
    )
    factories = _factories()
    selected = {name: factories[name] for name in selected_names}
    bundles, phases, pipeline_order = _benchmark_pipelines(
        selected,
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
    variants = {}
    for name, bundle in bundles.items():
        variants[name] = {
            "phases": phases[name],
            "reuse_assembly": reuse[name],
            "value_norm": _pipeline_value(bundle),
            "coefficient_tree": dg.CFStats(bundle[0]),
            "compiled_tree": dg.CFStats(bundle[1]),
            "simd_requested": simd,
            "simd_active": bool(bundle[3].simd_evaluate),
        }
    pairs = _paired_results(variants, pair_names) if not args.variant else {}
    if pairs and max(pair["relative_value_error"] for pair in pairs.values()) > args.tolerance:
        raise RuntimeError("compact/dense consumer values disagree")
    return {
        "benchmark": "compact_form_consumer_attribution",
        "label": args.label,
        "backend": args.backend,
        "group": args.group,
        "variants": selected_names,
        "ngsolve_version": ngsolve.__version__,
        "ngsdiffgeo_version": getattr(dg, "__version__", "unknown"),
        "ngsdiffgeo_extension": _extension_identity(),
        "ngsdiffgeo_wrapper": _wrapper_identity(),
        "benchmark_script": _file_identity(__file__),
        "repository": _repository_identity(),
        "platform": platform.platform(),
        "threads": args.threads,
        "maxh": args.maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "reuse_warmup": args.reuse_warmup,
        "reuse_iterations": args.reuse_iterations,
        "retained_pipelines": args.retained_pipelines,
        "cache_container_storage_bytes": (
            cpp._CompactFormCacheContainerStorageBytes()
        ),
        "cache_container_storage_breakdown_bytes": dict(
            cpp._CompactFormCacheContainerStorageBreakdown()
        ),
        "cache_storage_lower_bound": (
            _compact_consumer_cache_storage_lower_bound()
        ),
        "tolerance": args.tolerance,
        "measurement_boundaries": {
            "construction": "form factory plus selected scalar consumer",
            "compile": "Compile(realcompile=False)",
            "form_setup": "BilinearForm plus SymbolicBFI",
            "assembly": "first Assemble on the new pipeline",
            "reuse_assembly": "Assemble on a retained pipeline",
            "memory": (
                "fresh subprocess peak-RSS increase while retaining "
                "compiled, assembled pipelines; imports, mesh, space, "
                "inputs, and one warm pipeline excluded"
            ),
        },
        "results": {
            "variants": variants,
            "compact_over_dense": pairs,
            "pipeline_execution_order": pipeline_order,
            "reuse_execution_order": reuse_order,
            "retained_pipeline_memory": (
                _retained_pipeline_memory(args, selected_names)
                if args.retained_pipelines
                else None
            ),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument("--backend", choices=("scalar", "simd"), default="simd")
    parser.add_argument(
        "--group", choices=tuple(_PAIR_GROUPS), default="all"
    )
    parser.add_argument("--variant", action="append", choices=_VARIANT_NAMES)
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--reuse-warmup", type=int, default=3)
    parser.add_argument("--reuse-iterations", type=int, default=31)
    parser.add_argument("--retained-pipelines", type=int, default=0)
    parser.add_argument(
        "--memory-worker", choices=_VARIANT_NAMES, help=argparse.SUPPRESS
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.maxh <= 0 or args.tolerance <= 0:
        parser.error("--maxh and --tolerance must be positive")
    for name in ("warmup", "reuse_warmup", "retained_pipelines"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("iterations", "reuse_iterations", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.memory_worker and args.retained_pipelines < 1:
        parser.error("memory worker requires --retained-pipelines > 0")
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
