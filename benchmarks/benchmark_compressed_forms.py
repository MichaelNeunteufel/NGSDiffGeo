"""Dense/compact comparison benchmark for the compressed-forms MVP.

Each workload has an optimized public factory, an exact dense implementation,
and a separately constructed mathematical reference. Reports contain isolated
construction costs, coefficient-graph sizes, graph-compilation time, and
interpreted/compiled assembly timings. Timing thresholds deliberately do not
belong in CI.

Dimension-four data is constant and construction-only.  Physical evaluation
uses nonzero, spatially varying coefficients on a three-dimensional mesh.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
from ngsdiffgeo import ngsdiffgeo as cpp
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

from benchmark_common import (
    _alternating_order,
    _extension_identity,
    _file_identity,
    _peak_rss_bytes,
    _repository_identity,
    _retained_memory_sample,
    _statistics,
    _subprocess_json,
    _wrapper_identity,
)

_DEFAULT_WORKLOAD_NAMES = (
    "kform_wedge_d_3d",
    "double_form_wedge_22_3d",
    "double_form_arithmetic_22_3d",
    "double_form_wedge_chain_22_3d",
    "double_form_wedge_22_4d",
    "double_form_wedge_chain_22_4d",
)

_HIGH_ORDER_WORKLOAD_NAMES = (
    "kform_wedge_4_4d",
    "double_form_wedge_31_4d",
    "double_form_wedge_33_4d",
    "double_form_wedge_44_4d",
)

_WORKLOAD_NAMES = _DEFAULT_WORKLOAD_NAMES + _HIGH_ORDER_WORKLOAD_NAMES


def _workloads():
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    beta = dg.OneForm(CF((2 - x, 1 + y * z, 1 + z)))
    gamma = dg.OneForm(CF((1 + y, 2 + z, 3 + x * y)))
    delta = dg.OneForm(CF((2 + x * y, 1 - z, 2 + y)))

    def kform_factory():
        pair = dg.Wedge(alpha, beta)
        return dg.Wedge(pair, gamma) + 0.25 * dg.d(pair)

    def native_kform_factory():
        pair = cpp.Wedge(alpha, beta)
        leading = cpp.Wedge(pair, gamma)
        exterior = cpp.d(pair)
        return cpp._AddKForms(
            leading, cpp._ScaleKFormConstant(exterior, 0.25)
        )

    def dense_kform_factory():
        pair = cpp._WedgeDenseKForms(alpha, beta)
        leading = cpp._WedgeDenseKForms(pair, gamma)
        exterior = cpp._ExteriorDerivativeDenseKForm(pair)
        return cpp._AddKForms(
            leading, cpp._ScaleKFormConstant(exterior, 0.25)
        )

    def kform_reference_factory():
        pair = dg.Wedge(alpha, beta)
        return dg.Wedge(pair, gamma) + 0.25 * (
            dg.Wedge(dg.d(alpha), beta) - dg.Wedge(alpha, dg.d(beta))
        )

    left = dg.DoubleForm(dg.Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=3)
    right = dg.DoubleForm(dg.Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=3)

    def double_form_factory():
        return dg.Wedge(left, right)

    def native_double_form_factory():
        return cpp.Wedge(left, right)

    def dense_double_form_factory():
        return cpp._WedgeDenseDoubleForms(left, right)

    def double_form_arithmetic_factory():
        forward = dg.Wedge(left, right)
        reverse = dg.Wedge(right, left)
        return 0.5 * (forward + reverse)

    def native_double_form_arithmetic_factory():
        forward = cpp.Wedge(left, right)
        reverse = cpp.Wedge(right, left)
        return cpp._ScaleDoubleFormConstant(
            cpp._AddDoubleForms(forward, reverse), 0.5
        )

    def dense_double_form_arithmetic_factory():
        forward = cpp._WedgeDenseDoubleForms(left, right)
        reverse = cpp._WedgeDenseDoubleForms(right, left)
        return cpp._ScaleDoubleFormConstant(
            cpp._AddDoubleForms(forward, reverse), 0.5
        )

    alpha_left = dg.DoubleForm(alpha.coef, p=1, q=0, dim=3)
    beta_right = dg.DoubleForm(beta.coef, p=0, q=1, dim=3)
    gamma_left = dg.DoubleForm(gamma.coef, p=1, q=0, dim=3)
    delta_right = dg.DoubleForm(delta.coef, p=0, q=1, dim=3)

    def double_form_chain_factory():
        first = dg.Wedge(alpha_left, beta_right)
        second = dg.Wedge(first, gamma_left)
        return dg.Wedge(second, delta_right)

    def native_double_form_chain_factory():
        first = cpp.Wedge(alpha_left, beta_right)
        second = cpp.Wedge(first, gamma_left)
        return cpp.Wedge(second, delta_right)

    def dense_double_form_chain_factory():
        first = cpp._WedgeDenseDoubleForms(alpha_left, beta_right)
        second = cpp._WedgeDenseDoubleForms(first, gamma_left)
        return cpp._WedgeDenseDoubleForms(second, delta_right)

    def double_form_reference_factory():
        return dg.DoubleForm(
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

    def native_double_form_4d_factory():
        return cpp.Wedge(left4, right4)

    def dense_double_form_4d_factory():
        return cpp._WedgeDenseDoubleForms(left4, right4)

    alpha4_left = dg.DoubleForm(alpha4.coef, p=1, q=0, dim=4)
    beta4_left = dg.DoubleForm(beta4.coef, p=1, q=0, dim=4)
    gamma4_left = dg.DoubleForm(gamma4.coef, p=1, q=0, dim=4)
    delta4_left = dg.DoubleForm(delta4.coef, p=1, q=0, dim=4)
    alpha4_right = dg.DoubleForm(alpha4.coef, p=0, q=1, dim=4)
    beta4_right = dg.DoubleForm(beta4.coef, p=0, q=1, dim=4)
    gamma4_right = dg.DoubleForm(gamma4.coef, p=0, q=1, dim=4)
    delta4_right = dg.DoubleForm(delta4.coef, p=0, q=1, dim=4)

    def double_form_4d_chain_factory():
        first = dg.Wedge(alpha4_left, beta4_right)
        second = dg.Wedge(first, gamma4_left)
        return dg.Wedge(second, delta4_right)

    def native_double_form_4d_chain_factory():
        first = cpp.Wedge(alpha4_left, beta4_right)
        second = cpp.Wedge(first, gamma4_left)
        return cpp.Wedge(second, delta4_right)

    def dense_double_form_4d_chain_factory():
        first = cpp._WedgeDenseDoubleForms(alpha4_left, beta4_right)
        second = cpp._WedgeDenseDoubleForms(first, gamma4_left)
        return cpp._WedgeDenseDoubleForms(second, delta4_right)

    def double_form_4d_reference_factory():
        return dg.DoubleForm(
            dg.Einsum(
                "ij,kl->ijkl",
                dg.Wedge(alpha4, gamma4),
                dg.Wedge(beta4, delta4),
            ),
            p=2,
            q=2,
            dim=4,
        )

    def kform_4d_factory():
        first = dg.Wedge(alpha4, beta4)
        second = dg.Wedge(first, gamma4)
        return dg.Wedge(second, delta4)

    def native_kform_4d_factory():
        first = cpp.Wedge(alpha4, beta4)
        second = cpp.Wedge(first, gamma4)
        return cpp.Wedge(second, delta4)

    def dense_kform_4d_factory():
        first = cpp._WedgeDenseKForms(alpha4, beta4)
        second = cpp._WedgeDenseKForms(first, gamma4)
        return cpp._WedgeDenseKForms(second, delta4)

    def _double_wedge_chain(wedge, factors):
        result = factors[0]
        for factor in factors[1:]:
            result = wedge(result, factor)
        return result

    double_form_31_factors = (
        alpha4_left,
        beta4_left,
        gamma4_left,
        delta4_right,
    )
    double_form_33_factors = double_form_31_factors + (
        gamma4_right,
        beta4_right,
    )
    double_form_44_factors = (
        alpha4_left,
        beta4_left,
        gamma4_left,
        delta4_left,
        alpha4_right,
        beta4_right,
        gamma4_right,
        delta4_right,
    )

    def double_form_31_factory():
        return _double_wedge_chain(dg.Wedge, double_form_31_factors)

    def native_double_form_31_factory():
        return _double_wedge_chain(cpp.Wedge, double_form_31_factors)

    def dense_double_form_31_factory():
        return _double_wedge_chain(
            cpp._WedgeDenseDoubleForms, double_form_31_factors
        )

    def double_form_33_factory():
        return _double_wedge_chain(dg.Wedge, double_form_33_factors)

    def native_double_form_33_factory():
        return _double_wedge_chain(cpp.Wedge, double_form_33_factors)

    def dense_double_form_33_factory():
        return _double_wedge_chain(
            cpp._WedgeDenseDoubleForms, double_form_33_factors
        )

    def double_form_44_factory():
        return _double_wedge_chain(dg.Wedge, double_form_44_factors)

    def native_double_form_44_factory():
        return _double_wedge_chain(cpp.Wedge, double_form_44_factors)

    def dense_double_form_44_factory():
        return _double_wedge_chain(
            cpp._WedgeDenseDoubleForms, double_form_44_factors
        )

    def double_form_31_reference_factory():
        left = dg.Wedge(dg.Wedge(alpha4, beta4), gamma4)
        return dg.DoubleForm(
            dg.Einsum("ijk,l->ijkl", left, delta4),
            p=3,
            q=1,
            dim=4,
        )

    def double_form_33_reference_factory():
        left = dg.Wedge(dg.Wedge(alpha4, beta4), gamma4)
        right = dg.Wedge(dg.Wedge(delta4, gamma4), beta4)
        return dg.DoubleForm(
            dg.Einsum("ijk,lmn->ijklmn", left, right),
            p=3,
            q=3,
            dim=4,
        )

    def double_form_44_reference_factory():
        left = kform_4d_factory()
        right = dg.Wedge(
            dg.Wedge(dg.Wedge(alpha4, beta4), gamma4), delta4
        )
        return dg.DoubleForm(
            dg.Einsum("ijkl,mnop->ijklmnop", left, right),
            p=4,
            q=4,
            dim=4,
        )

    return {
        "kform_wedge_d_3d": {
            "factory": kform_factory,
            "native_factory": native_kform_factory,
            "dense_factory": dense_kform_factory,
            "reference_factory": kform_reference_factory,
            "physical": True,
            "space_dimension": 3,
            "form_degrees": [3],
            "full_components": 27,
            "independent_components": 1,
            "cache_basis_degrees": (0, 1, 2, 3),
            "cache_wedge_degrees": ((1, 0, 1, 0), (2, 0, 1, 0)),
            "cache_exterior_degrees": (2,),
            "cache_double_expansion_degrees": (),
        },
        "double_form_wedge_22_3d": {
            "factory": double_form_factory,
            "native_factory": native_double_form_factory,
            "dense_factory": dense_double_form_factory,
            "reference_factory": double_form_reference_factory,
            "physical": True,
            "space_dimension": 3,
            "form_degrees": [2, 2],
            "full_components": 81,
            "independent_components": 9,
            "cache_basis_degrees": (1, 2),
            "cache_wedge_degrees": ((1, 1, 1, 1),),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "double_form_arithmetic_22_3d": {
            "factory": double_form_arithmetic_factory,
            "native_factory": native_double_form_arithmetic_factory,
            "dense_factory": dense_double_form_arithmetic_factory,
            "reference_factory": double_form_reference_factory,
            "physical": True,
            "space_dimension": 3,
            "form_degrees": [2, 2],
            "full_components": 81,
            "independent_components": 9,
            "cache_basis_degrees": (1, 2),
            "cache_wedge_degrees": ((1, 1, 1, 1),),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": ((2, 2),),
        },
        "double_form_wedge_chain_22_3d": {
            "factory": double_form_chain_factory,
            "native_factory": native_double_form_chain_factory,
            "dense_factory": dense_double_form_chain_factory,
            "reference_factory": double_form_reference_factory,
            "physical": True,
            "space_dimension": 3,
            "form_degrees": [2, 2],
            "full_components": 81,
            "independent_components": 9,
            "cache_basis_degrees": (0, 1, 2),
            "cache_wedge_degrees": (
                (1, 0, 0, 1),
                (1, 1, 1, 0),
                (2, 1, 0, 1),
            ),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "double_form_wedge_22_4d": {
            "factory": double_form_4d_factory,
            "native_factory": native_double_form_4d_factory,
            "dense_factory": dense_double_form_4d_factory,
            "reference_factory": double_form_4d_reference_factory,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [2, 2],
            "full_components": 256,
            "independent_components": 36,
            "cache_basis_degrees": (1, 2),
            "cache_wedge_degrees": ((1, 1, 1, 1),),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "double_form_wedge_chain_22_4d": {
            "factory": double_form_4d_chain_factory,
            "native_factory": native_double_form_4d_chain_factory,
            "dense_factory": dense_double_form_4d_chain_factory,
            "reference_factory": double_form_4d_reference_factory,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [2, 2],
            "full_components": 256,
            "independent_components": 36,
            "cache_basis_degrees": (0, 1, 2),
            "cache_wedge_degrees": (
                (1, 0, 0, 1),
                (1, 1, 1, 0),
                (2, 1, 0, 1),
            ),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "kform_wedge_4_4d": {
            "factory": kform_4d_factory,
            "native_factory": native_kform_4d_factory,
            "dense_factory": dense_kform_4d_factory,
            "reference_factory": kform_4d_factory,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [4],
            "full_components": 4**4,
            "independent_components": 1,
            "cache_basis_degrees": (1, 2, 3, 4),
            "cache_wedge_degrees": (
                (1, 0, 1, 0),
                (2, 0, 1, 0),
                (3, 0, 1, 0),
            ),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "double_form_wedge_31_4d": {
            "factory": double_form_31_factory,
            "native_factory": native_double_form_31_factory,
            "dense_factory": dense_double_form_31_factory,
            "reference_factory": double_form_31_reference_factory,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [3, 1],
            "full_components": 4**4,
            "independent_components": math.comb(4, 3) * math.comb(4, 1),
            "cache_basis_degrees": (0, 1, 2, 3),
            "cache_wedge_degrees": (
                (1, 0, 1, 0),
                (2, 0, 1, 0),
                (3, 0, 0, 1),
            ),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "double_form_wedge_33_4d": {
            "factory": double_form_33_factory,
            "native_factory": native_double_form_33_factory,
            "dense_factory": dense_double_form_33_factory,
            "reference_factory": double_form_33_reference_factory,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [3, 3],
            "full_components": 4**6,
            "independent_components": math.comb(4, 3) ** 2,
            "cache_basis_degrees": (0, 1, 2, 3),
            "cache_wedge_degrees": (
                (1, 0, 1, 0),
                (2, 0, 1, 0),
                (3, 0, 0, 1),
                (3, 1, 0, 1),
                (3, 2, 0, 1),
            ),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
        "double_form_wedge_44_4d": {
            "factory": double_form_44_factory,
            "native_factory": native_double_form_44_factory,
            "dense_factory": dense_double_form_44_factory,
            "reference_factory": double_form_44_reference_factory,
            "physical": False,
            "space_dimension": 4,
            "form_degrees": [4, 4],
            "full_components": 4**8,
            "independent_components": 1,
            "cache_basis_degrees": (0, 1, 2, 3, 4),
            "cache_wedge_degrees": (
                (1, 0, 1, 0),
                (2, 0, 1, 0),
                (3, 0, 1, 0),
                (4, 0, 0, 1),
                (4, 1, 0, 1),
                (4, 2, 0, 1),
                (4, 3, 0, 1),
            ),
            "cache_exterior_degrees": (),
            "cache_double_expansion_degrees": (),
        },
    }


def _cache_storage_lower_bound(workload):
    dim = workload["space_dimension"]
    basis = [
        {
            "dim": dim,
            "degree": degree,
            "bytes": cpp._FormBasisStorageBytesLowerBound(dim, degree),
        }
        for degree in workload["cache_basis_degrees"]
    ]
    wedge = [
        {
            "dim": dim,
            "degrees": list(degrees),
            "bytes": cpp._CompactWedgeTableStorageBytesLowerBound(
                dim, *degrees
            ),
        }
        for degrees in workload["cache_wedge_degrees"]
    ]
    exterior = [
        {
            "dim": dim,
            "degree": degree,
            "bytes": cpp._CompactExteriorDerivativeTableStorageBytesLowerBound(
                dim, degree
            ),
        }
        for degree in workload["cache_exterior_degrees"]
    ]
    double_expansion = [
        {
            "dim": dim,
            "degrees": list(degrees),
            "bytes": (
                cpp._CompactDoubleFormExpansionTableStorageBytesLowerBound(
                    dim, *degrees
                )
            ),
        }
        for degrees in workload["cache_double_expansion_degrees"]
    ]
    return {
        "kind": "owned_storage_lower_bound",
        "form_basis": basis,
        "wedge_tables": wedge,
        "exterior_derivative_tables": exterior,
        "double_form_expansion_tables": double_expansion,
        "total_bytes": sum(
            item["bytes"]
            for family in (basis, wedge, exterior, double_expansion)
            for item in family
        ),
        "excluded": (
            "allocator bookkeeping, shared_ptr control blocks, and cache "
            "container storage"
        ),
    }


def _benchmark_factory_pair(factories, warmup, iterations):
    names = tuple(factories)
    candidates = {name: None for name in names}
    samples = {name: [] for name in names}

    for iteration in range(warmup):
        for name in _alternating_order(names, iteration):
            candidates[name] = factories[name]()

    execution_order = []
    for iteration in range(iterations):
        order = _alternating_order(names, iteration)
        execution_order.append(list(order))
        for name in order:
            # Destruction belongs outside the construction measurement.
            candidates[name] = None
            started = time.perf_counter_ns()
            candidates[name] = factories[name]()
            samples[name].append((time.perf_counter_ns() - started) * 1e-9)

    return (
        candidates,
        {name: _statistics(values) for name, values in samples.items()},
        execution_order,
    )


def _benchmark_compile_pair(factories, warmup, iterations):
    names = tuple(factories)
    compiled = {name: None for name in names}
    samples = {name: [] for name in names}

    for iteration in range(warmup):
        for name in _alternating_order(names, iteration):
            compiled[name] = factories[name]().coef.Compile(
                realcompile=False,
                wait=True,
                maxderiv=0,
            )

    execution_order = []
    for iteration in range(iterations):
        order = _alternating_order(names, iteration)
        execution_order.append(list(order))
        for name in order:
            coefficient = factories[name]().coef
            compiled[name] = None
            started = time.perf_counter_ns()
            compiled[name] = coefficient.Compile(
                realcompile=False,
                wait=True,
                maxderiv=0,
            )
            samples[name].append((time.perf_counter_ns() - started) * 1e-9)

    return (
        compiled,
        {
            name: {"kind": "ngsolve_graph", **_statistics(values)}
            for name, values in samples.items()
        },
        execution_order,
    )


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


def _benchmark_evaluation(
    output,
    compiled,
    dense_output,
    compiled_dense_output,
    mesh,
    backends,
    warmup,
    iterations,
):
    forms = {}
    for backend in backends:
        simd = backend == "simd"
        forms[(backend, "interpreted")] = _make_form(output, mesh, simd)
        forms[(backend, "compiled")] = _make_form(compiled, mesh, simd)
        forms[(backend, "dense_interpreted")] = _make_form(
            dense_output, mesh, simd
        )
        forms[(backend, "dense_compiled")] = _make_form(
            compiled_dense_output, mesh, simd
        )

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
        for variant in (
            "interpreted",
            "compiled",
            "dense_interpreted",
            "dense_compiled",
        ):
            key = (backend, variant)
            form, integrator = forms[key]
            variants[variant] = {
                **_statistics(timings[key]),
                "value_norm": Norm(form.mat.AsVector()),
                "simd_requested": backend == "simd",
                "simd_active": bool(integrator.simd_evaluate),
            }
        reference = variants["interpreted"]["value_norm"]
        for variant in (
            "compiled",
            "dense_interpreted",
            "dense_compiled",
        ):
            variants[variant]["relative_value_error"] = abs(
                variants[variant]["value_norm"] - reference
            ) / max(1.0, abs(reference))
        report[backend] = variants
    return report


def _backend_equivalence(evaluation):
    if set(evaluation) != {"scalar", "simd"}:
        return {}
    result = {}
    for variant in evaluation["scalar"]:
        scalar = evaluation["scalar"][variant]["value_norm"]
        simd = evaluation["simd"][variant]["value_norm"]
        result[variant] = abs(simd - scalar) / max(1.0, abs(scalar))
    return result


def _retained_graph_memory_sample(workload_name, variant, count):
    workload = _workloads()[workload_name]
    factory = (
        workload["factory"]
        if variant == "compact"
        else workload["dense_factory"]
    )
    return _retained_memory_sample(
        factory,
        count,
        count_key="retained_graphs",
        per_item_key="estimated_incremental_bytes_per_graph",
    )


def _retained_graph_memory(workload_name, compact_count, dense_count):
    samples = {}
    for variant, count in (
        ("compact", compact_count),
        ("dense", dense_count),
    ):
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--memory-worker",
                workload_name,
                variant,
                "--retained-graphs",
                str(count),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        samples[variant] = json.loads(process.stdout)

    compact = samples["compact"]
    dense = samples["dense"]
    if compact["supported"] and dense["supported"]:
        compact_bytes = compact["estimated_incremental_bytes_per_graph"]
        dense_bytes = dense["estimated_incremental_bytes_per_graph"]
        samples["dense_over_compact_estimate"] = (
            dense_bytes / compact_bytes if compact_bytes else None
        )
    else:
        samples["dense_over_compact_estimate"] = None
    return samples


def _selected_backends(mode):
    if mode == "construction":
        return ()
    if mode == "all":
        return ("scalar", "simd")
    return (mode,)


def _construction_factories(workload, path):
    factories = {}
    if path in {"public", "both"}:
        factories["public"] = workload["factory"]
    if path in {"native", "both"}:
        factories["native"] = workload["native_factory"]
    factories["dense"] = workload["dense_factory"]
    return factories


def _cold_construction_sample(workload_name, variant, threads):
    return _subprocess_json(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cold-construction-worker",
            workload_name,
            variant,
            "--threads",
            str(threads),
        ]
    )


def _benchmark_factory_pair_cold(
    workload_name, workload, path, iterations, threads
):
    factories = _construction_factories(workload, path)
    names = tuple(factories)
    samples = {name: [] for name in names}
    execution_order = []
    for iteration in range(iterations):
        order = _alternating_order(names, iteration)
        execution_order.append(list(order))
        for name in order:
            sample = _cold_construction_sample(
                workload_name, name, threads
            )
            samples[name].append(sample["seconds"])

    # Results cannot cross the subprocess boundary. Construct one instance in
    # the parent after timing for equivalence, graph, compile, and evaluation
    # checks. These constructions are deliberately outside the cold samples.
    candidates = {name: factory() for name, factory in factories.items()}
    return (
        candidates,
        {name: _statistics(values) for name, values in samples.items()},
        execution_order,
    )


def _cold_construction_worker(workload_name, variant):
    workload = _workloads()[workload_name]
    factory = _construction_factories(workload, "both")[variant]
    gc.collect()
    started = time.perf_counter_ns()
    candidate = factory()
    seconds = (time.perf_counter_ns() - started) * 1e-9
    return {
        "seconds": seconds,
        "variant": variant,
        "workload": workload_name,
        "unique_nodes": dg.CFStats(candidate)["unique_nodes"],
    }


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    backends = _selected_backends(args.mode)
    results = {}

    workloads = _workloads()
    selected_names = args.workload or list(_DEFAULT_WORKLOAD_NAMES)
    for name in selected_names:
        workload = workloads[name]
        factories = _construction_factories(
            workload, args.construction_path
        )
        if args.cache_state == "cold":
            outputs, construction_statistics, construction_order = (
                _benchmark_factory_pair_cold(
                    name,
                    workload,
                    args.construction_path,
                    args.construction_iterations,
                    args.threads,
                )
            )
        else:
            outputs, construction_statistics, construction_order = (
                _benchmark_factory_pair(
                    factories,
                    args.construction_warmup,
                    args.construction_iterations,
                )
            )
        primary_variant = (
            "native" if args.construction_path == "native" else "public"
        )
        output = outputs[primary_variant]
        dense_output = outputs["dense"]
        construction = construction_statistics[primary_variant]
        dense_construction = construction_statistics["dense"]
        reference = workload["reference_factory"]()
        l2_error = _l2_error(output, reference, mesh)
        dense_l2_error = _l2_error(dense_output, reference, mesh)
        compact_dense_l2_error = _l2_error(output, dense_output, mesh)
        if max(l2_error, dense_l2_error, compact_dense_l2_error) > args.tolerance:
            raise RuntimeError(
                f"{name} compact/dense/reference disagreement: "
                f"{l2_error:.3e}, {dense_l2_error:.3e}, "
                f"{compact_dense_l2_error:.3e}"
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
            "dense_construction": dense_construction,
            "public_construction": construction_statistics.get("public"),
            "native_construction": construction_statistics.get("native"),
            "construction_execution_order": construction_order,
            "wrapper_tree": dg.CFStats(output),
            "dense_wrapper_tree": dg.CFStats(dense_output),
            "mathematical_reference_tree": dg.CFStats(reference),
            "cache_storage_lower_bound": _cache_storage_lower_bound(workload),
            "l2_error": l2_error,
            "dense_l2_error": dense_l2_error,
            "compact_dense_l2_error": compact_dense_l2_error,
            "construction_speedup": (
                dense_construction["median_seconds"]
                / construction["median_seconds"]
            ),
            "compile": None,
            "dense_compile": None,
            "compile_execution_order": [],
            "evaluation": {},
            "compact_speedup_over_dense": {},
            "scalar_simd_relative_value_error": {},
            "retained_graph_memory": None,
        }
        if args.retained_graphs or args.retained_compact_graphs:
            result["retained_graph_memory"] = _retained_graph_memory(
                name,
                args.retained_compact_graphs or args.retained_graphs,
                args.retained_dense_graphs or args.retained_graphs,
            )
        if workload["physical"] and backends:
            compiled_outputs, compile_statistics, compile_order = (
                _benchmark_compile_pair(
                    {
                        "compact": factories[primary_variant],
                        "dense": workload["dense_factory"],
                    },
                    args.compile_warmup,
                    args.compile_iterations,
                )
            )
            compiled = compiled_outputs["compact"]
            compiled_dense_output = compiled_outputs["dense"]
            result["compile"] = compile_statistics["compact"]
            result["compiled_tree"] = dg.CFStats(compiled)
            result["dense_compile"] = compile_statistics["dense"]
            result["compile_execution_order"] = compile_order
            result["dense_compiled_tree"] = dg.CFStats(
                compiled_dense_output
            )
            result["compile_speedup"] = (
                compile_statistics["dense"]["median_seconds"]
                / compile_statistics["compact"]["median_seconds"]
            )
            result["evaluation"] = _benchmark_evaluation(
                output,
                compiled,
                dense_output,
                compiled_dense_output,
                mesh,
                backends,
                args.warmup,
                args.iterations,
            )
            result["compact_speedup_over_dense"] = {
                backend: {
                    "interpreted": (
                        variants["dense_interpreted"]["median_seconds"]
                        / variants["interpreted"]["median_seconds"]
                    ),
                    "compiled": (
                        variants["dense_compiled"]["median_seconds"]
                        / variants["compiled"]["median_seconds"]
                    ),
                }
                for backend, variants in result["evaluation"].items()
            }
            result["scalar_simd_relative_value_error"] = _backend_equivalence(
                result["evaluation"]
            )
            for backend, variants in result["evaluation"].items():
                for variant in (
                    "compiled",
                    "dense_interpreted",
                    "dense_compiled",
                ):
                    relative_error = variants[variant]["relative_value_error"]
                    if relative_error > args.tolerance:
                        raise RuntimeError(
                            f"{name} {variant} {backend} evaluation disagrees "
                            f"with interpreted evaluation: {relative_error:.3e}"
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
        "workloads": selected_names,
        "cache_state": args.cache_state,
        "construction_path": args.construction_path,
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
        "clock": "time.perf_counter",
        "maxh": args.maxh,
        "elements": mesh.ne,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "construction_warmup": (
            0 if args.cache_state == "cold" else args.construction_warmup
        ),
        "construction_requested_warmup": args.construction_warmup,
        "construction_iterations": args.construction_iterations,
        "compile_warmup": args.compile_warmup,
        "compile_iterations": args.compile_iterations,
        "retained_graphs": args.retained_graphs,
        "retained_compact_graphs": (
            args.retained_compact_graphs or args.retained_graphs
        ),
        "retained_dense_graphs": (
            args.retained_dense_graphs or args.retained_graphs
        ),
        "cache_container_storage_bytes": (
            cpp._CompactFormCacheContainerStorageBytes()
        ),
        "cache_container_storage_breakdown_bytes": dict(
            cpp._CompactFormCacheContainerStorageBreakdown()
        ),
        "measurement_boundaries": {
            "construction": (
                "counterbalanced factory calls; previous graph destruction "
                "excluded; cold mode uses one call per fresh subprocess"
            ),
            "compile": (
                "counterbalanced CoefficientFunction.Compile calls; factory "
                "and previous compiled graph destruction excluded"
            ),
            "evaluation": "BilinearForm.Assemble",
            "retained_graph_memory": (
                "fresh subprocess peak-RSS increase while retaining N graphs; "
                "shared workload inputs and warmed one-time caches excluded"
            ),
            "cache_storage_lower_bound": (
                "sum of owned table/container payloads used by the compact "
                "workload; fixed cache-container storage is reported once at "
                "top level, while allocator overhead is excluded"
            ),
        },
        "tolerance": args.tolerance,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument(
        "--workload",
        action="append",
        choices=_WORKLOAD_NAMES,
        help="run only this workload; repeat to select multiple workloads",
    )
    parser.add_argument(
        "--mode",
        choices=("construction", "scalar", "simd", "all"),
        default="all",
    )
    parser.add_argument(
        "--construction-path",
        choices=("public", "native", "both"),
        default="both",
        help=(
            "measure the public wrapper, direct native factory, or both; "
            "the dense control is always measured"
        ),
    )
    parser.add_argument(
        "--cache-state",
        choices=("warm", "cold"),
        default="warm",
        help=(
            "warm measures repeated calls in this process; cold measures each "
            "construction in a fresh subprocess"
        ),
    )
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--construction-warmup", type=int, default=10)
    parser.add_argument("--construction-iterations", type=int, default=100)
    parser.add_argument("--compile-warmup", type=int, default=2)
    parser.add_argument("--compile-iterations", type=int, default=5)
    parser.add_argument(
        "--retained-graphs",
        type=int,
        default=0,
        help=(
            "also estimate incremental graph memory from peak RSS in fresh "
            "subprocesses; 0 disables the measurement"
        ),
    )
    parser.add_argument(
        "--retained-compact-graphs",
        type=int,
        default=0,
        help=(
            "override --retained-graphs for the compact worker; specify it "
            "together with --retained-dense-graphs"
        ),
    )
    parser.add_argument(
        "--retained-dense-graphs",
        type=int,
        default=0,
        help=(
            "override --retained-graphs for the dense worker; specify it "
            "together with --retained-compact-graphs"
        ),
    )
    parser.add_argument(
        "--memory-worker",
        nargs=2,
        metavar=("WORKLOAD", "VARIANT"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cold-construction-worker",
        nargs=2,
        metavar=("WORKLOAD", "VARIANT"),
        help=argparse.SUPPRESS,
    )
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
    if args.construction_warmup < 0:
        parser.error("--construction-warmup must be nonnegative")
    if args.compile_warmup < 0:
        parser.error("--compile-warmup must be nonnegative")
    if args.construction_iterations < 1 or args.compile_iterations < 1:
        parser.error("construction and compile iteration counts must be positive")
    for name in (
        "retained_graphs",
        "retained_compact_graphs",
        "retained_dense_graphs",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if bool(args.retained_compact_graphs) != bool(args.retained_dense_graphs):
        parser.error(
            "--retained-compact-graphs and --retained-dense-graphs must be "
            "specified together"
        )
    if args.memory_worker:
        workload, variant = args.memory_worker
        if workload not in _WORKLOAD_NAMES:
            parser.error(f"unknown memory workload: {workload}")
        if variant not in {"compact", "dense"}:
            parser.error("memory worker variant must be compact or dense")
        if args.retained_graphs < 1:
            parser.error("memory worker requires --retained-graphs > 0")
    if args.cold_construction_worker:
        workload, variant = args.cold_construction_worker
        if workload not in _WORKLOAD_NAMES:
            parser.error(f"unknown cold-construction workload: {workload}")
        if variant not in {"public", "native", "dense"}:
            parser.error(
                "cold-construction worker variant must be public, native, or dense"
            )
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    return args


def main():
    args = parse_args()
    if args.memory_worker:
        workload, variant = args.memory_worker
        ngsolve.SetNumThreads(1)
        print(
            json.dumps(
                _retained_graph_memory_sample(
                    workload, variant, args.retained_graphs
                ),
                sort_keys=True,
            )
        )
        return
    if args.cold_construction_worker:
        workload, variant = args.cold_construction_worker
        ngsolve.SetNumThreads(args.threads)
        print(
            json.dumps(
                _cold_construction_worker(workload, variant),
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
