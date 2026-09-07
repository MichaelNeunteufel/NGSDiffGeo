"""Core form contracts, independent component references, and audit regressions."""

import itertools
import pickle
import subprocess
import sys

import numpy as np
import pytest

import ngsdiffgeo as dg
from ngsdiffgeo import ngsdiffgeo as cpp
from ngsolve import BBND, BND, VOL, CF, Id, Norm, OuterProduct, x, y

from tests._helpers import assert_l2_close


def _sign(indices):
    inversions = sum(
        indices[i] > indices[j]
        for i in range(len(indices))
        for j in range(i + 1, len(indices))
    )
    return (-1) ** inversions


def _alternating_components(n, k, offset):
    independent = {
        indices: offset + sum((j + 1) * (i + 1) for j, i in enumerate(indices))
        for indices in itertools.combinations(range(n), k)
    }

    def component(indices):
        if len(set(indices)) != k:
            return 0
        return _sign(indices) * independent[tuple(sorted(indices))]

    values = [component(indices) for indices in itertools.product(range(n), repeat=k)]
    coefficient = CF(values[0]) if k == 0 else CF(tuple(values), dims=(n,) * k)
    return coefficient, component


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_kform_wedge_matches_independent_shuffle_formula(n, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    point = mesh(0.2, 0.3)
    for k in range(n + 1):
        for ell in range(n - k + 1):
            a, ca = _alternating_components(n, k, 2)
            b, cb = _alternating_components(n, ell, 7)
            result = cpp.Wedge(cpp.KForm(a, k, n), cpp.KForm(b, ell, n))
            assert tuple(result.dims) == (n,) * (k + ell)
            expected = []
            for indices in itertools.product(range(n), repeat=k + ell):
                value = 0
                for left in itertools.combinations(range(k + ell), k):
                    right = tuple(i for i in range(k + ell) if i not in left)
                    value += (
                        _sign(left + right)
                        * ca(tuple(indices[i] for i in left))
                        * cb(tuple(indices[i] for i in right))
                    )
                expected.append(value)
            np.testing.assert_allclose(
                np.asarray(result(point)).reshape(-1), expected, atol=1e-12
            )


@pytest.mark.parametrize("vb", [VOL, BND, BBND])
@pytest.mark.parametrize("slot", ["left", "right", "both"])
@pytest.mark.parametrize("inverse", [False, True])
def test_zero_doubleform_star_preserves_unselected_degree(vb, slot, inverse):
    manifold = dg.RiemannianManifold(Id(3))
    # The asymmetric (0,1) degrees also distinguish slots on a 1D edge.
    zero = dg.DoubleForm(CF((0, 0, 0)), p=0, q=1, dim=3)
    n = {VOL: 3, BND: 2, BBND: 1}[vb]
    operation = dg.inv_star if inverse else dg.star
    result = operation(zero, manifold, vb=vb, slot=slot)
    expected = (n if slot != "right" else 0, n - 1 if slot != "left" else 1)
    assert (result.degree_left, result.degree_right) == expected
    assert result.dim_space == 3
    assert tuple(result.dims) == (3,) * sum(expected)
    assert result.is_zero


@pytest.mark.parametrize("kind", ["kform", "doubleform"])
@pytest.mark.parametrize("entry", ["function", "method", "inverse_function", "inverse_method"])
@pytest.mark.parametrize("value", [0, 1])
def test_star_entrypoints_reject_ambient_dimension_mismatch(kind, entry, value):
    manifold = dg.RiemannianManifold(Id(3))
    form = (
        dg.ScalarField(CF(value), dim=2)
        if kind == "kform"
        else dg.DoubleForm(CF(value), p=0, q=0, dim=2)
    )
    operations = {
        "function": lambda: dg.star(form, manifold),
        "method": lambda: form.star(manifold),
        "inverse_function": lambda: dg.inv_star(form, manifold),
        "inverse_method": lambda: form.inv_star(manifold),
    }
    with pytest.raises(Exception, match="dimension.*match"):
        operations[entry]()


@pytest.mark.parametrize("operand", ["left", "right"])
@pytest.mark.parametrize("variable_kind", ["wrapper", "coefficient"])
def test_double_wedge_replace_and_differentiate_operands(
    operand, variable_kind, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    a = cpp.DoubleForm(CF((1, 2, 3, 4), dims=(2, 2)), 1, 1, 2)
    b = cpp.DoubleForm(CF((5, 6, 7, 8), dims=(2, 2)), 1, 1, 2)
    wedge = cpp.Wedge(a, b)
    selected = a if operand == "left" else b
    variable = selected if variable_kind == "wrapper" else selected.coef

    assert_l2_close(wedge.Replace({variable: 2 * selected.coef}), 2 * wedge, mesh, tol=1e-12)
    assert_l2_close(wedge.Diff(variable, selected.coef), wedge, mesh, tol=1e-12)
    jacobian = wedge.Diff(variable)
    assert tuple(jacobian.dims) == (2,) * 6
    contracted = dg.Einsum("abcdij,ij->abcd", jacobian, selected.coef)
    assert_l2_close(contracted, wedge, mesh, tol=1e-12)


def test_scalar_double_wedge_jacobian_keeps_operand_identity(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    a = cpp.DoubleForm(x + 1, 0, 0, 2)
    b = cpp.DoubleForm(y + 2, 0, 0, 2)
    wedge = cpp.Wedge(a, b)
    assert_l2_close(wedge.Diff(a), b, mesh, tol=1e-12)
    assert_l2_close(wedge.Diff(b), a, mesh, tol=1e-12)


@pytest.mark.parametrize("rank,n", [(0, 2), (1, 1), (2, 2), (3, 3), (4, 4)])
@pytest.mark.parametrize("scale", [1, 1 + 2j])
def test_alternation_components_compilation_and_pickle(
    rank, n, scale, make_unit_square_mesh, monkeypatch
):
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    mesh = make_unit_square_mesh(maxh=0.7)
    point = mesh(0.2, 0.3)
    raw_values = np.arange(1, n**rank + 1).reshape((n,) * rank)
    raw = CF(scale) if rank == 0 else CF(
        tuple(scale * int(v) for v in raw_values.flat), dims=(n,) * rank
    )
    alternation = cpp.Alternation(raw, rank, n)
    expected = [
        sum(
            _sign(perm) * raw_values[tuple(indices[i] for i in perm)] * scale
            for perm in itertools.permutations(range(rank))
        )
        for indices in itertools.product(range(n), repeat=rank)
    ]
    for result in (
        alternation,
        pickle.loads(pickle.dumps(alternation)),
        alternation.Compile(realcompile=False, maxderiv=0),
        alternation.Compile(realcompile=True, maxderiv=0, wait=True),
    ):
        assert tuple(result.dims) == (n,) * rank
        assert result.is_complex == alternation.is_complex
        np.testing.assert_allclose(
            np.asarray(result(point)).reshape(-1), expected, atol=1e-12
        )


@pytest.mark.parametrize("kind", ["alternation", "block", "wedge"])
def test_custom_form_node_equivalence_keys_are_structural(kind):
    vector = CF((x, y))
    matrix = OuterProduct(vector, CF((1 + x, 2 + y)))
    a = cpp.DoubleForm(matrix, 1, 1, 2)
    b = cpp.DoubleForm(matrix + Id(2), 1, 1, 2)
    factories = {
        "alternation": lambda: cpp.Alternation(matrix, 2, 2),
        "block": lambda: dg.RiemannianManifold(Id(2)).d_cov(a, slot="left"),
        "wedge": lambda: cpp.Wedge(a, b),
    }
    assert factories[kind]()._equivalence_key == factories[kind]()._equivalence_key


@pytest.mark.parametrize(
    "expression",
    [
        "cpp.ScalarField(CF(1), 2).star(None)",
        "cpp.ScalarField(CF(1), 2).inv_star(None)",
        "cpp.DoubleForm(CF(1), 0, 0, 2).star(None)",
        "cpp.DoubleForm(CF(1), 0, 0, 2).inv_star(None)",
        "cpp.star(cpp.ScalarField(CF(1), 2), None)",
        "cpp.inv_star(cpp.ScalarField(CF(1), 2), None)",
        "cpp.star(cpp.DoubleForm(CF(1), 0, 0, 2), None)",
        "cpp.inv_star(cpp.DoubleForm(CF(1), 0, 0, 2), None)",
        "cpp.slot_inner_product(cpp.DoubleForm(CF(1), 0, 0, 2), None)",
        "cpp.delta(cpp.ScalarField(CF(1), 2), None)",
    ],
)
def test_null_manifold_raises_instead_of_crashing(expression):
    # Carry the imported package path into a subprocess, including isolated builds.
    code = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    "ngsdiffgeo", {dg.__file__!r}, submodule_search_locations={list(dg.__path__)!r}
)
package = importlib.util.module_from_spec(spec)
sys.modules["ngsdiffgeo"] = package
spec.loader.exec_module(package)
from ngsdiffgeo import ngsdiffgeo as cpp
from ngsolve import CF
try:
    {expression}
except Exception as error:
    assert "null" in str(error) or "non-null" in str(error), str(error)
else:
    raise AssertionError("null manifold accepted")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

@pytest.mark.parametrize("n,p,q,r,s", [
    (1, 0, 0, 1, 0), (2, 1, 1, 1, 1),
    (3, 2, 1, 1, 1), (3, 1, 2, 1, 1), (4, 2, 2, 1, 1),
])
def test_double_wedge_matches_independent_block_formulas(
    n, p, q, r, s, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    point = mesh(0.2, 0.3)
    _, a_left = _alternating_components(n, p, 2)
    _, a_right = _alternating_components(n, q, 3)
    _, b_left = _alternating_components(n, r, 5)
    _, b_right = _alternating_components(n, s, 7)

    def double_form(left, right, left_degree, right_degree):
        degree = left_degree + right_degree
        values = tuple(
            left(indices[:left_degree]) * right(indices[left_degree:])
            for indices in itertools.product(range(n), repeat=degree)
        )
        raw = CF(values[0]) if degree == 0 else CF(values, dims=(n,) * degree)
        return cpp.DoubleForm(raw, left_degree, right_degree, n)

    def block_wedge(indices, left, right, left_degree):
        value = 0
        for selected in itertools.combinations(range(len(indices)), left_degree):
            complement = tuple(i for i in range(len(indices)) if i not in selected)
            value += (
                _sign(selected + complement)
                * left(tuple(indices[i] for i in selected))
                * right(tuple(indices[i] for i in complement))
            )
        return value

    result = cpp.Wedge(
        double_form(a_left, a_right, p, q), double_form(b_left, b_right, r, s)
    )
    expected = [
        block_wedge(indices[:p + r], a_left, b_left, p)
        * block_wedge(indices[p + r:], a_right, b_right, q)
        for indices in itertools.product(range(n), repeat=p + q + r + s)
    ]
    assert (result.degree_left, result.degree_right) == (p + r, q + s)
    np.testing.assert_allclose(
        np.asarray(result(point)).reshape(-1), expected, atol=1e-12
    )


@pytest.mark.parametrize("kind", ["block_left", "block_right", "wedge", "scalar_wedge"])
@pytest.mark.parametrize("scale", [1, 1 + 2j])
def test_block_and_wedge_compilation_and_pickle(
    kind, scale, make_unit_square_mesh, monkeypatch
):
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    mesh = make_unit_square_mesh(maxh=0.7)
    matrix = scale * CF((x, y, 2 * y, 3 * x), dims=(2, 2))
    form = cpp.DoubleForm(matrix, 1, 1, 2)
    if kind.startswith("block"):
        slot = "left" if kind == "block_left" else "right"
        result = dg.RiemannianManifold(Id(2)).d_cov(form, slot=slot)
        # Independent derivatives of the linear matrix above; no permutation helper.
        values = (
            (0, 0, 0, 2, 0, -2, 0, 0)
            if slot == "left"
            else (0, 0, 0, 0, 0, 1, -1, 0)
        )
        expected = scale * CF(values, dims=(2, 2, 2))
    elif kind == "wedge":
        result = cpp.Wedge(form, cpp.DoubleForm(Id(2), 1, 1, 2))
        values = [0] * 16
        values[5] = values[10] = 4 * x
        values[6] = values[9] = -4 * x
        expected = scale * CF(tuple(values), dims=(2, 2, 2, 2))
    else:
        result = cpp.Wedge(
            cpp.DoubleForm(scale * x, 0, 0, 2),
            cpp.DoubleForm(y, 0, 0, 2),
        )
        expected = scale * x * y

    for candidate in (
        result, pickle.loads(pickle.dumps(result)),
        result.Compile(realcompile=False, maxderiv=0),
        result.Compile(realcompile=True, maxderiv=0, wait=True),
    ):
        assert tuple(candidate.dims) == tuple(expected.dims)
        assert_l2_close(Norm(candidate - expected), CF(0), mesh, tol=1e-10)
