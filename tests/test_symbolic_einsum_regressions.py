"""Regressions for semantic operands hidden by native einsum optimizations."""

import pickle

import numpy as np
import pytest

import ngsdiffgeo as dg
from ngsolve import CF, x, y

from tests._helpers import assert_l2_close


@pytest.mark.parametrize("p,q", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("variable_kind", ["wrapper", "coefficient"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_nested_transpose_replaces_optimized_operands(
    p, q, variable_kind, roundtrip, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    rank = p + q
    raw = CF(2 + x) if rank == 0 else CF((1 + x, 2 + y))
    if rank == 2:
        raw = CF((0, 0, 0, 0), dims=(2, 2))
    operand = dg.DoubleForm(raw, p=p, q=q, dim=2)
    result = operand.trans.trans
    variable = operand if variable_kind == "wrapper" else operand.coef
    if roundtrip:
        variable, result = pickle.loads(pickle.dumps((variable, result)))
    replacement = CF(3 + y) if rank == 0 else CF(
        tuple(i + 1 + x for i in range(2**rank)), dims=(2,) * rank
    )
    replaced = result.Replace({variable: replacement})
    assert tuple(replaced.dims) == tuple(replacement.dims)
    assert_l2_close(replaced, replacement, mesh, tol=1e-12)
    assert_l2_close(replaced.Compile(realcompile=False), replacement, mesh, tol=1e-12)


@pytest.mark.parametrize("p,q", [(0, 0), (1, 0), (1, 1)])
def test_zero_transpose_jacobian_matches_independent_directions(
    p, q, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    shape = (2,) * (p + q)
    size = 2 ** (p + q)
    raw = CF(0) if not shape else CF((0,) * size, dims=shape)
    operand = dg.DoubleForm(raw, p=p, q=q, dim=2)
    # A semantic zero form is a differentiation target; native ZeroCF itself
    # always differentiates to zero, including when it is its own variable.
    variable = operand
    result = operand.trans
    jacobian = result.Diff(variable)
    assert tuple(jacobian.dims) == shape + shape
    axes = tuple(range(p, p + q)) + tuple(range(p))
    for values in np.eye(size):
        direction = CF(float(values[0])) if not shape else CF(tuple(values), dims=shape)
        expected_values = values.reshape(shape).transpose(axes).reshape(-1)
        expected = CF(float(expected_values[0])) if not shape else CF(
            tuple(expected_values), dims=shape
        )
        contracted = (
            jacobian.Reshape((size, size)) * direction.Reshape((size,))
        ).Reshape(shape)
        assert_l2_close(result.Diff(variable, direction), expected, mesh, tol=1e-12)
        assert_l2_close(contracted, expected, mesh, tol=1e-12)
