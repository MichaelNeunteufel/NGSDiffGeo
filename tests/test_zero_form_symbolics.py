"""Zero values must not erase a form's identity as a symbolic operand."""

import pickle

import pytest

import ngsdiffgeo as dg
from ngsolve import CF, Id, Norm, x, y

from tests._helpers import assert_l2_close


def _zero_operation(name):
    manifold = dg.RiemannianManifold(Id(2))
    a = dg.OneForm(CF((0, 0)))
    direction = CF((y*y, x*y))
    if name == "scale":
        return a, 2*a, direction, 2*direction
    if name == "divide":
        return a, a/(1+x), direction, direction/(1+x)
    if name == "negate":
        return a, -a, direction, -direction
    if name == "wedge":
        value = 4*y*y - 3*x*y
        return a, dg.Wedge(a, dg.OneForm(CF((3, 4)))), direction, CF((0, value, -value, 0), dims=(2, 2))
    if name == "star":
        return a, dg.star(a, manifold), direction, CF((-x*y, y*y))
    if name == "inv_star":
        return a, dg.inv_star(a, manifold), direction, CF((x*y, -y*y))
    if name == "d":
        return a, dg.d(a), direction, CF((0, -y, y, 0), dims=(2, 2))
    if name == "scalar_star":
        a = dg.ScalarField(CF(0), dim=2)
        return a, dg.star(a, manifold), 1+x, CF((0, 1+x, -1-x, 0), dims=(2, 2))
    if name == "scalar_wedge":
        a = dg.ScalarField(CF(0), dim=2)
        return a, dg.Wedge(a, dg.ScalarField(1+y, dim=2)), 1+x, (1+x)*(1+y)
    if name == "top_star":
        a = dg.TwoForm(CF((0,)*4, dims=(2, 2)), dim=2)
        return a, dg.star(a, manifold), CF((0, 1+x, -1-x, 0), dims=(2, 2)), 1+x
    if name == "double_wedge":
        a = dg.DoubleForm(CF((0, 0)), p=1, q=0, dim=2)
        b = dg.DoubleForm(CF((3, 4)), p=1, q=0, dim=2)
        value = 4*y*y - 3*x*y
        return a, dg.Wedge(a, b), direction, CF((0, value, -value, 0), dims=(2, 2))
    if name == "double_star":
        a = dg.DoubleForm(CF((0,)*4, dims=(2, 2)), p=1, q=1, dim=2)
        direction = CF((1+x, y, x*y, y*y), dims=(2, 2))
        return a, dg.star(a, manifold), direction, CF((y*y, -x*y, -y, 1+x), dims=(2, 2))
    if name == "composed":
        return a, 3*dg.star(2*a, manifold), direction, CF((-6*x*y, 6*y*y))
    raise AssertionError(name)


@pytest.mark.parametrize("name", [
    "scale", "divide", "negate", "wedge", "star", "inv_star", "d",
    "scalar_star", "scalar_wedge", "top_star", "double_wedge", "double_star", "composed",
])
@pytest.mark.parametrize("action", ["diff", "replace", "jacobian"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_zero_form_operation_keeps_symbolic_operand(
    name, action, roundtrip, make_unit_square_mesh
):
    if name == "d" and action == "jacobian":
        pytest.skip("spatial differentiation has no pointwise matrix Jacobian")
    a, result, direction, expected = _zero_operation(name)
    direction, expected = CF(direction), CF(expected)
    mesh = make_unit_square_mesh(maxh=0.7)
    assert_l2_close(Norm(result), CF(0), mesh, tol=1e-12)
    if roundtrip:
        a, result = pickle.loads(pickle.dumps((a, result)))
    if action == "diff":
        actual = result.Diff(a, direction)
    elif action == "replace":
        actual = result.Replace({a: direction})
    else:
        jacobian = result.Diff(a)
        assert tuple(jacobian.dims) == tuple(result.dims) + tuple(a.dims)
        actual = (jacobian.Reshape((result.dim, a.dim)) * direction.Reshape((a.dim,))).Reshape(tuple(result.dims))
    assert tuple(actual.dims) == tuple(expected.dims)
    assert_l2_close(actual, expected, mesh, tol=1e-12)
    assert_l2_close(actual.Compile(realcompile=False), expected, mesh, tol=1e-12)


@pytest.mark.parametrize("first_method", [False, True])
@pytest.mark.parametrize("second_method", [False, True])
def test_default_scalar_delta_can_be_applied_repeatedly(first_method, second_method):
    manifold = dg.RiemannianManifold(Id(2))
    scalar = dg.ScalarField(1+x)
    first = manifold.delta(scalar) if first_method else dg.delta(scalar, manifold)
    second = manifold.delta(first) if second_method else dg.delta(first, manifold)
    assert isinstance(second, dg.FormalZeroKForm)
    assert (second.degree, second.dim_space) == (-2, 0)
    assert (first.degree, first.dim_space) == (-1, 0)
    assert scalar.dim_space == 0


@pytest.mark.parametrize("method", [False, True])
def test_formal_delta_still_rejects_known_dimension_mismatch(method):
    manifold = dg.RiemannianManifold(Id(2))
    zero = dg.FormalZeroKForm(-1, 3)
    with pytest.raises(ValueError, match="dimension"):
        manifold.delta(zero) if method else dg.delta(zero, manifold)
