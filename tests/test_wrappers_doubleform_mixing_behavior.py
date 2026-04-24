"""
Behavioral checks for implicit double-form coercions.

These tests are intentionally separate from the hard-guard dispatch tests:
they document cases where the implicit/coerced path is currently numerically
equivalent to the explicit path.
"""

import pytest

import ngsdiffgeo as dg
from ngsolve import CF, x, y, z

from tests._helpers import l2_error


def test_equivalent_coercion_doubleform_add_plain_matrix(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)

    df = dg.DoubleForm(
        CF((1 + x, x - y, x - y, 2 + y), dims=(2, 2)), p=1, q=1, dim=2
    )
    mat = CF((2 + x, y, x * y, 3 - y), dims=(2, 2))

    mixed = df + mat
    explicit = df.coef + mat

    assert isinstance(mixed, dg.DoubleForm)
    assert l2_error(mixed.coef, explicit, mesh) == pytest.approx(0.0)


def test_equivalent_coercion_doubleform_sub_plain_matrix(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)

    df = dg.DoubleForm(
        CF((1 + x, x - y, x - y, 2 + y), dims=(2, 2)), p=1, q=1, dim=2
    )
    mat = CF((2 + x, y, x * y, 3 - y), dims=(2, 2))

    mixed = df - mat
    explicit = df.coef - mat

    assert isinstance(mixed, dg.DoubleForm)
    assert l2_error(mixed.coef, explicit, mesh) == pytest.approx(0.0)


def test_equivalent_coercion_wedge_tensor11_autopromotion(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)

    tensor_11 = dg.TensorField(
        CF((1 + x, x * y, x * y, 2 + y), dims=(2, 2)),
        "11",
    )
    df = dg.DoubleForm(
        CF((2 - x, x + y, x + y, 3 + x), dims=(2, 2)), p=1, q=1, dim=2
    )

    mixed = dg.Wedge(tensor_11, df)
    explicit = dg.Wedge(dg.DoubleForm(tensor_11, p=1, q=1, dim=2), df)

    assert isinstance(mixed, dg.DoubleForm)
    assert l2_error(mixed.coef, explicit.coef, mesh) == pytest.approx(0.0)


def test_equivalent_coercion_sym_tensor11_autopromotion(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)

    tensor_11 = dg.TensorField(
        CF((1 + x, 2 * x - y, x + y, 2 + y), dims=(2, 2)),
        "11",
    )

    mixed = dg.Sym(tensor_11)
    explicit = dg.Sym(dg.DoubleForm(tensor_11, p=1, q=1, dim=2))

    assert isinstance(mixed, dg.DoubleForm)
    assert l2_error(mixed.coef, explicit.coef, mesh) == pytest.approx(0.0)


def test_equivalent_coercion_star_double_scalar_promotion(
    make_unit_cube_mesh, rm_euclidean_3d
):
    mesh = make_unit_cube_mesh(maxh=0.5)
    rm = rm_euclidean_3d

    scalar = 1 + x + y + z

    mixed = dg.star(scalar, rm, double=True)
    explicit = dg.star(dg.DoubleForm(scalar, p=0, q=0, dim=3), rm, double=True)

    assert isinstance(mixed, dg.DoubleForm)
    assert l2_error(mixed.coef, explicit.coef, mesh) == pytest.approx(0.0)


def test_equivalent_coercion_inv_star_double_scalar_promotion(
    make_unit_cube_mesh, rm_euclidean_3d
):
    mesh = make_unit_cube_mesh(maxh=0.5)
    rm = rm_euclidean_3d

    scalar = 1 + x + y + z

    mixed = dg.inv_star(scalar, rm, double=True)
    explicit = dg.inv_star(
        dg.DoubleForm(scalar, p=0, q=0, dim=3), rm, double=True
    )

    assert isinstance(mixed, dg.DoubleForm)
    assert l2_error(mixed.coef, explicit.coef, mesh) == pytest.approx(0.0)


def test_equivalent_coercion_trace_sigma_tensor11_autopromotion(
    make_unit_square_mesh, rm_euclidean_2d
):
    mesh = make_unit_square_mesh(maxh=0.3)
    rm = rm_euclidean_2d

    phi = dg.DoubleForm(
        CF((1 + x, x - y, x - y, 2 + y), dims=(2, 2)), p=1, q=1, dim=2
    )
    sigma_tensor = dg.TensorField(
        CF((2 + x, x * y, x * y, 3 - y), dims=(2, 2)),
        "11",
    )

    mixed = rm.TraceSigma(phi, sigma_tensor)
    explicit = rm.TraceSigma(phi, dg.DoubleForm(sigma_tensor, p=1, q=1, dim=2))

    assert l2_error(mixed, explicit, mesh) == pytest.approx(0.0)
