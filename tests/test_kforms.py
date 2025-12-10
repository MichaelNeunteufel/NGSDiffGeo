import pytest
from netgen.occ import unit_square, unit_cube
from ngsolve import *

import ngsdiffgeo as dg


def l2_error(a, b, mesh):
    return sqrt(Integrate(InnerProduct(a - b, a - b) * dx(bonus_intorder=3), mesh))


def l2_norm(a, mesh):
    return sqrt(Integrate(InnerProduct(a, a) * dx(bonus_intorder=3), mesh))


def test_kform_construction_and_metadata():
    dim = 2
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.35))

    f = CoefficientFunction(x**2 + y)
    v = CF((x + y, y**2))

    f0 = dg.ScalarField(f, dim=dim)
    alpha = dg.OneForm(v)
    one_form = dg.KForm(v, k=1, dim=dim)

    assert f0.degree == 0
    assert alpha.degree == 1
    assert f0.covariant_indices == ""
    assert alpha.covariant_indices == "1"
    assert l2_error(alpha, one_form, mesh) == pytest.approx(0)
    assert isinstance(f0, dg.KForm)
    assert isinstance(alpha, dg.KForm)


def test_wedge_algebra_and_overflow_zero_2d():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

    f = dg.ScalarField(x * y, dim=2)
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((y, x * x)))

    ab = dg.Wedge(alpha, beta)
    ba = dg.Wedge(beta, alpha)

    assert ab.degree == 2
    assert ab.covariant_indices == "11"
    assert l2_error(ab, -ba, mesh) == pytest.approx(0)

    f_ab = dg.Wedge(f, ab)
    assert l2_error(f_ab, f * ab, mesh) == pytest.approx(0)

    aa = dg.Wedge(alpha, alpha)
    assert l2_norm(aa, mesh) == pytest.approx(0)

    overflow = dg.Wedge(ab, alpha)
    assert overflow.degree == 3
    assert overflow.covariant_indices == "111"
    assert l2_norm(overflow, mesh) == pytest.approx(0)


def test_wedge_algebra_and_overflow_zero_3d():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))

    f = dg.ScalarField(x * y * z, dim=3)
    alpha = dg.OneForm(CF((x, y, z)))
    beta = dg.OneForm(CF((y, x * x, z * z)))
    gamma = dg.KForm(CF((0, z, -y, -z, 0, x, y, -x, 0), dims=(3, 3)), k=2, dim=3)

    ab = dg.Wedge(alpha, beta)
    ba = dg.Wedge(beta, alpha)

    assert ab.degree == 2
    assert ab.covariant_indices == "11"
    assert l2_error(ab, -ba, mesh) < 1e-11

    f_ab = dg.Wedge(f, ab)
    assert l2_error(f_ab, f * ab, mesh) < 1e-11

    aa = dg.Wedge(alpha, alpha)
    assert l2_norm(aa, mesh) < 1e-11

    a_gamma = dg.Wedge(alpha, gamma)
    gamma_a = dg.Wedge(gamma, alpha)
    assert l2_error(a_gamma, gamma_a, mesh) < 1e-11

    overflow = dg.Wedge(a_gamma, alpha)
    assert overflow.degree == 4
    assert overflow.covariant_indices == "1111"
    assert l2_norm(overflow, mesh) < 1e-11


def test_exterior_derivative_basic_identities_2d():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

    f = x * y
    f_form = dg.ScalarField(f, dim=2)
    alpha = dg.OneForm(CF((x**2, y**2)))
    beta = dg.OneForm(CF((2 * y, -(x**2))))

    df = dg.d(f_form)
    ddf = dg.d(df)

    expected_df = dg.OneForm(CF((y, x)))
    assert l2_error(df, expected_df, mesh) < 1e-8
    assert l2_norm(ddf, mesh) < 1e-10

    left = dg.d(dg.Wedge(f_form, alpha))
    right = dg.Wedge(df, alpha) + dg.Wedge(f_form, dg.d(alpha))
    assert l2_error(left, right, mesh) < 1e-8

    left = dg.d(dg.Wedge(alpha, beta))
    right = dg.Wedge(dg.d(alpha), beta) + dg.Wedge(alpha, dg.d(beta))
    assert l2_error(left, right, mesh) < 1e-8


def test_exterior_derivative_basic_identities_3d():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    f = x * y * z
    f_form = dg.ScalarField(f, dim=3)
    alpha = dg.OneForm(CF((x**2, y**2, z**2)))
    beta = dg.OneForm(CF((2 * y, -3 * x**2, x * z)))

    df = dg.d(f_form)
    ddf = dg.d(df)

    expected_df = dg.OneForm(CF((y * z, x * z, x * y)))
    assert l2_error(df, expected_df, mesh) < 1e-8
    assert l2_norm(ddf, mesh) < 1e-10

    left = dg.d(dg.Wedge(f_form, alpha))
    right = dg.Wedge(df, alpha) + dg.Wedge(f_form, dg.d(alpha))
    assert l2_error(left, right, mesh) < 1e-8

    left = dg.d(dg.Wedge(alpha, beta))
    right = dg.Wedge(dg.d(alpha), beta) - dg.Wedge(alpha, dg.d(beta))
    assert l2_error(left, right, mesh) < 1e-8


def test_inheritance_and_typed_zero():
    dim = 3
    zero_scalar = dg.ScalarField(CF(0), dim=dim)
    df = dg.d(zero_scalar)
    assert isinstance(df, dg.OneForm)
    assert df.degree == 1

    a = dg.OneForm(CF((x, y, z)))
    b = dg.OneForm(CF((y, -x, 0)))
    w = dg.Wedge(a, b)
    assert isinstance(w, dg.KForm)
    assert w.degree == 2


def test_manifold_kform_factory_sets_dim():
    metric = CF((1, 0, 0, 1), dims=(2, 2))
    rm = dg.RiemannianManifold(metric)

    f = x + y
    alpha_cf = CF((x, y))

    f_form = rm.KForm(f, 0)
    alpha = rm.KForm(alpha_cf, 1)

    assert isinstance(f_form, dg.ScalarField)
    assert isinstance(alpha, dg.OneForm)
    assert f_form.degree == 0
    assert alpha.degree == 1

    # Ensure typed zeros stay typed and dimension follows the manifold
    zero_alpha = rm.KForm(CF((0, 0)), 1)
    assert isinstance(zero_alpha, dg.OneForm)


def test_wedge_associativity_scaling_3d():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))

    left = dg.Wedge(dg.Wedge(dx, dy), dz)
    right = dg.Wedge(dx, dg.Wedge(dy, dz))

    assert left.degree == 3 and right.degree == 3
    assert l2_error(left, right, mesh) == pytest.approx(0)


def test_exterior_derivative_scaling_on_rot_field():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))

    dy_dz = dg.Wedge(dy, dz)
    dz_dx = dg.Wedge(dz, dx)
    dx_dy = dg.Wedge(dx, dy)

    two_form = x * dy_dz + y * dz_dx + z * dx_dy
    d_two_form = dg.d(two_form)
    volume_form = dg.Wedge(dx, dy_dz)
    expected = 3 * volume_form

    assert d_two_form.degree == 3
    assert l2_error(d_two_form, expected, mesh) < 1e-11


def test_hodge_star_involution_nonorthonormal_metric():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3
    metric = CF((2, 0, 0, 0, 3, 0, 0, 0, 5), dims=(3, 3))
    rm = dg.RiemannianManifold(metric)

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))

    for form in (dx, dy, dz):
        ss = dg.star(dg.star(form, rm), rm)
        assert l2_error(ss, form, mesh) == pytest.approx(0)

    one = dg.ScalarField(CF(1), dim=dim)
    ss_scalar = dg.star(dg.star(one, rm), rm)
    assert l2_error(ss_scalar, one, mesh) == pytest.approx(0)


def test_hodge_star_zero_preserves_degree():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    zero_two = dg.TwoForm(CF((0,) * 9, dims=(3, 3)), dim=dim)
    starred = dg.star(zero_two, rm)

    assert isinstance(starred, dg.KForm)
    assert starred.degree == 1
    assert l2_norm(starred, mesh) == pytest.approx(0)


if __name__ == "__main__":
    pytest.main([__file__])
