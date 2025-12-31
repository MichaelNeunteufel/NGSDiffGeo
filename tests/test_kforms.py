import pytest
from netgen.occ import unit_square, unit_cube
from ngsolve import *
from ngsolve.fem import Einsum

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


def test_doubleform_construction_metadata():
    dim = 2
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((x**2, y**2)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    assert df.degree_left == 1
    assert df.degree_right == 1
    assert df.dim_space == dim
    assert df.covariant_indices == "11"


def test_doubleform_wedge_blockwise_2d():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((x**2, y**2)))
    gamma = dg.OneForm(CF((y, x)))
    delta = dg.OneForm(CF((1 + x, 1 + y)))

    left = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    right = dg.DoubleForm(Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=dim)

    wedged = dg.Wedge(left, right)

    expected_left = dg.Wedge(alpha, gamma)
    expected_right = dg.Wedge(beta, delta)
    expected = dg.DoubleForm(
        Einsum("ij,kl->ijkl", expected_left, expected_right), p=2, q=2, dim=dim
    )

    assert wedged.degree_left == 2
    assert wedged.degree_right == 2
    assert l2_error(wedged, expected, mesh) == pytest.approx(0)

    left_big = dg.DoubleForm(
        Einsum("ij,k->ijk", expected_left, beta), p=2, q=1, dim=dim
    )
    overflow = dg.Wedge(left_big, right)
    assert overflow.degree_left == 3
    assert overflow.degree_right == 2
    assert l2_norm(overflow, mesh) == pytest.approx(0)


def test_doubleform_wedge_right_overflow_zero():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.TwoForm(CF((0, 1, -1, 0), dims=(2, 2)), dim=dim)
    left = dg.DoubleForm(Einsum("i,jk->ijk", alpha, beta), p=1, q=2, dim=dim)

    gamma = dg.OneForm(CF((y, x)))
    delta = dg.OneForm(CF((1 + x, 1 + y)))
    right = dg.DoubleForm(Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=dim)

    overflow = dg.Wedge(left, right)

    assert overflow.degree_left == 2
    assert overflow.degree_right == 3
    assert l2_norm(overflow, mesh) == pytest.approx(0)


def test_doubleform_wedge_right_slot_nonzero():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3

    rm = dg.RiemannianManifold(Id(dim))
    g_11 = dg.DoubleForm(rm.G, p=1, q=1, dim=dim)
    alpha = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
    tmp = dg.DoubleForm(alpha, p=0, q=1, dim=dim)

    wedged = dg.Wedge(g_11, tmp)
    swapped = dg.DoubleForm(Einsum("ijk->ikj", wedged.coef), p=1, q=2, dim=dim)

    assert wedged.degree_left == 1
    assert wedged.degree_right == 2
    assert l2_norm(wedged, mesh) > 1e-6
    assert l2_norm(wedged + swapped, mesh) == pytest.approx(0)


def test_doubleform_wedge_right_only_block():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3

    u = dg.OneForm(CF((x, y, z)))
    v = dg.OneForm(CF((1 + x, 1 + y, 1 + z)))
    du = dg.DoubleForm(u, p=0, q=1, dim=dim)
    dv = dg.DoubleForm(v, p=0, q=1, dim=dim)

    w = dg.Wedge(du, dv)
    expected = dg.DoubleForm(
        Einsum("i,j->ij", u, v) - Einsum("i,j->ji", u, v), p=0, q=2, dim=dim
    )

    assert l2_error(w, expected, mesh) == pytest.approx(0)


def test_doubleform_right_slot_consistency():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3

    alpha = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
    tmp01 = dg.DoubleForm(alpha, p=0, q=1, dim=dim)
    tmp10 = dg.DoubleForm(alpha, p=1, q=0, dim=dim)

    assert l2_error(tmp01, tmp10.trans, mesh) == pytest.approx(0)


def test_doubleform_hodge_star_involution():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x, y, z)))
    beta = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    ss = dg.star(dg.star(df, rm), rm)
    assert l2_error(ss, df, mesh) == pytest.approx(0)


def test_doubleform_inner_product_factorizes():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    a1 = dg.OneForm(CF((x, y)))
    b1 = dg.OneForm(CF((1 + x, 1 + y)))
    a2 = dg.OneForm(CF((x**2, y**2)))
    b2 = dg.OneForm(CF((2 + x, 3 + y)))

    df1 = dg.DoubleForm(Einsum("i,j->ij", a1, b1), p=1, q=1, dim=dim)
    df2 = dg.DoubleForm(Einsum("i,j->ij", a2, b2), p=1, q=1, dim=dim)

    ip_df = rm.InnerProduct(df1, df2)
    ip_expected = rm.InnerProduct(a1, a2) * rm.InnerProduct(b1, b2)

    assert l2_error(ip_df, ip_expected, mesh) == pytest.approx(0)


def test_doubleform_trace_contracts_first_slots():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    a = dg.OneForm(CF((x, y)))
    b = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", a, b), p=1, q=1, dim=dim)

    traced = rm.Trace(df)
    expected = rm.InnerProduct(a, b)

    assert isinstance(traced, dg.ScalarField)
    assert l2_error(traced, expected, mesh) == pytest.approx(0)


def test_doubleform_slot_inner_product_full_contraction():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 2 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    sip = dg.slot_inner_product(df, rm)
    expected = rm.InnerProduct(alpha, beta)

    assert l2_error(sip, expected, mesh) == pytest.approx(0)


def test_doubleform_slot_inner_product_degree_mismatch_raises():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.TwoForm(CF((0, 1, -1, 0), dims=(2, 2)), dim=dim)
    df = dg.DoubleForm(Einsum("i,jk->ijk", alpha, beta), p=1, q=2, dim=dim)

    with pytest.raises(Exception):
        dg.slot_inner_product(df, rm)


def test_doubleform_transpose_swaps_slots():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    df_t = df.trans
    expected = dg.DoubleForm(Einsum("i,j->ij", beta, alpha), p=1, q=1, dim=dim)

    assert df_t.degree_left == 1
    assert df_t.degree_right == 1
    assert l2_error(df_t, expected, mesh) == pytest.approx(0)


def test_doubleform_trace_l_parameter():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.TwoForm(
        CF((0, x, -x, 0), dims=(2, 2)),
        dim=dim,
    )
    beta = dg.TwoForm(
        CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)),
        dim=dim,
    )
    df = dg.DoubleForm(Einsum("ij,kl->ijkl", alpha, beta), p=2, q=2, dim=dim)

    trace1 = rm.Trace(df, l=1)
    trace1_default = rm.Trace(df)
    assert trace1.degree_left == 1
    assert trace1.degree_right == 1
    assert l2_error(trace1, trace1_default, mesh) == pytest.approx(0)

    trace2 = rm.Trace(df, l=2)
    expected2 = rm.InnerProduct(alpha, beta)
    assert l2_error(trace2, expected2, mesh) == pytest.approx(0)

    trace3 = rm.Trace(df, l=3)
    assert l2_norm(trace3, mesh) == pytest.approx(0)


def test_doubleform_trace_l0_identity():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    traced = rm.Trace(df, l=0)

    assert traced.degree_left == 1
    assert traced.degree_right == 1
    assert l2_error(traced, df, mesh) == pytest.approx(0)


def test_inner_product_forms_scaling_kforms():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=dim)

    ip = rm.InnerProduct(alpha, beta)
    ip_forms = rm.InnerProduct(alpha, beta, forms=True)

    assert l2_error(ip_forms, 0.5 * ip, mesh) == pytest.approx(0)


def test_inner_product_forms_scaling_doubleforms():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=dim)
    gamma = dg.TwoForm(CF((0, 2 + x, -(2 + x), 0), dims=(2, 2)), dim=dim)
    delta = dg.TwoForm(CF((0, 3 + y, -(3 + y), 0), dims=(2, 2)), dim=dim)

    df1 = dg.DoubleForm(Einsum("ij,kl->ijkl", alpha, beta), p=2, q=2, dim=dim)
    df2 = dg.DoubleForm(Einsum("ij,kl->ijkl", gamma, delta), p=2, q=2, dim=dim)

    ip = rm.InnerProduct(df1, df2)
    ip_forms = rm.InnerProduct(df1, df2, forms=True)

    assert l2_error(ip_forms, 0.25 * ip, mesh) == pytest.approx(0)


def test_doubleform_s_operator_matches_wedge():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 2 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    s_df = rm.s(df)
    expected_form = dg.Wedge(beta, alpha)
    expected = dg.DoubleForm(expected_form, p=2, q=0, dim=dim)

    assert s_df.degree_left == 2
    assert s_df.degree_right == 0
    assert l2_error(s_df, expected, mesh) == pytest.approx(0)


def test_wedge_tensor_with_doubleform():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(g, df)

    assert out.degree_left == 2
    assert out.degree_right == 2


def test_wedge_tensor_tensor_as_doubleform():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    h = dg.TensorField(CF((2, 0, 0, 3), dims=(2, 2)), covariant_indices="11")

    out = dg.Wedge(g, h)

    assert out.degree_left == 2
    assert out.degree_right == 2


def test_wedge_doubleform_tensor():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(df, g)

    assert out.degree_left == 2
    assert out.degree_right == 2


def test_wedge_tensor_doubleform_tensor_combinations():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out1 = dg.Wedge(g, df)
    out2 = dg.Wedge(df, g)
    out3 = dg.Wedge(g, g)

    assert out1.degree_left == 2
    assert out1.degree_right == 2
    assert out2.degree_left == 2
    assert out2.degree_right == 2
    assert out3.degree_left == 2
    assert out3.degree_right == 2


def test_wedge_with_scalarfield_kform():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    f = dg.ScalarField(x + y, dim=dim)
    alpha = dg.OneForm(CF((x, y)))

    out = dg.Wedge(f, alpha)
    expected = f * alpha

    assert isinstance(out, dg.KForm)
    assert out.degree == 1
    assert l2_error(out, expected, mesh) == pytest.approx(0)


def test_wedge_with_scalarfield_doubleform():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    f = dg.ScalarField(x + y, dim=dim)
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(f, df)
    expected = dg.DoubleForm(f * df, p=1, q=1, dim=dim)

    assert out.degree_left == 1
    assert out.degree_right == 1
    assert l2_error(out, expected, mesh) == pytest.approx(0)


def test_wedge_transposed_doubleform():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(df.trans, df)
    expected = dg.Wedge(
        dg.DoubleForm(Einsum("i,j->ij", beta, alpha), p=1, q=1, dim=dim), df
    )

    assert out.degree_left == 2
    assert out.degree_right == 2
    assert l2_error(out, expected, mesh) == pytest.approx(0)


def test_star_doubleform_flag_from_scalar():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    one = dg.ScalarField(CF(1), dim=dim)
    out = rm.star(one, double=True)

    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == dim
    assert out.degree_right == dim
    assert l2_norm(out, mesh) > 0


def test_inv_star_doubleform_flag_from_scalar():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    one = dg.ScalarField(CF(1), dim=dim)
    out = rm.inv_star(one, double=True)

    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == dim
    assert out.degree_right == dim
    assert l2_norm(out, mesh) > 0


if __name__ == "__main__":
    pytest.main([__file__])
