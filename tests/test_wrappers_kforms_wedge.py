from tests._kforms_support import (
    CF,
    Einsum,
    Integrate,
    Norm,
    VOL,
    dg,
    dx,
    l2_error,
    l2_norm,
    pytest,
    x,
    y,
    z,
)


def test_wedge_algebra_and_overflow_zero_2d(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)

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


def test_wedge_algebra_and_overflow_zero_3d(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.5)

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


def test_wedge_power_doubleform_edge_cases(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out0 = dg.WedgePower(df, 0)
    expected0 = dg.ScalarField(CF(1), dim=dim)
    assert l2_error(out0, expected0, mesh) == pytest.approx(0)

    out1 = dg.WedgePower(df, 1)
    assert l2_error(out1, df, mesh) == pytest.approx(0)


def test_sym_doubleform_properties(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    alpha = dg.OneForm(CF((x + 1, y)))
    beta = dg.OneForm(CF((2 * x, 3 * y + 1)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    symA = dg.Sym(A)
    skewA = 0.5 * (A - A.trans)

    assert isinstance(symA, dg.DoubleForm)
    assert symA.degree_left == 1
    assert symA.degree_right == 1
    assert l2_error(symA, symA.trans, mesh) < 1e-11
    assert l2_error(A, symA + skewA, mesh) < 1e-11
    assert l2_error(dg.Sym(symA), symA, mesh) < 1e-11


def test_sym_requires_square_doubleform():
    dim = 2
    alpha = dg.OneForm(CF((x, y)))
    omega = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    nonsquare = dg.DoubleForm(dg.TensorProduct(alpha, omega), p=1, q=2, dim=dim)
    assert nonsquare.degree_left == 1
    assert nonsquare.degree_right == 2

    with pytest.raises(ValueError, match=r"\(k,k\)"):
        dg.Sym(nonsquare)


def test_pow_operator_doubleform_and_tensorfield(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out0 = df**0
    expected0 = dg.ScalarField(CF(1), dim=dim)
    assert l2_error(out0, expected0, mesh) == pytest.approx(0)

    out1 = df**1
    assert l2_error(out1, df, mesh) == pytest.approx(0)

    out2 = df**2
    expected2 = dg.WedgePower(df, 2)
    assert l2_error(out2, expected2, mesh) == pytest.approx(0)

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    g0 = g**0
    assert l2_error(g0, expected0, mesh) == pytest.approx(0)

    g1 = g**1
    expected_g1 = dg.WedgePower(g, 1)
    assert isinstance(g1, dg.DoubleForm)
    assert l2_error(g1, expected_g1, mesh) == pytest.approx(0)

    g2 = g**2
    expected_g2 = dg.WedgePower(g, 2)
    assert l2_error(g2, expected_g2, mesh) == pytest.approx(0)


def test_exterior_derivative_basic_identities_2d(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)

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


def test_exterior_derivative_basic_identities_3d(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.5)
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


def test_wedge_associativity_scaling_3d(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.6)

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))

    left = dg.Wedge(dg.Wedge(dx, dy), dz)
    right = dg.Wedge(dx, dg.Wedge(dy, dz))

    assert left.degree == 3 and right.degree == 3
    assert l2_error(left, right, mesh) == pytest.approx(0)


def test_exterior_derivative_scaling_on_rot_field(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.6)

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


def test_doubleform_wedge_blockwise_2d(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
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

    left_big = dg.DoubleForm(Einsum("ij,k->ijk", expected_left, beta), p=2, q=1, dim=dim)
    overflow = dg.Wedge(left_big, right)
    assert overflow.degree_left == 3
    assert overflow.degree_right == 2
    assert l2_norm(overflow, mesh) == pytest.approx(0)


def test_doubleform_wedge_right_overflow_zero(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
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


def test_doubleform_wedge_high_rank_zero_remains_representable(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    alpha = dg.OneForm(CF((x, y, z)))
    beta = dg.OneForm(CF((1 + x, 1 + y, 1 + z)))
    gamma = dg.OneForm(CF((x * y, y * z, z * x)))

    left_11 = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    left_22 = dg.Wedge(left_11, left_11)
    right_33 = dg.Wedge(
        left_22,
        dg.DoubleForm(Einsum("i,j->ij", gamma, alpha), p=1, q=1, dim=dim),
    )

    overflow = dg.Wedge(left_22, right_33)

    assert overflow.degree_left == 5
    assert overflow.degree_right == 5
    assert Integrate(Norm(overflow) * dx(bonus_intorder=3), mesh) == pytest.approx(0)
    assert Integrate(rm.SlotInnerProduct(overflow) * dx(bonus_intorder=3), mesh) == pytest.approx(0)


def test_doubleform_wedge_right_slot_nonzero(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    g_11 = dg.DoubleForm(rm.G, p=1, q=1, dim=dim)
    alpha = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
    tmp = dg.DoubleForm(alpha, p=0, q=1, dim=dim)

    wedged = dg.Wedge(g_11, tmp)
    swapped = dg.DoubleForm(Einsum("ijk->ikj", wedged.coef), p=1, q=2, dim=dim)

    assert wedged.degree_left == 1
    assert wedged.degree_right == 2
    assert l2_norm(wedged, mesh) > 1e-6
    assert l2_norm(wedged + swapped, mesh) == pytest.approx(0)


def test_doubleform_wedge_right_only_block(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3

    u = dg.OneForm(CF((x, y, z)))
    v = dg.OneForm(CF((1 + x, 1 + y, 1 + z)))
    du = dg.DoubleForm(u, p=0, q=1, dim=dim)
    dv = dg.DoubleForm(v, p=0, q=1, dim=dim)

    w = dg.Wedge(du, dv)
    expected = dg.DoubleForm(Einsum("i,j->ij", u, v) - Einsum("i,j->ji", u, v), p=0, q=2, dim=dim)

    assert l2_error(w, expected, mesh) == pytest.approx(0)


def test_doubleform_right_slot_consistency(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3

    alpha = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
    tmp01 = dg.DoubleForm(alpha, p=0, q=1, dim=dim)
    tmp10 = dg.DoubleForm(alpha, p=1, q=0, dim=dim)

    assert l2_error(tmp01, tmp10.trans, mesh) == pytest.approx(0)


def test_doubleform_s_operator_matches_wedge(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 2 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    s_df = rm.s(df)
    expected_form = dg.Wedge(beta, alpha)
    expected = dg.DoubleForm(expected_form, p=2, q=0, dim=dim)

    assert s_df.degree_left == 2
    assert s_df.degree_right == 0
    assert l2_error(s_df, expected, mesh) == pytest.approx(0)


def test_wedge_tensor_with_doubleform(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(g, df)

    assert out.degree_left == 2
    assert out.degree_right == 2


def test_wedge_tensor_tensor_as_doubleform(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    h = dg.TensorField(CF((2, 0, 0, 3), dims=(2, 2)), covariant_indices="11")

    out = dg.Wedge(g, h)

    assert out.degree_left == 2
    assert out.degree_right == 2


def test_wedge_doubleform_tensor(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    g = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), covariant_indices="11")
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(df, g)

    assert out.degree_left == 2
    assert out.degree_right == 2


def test_wedge_tensor_doubleform_tensor_combinations(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
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


def test_wedge_with_scalarfield_kform(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    f = dg.ScalarField(x + y, dim=dim)
    alpha = dg.OneForm(CF((x, y)))

    out = dg.Wedge(f, alpha)
    expected = f * alpha

    assert isinstance(out, dg.KForm)
    assert out.degree == 1
    assert l2_error(out, expected, mesh) == pytest.approx(0)


def test_wedge_with_scalarfield_doubleform(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
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


def test_wedge_transposed_doubleform(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    out = dg.Wedge(df.trans, df)
    expected = dg.Wedge(dg.DoubleForm(Einsum("i,j->ij", beta, alpha), p=1, q=1, dim=dim), df)

    assert out.degree_left == 2
    assert out.degree_right == 2
    assert l2_error(out, expected, mesh) == pytest.approx(0)
