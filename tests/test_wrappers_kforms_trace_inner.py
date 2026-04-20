from tests._kforms_support import BBND, CF, Cross, Einsum, Id, acos, dg, l2_error, l2_error_bbnd, l2_norm, pytest, specialcf, sqrt, x, y, z


def test_doubleform_inner_product_factorizes(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    a1 = dg.OneForm(CF((x, y)))
    b1 = dg.OneForm(CF((1 + x, 1 + y)))
    a2 = dg.OneForm(CF((x**2, y**2)))
    b2 = dg.OneForm(CF((2 + x, 3 + y)))

    df1 = dg.DoubleForm(Einsum("i,j->ij", a1, b1), p=1, q=1, dim=dim)
    df2 = dg.DoubleForm(Einsum("i,j->ij", a2, b2), p=1, q=1, dim=dim)

    ip_df = rm.InnerProduct(df1, df2)
    ip_expected = rm.InnerProduct(a1, a2) * rm.InnerProduct(b1, b2)

    assert l2_error(ip_df, ip_expected, mesh) == pytest.approx(0)


def test_doubleform_trace_contracts_first_slots(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    a = dg.OneForm(CF((x, y)))
    b = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", a, b), p=1, q=1, dim=dim)

    traced = rm.Trace(df)
    expected = rm.InnerProduct(a, b)

    assert isinstance(traced, dg.ScalarField)
    assert l2_error(traced, expected, mesh) == pytest.approx(0)


def test_trace_scalarfield_returns_zero(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    f = dg.ScalarField(x + y, dim=dim)
    traced = rm.Trace(f)

    assert isinstance(traced, dg.ScalarField)
    assert l2_norm(traced, mesh) == pytest.approx(0)


def test_trace_scalarfield_l0_returns_input(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    f = dg.ScalarField(x + y, dim=dim)
    traced = rm.Trace(f, l=0)

    assert isinstance(traced, dg.ScalarField)
    assert l2_error(traced, f, mesh) == pytest.approx(0)


def test_doubleform_trace_sigma_raises_indices(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    metric = CF((2, 0, 0, 3), dims=(2, 2))
    rm = dg.RiemannianManifold(metric)

    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    beta = dg.OneForm(CF((2 - x, 3 - y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    traced = rm.TraceSigma(df, rm.G)
    expected = rm.InnerProduct(alpha, beta)

    assert l2_error(traced, expected, mesh) == pytest.approx(0)


def test_scalarfield_trace_sigma_returns_zero(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    f = dg.ScalarField(x + y, dim=dim)
    traced = rm.TraceSigma(f, rm.G)

    assert isinstance(traced, dg.ScalarField)
    assert l2_error(traced, dg.ScalarField(0, dim=dim), mesh) == pytest.approx(0)


def test_angle_defect_zero_for_euclidean_metric(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.5)
    rm = rm_euclidean_3d

    theta = rm.AngleDefect

    assert isinstance(theta, dg.ScalarField)
    assert l2_norm(theta, mesh, vb=BBND) == pytest.approx(0)


def test_angle_defect_matches_manual_formula(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.5)
    dim = 3
    metric = CF((2.0, 0, 0, 0, 3.0, 0, 0, 0, 5.0), dims=(3, 3))
    rm = dg.RiemannianManifold(metric)

    bbnd_tang = specialcf.EdgeFaceTangentialVectors(3)
    tE = specialcf.tangential(mesh.dim, True)
    tef1 = bbnd_tang[:, 0]
    tef2 = bbnd_tang[:, 1]
    n1 = Cross(tE, tef1)
    n2 = Cross(tE, tef2)

    n1g = 1 / sqrt(rm.G_inv[n1, n1]) * rm.Raise(dg.OneForm(n1))
    n2g = 1 / sqrt(rm.G_inv[n2, n2]) * rm.Raise(dg.OneForm(n2))
    theta_manual = dg.ScalarField(acos(n1 * n2) - acos(rm.G[n1g, n2g]), dim=dim)

    assert l2_error_bbnd(rm.AngleDefect, theta_manual, mesh) == pytest.approx(0)


def test_angle_defect_raises_in_2d():
    rm = dg.RiemannianManifold(Id(2))

    with pytest.raises(Exception, match="Angle defect only available in 3D"):
        _ = rm.AngleDefect


def test_doubleform_slot_inner_product_full_contraction(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 2 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    sip = dg.slot_inner_product(df, rm)
    expected = rm.InnerProduct(alpha, beta)

    assert l2_error(sip, expected, mesh) == pytest.approx(0)


def test_doubleform_slot_inner_product_degree_mismatch_raises(rm_euclidean_2d):
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.TwoForm(CF((0, 1, -1, 0), dims=(2, 2)), dim=dim)
    df = dg.DoubleForm(Einsum("i,jk->ijk", alpha, beta), p=1, q=2, dim=dim)

    with pytest.raises(Exception):
        dg.slot_inner_product(df, rm)


def test_slot_inner_product_scalar_field(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    f = dg.ScalarField(x + y, dim=dim)
    expected = f

    sip_function = dg.slot_inner_product(f, rm)
    sip_method = rm.SlotInnerProduct(f)

    assert l2_error(sip_function, expected, mesh) == pytest.approx(0)
    assert l2_error(sip_method, expected, mesh) == pytest.approx(0)


def test_doubleform_transpose_swaps_slots(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    df_t = df.trans
    expected = dg.DoubleForm(Einsum("i,j->ij", beta, alpha), p=1, q=1, dim=dim)

    assert df_t.degree_left == 1
    assert df_t.degree_right == 1
    assert l2_error(df_t, expected, mesh) == pytest.approx(0)


def test_doubleform_trace_l_parameter(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=dim)
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


def test_doubleform_trace_l0_identity(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((1 + x, 1 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    traced = rm.Trace(df, l=0)

    assert traced.degree_left == 1
    assert traced.degree_right == 1
    assert l2_error(traced, df, mesh) == pytest.approx(0)


def test_inner_product_forms_scaling_kforms(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=dim)

    ip = rm.InnerProduct(alpha, beta)
    ip_forms = rm.InnerProduct(alpha, beta, forms=True)

    assert l2_error(ip_forms, 0.5 * ip, mesh) == pytest.approx(0)


def test_inner_product_forms_scaling_doubleforms(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    dim = 2
    rm = rm_euclidean_2d

    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=dim)
    gamma = dg.TwoForm(CF((0, 2 + x, -(2 + x), 0), dims=(2, 2)), dim=dim)
    delta = dg.TwoForm(CF((0, 3 + y, -(3 + y), 0), dims=(2, 2)), dim=dim)

    df1 = dg.DoubleForm(Einsum("ij,kl->ijkl", alpha, beta), p=2, q=2, dim=dim)
    df2 = dg.DoubleForm(Einsum("ij,kl->ijkl", gamma, delta), p=2, q=2, dim=dim)

    ip = rm.InnerProduct(df1, df2)
    ip_forms = rm.InnerProduct(df1, df2, forms=True)

    assert l2_error(ip_forms, 0.25 * ip, mesh) == pytest.approx(0)
