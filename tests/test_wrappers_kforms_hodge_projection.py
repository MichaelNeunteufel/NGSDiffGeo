from tests._kforms_support import BND, CF, Einsum, VOL, dg, l2_error, l2_error_bbnd, l2_error_bnd, l2_norm, pytest, x, y, z


def test_hodge_star_involution_nonorthonormal_metric(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.6)
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


def test_hodge_star_zero_preserves_degree(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    zero_two = dg.TwoForm(CF((0,) * 9, dims=(3, 3)), dim=dim)
    starred = dg.star(zero_two, rm)

    assert isinstance(starred, dg.KForm)
    assert starred.degree == 1
    assert l2_norm(starred, mesh) == pytest.approx(0)


def test_doubleform_hodge_star_involution(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    alpha = dg.OneForm(CF((x, y, z)))
    beta = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    ss = dg.star(dg.star(df, rm), rm)
    assert l2_error(ss, df, mesh) == pytest.approx(0)


def test_doubleform_hodge_star_left_right_inverse(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    alpha = dg.OneForm(CF((x, y, z)))
    beta = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    star_left = dg.star(df, rm, slot="left")
    assert star_left.degree_left == dim - 1
    assert star_left.degree_right == 1
    back_left = dg.inv_star(star_left, rm, slot="left")
    assert l2_error(back_left, df, mesh) == pytest.approx(0)

    star_right = dg.star(df, rm, slot="right")
    assert star_right.degree_left == 1
    assert star_right.degree_right == dim - 1
    back_right = dg.inv_star(star_right, rm, slot="right")
    assert l2_error(back_right, df, mesh) == pytest.approx(0)


def test_project_tensor_tangent_and_normal(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    v = dg.VectorField(CF((x, y, z)))
    v_tan = rm.ProjectTensor(v, "F")
    vn = rm.ProjectTensor(v, "n")

    assert isinstance(v_tan, dg.VectorField)
    assert isinstance(vn, dg.ScalarField)

    ip_tan = rm.InnerProduct(v_tan, rm.normal, vb=VOL)
    assert l2_error_bnd(ip_tan, CF(0), mesh) == pytest.approx(0)

    ip_vn = rm.InnerProduct(v, rm.normal, vb=VOL)
    assert l2_error_bnd(vn, ip_vn, mesh) == pytest.approx(0)

    alpha = dg.OneForm(CF((x, y, z)))
    alpha_tan = rm.ProjectTensor(alpha, "F")
    alpha_n = rm.ProjectTensor(alpha, "n")

    assert isinstance(alpha_tan, dg.OneForm)
    assert isinstance(alpha_n, dg.ScalarField)

    beta = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    df_tan = rm.ProjectTensor(df, "F")
    df_n = rm.ProjectTensor(df, "n")

    assert isinstance(df_tan, dg.DoubleForm)
    assert isinstance(df_n, dg.DoubleForm)


def test_project_tensor_edge_mode_on_bbnd(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    rm = rm_euclidean_3d

    v = dg.VectorField(CF((x + 1, y - 2, z + 3)))
    v_edge = rm.ProjectTensor(v, "E")
    v_edge_int = rm.ProjectTensor(v, 3)
    assert isinstance(v_edge, dg.VectorField)
    assert l2_error_bbnd(v_edge, v_edge_int, mesh) < 1e-12

    ip_n = rm.InnerProduct(v_edge, rm.edge_normal(0), vb=VOL)
    ip_cn = rm.InnerProduct(v_edge, rm.edge_conormal(0), vb=VOL)
    assert l2_error_bbnd(ip_n, CF(0), mesh) < 1e-10
    assert l2_error_bbnd(ip_cn, CF(0), mesh) < 1e-10

    alpha = dg.OneForm(CF((x, 1 + y, 2 + z)))
    alpha_edge = rm.ProjectTensor(alpha, "edge")
    assert isinstance(alpha_edge, dg.OneForm)

    alpha_n = rm.InnerProduct(alpha_edge, rm.edge_normal(0), vb=VOL)
    alpha_cn = rm.InnerProduct(alpha_edge, rm.edge_conormal(0), vb=VOL)
    assert l2_error_bbnd(alpha_n, CF(0), mesh) < 1e-10
    assert l2_error_bbnd(alpha_cn, CF(0), mesh) < 1e-10


def test_doubleform_hodge_star_boundary_matches_contraction(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    alpha = dg.OneForm(CF((x, y, z)))
    beta = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    star_bnd = dg.star(df, rm, vb=BND)

    star_vol = dg.star(df, rm, vb=VOL)
    normal = rm.normal
    left_deg = star_vol.degree_left
    contracted_left = rm.Contraction(star_vol, normal, slot=0)
    contracted_right = rm.Contraction(contracted_left, normal, slot=left_deg - 1)
    expected = dg.DoubleForm(contracted_right, p=left_deg - 1, q=star_vol.degree_right - 1, dim=dim)

    assert l2_error_bnd(star_bnd, expected, mesh) == pytest.approx(0)


def test_star_doubleform_flag_from_scalar(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    one = dg.ScalarField(CF(1), dim=dim)
    out = rm.star(one, double=True)

    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == dim
    assert out.degree_right == dim
    assert l2_norm(out, mesh) > 0


def test_inv_star_doubleform_flag_from_scalar(make_unit_cube_mesh, rm_euclidean_3d):
    mesh = make_unit_cube_mesh(maxh=0.6)
    dim = 3
    rm = rm_euclidean_3d

    one = dg.ScalarField(CF(1), dim=dim)
    out = rm.inv_star(one, double=True)

    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == dim
    assert out.degree_right == dim
    assert l2_norm(out, mesh) > 0
