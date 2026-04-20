from tests._kforms_support import CF, CoefficientFunction, Einsum, Mesh, dg, l2_error, pytest, x, y, z


def test_kform_construction_and_metadata(make_unit_square_mesh):
    dim = 2
    mesh = make_unit_square_mesh(maxh=0.35)

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

    zero_alpha = rm.KForm(CF((0, 0)), 1)
    assert isinstance(zero_alpha, dg.OneForm)


def test_doubleform_construction_metadata():
    dim = 2
    alpha = dg.OneForm(CF((x, y)))
    beta = dg.OneForm(CF((x**2, y**2)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    assert df.degree_left == 1
    assert df.degree_right == 1
    assert df.dim_space == dim
    assert df.covariant_indices == "11"
