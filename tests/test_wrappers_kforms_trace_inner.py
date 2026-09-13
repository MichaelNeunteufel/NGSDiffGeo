from tests._kforms_support import BBND, CF, Cross, Einsum, Id, Norm, acos, dg, l2_error, l2_error_bbnd, l2_norm, pytest, specialcf, sqrt, x, y, z


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


@pytest.mark.parametrize("metric", [Id(2), (1 + x*x + y*y)*Id(2),
                                     CF((2+x*x, 0.3, 0.3, 1+y*y), dims=(2, 2))])
def test_angle_defect_2d_matches_vertex_angle(make_unit_square_mesh, metric):
    from ngsolve import Integrate, dx

    mesh = make_unit_square_mesh(maxh=0.5)
    rm = dg.RiemannianManifold(metric)
    tangents = specialcf.VertexTangentialVectors(2)
    a, b = tangents[:, 0], tangents[:, 1]
    expected = acos(a*b) - acos((metric*a*b) / sqrt((metric*a*a)*(metric*b*b)))
    theta = rm.AngleDefect
    assert isinstance(theta, dg.ScalarField)
    assert theta.dim_space == 2
    for value in (theta, theta.Compile()):
        error = sqrt(Integrate((value-expected)**2 * dx(element_vb=BBND), mesh))
        assert error < 1e-12


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


def test_compact_doubleform_trace_preserves_symbolic_operations(
    make_unit_square_mesh,
):
    import pickle

    from ngsolve import Parameter

    mesh = make_unit_square_mesh(maxh=0.4)
    dim = 2
    metric_parameter = Parameter(0.4)
    metric = CF(
        (2 + metric_parameter + x, 0.2, 0.2, 3 + y),
        dims=(dim, dim),
    )
    rm = dg.RiemannianManifold(metric)
    input_parameter = Parameter(1.25)
    scale = dg.ScalarField(input_parameter, dim=dim)
    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    beta = dg.OneForm(CF((2 - y, 1 + x)))
    gamma = dg.OneForm(CF((3 + y, 1 - x)))
    delta = dg.OneForm(CF((1 + x * y, 2 - y)))
    left = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    right = dg.DoubleForm(Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=dim)
    compact = scale * dg.Wedge(left, right)

    trace_once = rm.Trace(compact, l=1)
    trace_full = rm.Trace(compact, l=2)
    expected_once = dg.DoubleForm(
        Einsum("ijkl,ik->jl", compact, rm.G_inv),
        p=1,
        q=1,
        dim=dim,
    )
    expected_full = Einsum(
        "ijkl,ik,jl->", compact, rm.G_inv, rm.G_inv
    )

    assert any(
        "CompactDoubleTraceIndependent" in name
        for name in dg.CFStats(trace_once)["types"]
    )
    assert l2_error(trace_once, expected_once, mesh) < 1e-10
    assert l2_error(trace_full, expected_full, mesh) < 1e-10
    assert l2_error(
        trace_once.Diff(input_parameter, CF(1)),
        expected_once / input_parameter,
        mesh,
    ) < 1e-9
    assert l2_error(
        trace_once.Diff(metric_parameter, CF(1)),
        expected_once.Diff(metric_parameter, CF(1)),
        mesh,
    ) < 1e-9
    assert l2_error(
        trace_once.Replace({input_parameter: CF(2)}),
        (2 / input_parameter) * expected_once,
        mesh,
    ) < 1e-9
    assert l2_error(
        pickle.loads(pickle.dumps(trace_once)), expected_once, mesh
    ) < 1e-9
    for value, expected in (
        (trace_once, expected_once),
        (trace_full, expected_full),
    ):
        assert l2_error(
            value.Compile(realcompile=False, wait=True, maxderiv=0),
            expected,
            mesh,
        ) < 1e-9

    # Native code generation must preserve a negative first contraction term.
    # A norm-only comparison can hide this as an off-diagonal sign change.
    assert l2_error(
        trace_once.Compile(realcompile=True, wait=True, maxderiv=0),
        expected_once,
        mesh,
    ) < 1e-9


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


def test_compact_metric_inner_products_keep_compact_consumers(
    make_unit_square_mesh,
):
    import pickle

    from ngsolve import (
        BilinearForm,
        Norm,
        NumberSpace,
        Parameter,
        SymbolicBFI,
        TaskManager,
    )

    mesh = make_unit_square_mesh(maxh=0.4)
    dim = 2
    metric = CF((2 + x, 0.2, 0.2, 3 + y), dims=(dim, dim))
    rm = dg.RiemannianManifold(metric)
    parameter = Parameter(1.25)
    parameter_form = dg.ScalarField(parameter, dim=dim)
    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    beta = dg.OneForm(CF((2 - y, 1 + x)))
    gamma = dg.OneForm(CF((3 + y, 1 - x)))
    delta = dg.OneForm(CF((1 + x * y, 2 - y)))

    left_k = parameter_form * dg.Wedge(alpha, beta)
    right_k = dg.Wedge(gamma, delta)
    compact_k = rm.InnerProduct(left_k, right_k, forms=True)
    expected_k = 0.5 * dg.Einsum(
        "ij,kl,ik,jl->",
        left_k,
        right_k,
        rm.G_inv,
        rm.G_inv,
    )

    left_df = dg.DoubleForm(
        dg.Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim
    )
    right_df = dg.DoubleForm(
        dg.Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=dim
    )
    compact_left = parameter_form * dg.Wedge(left_df, right_df)
    compact_right = dg.Wedge(right_df, left_df)
    compact_double = rm.InnerProduct(
        compact_left, compact_right, forms=True
    )
    expected_double = 0.25 * dg.Einsum(
        "ijkl,mnop,im,jn,ko,lp->",
        compact_left,
        compact_right,
        rm.G_inv,
        rm.G_inv,
        rm.G_inv,
        rm.G_inv,
    )

    assert any(
        "CompactKFormInnerProduct" in name
        for name in dg.CFStats(compact_k)["types"]
    )
    assert any(
        "CompactDoubleFormInnerProduct" in name
        for name in dg.CFStats(compact_double)["types"]
    )
    assert l2_error(compact_k, expected_k, mesh) < 1e-11
    assert l2_error(compact_double, expected_double, mesh) < 1e-10
    assert l2_error(
        compact_k.Diff(parameter, CF(1)), expected_k / parameter, mesh
    ) < 1e-10
    assert l2_error(
        compact_double.Diff(parameter, CF(1)),
        expected_double / parameter,
        mesh,
    ) < 1e-9
    assert l2_error(
        compact_k.Replace({parameter: CF(2)}),
        2 / parameter * expected_k,
        mesh,
    ) < 1e-10
    assert l2_error(
        compact_double.Replace({parameter: CF(2)}),
        2 / parameter * expected_double,
        mesh,
    ) < 1e-9
    assert l2_error(
        pickle.loads(pickle.dumps(compact_k)), expected_k, mesh
    ) < 1e-10
    assert l2_error(
        pickle.loads(pickle.dumps(compact_double)), expected_double, mesh
    ) < 1e-9

    for value, expected in (
        (compact_k, expected_k),
        (compact_double, expected_double),
    ):
        compiled = value.Compile(realcompile=False, wait=True, maxderiv=0)
        space = NumberSpace(mesh)
        trial, test = space.TnT()
        assembled_errors = []
        for simd in (False, True):
            form = BilinearForm(space)
            integrator = SymbolicBFI(
                (compiled - expected) ** 2 * trial * test,
                bonus_intorder=2,
                simd_evaluate=simd,
            )
            form += integrator
            with TaskManager():
                form.Assemble()
            assembled_errors.append(Norm(form.mat.AsVector()))
            assert integrator.simd_evaluate is simd
        assert assembled_errors[0] < 1e-18
        assert assembled_errors[1] == pytest.approx(
            assembled_errors[0], rel=1e-11, abs=1e-18
        )


def test_metric_inner_product_keeps_dense_fallback_for_unproven_forms(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.4)
    metric = CF((2 + x, 0.2, 0.2, 3 + y), dims=(2, 2))
    rm = dg.RiemannianManifold(metric)

    left_k = dg.TwoForm(CF((x, 1 + y, 2 - x, y), dims=(2, 2)), dim=2)
    right_k = dg.TwoForm(
        CF((1 + x, 3 - y, x * y, 2 + y), dims=(2, 2)), dim=2
    )
    result_k = rm.InnerProduct(left_k, right_k, forms=True)
    expected_k = 0.5 * dg.Einsum(
        "ij,kl,ik,jl->", left_k, right_k, rm.G_inv, rm.G_inv
    )

    left_df = dg.DoubleForm(
        CF((1 + x, y, 2 - y, x * y), dims=(2, 2)),
        p=1,
        q=1,
        dim=2,
    )
    right_df = dg.DoubleForm(
        CF((x, 3 + y, 1 - x, 2 + x * y), dims=(2, 2)),
        p=1,
        q=1,
        dim=2,
    )
    result_df = rm.InnerProduct(left_df, right_df, forms=True)
    expected_df = dg.Einsum(
        "ij,kl,ik,jl->", left_df, right_df, rm.G_inv, rm.G_inv
    )

    assert not any(
        "CompactKFormInnerProduct" in name
        for name in dg.CFStats(result_k)["types"]
    )
    assert not any(
        "CompactDoubleFormInnerProduct" in name
        for name in dg.CFStats(result_df)["types"]
    )
    assert l2_error(result_k, expected_k, mesh) < 1e-11
    assert l2_error(result_df, expected_df, mesh) < 1e-11


def test_compact_metric_inner_product_metric_derivative_realcompiles(
    make_unit_square_mesh,
):
    from ngsolve import Parameter

    mesh = make_unit_square_mesh(maxh=0.4)
    metric_parameter = Parameter(0.4)
    metric = CF(
        (2 + metric_parameter + x, 0.2, 0.2, 3 + y),
        dims=(2, 2),
    )
    rm = dg.RiemannianManifold(metric)
    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    beta = dg.OneForm(CF((2 - y, 1 + x)))
    gamma = dg.OneForm(CF((3 + y, 1 - x)))
    delta = dg.OneForm(CF((1 + x * y, 2 - y)))
    compact = dg.Wedge(
        dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=2),
        dg.DoubleForm(Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=2),
    )
    result = rm.InnerProduct(compact, compact, forms=True).Diff(
        metric_parameter, CF(1)
    )
    expected = (
        0.25
        * dg.Einsum(
            "ijkl,mnop,im,jn,ko,lp->",
            compact,
            compact,
            rm.G_inv,
            rm.G_inv,
            rm.G_inv,
            rm.G_inv,
        )
    ).Diff(metric_parameter, CF(1))

    assert l2_error(result, expected, mesh) < 1e-9
    assert l2_error(
        result.Compile(realcompile=True, wait=True, maxderiv=0),
        expected,
        mesh,
    ) < 1e-9


def test_compact_metric_inner_products_preserve_complex_values(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.4)
    rm = dg.RiemannianManifold(
        CF((2 + x, 0.2, 0.2, 3 + y), dims=(2, 2))
    )
    alpha = dg.OneForm(CF(((1 + 2j) * (1 + x), (1 + 2j) * y)))
    beta = dg.OneForm(CF((2 - y, 1 + x)))
    left = dg.Wedge(alpha, beta)
    right = dg.Wedge(beta, alpha)
    result = rm.InnerProduct(left, right, forms=True)
    expected = 0.5 * dg.Einsum(
        "ij,kl,ik,jl->", left, right, rm.G_inv, rm.G_inv
    )

    assert result.is_complex
    assert l2_norm(Norm(result - expected), mesh) < 1e-10
    assert l2_norm(
        Norm(
            result.Compile(realcompile=False, wait=True, maxderiv=0)
            - expected
        ),
        mesh,
    ) < 1e-10
