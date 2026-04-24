import pytest
from netgen.csg import unit_cube
from netgen.occ import unit_square
from ngsolve import BBND, BND, VOL, CF, Id, Integrate, Mesh, x, y, z
from ngsolve.fem import Einsum

import ngsdiffgeo as dg
from tests._helpers import l2_norm


def test_contract_slot_formal_preserves_negative_degree_propagation():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    phi = dg.DoubleForm(x + y, p=0, q=0, dim=dim)
    X = rm.normal

    out_left = rm.ContractSlot(phi, X, slot="left")
    out_both = rm.ContractSlot(out_left, X, slot="right")

    assert isinstance(out_left, dg.FormalZeroDoubleForm)
    assert out_left.degree_left == -1
    assert out_left.degree_right == 0
    assert isinstance(out_both, dg.FormalZeroDoubleForm)
    assert out_both.degree_left == -1
    assert out_both.degree_right == -1


def test_project_doubleform_formal_preserves_degree_effect_on_empty_slots():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    phi = dg.DoubleForm(x + y, p=0, q=0, dim=dim)

    out = rm.ProjectDoubleForm(phi, left="n", right="n")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == -1


def test_star_formal_recovers_nonnegative_degree_from_negative_formal_zero():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    phi = dg.FormalZeroDoubleForm(-1, -1, dim)

    out = dg.star(phi, rm, vb=BND, double=True)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 3
    assert out.degree_right == 3


def test_star_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    phi = dg.FormalZeroDoubleForm(-1, -1, dim)

    out = dg.star(phi, rm, vb=BND, double=True)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 3
    assert out.degree_right == 3


def test_inv_star_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    phi = dg.FormalZeroDoubleForm(-1, -1, dim)

    out = dg.inv_star(phi, rm, vb=BND, double=True)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 3
    assert out.degree_right == 3


def test_star_dispatches_concrete_overdegree_doubleform_to_formal_zero():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    twoform_cf = CF(
        (
            0,
            -(1 + x),
            2 + y,
            1 + x,
            0,
            -(3 + z),
            -(2 + y),
            3 + z,
            0,
        ),
        dims=(3, 3),
    )
    phi = dg.DoubleForm(twoform_cf, p=0, q=2, dim=dim)

    out = dg.star(phi, rm, vb=BBND, double=True)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 1
    assert out.degree_right == -1


def test_inv_star_dispatches_concrete_overdegree_doubleform_to_formal_zero():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    twoform_cf = CF(
        (
            0,
            -(1 + x),
            2 + y,
            1 + x,
            0,
            -(3 + z),
            -(2 + y),
            3 + z,
            0,
        ),
        dims=(3, 3),
    )
    phi = dg.DoubleForm(twoform_cf, p=2, q=0, dim=dim)

    out = dg.inv_star(phi, rm, vb=BBND, double=True)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 1


def test_materialize_zero_preserves_representable_type():
    dim = 2
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

    zdf = dg.FormalZeroDoubleForm(1, 1, dim)
    mat = dg.materialize_zero(zdf)

    assert isinstance(mat, dg.DoubleForm)
    assert mat.degree_left == 1
    assert mat.degree_right == 1
    assert l2_norm(mat, mesh) < 1e-12


def test_materialize_zero_rejects_negative_degree():
    with pytest.raises(ValueError, match="negative degree"):
        dg.materialize_zero(dg.FormalZeroDoubleForm(-1, 0, 3))


def test_wedge_formal_preserves_degree_sum():
    dim = 3
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, alpha), p=1, q=1, dim=dim)
    zdf = dg.FormalZeroDoubleForm(-1, 0, dim)

    out = dg.Wedge(zdf, df)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 1


def test_wedge_dispatches_formal_zero_input():
    dim = 3
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, alpha), p=1, q=1, dim=dim)
    zdf = dg.FormalZeroDoubleForm(-1, 0, dim)

    out = dg.Wedge(zdf, df)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 1


def test_inner_product_formal_collapses_formal_zero_to_scalar_zero():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    beta = dg.OneForm(CF((2 + x, 3 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    zdf = dg.FormalZeroDoubleForm(1, 1, dim)

    out = rm.InnerProduct(zdf, df, forms=True)

    assert Integrate(out, mesh) == pytest.approx(0)


def test_formal_zero_doubleform_innerproduct_returns_scalar_cf():
    dim = 3
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)
    df = dg.DoubleForm(1 + x + y + z, p=0, q=0, dim=dim)

    out = zdf.InnerProduct(df)

    assert hasattr(out, "dim")
    assert out.dim == 1


def test_formal_zero_doubleform_innerproduct_rejects_degree_mismatch():
    dim = 3
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(alpha, p=1, q=0, dim=dim)

    with pytest.raises(TypeError, match="matching double-form degrees"):
        zdf.InnerProduct(df)


def test_formal_zero_kform_innerproduct_rejects_degree_mismatch():
    dim = 3
    zkf = dg.FormalZeroKForm(1, dim)
    beta = dg.TwoForm(
        CF(
            (
                0,
                -(1 + x),
                2 + y,
                1 + x,
                0,
                -(3 + z),
                -(2 + y),
                3 + z,
                0,
            ),
            dims=(3, 3),
        ),
        dim=dim,
    )

    with pytest.raises(TypeError, match="matching form degrees"):
        zkf.InnerProduct(beta)


def test_slot_inner_product_dispatches_formal_zero_input():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.35))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(1, 1, dim)

    out = rm.SlotInnerProduct(zdf, vb=VOL, forms=True)

    assert Integrate(out, mesh) == pytest.approx(0)


def test_slot_inner_product_top_level_dispatches_formal_zero_input():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.35))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(1, 1, dim)

    out = dg.slot_inner_product(zdf, rm, vb=VOL, forms=True)

    assert Integrate(out, mesh) == pytest.approx(0)


def test_inner_product_dispatches_formal_zero_input():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    beta = dg.OneForm(CF((2 + x, 3 + y)))
    df = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    zdf = dg.FormalZeroDoubleForm(1, 1, dim)

    out = rm.InnerProduct(zdf, df, forms=True)

    assert Integrate(out, mesh) == pytest.approx(0)


def test_inner_product_formal_zero_preserves_rank_mismatch_error_for_kforms():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zkf = dg.FormalZeroKForm(1, dim)
    beta = dg.TwoForm(
        CF(
            (
                0,
                -(1 + x),
                2 + y,
                1 + x,
                0,
                -(3 + z),
                -(2 + y),
                3 + z,
                0,
            ),
            dims=(3, 3),
        ),
        dim=dim,
    )

    with pytest.raises(TypeError, match="matching form degrees"):
        rm.InnerProduct(zkf, beta)


def test_inner_product_formal_zero_preserves_rank_mismatch_error_for_doubleforms():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(1, 1, dim)
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    df = dg.DoubleForm(alpha, p=1, q=0, dim=dim)

    with pytest.raises(TypeError, match="matching double-form degrees"):
        rm.InnerProduct(zdf, df, forms=True)


def test_projectdoubleform_dispatches_formal_zero_input():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)

    out = rm.ProjectDoubleForm(zdf, left="n", right="n")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == -1


def test_project_doubleform_formal_keeps_concrete_00_under_noncontracting_projection():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    beta = dg.OneForm(CF((2 + x, 3 + y, 4 + z)))
    df11 = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    df00 = rm.ProjectDoubleForm(df11, left="n", right="m", conormal=rm.edge_conormals[0])
    expected = rm.ProjectDoubleForm(df00, left="E", right="E")

    out = rm.ProjectDoubleForm(df00, left="E", right="E")

    assert isinstance(df00, dg.DoubleForm)
    assert df00.degree_left == 0
    assert df00.degree_right == 0
    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0
    assert isinstance(expected, dg.DoubleForm)
    assert expected.degree_left == 0
    assert expected.degree_right == 0
    assert out.coef.dim == expected.coef.dim == 1


def test_contractslot_dispatches_formal_zero_input():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)

    out = rm.ContractSlot(zdf, rm.normal, slot="left")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 0


def test_contraction_formal_preserves_kform_degree_drop():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zkf = dg.FormalZeroKForm(2, dim)

    out = rm.Contraction(zkf, rm.normal, slot=0)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == 1
    assert out.dim_space == dim


def test_contraction_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zkf = dg.FormalZeroKForm(1, dim)

    out = rm.Contraction(rm.normal, zkf, slot=0)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == 0
    assert out.dim_space == dim


def test_contractionformal_method_preserves_kform_degree_drop():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zkf = dg.FormalZeroKForm(2, dim)

    out = rm.Contraction(zkf, rm.normal, slot=0)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == 1
    assert out.dim_space == dim


def test_d_formal_preserves_kform_degree_propagation():
    dim = 3
    zkf = dg.FormalZeroKForm(-1, dim)

    out = dg.d(zkf)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == 0
    assert out.dim_space == dim


def test_d_dispatches_formal_zero_input():
    dim = 3
    zkf = dg.FormalZeroKForm(1, dim)

    out = dg.d(zkf)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == 2
    assert out.dim_space == dim


def test_delta_formal_preserves_kform_degree_propagation():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zkf = dg.FormalZeroKForm(0, dim)

    out = dg.delta(zkf, rm)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == -1
    assert out.dim_space == dim


def test_delta_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zkf = dg.FormalZeroKForm(2, dim)

    out = rm.delta(zkf)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == 1
    assert out.dim_space == dim


def test_delta_dispatches_concrete_scalar_to_formal_zero():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    f = dg.ScalarField(x + y, dim=dim)

    out = rm.delta(f)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == -1
    assert out.dim_space == dim


def test_d_cov_formal_preserves_degree_propagation():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(-1, 0, dim)

    out_left = rm.d_cov(zdf, slot="left")
    out_right = rm.d_cov(zdf, slot="right")

    assert isinstance(out_left, dg.FormalZeroDoubleForm)
    assert out_left.degree_left == 0
    assert out_left.degree_right == 0
    assert isinstance(out_right, dg.FormalZeroDoubleForm)
    assert out_right.degree_left == -1
    assert out_right.degree_right == 1


def test_delta_cov_formal_preserves_degree_propagation():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(-1, 0, dim)

    out_left = rm.delta_cov(zdf, slot="left")
    out_right = rm.delta_cov(zdf, slot="right")

    assert isinstance(out_left, dg.FormalZeroDoubleForm)
    assert out_left.degree_left == -2
    assert out_left.degree_right == 0
    assert isinstance(out_right, dg.FormalZeroDoubleForm)
    assert out_right.degree_left == -1
    assert out_right.degree_right == -1


def test_d_cov_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(-1, 0, dim)

    out = rm.d_cov(zdf, slot="left")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0


def test_delta_cov_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(-1, 0, dim)

    out = rm.delta_cov(zdf, slot="right")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == -1


def test_delta_cov_dispatches_concrete_00_to_formal_zero():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    df = dg.DoubleForm(x + y + z, p=0, q=0, dim=dim)

    out = rm.delta_cov(df, slot="left")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 0


def test_covdiv_formal_preserves_degree_propagation():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 1, dim)

    out = rm.CovDiv(zdf, slot="right")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_covdiv_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)

    out = rm.CovDiv(zdf, slot="left")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_covdivformal_method_preserves_degree_propagation():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(1, 0, dim)

    out = rm.CovDiv(zdf, slot="left")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_trace_formal_preserves_doubleform_degree_drop():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(1, 2, dim)

    out = rm.Trace(zdf, l=1)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 1
    assert out.dim_space == dim


def test_trace_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)

    out = rm.Trace(zdf)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == -1
    assert out.dim_space == dim


def test_traceformal_method_preserves_doubleform_degree_drop():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(2, 2, dim)

    out = rm.Trace(zdf, l=2)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_trace_sigma_formal_preserves_doubleform_degree_drop():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(1, 1, dim)

    out = rm.TraceSigma(zdf, rm.G)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_tracesigmaformal_method_preserves_doubleform_degree_drop():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 1, dim)

    out = rm.TraceSigma(zdf, rm.G)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_trace_sigma_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(0, 1, dim)

    out = rm.TraceSigma(zdf, rm.G)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_trace_dispatches_concrete_overtrace_to_formal_zero():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=dim)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=dim)
    df = dg.DoubleForm(Einsum("ij,kl->ijkl", alpha, beta), p=2, q=2, dim=dim)

    out = rm.Trace(df, l=3)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == -1


def test_trace_sigma_dispatches_concrete_00_to_formal_zero():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    df = dg.DoubleForm(x + y, p=0, q=0, dim=dim)

    out = rm.TraceSigma(df, rm.G)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == -1


def test_covdiv_dispatches_concrete_00_to_formal_zero():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    df = dg.DoubleForm(x + y, p=0, q=0, dim=dim)

    out = rm.CovDiv(df, slot="left")

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == -1
    assert out.degree_right == 0


def test_contraction_dispatches_concrete_scalar_to_formal_zero():
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))
    f = dg.ScalarField(x + y, dim=dim)

    out = rm.Contraction(f, rm.normal, slot=0)

    assert isinstance(out, dg.FormalZeroKForm)
    assert out.degree == -1
    assert out.dim_space == dim


def test_s_dispatches_concrete_overdegree_to_formal_zero():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))
    vol = dg.Wedge(dx, dg.Wedge(dy, dz))
    df = dg.DoubleForm(Einsum("ijk,lmn->ijklmn", vol, vol), p=3, q=3, dim=dim)

    out = rm.s(df)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 4
    assert out.degree_right == 2
    assert out.dim_space == dim


def test_s_dispatches_formal_zero_input():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))
    zdf = dg.FormalZeroDoubleForm(2, 1, dim)

    out = rm.s(zdf)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 3
    assert out.degree_right == 0
    assert out.dim_space == dim


def test_formal_chain_matches_option_a_degree_recovery():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.35))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))
    vol = dg.Wedge(dx, dg.Wedge(dy, dz))
    phi = dg.DoubleForm(Einsum("ijk,lmn->ijklmn", vol, vol), p=3, q=3, dim=dim)

    tmp = dg.inv_star(phi, rm, vb=VOL, double=True)
    assert isinstance(tmp, dg.DoubleForm)
    assert tmp.degree_left == 0
    assert tmp.degree_right == 0

    tmp = rm.ProjectDoubleForm(tmp, left="n", right="n")
    assert isinstance(tmp, dg.FormalZeroDoubleForm)
    assert tmp.degree_left == -1
    assert tmp.degree_right == -1

    tmp = dg.star(tmp, rm, vb=BND, double=True)
    assert isinstance(tmp, dg.FormalZeroDoubleForm)
    assert tmp.degree_left == 3
    assert tmp.degree_right == 3


def test_formal_zero_adds_cleanly_to_concrete_doubleform():
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    phi = dg.DoubleForm(x + y + z, p=0, q=0, dim=dim)
    alpha = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    beta = dg.OneForm(CF((2 + x, 3 + y, 4 + z)))
    psi = dg.DoubleForm(Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)

    term1 = dg.Wedge(rm.ContractSlot(phi, rm.normal, slot="left"), psi)
    term2 = dg.Wedge(phi, rm.ContractSlot(psi, rm.normal, slot="left"))
    out = term1 + term2

    assert isinstance(term1, dg.FormalZeroDoubleForm)
    assert isinstance(term2, dg.DoubleForm)
    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 1


def test_scalarfield_add_formal_zero_doubleform_keeps_00_doubleform():
    dim = 3
    f = dg.ScalarField(x + y + z, dim=dim)
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)

    out = f + zdf

    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == 0
    assert out.degree_right == 0


def test_typed_kform_add_formal_zero_kform_keeps_type():
    dim = 2
    alpha = dg.OneForm(CF((1 + x, 2 + y)))
    zkf = dg.FormalZeroKForm(1, dim)

    out = alpha + zkf

    assert isinstance(out, dg.OneForm)
    assert out.degree == 1


def test_wedge_formal_zero_accepts_covariant_11_tensorfield():
    dim = 3
    zdf = dg.FormalZeroDoubleForm(0, 0, dim)
    metric_tf = dg.TensorField(CF((1, 0, 0, 0, 1, 0, 0, 0, 1), dims=(3, 3)), "11")

    out = dg.Wedge(zdf, metric_tf)

    assert isinstance(out, dg.FormalZeroDoubleForm)
    assert out.degree_left == 1
    assert out.degree_right == 1
