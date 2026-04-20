import pytest
from netgen.csg import unit_cube
from netgen.occ import unit_square
from ngsolve import BBND, BND, CF, Id, InnerProduct, Integrate, Mesh, dx, sqrt, x, y, z

import ngsdiffgeo as dg
from tests._helpers import l2_error, l2_error_bbnd, l2_error_bnd, l2_norm_bnd


def two_form_3d(a12, a13, a23):
    return dg.TwoForm(
        CF((0, a12, a13, -a12, 0, a23, -a13, -a23, 0), dims=(3, 3)),
        dim=3,
    )


def test_double_form_tangent_projection_kills_normal_left():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x + 1, y + 2)))
    beta = dg.OneForm(CF((2 * x, 3 * y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(A, left="F", right="none")
    normal_left = rm.Contraction(proj, rm.normal, slot=0)
    assert l2_norm_bnd(normal_left, mesh) < 1e-8


def test_double_form_normal_projection_preserves_normal_left():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x + 1, y + 2)))
    beta = dg.OneForm(CF((2 * x, 3 * y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(A, left="n", right="none")
    assert proj.degree_left == 0
    expected = dg.DoubleForm(rm.Contraction(A, rm.normal, slot=0), p=0, q=1, dim=dim)
    assert l2_error_bnd(proj, expected, mesh) < 1e-8


def test_double_form_tangent_projection_kills_normal_right():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x + 1, y + 2)))
    beta = dg.OneForm(CF((2 * x, 3 * y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(A, left="none", right="F")
    normal_right = rm.Contraction(proj, rm.normal, slot=1)
    assert l2_norm_bnd(normal_right, mesh) < 1e-8


def test_double_form_normal_projection_preserves_normal_right():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x + 1, y + 2)))
    beta = dg.OneForm(CF((2 * x, 3 * y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(A, left="none", right="n")
    assert proj.degree_right == 0
    expected = dg.DoubleForm(rm.Contraction(A, rm.normal, slot=1), p=1, q=0, dim=dim)
    assert l2_error_bnd(proj, expected, mesh) < 1e-8


def test_double_form_contract_slot_left_matches_tensor_contraction():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x + 1, y + 2)))
    beta = dg.OneForm(CF((2 * x, 3 * y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    contracted = rm.ContractSlot(A, rm.normal, slot="left")
    assert contracted.degree_left == 0
    expected = dg.DoubleForm(rm.Contraction(A, rm.normal, slot=0), p=0, q=1, dim=dim)
    assert l2_error_bnd(contracted, expected, mesh) < 1e-8


def test_double_form_contract_slot_right_matches_tensor_contraction():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x + 1, y + 2)))
    beta = dg.OneForm(CF((2 * x, 3 * y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    contracted = rm.ContractSlot(A, rm.normal, slot="right")
    assert contracted.degree_right == 0
    expected = dg.DoubleForm(rm.Contraction(A, rm.normal, slot=1), p=1, q=0, dim=dim)
    assert l2_error_bnd(contracted, expected, mesh) < 1e-8


def test_project_doubleform_scalar_with_normal_returns_zero():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    v = dg.ScalarField(x + y, dim=dim)
    proj = rm.ProjectDoubleForm(v, left="n", right="F")
    assert l2_norm_bnd(proj, mesh) < 1e-10


def test_project_doubleform_00_with_normal_returns_zero():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    v00 = dg.DoubleForm(x + y, p=0, q=0, dim=dim)
    proj = rm.ProjectDoubleForm(v00, left="n", right="F")
    assert proj.degree_left == 0
    assert proj.degree_right == 0
    assert l2_norm_bnd(proj, mesh) < 1e-10


def test_double_form_normal_projection_is_pure_contraction_left():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    alpha = two_form_3d(x + y, y + z, z + x)
    beta = dg.OneForm(CF((1 + x, 2 + y, 3 + z)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=2, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(A, left="n", right="none")
    expected = dg.DoubleForm(rm.Contraction(A, rm.normal, slot=0), p=1, q=1, dim=dim)

    assert proj.degree_left == 1
    assert l2_error_bnd(proj, expected, mesh) < 1e-8


def test_double_form_conormal_projection_is_pure_contraction_left():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    alpha = two_form_3d(x + 2 * y, y + 3 * z, z + 4 * x)
    beta = dg.OneForm(CF((1 + 2 * x, 2 + 3 * y, 3 + 4 * z)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=2, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(
        A,
        left="m",
        right="none",
        conormal=rm.edge_conormals[0],
        project_remaining=False,
    )
    expected = dg.DoubleForm(
        rm.Contraction(A, rm.edge_conormals[0], slot=0), p=1, q=1, dim=dim
    )

    assert proj.degree_left == 1
    assert l2_error_bbnd(proj, expected, mesh) < 1e-8


def test_double_form_conormal_projection_projects_remaining_indices_by_default():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    alpha = two_form_3d(x + 2 * y, y + 3 * z, z + 4 * x)
    beta = dg.OneForm(CF((1 + 2 * x, 2 + 3 * y, 3 + 4 * z)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=2, q=1, dim=dim)

    proj = rm.ProjectDoubleForm(
        A, left="m", right="none", conormal=rm.edge_conormals[0]
    )
    contracted = dg.DoubleForm(
        rm.Contraction(A, rm.edge_conormals[0], slot=0), p=1, q=1, dim=dim
    )
    expected = rm.ProjectDoubleForm(contracted, left="E", right="none")

    assert proj.degree_left == 1
    assert l2_error_bbnd(proj, expected, mesh) < 1e-8


def test_inner_product_forms_accepts_traced_zero_zero_doubleform():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
    beta = dg.OneForm(CF((0.3 * z * y, x * z**2, y**2)))
    phi = dg.DoubleForm(dg.Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=dim)
    psi = dg.DoubleForm(1.0 + x - 0.2 * y, p=0, q=0, dim=dim)

    traced = rm.Trace(rm.star(phi), l=dim - phi.degree_left - psi.degree_left)

    out = rm.InnerProduct(traced, psi, forms=True)
    expected = rm.InnerProduct(
        dg.DoubleForm(traced, p=0, q=0, dim=dim),
        psi,
        forms=True,
    )

    assert l2_error(out, expected, mesh) == pytest.approx(0)


if __name__ == "__main__":
    pytest.main([__file__])
