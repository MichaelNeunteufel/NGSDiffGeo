import pytest
from netgen.occ import unit_square
from ngsolve import *

import ngsdiffgeo as dg


def l2_error_bnd(a, b, mesh):
    return sqrt(Integrate(InnerProduct(a - b, a - b), mesh, BND))


def l2_norm_bnd(a, mesh):
    return sqrt(Integrate(InnerProduct(a, a), mesh, BND))


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


if __name__ == "__main__":
    pytest.main([__file__])
