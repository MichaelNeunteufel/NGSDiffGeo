import pytest
from netgen.occ import unit_square
from ngsolve import *

import ngsdiffgeo as dg


def l2_error(a, b, mesh):
    return sqrt(Integrate(InnerProduct(a - b, a - b), mesh))


def l2_norm(a, mesh):
    return sqrt(Integrate(InnerProduct(a, a), mesh))


def l2_norm_bnd(a, mesh):
    return sqrt(Integrate(InnerProduct(a, a) * dx(element_boundary=True), mesh))


def test_double_form_covariant_derivatives_constant_zero():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((1, 0)))
    beta = dg.OneForm(CF((0, 1)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    d1 = rm.d_cov(A, slot="left")
    d2 = rm.d_cov(A, slot="right")
    delta1 = rm.delta_cov(A, slot="left")
    delta2 = rm.delta_cov(A, slot="right")

    assert d1.degree_left == 2
    assert d1.degree_right == 1
    assert d2.degree_left == 1
    assert d2.degree_right == 2
    assert delta1.degree_left == 0
    assert delta1.degree_right == 1
    assert delta2.degree_left == 1
    assert delta2.degree_right == 0

    assert l2_norm(d1, mesh) < 1e-12
    assert l2_norm(d2, mesh) < 1e-12
    assert l2_norm(delta1, mesh) < 1e-12
    assert l2_norm(delta2, mesh) < 1e-12


def test_double_form_covariant_d1_matches_left_exterior_derivative():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((0, x)))
    beta = dg.OneForm(CF((1, 0)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    d1 = rm.d_cov(A, slot="left")
    expected = dg.DoubleForm(dg.TensorProduct(dg.d(alpha), beta), p=2, q=1, dim=dim)
    assert l2_error(d1, expected, mesh) < 1e-8


def test_double_form_covariant_d2_matches_right_exterior_derivative():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((1, 0)))
    beta = dg.OneForm(CF((0, x)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    d2 = rm.d_cov(A, slot="right")
    expected = dg.DoubleForm(dg.TensorProduct(alpha, dg.d(beta)), p=1, q=2, dim=dim)
    assert l2_error(d2, expected, mesh) < 1e-8


def test_double_form_covariant_codifferentials_divergence():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((x, 0)))
    beta = dg.OneForm(CF((0, y)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    delta1 = rm.delta_cov(A, slot="left")
    delta2 = rm.delta_cov(A, slot="right")

    expected_delta1 = dg.DoubleForm((-1.0) * beta, p=0, q=1, dim=dim)
    expected_delta2 = dg.DoubleForm((-1.0) * alpha, p=1, q=0, dim=dim)

    assert l2_error(delta1, expected_delta1, mesh) < 1e-8
    assert l2_error(delta2, expected_delta2, mesh) < 1e-8


def test_double_form_covariant_derivatives_boundary_constant_zero():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    alpha = dg.OneForm(CF((1, 0)))
    beta = dg.OneForm(CF((0, 1)))
    A = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=dim)

    d1 = rm.d_cov(A, slot="left", vb=BND)
    d2 = rm.d_cov(A, slot="right", vb=BND)
    delta1 = rm.delta_cov(A, slot="left", vb=BND)
    delta2 = rm.delta_cov(A, slot="right", vb=BND)

    assert l2_norm_bnd(d1, mesh) < 1e-12
    assert l2_norm_bnd(d2, mesh) < 1e-12
    assert l2_norm_bnd(delta1, mesh) < 1e-12
    assert l2_norm_bnd(delta2, mesh) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
