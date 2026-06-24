import pytest

from ngsolve import (
    BND,
    BilinearForm,
    Grad,
    GridFunction,
    H1,
    InnerProduct,
    Mesh,
    Norm,
    SymbolicBFI,
    Variation,
    dx,
    x,
    y,
)
from netgen.occ import unit_square

import ngsdiffgeo as ngd


def _mesh_and_space():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=2, dirichlet="bottom|right|top|left")
    return mesh, fes


def _assembled_norm(fes, cf, vb=None):
    a = BilinearForm(fes, symmetric=False)
    if vb is None:
        a += SymbolicBFI(cf, simd_evaluate=False, bonus_intorder=3)
    else:
        a += SymbolicBFI(cf, vb, simd_evaluate=False, bonus_intorder=3)
    a.Assemble()
    return Norm(a.mat.AsVector())


def _assert_zero_form(fes, cf, tol=1e-8, vb=None):
    assert _assembled_norm(fes, cf, vb=vb) < tol


def test_gradcf_symbolic_proxy_linear_and_product_rules():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    grad_linear = ngd.GradCF(2 * u + 1, 2)
    _assert_zero_form(fes, (grad_linear[0] - 2 * Grad(u)[0]) * v)
    _assert_zero_form(fes, (grad_linear[1] - 2 * Grad(u)[1]) * v)

    grad_square = ngd.GradCF(u * u, 2)
    ref_square = 2 * u * ngd.GradCF(u, 2)
    _assert_zero_form(fes, (grad_square[0] - ref_square[0]) * v)
    _assert_zero_form(fes, (grad_square[1] - ref_square[1]) * v)


def test_gradcf_symbolic_gridfunction_times_proxy_product_rule():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    g = GridFunction(fes)
    g.Set(x * x + y)

    grad_expr = ngd.GradCF(g * u, 2)
    grad_g = ngd.GradCF(g, 2)
    grad_u = ngd.GradCF(u, 2)

    for i in range(2):
        ref = grad_g[i] * u + g * grad_u[i]
        _assert_zero_form(fes, (grad_expr[i] - ref) * v)


def test_gradcf_symbolic_energy_variation():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    energy = InnerProduct(ngd.GradCF(u, 2), ngd.GradCF(u, 2))
    variation = energy.Diff(u, v)
    ref = 2 * InnerProduct(ngd.GradCF(u, 2), ngd.GradCF(v, 2))

    _assert_zero_form(fes, variation - ref)


def test_gradcf_variation_assemble_linearization_matches_reference_bilinear_form():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    energy = InnerProduct(ngd.GradCF(u, 2), ngd.GradCF(u, 2))
    variation_form = BilinearForm(fes, symmetric=False)
    variation_form += Variation(energy * dx)

    gfu = GridFunction(fes)
    gfu.Set(x * y + 0.2 * x)
    variation_form.AssembleLinearization(gfu.vec)

    reference = BilinearForm(fes, symmetric=False)
    reference += SymbolicBFI(
        2 * InnerProduct(ngd.GradCF(u, 2), ngd.GradCF(v, 2)),
        simd_evaluate=False,
        bonus_intorder=3,
    )
    reference.Assemble()

    diff = variation_form.mat.CreateMatrix()
    diff.AsVector().data = variation_form.mat.AsVector() - reference.mat.AsVector()
    assert Norm(diff.AsVector()) < 1e-8


def test_gradcf_variation_of_nonlinear_squared_gradient_expression():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    energy = InnerProduct(ngd.GradCF(u**2, 2), ngd.GradCF(u**2, 2))
    variation_form = BilinearForm(fes, symmetric=False)
    variation_form += Variation(energy * dx(bonus_intorder=4))

    gfu = GridFunction(fes)
    gfu.Set(x * y + 0.3 * x + 0.2)
    variation_form.AssembleLinearization(gfu.vec)

    grad_u = ngd.GradCF(u, 2)
    grad_v = ngd.GradCF(v, 2)
    grad_u2 = ngd.GradCF(u**2, 2)
    first_variation = 2 * InnerProduct(
        grad_u2,
        2 * v * grad_u + 2 * u * grad_v,
    )

    reference = BilinearForm(fes, symmetric=False)
    reference += SymbolicBFI(first_variation, simd_evaluate=False, bonus_intorder=4)
    reference.AssembleLinearization(gfu.vec)

    diff = variation_form.mat.CreateMatrix()
    diff.AsVector().data = variation_form.mat.AsVector() - reference.mat.AsVector()
    assert Norm(diff.AsVector()) < 1e-8


def test_gradcf_rejects_trial_and_test_inside_same_gradcf():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    with pytest.raises(Exception, match="trial and test"):
        ngd.GradCF(u * v, 2)


def test_hessecf_symbolic_h1_smoke():
    _, fes = _mesh_and_space()
    u, _ = fes.TnT()

    hesse = ngd.HesseCF(u, 2)
    assert hesse.dim == 4
    assert tuple(hesse.dims) == (2, 2)


def test_hessecf_pure_coefficient_function_matches_polynomial_hessian():
    mesh, _ = _mesh_and_space()
    cf = x**3 * y**2 - x * y

    hesse = ngd.HesseCF(cf, 2)
    values = hesse(mesh(0.2, 0.3))

    assert hesse.dim == 4
    assert tuple(hesse.dims) == (2, 2)
    assert values[0] == pytest.approx(6 * 0.2 * 0.3**2, abs=1e-6)
    assert values[1] == pytest.approx(6 * 0.2**2 * 0.3 - 1, abs=1e-6)
    assert values[2] == pytest.approx(6 * 0.2**2 * 0.3 - 1, abs=1e-6)
    assert values[3] == pytest.approx(2 * 0.2**3, abs=1e-6)


def test_hessecf_pure_gridfunction_smoke():
    mesh, fes = _mesh_and_space()
    g = GridFunction(fes)
    g.Set(x**2 + y)

    hesse = ngd.HesseCF(g, 2)
    values = hesse(mesh(0.2, 0.3))

    assert hesse.dim == 4
    assert tuple(hesse.dims) == (2, 2)
    assert len(values) == 4


def test_gradcf_and_hessecf_boundary_operators_smoke():
    _, fes = _mesh_and_space()
    u, v = fes.TnT()

    grad_boundary = ngd.GradCF(u, 2, True)
    ref_grad_boundary = u.Operator("Gradboundary")
    _assert_zero_form(fes, (grad_boundary[0] - ref_grad_boundary[0]) * v, vb=BND)
    _assert_zero_form(fes, (grad_boundary[1] - ref_grad_boundary[1]) * v, vb=BND)

    hesse_boundary = ngd.HesseCF(u, 2, True)
    assert hesse_boundary.dim == 4
    assert tuple(hesse_boundary.dims) == (2, 2)
    _assembled_norm(fes, hesse_boundary[0, 0] * v, vb=BND)


def test_gradcf_tensor_valued_h1_proxy_prepends_derivative_index():
    mesh, _ = _mesh_and_space()
    fes = H1(mesh, order=2, dim=4)
    u, _ = fes.TnT()

    grad_u = ngd.GradCF(u, 2)
    raw_grad_u = u.Operator("Grad")

    assert grad_u.dim == 8
    assert tuple(grad_u.dims) == (2, 4)
    assert tuple(raw_grad_u.dims) == (4, 2)


def test_hessecf_product_without_symbolic_hesse_operator_fails_clearly():
    _, fes = _mesh_and_space()
    u, _ = fes.TnT()

    with pytest.raises(Exception, match='Operator\\("hesse"\\)'):
        ngd.HesseCF(u * u, 2)
