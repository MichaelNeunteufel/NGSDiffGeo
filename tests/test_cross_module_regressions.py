"""Cross-module regressions: semantic derivatives, proxy dispatch and adapters."""

import pickle

import pytest
from ngsolve import (
    BBND, BilinearForm, CF, Grad, H1, Id, InnerProduct, Integrate,
    Norm, SymbolicBFI, x, y,
)

from netgen.libngpy._meshing import NgException

import ngsdiffgeo as dg
from ngsdiffgeo import wrappers
from tests._helpers import assert_l2_close, l2_error_bnd


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_zero_hessian_retains_input(nested, action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    f = dg.ScalarField(CF(0), dim=2)
    result = dg.GradCF(dg.GradCF(f, 2), 2) if nested else dg.HesseCF(f, 2)
    h = x*x+y*y
    actual = result.Diff(f, h) if action == "diff" else result.Replace({f: h})
    assert_l2_close(actual, 2*Id(2), mesh, tol=2e-6)


@pytest.mark.parametrize("operation", ["nested", "component", "covariant"])
@pytest.mark.parametrize("simd", [False, True])
@pytest.mark.parametrize("custom_gradient", [False, True])
def test_proxy_hessian_matches_native(operation, simd, custom_gradient, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    fes = H1(mesh, order=3)
    u, v = fes.TnT()
    native = u.Operator("hesse")
    gradient = dg.GradCF(dg.ScalarField(u, dim=2), 2) if custom_gradient else Grad(u)
    if operation == "covariant":
        actual = dg.RiemannianManifold(Id(2)).CovHesse(dg.ScalarField(u, dim=2))
    elif operation == "component":
        actual = dg.GradCF(gradient[0], 2)
        native = CF((native[0, 0], native[1, 0]))
    else:
        actual = dg.GradCF(gradient, 2)
    error = BilinearForm(fes)
    error += SymbolicBFI(InnerProduct(actual-native, native.Diff(u, v)), simd_evaluate=simd)
    error.Assemble()
    assert Norm(error.mat.AsVector()) < 1e-9


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("wrapper_target", [False, True])
def test_projection_replace_rebuilds_operand(double, wrapper_target, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    a = dg.DoubleForm(CF((1+x, 2, 3, 4+y), dims=(2, 2)), p=1, q=1, dim=2)
    project = (lambda t: manifold.ProjectDoubleForm(t, left="F")) if double else (lambda t: manifold.ProjectTensor(t, "F"))
    replacement = dg.DoubleForm(2*a.coef, p=1, q=1, dim=2)
    result = project(a).Replace({a if wrapper_target else a.coef: replacement})
    assert l2_error_bnd(result, project(replacement), mesh) < 1e-12


@pytest.mark.parametrize("hessian", [False, True])
def test_proxy_derivative_replacement_can_be_archived(hessian, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    f = dg.ScalarField(H1(mesh, order=3).TrialFunction(), dim=2)
    expression = dg.HesseCF(f, 2) if hessian else dg.GradCF(f, 2)
    direction = x*x+y*y
    expected = 2*Id(2) if hessian else CF((2*x, 2*y))
    # Native ProxyFunction is not registered for NGSolve archives. Replacing
    # the semantic input must remove every proxy, including the old evaluator.
    expression = expression.Replace({f: direction})
    restored = pickle.loads(pickle.dumps(expression))
    assert_l2_close(restored, expected, mesh, tol=2e-6)


@pytest.mark.parametrize("hessian", [False, True])
@pytest.mark.parametrize("realcompile", [False, True])
def test_compiled_proxy_derivative_matches_native(hessian, realcompile, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    fes = H1(mesh, order=3)
    u, v = fes.TnT()
    f = dg.ScalarField(u, dim=2)
    expression = dg.HesseCF(f, 2) if hessian else dg.GradCF(f, 2)
    native = u.Operator("hesse") if hessian else Grad(u)
    error = BilinearForm(fes)
    error += SymbolicBFI(InnerProduct(expression.Compile(realcompile=realcompile, wait=True)-native,
                                     native.Diff(u, v)))
    error.Assemble()
    assert Norm(error.mat.AsVector()) < 1e-9
