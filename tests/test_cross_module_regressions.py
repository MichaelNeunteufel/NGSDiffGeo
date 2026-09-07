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


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("action", ["diff", "replace"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_zero_wedge_jacobian_retains_mixed_dependencies(vector, action, roundtrip, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    factory = (lambda: dg.OneForm(CF((0, 0)))) if vector else (lambda: dg.ScalarField(CF(0), dim=2))
    a, b = factory(), factory()
    result = dg.Wedge(a, b)
    da, db = (CF((1, 0)), CF((0, 1))) if vector else (CF(1), CF(1))
    expected = CF((0, 1, -1, 0), dims=(2, 2)) if vector else CF(1)
    jacobian = result.Diff(a)
    if roundtrip:
        b, jacobian = pickle.loads(pickle.dumps((b, jacobian)))
    actual = jacobian.Diff(b, db) if action == "diff" else jacobian.Replace({b: db})
    actual = (actual.Reshape((result.dim, a.dim)) * da.Reshape((a.dim,))).Reshape(tuple(result.dims))
    assert_l2_close(actual, expected, mesh, tol=1e-12)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_zero_hessian_retains_input(nested, action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    f = dg.ScalarField(CF(0), dim=2)
    result = dg.GradCF(dg.GradCF(f, 2), 2) if nested else dg.HesseCF(f, 2)
    h = x*x+y*y
    actual = result.Diff(f, h) if action == "diff" else result.Replace({f: h})
    assert_l2_close(actual, 2*Id(2), mesh, tol=2e-6)


@pytest.mark.parametrize("exterior", [False, True])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_proxy_form_gradient_retains_wrapper(exterior, action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    u = H1(mesh, order=3).TrialFunction()
    f = dg.ScalarField(u, dim=2)
    result = dg.d(f) if exterior else dg.GradCF(f, 2)
    actual = result.Diff(f, x*x) if action == "diff" else result.Replace({f: x*x})
    assert_l2_close(actual, CF((2*x, 0)), mesh, tol=1e-8)


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
@pytest.mark.parametrize("operation", ["add", "subtract", "repeat", "reverse_subtract"])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_zero_additive_form_operands(double, operation, action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    wrap = (lambda cf: dg.DoubleForm(cf, p=1, q=0, dim=2)) if double else dg.OneForm
    a, b = wrap(CF((0, 0))), wrap(CF((3, 4)))
    direction = CF((1, 2))
    if operation == "reverse_subtract":
        a = dg.DoubleForm(CF(0), p=0, q=0, dim=2) if double else dg.ScalarField(CF(0), dim=2)
        b = dg.ScalarField(CF(3), dim=2)
        direction = CF(1)
    result = {"add": lambda: a+b, "subtract": lambda: a-b, "repeat": lambda: a+a,
              "reverse_subtract": lambda: 3-a}[operation]()
    expected = -direction if operation == "reverse_subtract" else (2*direction if operation == "repeat" else direction)
    if action == "replace" and operation != "repeat":
        expected = expected + (-b.coef if operation == "subtract" else b.coef)
    actual = result.Diff(a, direction) if action == "diff" else result.Replace({a: direction})
    assert_l2_close(actual, expected, mesh, tol=1e-12)


@pytest.mark.parametrize("action", ["diff", "replace"])
def test_sym_zero_operand(action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    a = dg.DoubleForm(CF((0,)*4, dims=(2, 2)), p=1, q=1, dim=2)
    direction = CF((1, 2, 3, 4), dims=(2, 2))
    result = dg.Sym(a)
    actual = result.Diff(a, direction) if action == "diff" else result.Replace({a: direction})
    assert_l2_close(actual, CF((1, 2.5, 2.5, 4), dims=(2, 2)), mesh, tol=1e-12)


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


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("slot,p,q", [("left", 0, 0), ("right", 0, 0), ("left", 0, 1), ("right", 1, 0)])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_point_boundary_hodge_preserves_identity(inverse, slot, p, q, action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    a = dg.DoubleForm(CF(3) if p+q == 0 else CF((3, 4)), p=p, q=q, dim=2)
    direction = CF(7) if p+q == 0 else CF((7, 8))
    result = (dg.inv_star if inverse else dg.star)(a, dg.RiemannianManifold(Id(2)), vb=BBND, slot=slot)
    actual = result.Diff(a, direction) if action == "diff" else result.Replace({a: direction})
    assert_l2_close(actual, direction, mesh, tol=1e-12)


def test_tensor_adapter_honors_explicit_variance():
    tensor = dg.TensorField(Id(2), covariant_indices="11")
    assert wrappers.as_tensorfield(tensor) is tensor
    assert wrappers.as_tensorfield(tensor, covariant_indices="11") is tensor
    assert wrappers.as_tensorfield(tensor, covariant_indices="00").covariant_indices == "00"
    with pytest.raises((ValueError, NgException)):
        wrappers.as_tensorfield(tensor, covariant_indices="nonsense")


def test_tensor_complex_left_multiplication(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    tensor = dg.TensorField(Id(2), covariant_indices="11")
    assert abs(Integrate(Norm(1j*tensor-tensor*1j), mesh)) < 1e-12


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
