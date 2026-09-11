"""Regression tests for RiemannianManifold expression-graph contracts."""

import importlib
import pickle

import pytest
from netgen.libngpy._meshing import NgException
from ngsolve import CF, Id, x, y, z

import ngsdiffgeo as dg
import ngsolve
from tests._helpers import assert_l2_close


def test_metric_rejects_test_proxy_hidden_by_native_zero_simplification(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.7)
    _, test = ngsolve.H1(mesh).TnT()
    cpp = importlib.import_module("ngsdiffgeo.ngsdiffgeo")
    zero_matrix = CF((0, 0, 0, 0), dims=(2, 2))
    hidden_test = cpp._ScaleCoefficient(zero_matrix, test)
    metric = cpp._SumCoefficients(Id(2), hidden_test)

    with pytest.raises(Exception, match="test function not allowed"):
        dg.RiemannianManifold(metric)


@pytest.mark.parametrize("operation", ["transpose", "trace", "inner", "S", "J"])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_tensor_algebra_preserves_wrapper_transformations(
    operation, action, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    tensor = dg.TensorField(CF((0, 0, 0, 0), dims=(2, 2)), "11")
    replacement = dg.TensorField(CF((x, 1, 2, y), dims=(2, 2)), "11")
    other = dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), "11")

    operations = {
        "transpose": lambda value: manifold.Transpose(value),
        "trace": lambda value: manifold.Trace(value),
        "inner": lambda value: manifold.InnerProduct(value, other),
        "S": lambda value: manifold.S(value),
        "J": lambda value: manifold.J(value),
    }
    expression = operations[operation](tensor)
    actual = (
        expression.Diff(tensor, replacement)
        if action == "diff"
        else expression.Replace({tensor: replacement})
    )
    expected = operations[operation](replacement)

    assert_l2_close(actual, expected, mesh, tol=1e-12)


@pytest.mark.parametrize("action", ["diff", "replace"])
def test_lower_preserves_wrapper_transformations(action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    vector = dg.VectorField(CF((0, 0)))
    replacement = dg.VectorField(CF((x, y)))

    expression = manifold.Lower(vector)
    actual = (
        expression.Diff(vector, replacement)
        if action == "diff"
        else expression.Replace({vector: replacement})
    )

    assert_l2_close(actual, manifold.Lower(replacement), mesh, tol=1e-12)


@pytest.mark.parametrize("action", ["diff", "replace"])
def test_covariant_derivative_preserves_zero_wrapper_transformations(
    action, make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    vector = dg.VectorField(CF((0, 0)))
    replacement = dg.VectorField(CF((x, y)))

    expression = manifold.CovDerivative(vector)
    actual = (
        expression.Diff(vector, replacement)
        if action == "diff"
        else expression.Replace({vector: replacement})
    )

    assert_l2_close(actual, manifold.CovDerivative(replacement), mesh, tol=2e-6)


@pytest.mark.parametrize("action", ["diff", "replace"])
def test_cross_preserves_zero_wrapper_transformations(action, make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.8)
    manifold = dg.RiemannianManifold(Id(3))
    left = dg.VectorField(CF((0, 0, 0)))
    replacement = dg.VectorField(CF((x, y, z)))
    right = dg.VectorField(CF((1, 2, 3)))

    expression = manifold.Cross(left, right)
    actual = (
        expression.Diff(left, replacement)
        if action == "diff"
        else expression.Replace({left: replacement})
    )

    assert_l2_close(actual, manifold.Cross(replacement, right), mesh, tol=1e-12)


@pytest.mark.parametrize("operation", ["trace_sigma", "slot_inner", "s"])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_double_form_algebra_preserves_wrapper_transformations(
    operation, action, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    form = dg.DoubleForm(CF((0, 0, 0, 0), dims=(2, 2)), p=1, q=1, dim=2)
    replacement = dg.DoubleForm(
        CF((x, 1, 2, y), dims=(2, 2)), p=1, q=1, dim=2
    )
    sigma = dg.DoubleForm(Id(2), p=1, q=1, dim=2)
    operations = {
        "trace_sigma": lambda value: manifold.TraceSigma(value, sigma),
        "slot_inner": lambda value: manifold.SlotInnerProduct(value),
        "s": lambda value: manifold.s(value),
    }

    expression = operations[operation](form)
    actual = (
        expression.Diff(form, replacement)
        if action == "diff"
        else expression.Replace({form: replacement})
    )
    expected = operations[operation](replacement)

    assert_l2_close(actual, expected, mesh, tol=1e-12)


def test_project_tensor_validates_mode_before_scalar_identity_shortcut():
    manifold = dg.RiemannianManifold(Id(2))

    with pytest.raises((ValueError, NgException), match="mode"):
        manifold.ProjectTensor(dg.ScalarField(1, dim=2), 99)


@pytest.mark.parametrize("operation", ["lower", "inner", "trace", "transpose"])
def test_optimized_low_rank_nodes_pickle_and_preserve_replacement(
    operation, make_unit_square_mesh
):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    tensor = dg.TensorField(CF((0, 0, 0, 0), dims=(2, 2)), "11")
    tensor_replacement = dg.TensorField(CF((x, 1, 2, y), dims=(2, 2)), "11")
    vector = dg.VectorField(CF((0, 0)))
    vector_replacement = dg.VectorField(CF((x, y)))
    other = dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), "11")

    if operation == "lower":
        source = vector
        replacement = vector_replacement
        apply = manifold.Lower
    else:
        source = tensor
        replacement = tensor_replacement
        apply = {
            "inner": lambda value: manifold.InnerProduct(value, other),
            "trace": manifold.Trace,
            "transpose": manifold.Transpose,
        }[operation]

    source, expression = pickle.loads(pickle.dumps((source, apply(source))))
    actual = expression.Replace({source: replacement})
    assert_l2_close(actual, apply(replacement), mesh, tol=1e-12)
