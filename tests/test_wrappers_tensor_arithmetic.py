"""Typed tensor-field arithmetic must preserve variance and metric semantics."""

import pytest
from ngsolve import CF

import ngsdiffgeo as dg
from ngsdiffgeo import wrappers


def test_wedge_power_is_part_of_wrapper_public_api():
    assert "WedgePower" in wrappers.__all__


def test_addition_rejects_different_variance():
    vector = dg.VectorField(CF((1, 2)))
    one_form = dg.OneForm(CF((3, 4)))

    with pytest.raises(TypeError, match="variance"):
        _ = vector + one_form
    with pytest.raises(TypeError, match="variance"):
        _ = one_form + vector


def test_rank_two_addition_rejects_different_variance():
    values = CF((1, 2, 3, 4), dims=(2, 2))
    covariant = dg.TensorField(values, "11")
    contravariant = dg.TensorField(values, "00")

    with pytest.raises(TypeError, match="variance"):
        _ = covariant + contravariant


def test_form_plus_generic_covariant_tensor_loses_form_refinement():
    form = dg.TwoForm(CF((0, 1, -1, 0), dims=(2, 2)))
    tensor = dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), "11")

    result = form + tensor

    assert isinstance(result, dg.TensorField)
    assert not isinstance(result, dg.KForm)
    assert result.covariant_indices == "11"


def test_tensor_multiplication_only_accepts_scalars():
    vector = dg.VectorField(CF((1, 2)))
    tensor = dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), "11")

    with pytest.raises(TypeError, match="scalar operands"):
        _ = vector * vector
    with pytest.raises(TypeError, match="scalar operands"):
        _ = tensor * tensor


def test_rank_two_product_contracts_only_opposite_variance_axes(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.8)
    covariant = dg.TensorField(CF((2, 0, 0, 3), dims=(2, 2)), "11")
    contravariant = dg.TensorField(CF((4, 1, 2, 5), dims=(2, 2)), "00")

    product = covariant * contravariant

    assert product.covariant_indices == "10"
    assert tuple(product(mesh(0.2, 0.3))) == pytest.approx((8, 2, 6, 15))


def test_metric_inner_product_is_explicit(make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.8)
    manifold = dg.RiemannianManifold(CF((2, 0, 0, 3), dims=(2, 2)))
    left = dg.VectorField(CF((1, 2)))
    right = dg.VectorField(CF((3, 4)))

    assert manifold.InnerProduct(left, right)(mesh(0.2, 0.3)) == pytest.approx(30)


def test_scalar_arithmetic_preserves_tensor_metadata():
    vector = dg.VectorField(CF((1, 2)))
    tensor = dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), "10")

    assert (2 * vector).covariant_indices == "0"
    assert (tensor / 2).covariant_indices == "10"
