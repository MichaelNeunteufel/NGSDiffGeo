import pytest

import ngsdiffgeo as dg
import ngsdiffgeo.wrappers as dg_wrappers
from netgen.occ import unit_cube, unit_square
from ngsolve import BBND, BND, CF, Id, Mesh, x, y, z
from ngsolve.fem import Einsum

from tests._helpers import l2_error, l2_norm


def test_as_kform_preserves_wrapped_identity():
    alpha = dg.OneForm(CF((x, y)))
    out = dg_wrappers.as_kform(alpha, k=1, dim=2)
    assert out is alpha


def test_as_kform_requires_inferable_dim():
    with pytest.raises(TypeError, match="dim must be provided or inferable"):
        dg_wrappers.as_kform(1, k=2)


def test_as_kform_high_degree_invalid_rank_raises():
    cf = CF((1, 2, 3, 4), dims=(2, 2))
    with pytest.raises(Exception, match="degree k requires rank-k coefficient"):
        dg_wrappers.as_kform(cf, k=4, dim=2)


def test_as_doubleform_infers_degrees_from_doubleform():
    df = dg.DoubleForm(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), p=1, q=1, dim=2)
    out = dg_wrappers.as_doubleform(df)
    assert out is df


def test_as_doubleform_requires_inferable_metadata():
    with pytest.raises(TypeError, match="p and q must be provided or inferable"):
        dg_wrappers.as_doubleform(CF((x, y)))


def test_as_doubleform_requires_inferable_dim():
    with pytest.raises(TypeError, match="dim must be provided or inferable"):
        dg_wrappers.as_doubleform(1, p=0, q=0)


def test_as_tensorfield_dispatches_scalar_vector_and_oneform():
    scalar = dg_wrappers.as_tensorfield(x + y, covariant_indices="", dim=2)
    vector = dg_wrappers.as_tensorfield(CF((x, y)), covariant_indices="0")
    oneform = dg_wrappers.as_tensorfield(CF((x, y)), covariant_indices="1")

    assert isinstance(scalar, dg.ScalarField)
    assert isinstance(vector, dg.VectorField)
    assert isinstance(oneform, dg.OneForm)


def test_as_tensorfield_scalar_requires_dim_if_not_inferable():
    with pytest.raises(TypeError, match="dim must be provided or inferable"):
        dg_wrappers.as_tensorfield(1, covariant_indices="", dim=None)


def test_wedge_rejects_non_covariant_tensorfield_as_doubleform():
    dim = 2
    tensor_00 = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), "00")
    df = dg.DoubleForm(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), p=1, q=1, dim=dim)

    with pytest.raises(TypeError, match="expected DoubleForm or covariant"):
        dg.Wedge(tensor_00, df)


def test_wedgepower_rejects_non_integer_power():
    df = dg.DoubleForm(CF((1, 0, 0, 1), dims=(2, 2)), p=1, q=1, dim=2)
    with pytest.raises(TypeError, match="non-negative integer"):
        dg.WedgePower(df, 1.5)


def test_wedgepower_rejects_negative_power():
    df = dg.DoubleForm(CF((1, 0, 0, 1), dims=(2, 2)), p=1, q=1, dim=2)
    with pytest.raises(ValueError, match="non-negative"):
        dg.WedgePower(df, -1)


def test_wedgepower_l0_requires_inferable_dim():
    tensor_11 = dg.TensorField(CF((1, 0, 0, 1), dims=(2, 2)), "11")
    out = dg.WedgePower(tensor_11, 0)
    assert isinstance(out, dg.ScalarField)
    assert out.dim_space == 2


def test_wedgepower_requires_11_doubleform():
    df = dg.DoubleForm(CF((x, y), dims=(2,)), p=0, q=1, dim=2)
    with pytest.raises(ValueError, match=r"\(1,1\)"):
        dg.WedgePower(df, 2)


def test_slot_inner_product_forms_false_matches_method(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    rm = rm_euclidean_2d

    alpha = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=2)
    beta = dg.TwoForm(CF((0, 1 + y, -(1 + y), 0), dims=(2, 2)), dim=2)
    df = dg.DoubleForm(Einsum("ij,kl->ijkl", alpha, beta), p=2, q=2, dim=2)

    function_out = dg.slot_inner_product(df, rm, forms=False)
    method_out = rm.SlotInnerProduct(df, forms=False)

    assert l2_error(function_out, method_out, mesh) == pytest.approx(0)


def test_projectdoubleform_rejects_invalid_string_mode(rm_euclidean_2d):
    df = dg.DoubleForm(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), p=1, q=1, dim=2)
    with pytest.raises(ValueError, match="mode must be"):
        rm_euclidean_2d.ProjectDoubleForm(df, left="bad")


def test_projectdoubleform_rejects_invalid_integer_mode(rm_euclidean_2d):
    df = dg.DoubleForm(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), p=1, q=1, dim=2)
    with pytest.raises(ValueError, match="mode must be"):
        rm_euclidean_2d.ProjectDoubleForm(df, left=9)


def test_projectdoubleform_rejects_invalid_mode_type(rm_euclidean_2d):
    df = dg.DoubleForm(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), p=1, q=1, dim=2)
    with pytest.raises(ValueError, match="mode must be string or int"):
        rm_euclidean_2d.ProjectDoubleForm(df, left=object())


def test_projectdoubleform_rejects_wrong_tensor_type(rm_euclidean_2d):
    with pytest.raises(TypeError, match="expects a DoubleForm"):
        rm_euclidean_2d.ProjectDoubleForm(dg.VectorField(CF((x, y))))


def test_projecttensor_normal_projection_drops_form_degree(rm_euclidean_2d):
    oneform = dg.OneForm(CF((x, y)))
    projected = rm_euclidean_2d.ProjectTensor(oneform, "n")
    assert isinstance(projected, dg.ScalarField)
    assert projected.degree == 0


def test_projecttensor_normal_projection_drops_doubleform_left_degree(rm_euclidean_2d):
    df = dg.DoubleForm(dg.OneForm(CF((x, y))), p=1, q=0, dim=2)
    projected = rm_euclidean_2d.ProjectTensor(df, "n")
    assert isinstance(projected, dg.DoubleForm)
    assert projected.degree_left == 0
    assert projected.degree_right == 0


def test_projecttensor_normal_projection_drops_doubleform_right_degree(rm_euclidean_2d):
    df = dg.DoubleForm(dg.OneForm(CF((x, y))), p=0, q=1, dim=2)
    projected = rm_euclidean_2d.ProjectTensor(df, "n")
    assert isinstance(projected, dg.DoubleForm)
    assert projected.degree_left == 0
    assert projected.degree_right == 0


def test_contraction_argument_order_is_symmetric_for_tensor_and_vector(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    tensor = dg.TensorField(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), "11")
    vector = dg.VectorField(CF((1 + x, 2 + y)))

    out1 = rm_euclidean_2d.Contraction(tensor, vector, 0)
    out2 = rm_euclidean_2d.Contraction(vector, tensor, 0)

    assert l2_error(out1, out2, mesh) == pytest.approx(0)


def test_contraction_rejects_two_vectors(rm_euclidean_2d):
    v1 = dg.VectorField(CF((x, y)))
    v2 = dg.VectorField(CF((1 + x, 1 + y)))
    with pytest.raises(TypeError, match="exactly one vector field"):
        rm_euclidean_2d.Contraction(v1, v2)


def test_contraction_rejects_two_non_vectors(rm_euclidean_2d):
    a = dg.OneForm(CF((x, y)))
    b = dg.OneForm(CF((1 + x, 1 + y)))
    with pytest.raises(TypeError, match="exactly one vector field"):
        rm_euclidean_2d.Contraction(a, b)


def test_contraction_preserves_kform_typing(make_unit_square_mesh, rm_euclidean_2d):
    mesh = make_unit_square_mesh(maxh=0.3)
    twoform = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=2)
    vector = dg.VectorField(CF((1 + x, 2 + y)))

    out = rm_euclidean_2d.Contraction(twoform, vector, 0)

    assert isinstance(out, dg.KForm)
    assert out.degree == 1
    assert l2_norm(out, mesh) > 0


def test_manifold_property_wrappers_have_expected_types(rm_euclidean_2d, rm_euclidean_3d):
    assert isinstance(rm_euclidean_2d.Curvature, dg.ScalarField)
    assert isinstance(rm_euclidean_2d.Gauss, dg.ScalarField)
    assert isinstance(rm_euclidean_2d.SFF, dg.DoubleForm)
    assert isinstance(rm_euclidean_2d.GeodesicCurvature, dg.ScalarField)
    assert isinstance(rm_euclidean_3d.MeanCurvature, dg.ScalarField)
    assert isinstance(rm_euclidean_3d.Curvature, dg.TensorField)


def test_doubleform_matrix_multiplication_requires_explicit_coef(rm_euclidean_2d):
    projector = Id(2)

    with pytest.raises(TypeError, match=r"DoubleForm '\*' only supports scalar operands"):
        projector * rm_euclidean_2d.SFF

    with pytest.raises(TypeError, match=r"DoubleForm '\*' only supports scalar operands"):
        rm_euclidean_2d.SFF * projector


def test_doubleform_matrix_division_requires_scalar_operand(make_unit_square_mesh):
    df = dg.DoubleForm(CF((1 + x, x - y, x - y, 2 + y), dims=(2, 2)), p=1, q=1, dim=2)
    mat = CF((2 + x, y, x * y, 3 - y), dims=(2, 2))

    with pytest.raises(TypeError, match=r"DoubleForm '/' only supports scalar operands"):
        df / mat


def test_edge_normal_and_conormal_wrappers_have_expected_types(rm_euclidean_3d):
    edge_normals = rm_euclidean_3d.edge_normals
    edge_conormals = rm_euclidean_3d.edge_conormals

    assert len(edge_normals) == 2
    assert len(edge_conormals) == 2
    assert all(isinstance(v, dg.VectorField) for v in edge_normals)
    assert all(isinstance(v, dg.VectorField) for v in edge_conormals)


def test_s_returns_tensorfield_for_tensor_input(rm_euclidean_2d):
    tensor = dg.TensorField(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), "11")
    out = rm_euclidean_2d.S(tensor)
    assert isinstance(out, dg.TensorField)


def test_j_returns_tensorfield_for_tensor_input(rm_euclidean_2d):
    tensor = dg.TensorField(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), "11")
    out = rm_euclidean_2d.J(tensor)
    assert isinstance(out, dg.TensorField)


def test_s_lowercase_always_returns_doubleform(rm_euclidean_2d):
    df = dg.DoubleForm(CF((x, y, 1 + x, 1 + y), dims=(2, 2)), p=1, q=1, dim=2)
    out = rm_euclidean_2d.s(df)
    assert isinstance(out, dg.DoubleForm)
    assert out.degree_left == 2
    assert out.degree_right == 0


def test_star_double_promotion_for_oneform_raises(rm_euclidean_2d):
    alpha = dg.OneForm(CF((x, y)))
    with pytest.raises(Exception):
        dg.star(alpha, rm_euclidean_2d, double=True, slot="left")


def test_inv_star_double_promotion_for_oneform_raises(rm_euclidean_2d):
    alpha = dg.OneForm(CF((x, y)))
    with pytest.raises(Exception):
        dg.inv_star(alpha, rm_euclidean_2d, double=True, slot="right")
