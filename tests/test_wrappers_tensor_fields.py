import pickle

import pytest

from netgen.csg import unit_cube
from netgen.occ import unit_square
from ngsolve import (
    BND,
    CF,
    CoefficientFunction,
    Id,
    InnerProduct,
    Integrate,
    Mesh,
    OuterProduct,
    Parameter,
    cos,
    sin,
    sqrt,
    x,
    y,
    z,
)
import ngsdiffgeo as dg
from ngsdiffgeo import ngsdiffgeo as _cpp

from tests._helpers import l2_inner


def _constant_tensor(dim, rank):
    size = dim**rank
    return CF(tuple(float((i % 7) + 1) for i in range(size)), dims=(dim,) * rank)


def _tensor_field_hierarchy():
    return [
        (
            dg.ScalarField(x + y, dim=3),
            _cpp.ScalarField,
            {"degree": 0, "dim_space": 3},
        ),
        (
            dg.OneForm(CF((x, y, x + y))),
            _cpp.OneForm,
            {"degree": 1, "dim_space": 3},
        ),
        (
            dg.TwoForm(_constant_tensor(3, 2), dim=3),
            _cpp.TwoForm,
            {"degree": 2, "dim_space": 3},
        ),
        (
            dg.ThreeForm(_constant_tensor(3, 3), dim=3),
            _cpp.ThreeForm,
            {"degree": 3, "dim_space": 3},
        ),
        (
            dg.KForm(_constant_tensor(4, 4), k=4, dim=4),
            _cpp.KForm,
            {"degree": 4, "dim_space": 4},
        ),
        (
            dg.DoubleForm(_constant_tensor(3, 3), p=1, q=2, dim=3),
            _cpp.DoubleForm,
            {
                "degree_left": 1,
                "degree_right": 2,
                "dim_space": 3,
            },
        ),
    ]


def test_tensorfield_constructors_and_metadata():
    f = CoefficientFunction(x**2 + 0.3 * y)
    v = CF((x + y**2, sin(x * y)))
    A = CF((x, y, sin(x), cos(y)), dims=(2, 2))

    fs = dg.ScalarField(f, dim=2)
    vv = dg.VectorField(v)
    oo = dg.OneForm(v)
    A00 = dg.TensorField(A, "00")
    A11 = dg.TensorField(A, "11")

    assert isinstance(fs, CoefficientFunction)
    assert isinstance(vv, CoefficientFunction)
    assert isinstance(oo, CoefficientFunction)
    assert isinstance(A00, CoefficientFunction)

    assert fs.covariant_indices == ""
    assert vv.covariant_indices == "0"
    assert oo.covariant_indices == "1"
    assert A00.covariant_indices == "00"
    assert A11.covariant_indices == "11"

    with pytest.raises(Exception, match="Received length 1"):
        dg.TensorField(A, "0")
    with pytest.raises(Exception):
        dg.TensorField(A, "0x")


def test_tensorfield_rejects_null_coefficients():
    v = dg.VectorField(CF((x, y)))
    one_form = dg.OneForm(CF((x, y)))

    with pytest.raises(Exception, match="input coefficient is null"):
        dg.TensorField(None, "")
    with pytest.raises(Exception, match="input coefficient is null"):
        dg.VectorField(None)
    with pytest.raises(Exception, match="input coefficient is null"):
        dg.TensorProduct(None, v)
    with pytest.raises(Exception, match="input coefficient is null"):
        dg.TensorProduct(v, None)
    with pytest.raises(Exception, match="input coefficient is null"):
        _cpp.Alternation(None, rank=2, dim=2)
    with pytest.raises(Exception, match="input coefficient is null"):
        _cpp.Wedge(None, one_form)


def test_form_factories_validate_dimension_before_narrowing():
    with pytest.raises(Exception, match=r"dim must be in \{1,...,4\}"):
        dg.ScalarField(x, dim=256)
    with pytest.raises(Exception, match=r"dim must be in \{1,...,4\}"):
        dg.ScalarField(x, dim=257)
    with pytest.raises(Exception, match=r"dim must be in \{1,...,4\}"):
        dg.DoubleForm(x, p=0, q=0, dim=256)


def test_scalarfield_unknown_dimension_can_be_promoted(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.4)
    unknown = dg.ScalarField(x + y)

    assert unknown.dim_space == 0

    promoted = dg.ScalarField(unknown, dim=2)
    assert promoted.dim_space == 2
    assert l2_inner(promoted.coef, unknown.coef, mesh) < 1e-12

    alpha = dg.OneForm(CF((x, y)))
    product = dg.Wedge(unknown, alpha)
    assert product.degree == 1
    assert product.dim_space == 2

    with pytest.raises(Exception, match="requested dim does not match"):
        dg.ScalarField(promoted, dim=3)


def test_doubleform_rank_validation_uses_checked_addition():
    with pytest.raises(Exception, match="combined rank must not exceed"):
        dg.DoubleForm(CF(0), p=2_000_000_000, q=2_000_000_000, dim=2)


def test_legacy_tensorfield_factories_emit_deprecation_warnings():
    with pytest.warns(DeprecationWarning, match="TensorField.from_cf"):
        _cpp.TensorField.from_cf(CF((x, y)), "0")
    with pytest.warns(DeprecationWarning, match="MakeVectorField"):
        _cpp.MakeVectorField(CF((x, y)))
    with pytest.warns(DeprecationWarning, match="ScalarField.from_cf"):
        _cpp.ScalarField.from_cf(x, 2)


def test_tensorfield_pickle_roundtrip_preserves_type_metadata_and_operations():
    objects = [
        dg.TensorField(CF((x, y, x + y, x - y), dims=(2, 2)), "11"),
        dg.VectorField(CF((x, y))),
    ]

    for original in objects:
        restored = pickle.loads(pickle.dumps(original))

        assert type(restored) is type(original)
        assert tuple(restored.dims) == tuple(original.dims)
        assert restored.covariant_indices == original.covariant_indices

        result = restored + restored
        assert isinstance(result, type(original))
        assert result.covariant_indices == original.covariant_indices


def test_derived_tensorfield_pickle_roundtrip_preserves_type_metadata_and_operations():
    for original, cpp_type, metadata in _tensor_field_hierarchy():
        restored = pickle.loads(pickle.dumps(original))

        assert type(restored) is type(original)
        assert isinstance(restored, cpp_type)
        assert tuple(restored.dims) == tuple(original.dims)
        for name, value in metadata.items():
            assert getattr(restored, name) == value

        result = restored + restored
        assert type(result) is type(original)
        for name, value in metadata.items():
            assert getattr(result, name) == value

        for duplicate_name in ("_k", "_p", "_q", "_dim"):
            assert not hasattr(restored, duplicate_name)


def test_pickling_reconstructs_internal_tensorfield_expression_nodes(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.4)

    left = dg.DoubleForm(
        OuterProduct(CF((x, y)), CF((1 + x, 2 + y))),
        p=1,
        q=1,
        dim=2,
    )
    right = dg.DoubleForm(
        OuterProduct(CF((2 + x, y)), CF((x, 1 + y))),
        p=1,
        q=1,
        dim=2,
    )

    objects = [
        dg.Wedge(left, right),
    ]

    for original in objects:
        restored = pickle.loads(pickle.dumps(original))

        assert type(restored) is type(original)
        assert tuple(restored.dims) == tuple(original.dims)
        assert restored.covariant_indices == original.covariant_indices
        assert l2_inner(restored.coef, original.coef, mesh) < 1e-12

    alternation = _cpp.Alternation(
        OuterProduct(CF((x, y)), CF((1 + x, 2 + y))),
        rank=2,
        dim=2,
    )
    restored_alternation = pickle.loads(pickle.dumps(alternation))
    assert restored_alternation.rank == alternation.rank
    assert restored_alternation.dim == alternation.dim
    assert l2_inner(restored_alternation, alternation, mesh) < 1e-12


def test_tensorfield_retyping_flattens_metadata_wrappers():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    raw = CF((x, y, x + y, x - y), dims=(2, 2))
    tensor = dg.TensorField(raw, "00")

    for covariance in ("11", "01", "10", "00"):
        tensor = dg.TensorField(tensor, covariance)
        assert tensor.covariant_indices == covariance
        assert not isinstance(tensor.coef, _cpp.TensorField)
        assert l2_inner(tensor.coef, raw, mesh) < 1e-12


def test_form_retyping_flattens_tensorfield_metadata_wrappers():
    vector = CF((x, y))
    matrix = CF((x, y, x + y, x - y), dims=(2, 2))

    one_form = dg.OneForm(dg.TensorField(vector, "1"))
    double_form = dg.DoubleForm(
        dg.TensorField(matrix, "11"),
        p=1,
        q=1,
        dim=2,
    )

    assert not isinstance(one_form.coef, _cpp.TensorField)
    assert not isinstance(double_form.coef, _cpp.TensorField)


def test_form_full_coefficient_preserves_unverified_dense_input(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.4)
    raw = CF(
        (x, 1 + y, 2 + x, 3 + y),
        dims=(2, 2),
    )

    # Form construction currently records semantics without projecting an
    # arbitrary tensor onto its alternating part.
    forms = [
        dg.TwoForm(raw, dim=2),
        dg.DoubleForm(raw, p=1, q=1, dim=2),
    ]
    for form in forms:
        assert tuple(form.coef.dims) == tuple(form.dims) == (2, 2)
        assert l2_inner(form.coef, raw, mesh) < 1e-12

    # Generic retyping may remove the semantic form wrapper, but it must retain
    # the complete tensor value rather than any future compact storage.
    retyped = dg.TensorField(forms[0], "00")
    assert tuple(retyped.coef.dims) == (2, 2)
    assert l2_inner(retyped.coef, raw, mesh) < 1e-12


def test_tensorfield_equivalence_key_includes_covariance_metadata():
    raw = CF((x, y, x + y, x - y), dims=(2, 2))
    tensor_00 = dg.TensorField(raw, "00")
    tensor_11 = dg.TensorField(raw, "11")
    tensor_00_again = dg.TensorField(tensor_11, "00")

    assert tensor_00._equivalence_key != tensor_11._equivalence_key
    assert tensor_00._equivalence_key == tensor_00_again._equivalence_key

    scalar_2d = dg.ScalarField(x, dim=2)
    scalar_3d = dg.ScalarField(x, dim=3)
    double_scalar_2d = dg.DoubleForm(x, p=0, q=0, dim=2)
    double_scalar_3d = dg.DoubleForm(x, p=0, q=0, dim=3)

    assert scalar_2d._equivalence_key != scalar_3d._equivalence_key
    assert double_scalar_2d._equivalence_key != double_scalar_3d._equivalence_key


def test_vectorfield_transform_and_scalar_derivatives_preserve_cpp_type():
    parameter = Parameter(2)
    parameter.MakeVariable()
    vector = dg.VectorField(parameter * CF((x, y)))

    results = [
        vector.Replace({parameter: CF(3)}),
        vector.Diff(parameter, 1),
        vector.Diff(parameter),
    ]

    for result in results:
        assert isinstance(result, _cpp.VectorField)
        assert tuple(result.dims) == (2,)
        assert result.covariant_indices == "0"


def test_derived_tensorfield_operations_preserve_concrete_type_and_metadata():
    parameter = Parameter(2)
    parameter.MakeVariable()

    for prototype, cpp_type, metadata in _tensor_field_hierarchy():
        original = prototype * parameter
        results = [
            original.Replace({parameter: CF(3)}),
            original.Diff(parameter, 1),
            original.Diff(parameter),
        ]

        for result in results:
            assert isinstance(result, cpp_type)
            assert tuple(result.dims) == tuple(original.dims)
            for name, value in metadata.items():
                assert getattr(result, name) == value


def test_non_scalar_jacobians_return_untyped_coefficient_functions():
    variable = CF((x, y))
    variable.MakeVariable()

    tensor = dg.TensorField(OuterProduct(variable, variable), "00")
    one_form = dg.OneForm(variable)
    double_form = dg.DoubleForm(
        OuterProduct(variable, variable),
        p=1,
        q=1,
        dim=2,
    )

    jacobians = [
        (tensor.Diff(variable), (2, 2, 2)),
        (one_form.Diff(variable), (2, 2)),
        (double_form.Diff(variable), (2, 2, 2)),
    ]

    for jacobian, expected_dims in jacobians:
        assert tuple(jacobian.dims) == expected_dims
        assert not isinstance(jacobian, _cpp.TensorField)
        assert not hasattr(jacobian, "covariant_indices")


def test_internal_tensor_operators_support_non_scalar_jacobians(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.4)
    variable = CF((x, y))
    variable.MakeVariable()
    direction = CF((1 + x, 2 - y))

    alternation = _cpp.Alternation(
        OuterProduct(variable, variable),
        rank=2,
        dim=2,
    )
    left = dg.DoubleForm(
        OuterProduct(variable, CF((1 + x, 2 + y))),
        p=1,
        q=1,
        dim=2,
    )
    right = dg.DoubleForm(
        OuterProduct(CF((2 * x, 3 * y)), variable),
        p=1,
        q=1,
        dim=2,
    )
    wedge = dg.Wedge(left, right)
    block_alternation = dg.RiemannianManifold(Id(2)).d_cov(
        left,
        slot="left",
    )

    cases = [
        (alternation, (2, 2, 2), "abi,i->ab"),
        (wedge, (2, 2, 2, 2, 2), "abcdi,i->abcd"),
        (block_alternation, (2, 2, 2, 2), "abci,i->abc"),
    ]
    for expression, expected_dims, contraction_signature in cases:
        jacobian = expression.Diff(variable)
        contracted = dg.Einsum(contraction_signature, jacobian, direction)
        directional = expression.Diff(variable, direction)

        assert tuple(jacobian.dims) == expected_dims
        assert not isinstance(jacobian, _cpp.TensorField)
        assert l2_inner(contracted, directional, mesh) < 1e-9


def test_scalar_jacobians_preserve_geometric_metadata():
    parameter = Parameter(2)
    parameter.MakeVariable()

    tensor_jacobian = dg.TensorField(parameter * Id(2), "11").Diff(parameter)
    one_form_jacobian = dg.OneForm(parameter * CF((x, y))).Diff(parameter)
    double_form_jacobian = dg.DoubleForm(
        parameter * Id(2),
        p=1,
        q=1,
        dim=2,
    ).Diff(parameter)

    assert isinstance(tensor_jacobian, _cpp.TensorField)
    assert tensor_jacobian.covariant_indices == "11"
    assert isinstance(one_form_jacobian, _cpp.OneForm)
    assert one_form_jacobian.degree == 1
    assert isinstance(double_form_jacobian, _cpp.DoubleForm)
    assert (double_form_jacobian.degree_left, double_form_jacobian.degree_right) == (1, 1)


def test_tensorproduct_signature_rank_boundary():
    left = dg.TensorField(CF((2,), dims=(1,) * 26), "0" * 26)
    right = dg.TensorField(CF((3,), dims=(1,) * 26), "1" * 26)
    product = dg.TensorProduct(left, right)

    assert len(product.dims) == 52
    assert product.covariant_indices == "0" * 26 + "1" * 26

    overflow = dg.TensorField(CF((3,), dims=(1,) * 27), "1" * 27)
    with pytest.raises(Exception, match="concat overflow"):
        dg.TensorProduct(left, overflow)

    with pytest.raises(Exception, match="rank overflow"):
        dg.TensorField(CF((0,), dims=(1,) * 53), "0" * 53)


def test_projected_tensor_archives_its_semantic_operand(rm_euclidean_2d):
    vector = dg.VectorField(CF((x, y)))
    projected = rm_euclidean_2d.ProjectTensor(vector, "F")
    projected_twice = rm_euclidean_2d.ProjectTensor(projected, "F")

    for operand, result in ((vector, projected), (projected, projected_twice)):
        assert isinstance(result, dg.VectorField)
        assert not isinstance(result.coef, _cpp.TensorField)
        archive_children = result.coef.data["childs"]
        # Archives retain the original operand for Diff/Replace; metadata is
        # stripped only from the native evaluation graph, not this semantic graph.
        tensor_children = [child for child in archive_children if isinstance(child, _cpp.TensorField)]
        assert tensor_children == [operand]


def test_complex_scalar_tensorfield_evaluates_like_wrapped_coefficient():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    tensor = dg.TensorField((1 + 2j) * (x + y), "")

    for point in (mesh(0.2, 0.3), mesh(0.8, 0.1)):
        assert tensor(point) == pytest.approx(tensor.coef(point))


@pytest.mark.parametrize("scale", [1, 1 + 2j])
def test_tensorfield_compile_preserves_values(scale, monkeypatch):
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    tensor = dg.TensorField(
        scale * CF((x, y, x + y, x - y), dims=(2, 2)),
        "11",
    )

    for realcompile in (False, True):
        compiled = tensor.Compile(
            realcompile=realcompile,
            maxderiv=2,
            wait=True,
        )
        assert tuple(compiled.dims) == tuple(tensor.dims)
        assert compiled.is_complex is tensor.is_complex
        assert abs(l2_inner(compiled, tensor.coef, mesh)) < 1e-12


def test_derived_tensorfield_compile_preserves_values(
    make_unit_square_mesh,
    monkeypatch,
):
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    mesh = make_unit_square_mesh(maxh=0.4)

    for original, _, _ in _tensor_field_hierarchy():
        for realcompile in (False, True):
            compiled = original.Compile(
                realcompile=realcompile,
                maxderiv=0,
                wait=True,
            )

            assert tuple(compiled.dims) == tuple(original.dims)
            assert compiled.is_complex is original.is_complex
            assert abs(l2_inner(compiled, original.coef, mesh)) < 1e-12


def test_typed_zeros_preserved():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.35))

    v = CF((x + y, x - 2 * y))
    w = CF((sin(x), cos(y)))

    wv = dg.VectorField(w)

    vv0 = dg.VectorField(0 * v)
    assert vv0.covariant_indices == "0"
    assert isinstance(vv0, dg.VectorField)

    B = dg.TensorProduct(vv0, wv)
    assert B.covariant_indices == "00"
    assert l2_inner(B.coef, 0 * OuterProduct(v, w), mesh) == pytest.approx(0)


def test_tensorproduct_matches_outerproduct_and_covariance():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))

    v = CF((x**2 + y, sin(x * y)))
    w = CF((y - x, cos(x)))

    vv = dg.VectorField(v)
    wv = dg.VectorField(w)
    vo = dg.OneForm(v)
    wo = dg.OneForm(w)

    out = OuterProduct(v, w)

    B00 = dg.TensorProduct(vv, wv)
    B11 = dg.TensorProduct(vo, wo)
    B10 = dg.TensorProduct(vo, wv)
    B01 = dg.TensorProduct(vv, wo)

    assert B00.covariant_indices == "00"
    assert B11.covariant_indices == "11"
    assert B10.covariant_indices == "10"
    assert B01.covariant_indices == "01"

    for B in [B00, B11, B10, B01]:
        assert l2_inner(B.coef, out, mesh) < 1e-12


def test_tensorproduct_requires_tensorfields():
    v = CF((x, y))
    A = CF((x, y, x + y, x - y), dims=(2, 2))

    vv = dg.VectorField(v)
    A00 = dg.TensorField(A, "00")

    with pytest.raises(TypeError):
        dg.TensorProduct(v, v)

    with pytest.raises(TypeError):
        dg.TensorProduct(vv, v)

    with pytest.raises(TypeError):
        dg.TensorProduct(A, A00)


def test_nested_wrapping_is_idempotent_in_value():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

    v = CF((x + y, x - y))
    vv = dg.VectorField(v)
    vv2 = dg.VectorField(vv)

    assert vv2.covariant_indices == "0"
    assert l2_inner(vv.coef, vv2.coef, mesh) == pytest.approx(0)


def test_J_and_S_preserve_doubleform_type_for_11_inputs():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    rm = dg.RiemannianManifold(Id(2))

    A = CF((x, y, x + y, x - y), dims=(2, 2))
    df = dg.DoubleForm(A, p=1, q=1, dim=2)

    Jdf = rm.J(df)
    Sdf = rm.S(df)

    trace_A = 2 * x - y
    expected_J = CF((x, x + y, y, x - y), dims=(2, 2)) - 0.5 * trace_A * Id(2)
    expected_S = CF((x, x + y, y, x - y), dims=(2, 2)) - trace_A * Id(2)

    assert isinstance(Jdf, dg.DoubleForm)
    assert isinstance(Sdf, dg.DoubleForm)
    assert Jdf.degree_left == 1
    assert Jdf.degree_right == 1
    assert Sdf.degree_left == 1
    assert Sdf.degree_right == 1
    assert l2_inner(Jdf.coef, expected_J, mesh) == pytest.approx(0)
    assert l2_inner(Sdf.coef, expected_S, mesh) == pytest.approx(0)


def test_S_and_J_on_bnd_match_projected_doubleform_for_non_euclidean_metric():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.8))
    g = CF((2 + x, 0.2, 0.1, 0.2, 3 + y, 0.3, 0.1, 0.3, 4 + z), dims=(3, 3))
    rm = dg.RiemannianManifold(g)

    alpha = dg.OneForm(CF((x + 1, y + 2, z + 3)))
    beta = dg.OneForm(CF((2 * x + 1, 3 * y + 1, 4 * z + 1)))
    phi = dg.DoubleForm(dg.TensorProduct(alpha, beta), p=1, q=1, dim=3)

    projected = rm.ProjectDoubleForm(phi, left="F", right="F")
    expected_S = dg.DoubleForm(
        projected.trans - rm.G_F * rm.Trace(projected, vb=BND),
        p=1,
        q=1,
        dim=3,
    )
    expected_J = dg.DoubleForm(
        projected.trans - 0.5 * rm.G_F * rm.Trace(projected, vb=BND),
        p=1,
        q=1,
        dim=3,
    )

    Sdf = rm.S(phi, vb=BND)
    Jdf = rm.J(phi, vb=BND)

    assert sqrt(Integrate(InnerProduct(Sdf - expected_S, Sdf - expected_S), mesh, BND)) < 1e-8
    assert sqrt(Integrate(InnerProduct(Jdf - expected_J, Jdf - expected_J), mesh, BND)) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__])
