"""Adversarial public-API combinations for semantic tensor wrappers.

These tests deliberately use only the Python API.  The compatibility matrices
guard against falling through to an untyped NGSolve operation when variance,
form degree, tensor rank, or manifold dimension makes an operation ambiguous.
"""

import operator
import pickle
import subprocess
import sys

import pytest
from ngsolve import CF, Id

import ngsdiffgeo as dg


_MATRIX_VALUES = (1, 2, 3, 4)


def _tensor(covariance):
    return dg.TensorField(CF(_MATRIX_VALUES, dims=(2, 2)), covariance)


def _kform(degree):
    if degree == 0:
        coefficient = CF(1)
    else:
        dims = (3,) * degree
        coefficient = CF(tuple(range(1, 3**degree + 1)), dims=dims)
    return dg.KForm(coefficient, k=degree, dim=3)


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
@pytest.mark.parametrize("left_covariance", ["00", "01", "10", "11"])
@pytest.mark.parametrize("right_covariance", ["00", "01", "10", "11"])
def test_rank_two_add_subtract_requires_identical_variance(
    operation, left_covariance, right_covariance
):
    left = _tensor(left_covariance)
    right = _tensor(right_covariance)

    if left_covariance == right_covariance:
        result = operation(left, right)
        assert isinstance(result, dg.TensorField)
        assert result.covariant_indices == left_covariance
    else:
        with pytest.raises(TypeError, match="variance"):
            operation(left, right)


@pytest.mark.parametrize("left_covariance", ["00", "01", "10", "11"])
@pytest.mark.parametrize("right_covariance", ["00", "01", "10", "11"])
def test_rank_two_product_contracts_exactly_opposite_inner_variance(
    left_covariance, right_covariance
):
    left = _tensor(left_covariance)
    right = _tensor(right_covariance)
    contraction_is_defined = left_covariance[1] != right_covariance[0]

    if contraction_is_defined:
        result = left * right
        assert isinstance(result, dg.TensorField)
        assert result.covariant_indices == (
            left_covariance[0] + right_covariance[1]
        )
        assert tuple(result.dims) == (2, 2)
    else:
        with pytest.raises(TypeError, match="opposite-variance axes"):
            _ = left * right


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
@pytest.mark.parametrize("left_degree", [0, 1, 2, 3])
@pytest.mark.parametrize("right_degree", [0, 1, 2, 3])
def test_kform_add_subtract_requires_identical_degree(
    operation, left_degree, right_degree
):
    left = _kform(left_degree)
    right = _kform(right_degree)

    if left_degree == right_degree:
        result = operation(left, right)
        assert isinstance(result, dg.KForm)
        assert result.degree == left_degree
        assert result.dim_space == 3
    else:
        with pytest.raises(TypeError, match="degree|variance|shape"):
            operation(left, right)


@pytest.mark.parametrize("left_degree", [0, 1, 2, 3])
@pytest.mark.parametrize("right_degree", [0, 1, 2, 3])
def test_wedge_is_the_explicit_product_for_kforms(left_degree, right_degree):
    result = dg.Wedge(_kform(left_degree), _kform(right_degree))

    assert result.degree == left_degree + right_degree
    assert result.dim_space == 3


def test_overflow_kform_wedges_keep_a_compact_zero_tree():
    for left_degree, right_degree in ((1, 3), (2, 2), (2, 3), (3, 3)):
        result = dg.Wedge(_kform(left_degree), _kform(right_degree))

        if dg.is_formal_zero(result):
            continue
        tree = str(result)
        assert tree.count("ZeroCF") == 1
        assert len(tree.splitlines()) == 2


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(
            lambda: dg.VectorField(CF((1, 2))),
            lambda: dg.VectorField(CF((3, 4))),
            id="vector-vector",
        ),
        pytest.param(
            lambda: dg.VectorField(CF((1, 2))),
            lambda: dg.OneForm(CF((3, 4))),
            id="vector-one-form",
        ),
        pytest.param(
            lambda: dg.OneForm(CF((1, 2))),
            lambda: dg.VectorField(CF((3, 4))),
            id="one-form-vector",
        ),
        pytest.param(
            lambda: dg.OneForm(CF((1, 2))),
            lambda: dg.OneForm(CF((3, 4))),
            id="one-form-one-form",
        ),
    ],
)
def test_rank_one_product_never_implicitly_selects_an_inner_product(left, right):
    with pytest.raises(TypeError, match="scalar operands"):
        _ = left() * right()


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        pytest.param(
            lambda: dg.VectorField(CF((1, 2))),
            lambda: dg.OneForm(CF((3, 4))),
            "variance",
            id="vector-and-one-form",
        ),
        pytest.param(
            lambda: dg.ScalarField(1, dim=2),
            lambda: dg.VectorField(CF((1, 2))),
            "scalar|variance|shape",
            id="scalar-and-vector",
        ),
        pytest.param(
            lambda: dg.OneForm(CF((1, 2))),
            lambda: dg.TwoForm(CF((0, 1, -1, 0), dims=(2, 2)), dim=2),
            "different degree|shape",
            id="different-k-form-degree",
        ),
        pytest.param(
            lambda: dg.OneForm(CF((1, 2))),
            lambda: dg.OneForm(CF((1, 2, 3))),
            "dimensions must match",
            id="different-ambient-dimension",
        ),
        pytest.param(
            lambda: dg.DoubleForm(
                CF((0, 1, -1, 0), dims=(2, 2)), p=2, q=0, dim=2
            ),
            lambda: dg.DoubleForm(
                CF((1, 2, 3, 4), dims=(2, 2)), p=1, q=1, dim=2
            ),
            "different left/right degree",
            id="different-double-form-block-degree",
        ),
    ],
)
def test_incompatible_add_subtract_is_rejected_in_both_orders(
    operation, left, right, message
):
    a, b = left(), right()
    with pytest.raises((TypeError, ValueError), match=message):
        operation(a, b)
    with pytest.raises((TypeError, ValueError), match=message):
        operation(b, a)


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_form_plus_unstructured_tensor_loses_unproved_alternation_in_both_orders(
    operation,
):
    form = dg.TwoForm(CF((0, 1, -1, 0), dims=(2, 2)), dim=2)
    tensor = _tensor("11")

    for result in (operation(form, tensor), operation(tensor, form)):
        assert isinstance(result, dg.TensorField)
        assert not isinstance(result, dg.KForm)
        assert result.covariant_indices == "11"


@pytest.mark.parametrize("operation_name", ["Raise", "Lower"])
@pytest.mark.parametrize("covariance", ["00", "01", "10", "11"])
@pytest.mark.parametrize("index", [0, 1])
def test_raise_lower_acceptance_follows_variance_at_selected_index(
    operation_name, covariance, index
):
    manifold = dg.RiemannianManifold(Id(2))
    tensor = _tensor(covariance)
    operation = getattr(manifold, operation_name)
    expected_input_variance = "1" if operation_name == "Raise" else "0"

    if covariance[index] == expected_input_variance:
        result = operation(tensor, index)
        expected = list(covariance)
        expected[index] = "0" if operation_name == "Raise" else "1"
        assert result.covariant_indices == "".join(expected)
    else:
        with pytest.raises(Exception, match="already"):
            operation(tensor, index)


def test_raw_coefficient_product_is_an_explicit_untyped_escape_hatch():
    vector = dg.VectorField(CF((1, 2)))
    raw_vector = CF((3, 4))

    result = vector * raw_vector

    assert not isinstance(result, dg.TensorField)
    assert tuple(result.dims) == ()


def test_valid_mixed_tree_survives_replace_pickle_compile_and_evaluation(
    make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.8)
    point = mesh(0.2, 0.3)
    source = dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), "11")
    replacement = dg.TensorField(CF((5, 6, 7, 8), dims=(2, 2)), "11")
    multiplier = dg.TensorField(CF((2, 0, 0, 3), dims=(2, 2)), "00")
    offset = dg.TensorField(CF((1, 1, 1, 1), dims=(2, 2)), "10")
    expression = 0.5 * (source * multiplier + offset)
    reference = 0.5 * (replacement * multiplier + offset)

    replaced = expression.Replace({source: replacement})
    restored_source, restored_expression = pickle.loads(
        pickle.dumps((source, expression))
    )
    restored_replaced = restored_expression.Replace(
        {restored_source: replacement}
    )
    compiled = replaced.Compile(realcompile=False, maxderiv=0)

    expected = tuple(reference(point))
    assert tuple(replaced(point)) == pytest.approx(expected)
    assert tuple(restored_replaced(point)) == pytest.approx(expected)
    assert tuple(compiled(point)) == pytest.approx(expected)


def test_malformed_public_api_matrix_raises_without_native_crash():
    """Keep constructor and dispatch abuse outside the pytest process."""
    expressions = [
        "dg.VectorField(CF(1))",
        "dg.OneForm(CF((1, 2, 3, 4), dims=(2, 2)))",
        "dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), 'x')",
        "dg.TensorField(CF((1, 2, 3, 4), dims=(2, 2)), '1')",
        "dg.KForm(CF((1, 2)), k=-1, dim=2)",
        "dg.KForm(CF((1, 2)), k=1, dim=3)",
        "dg.DoubleForm(CF((1, 2, 3, 4), dims=(2, 2)), p=1, q=0, dim=2)",
        "dg.RiemannianManifold(CF((1, 2, 3, 4, 5, 6), dims=(2, 3)))",
        "dg.VectorField(CF((1, 2))) + dg.OneForm(CF((3, 4)))",
        "dg.VectorField(CF((1, 2))) * dg.VectorField(CF((3, 4)))",
    ]
    code = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    "ngsdiffgeo", {dg.__file__!r}, submodule_search_locations={list(dg.__path__)!r}
)
package = importlib.util.module_from_spec(spec)
sys.modules["ngsdiffgeo"] = package
spec.loader.exec_module(package)
import ngsdiffgeo as dg
from ngsolve import CF

expressions = {expressions!r}
for expression in expressions:
    try:
        eval(expression)
    except Exception:
        pass
    else:
        raise AssertionError("invalid expression was accepted: " + expression)
print("all-invalid-inputs-rejected")
"""

    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "all-invalid-inputs-rejected" in result.stdout


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_doubleform_left_rejects_contravariant_rank_one_tensor(operation):
    double_form = dg.DoubleForm(CF((1, 2)), p=1, q=0, dim=2)
    vector = dg.VectorField(CF((3, 4)))

    with pytest.raises(TypeError, match="variance"):
        operation(double_form, vector)


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_doubleform_and_unstructured_tensor_lose_block_form_metadata_in_both_orders(
    operation, make_unit_square_mesh,
):
    mesh = make_unit_square_mesh(maxh=0.8)
    point = mesh(0.2, 0.3)
    double_form = dg.DoubleForm(
        CF((0, 1, -1, 0), dims=(2, 2)), p=2, q=0, dim=2
    )
    tensor = _tensor("11")

    forward = operation(double_form, tensor)
    reverse = operation(tensor, double_form)
    for result in (forward, reverse):
        assert isinstance(result, dg.TensorField)
        assert not isinstance(result, dg.DoubleForm)
        assert result.covariant_indices == "11"

    if operation is operator.add:
        assert tuple(forward(point)) == pytest.approx((1, 3, 2, 4))
        assert tuple(reverse(point)) == pytest.approx((1, 3, 2, 4))
    else:
        assert tuple(forward(point)) == pytest.approx((-1, -1, -4, -4))
        assert tuple(reverse(point)) == pytest.approx((1, 1, 4, 4))


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_11_doubleform_and_covariant_matrix_preserve_doubleform_in_both_orders(
    operation,
):
    double_form = dg.DoubleForm(
        CF((1, 2, 3, 4), dims=(2, 2)), p=1, q=1, dim=2
    )
    tensor = _tensor("11")

    for result in (
        operation(double_form, tensor),
        operation(tensor, double_form),
    ):
        assert isinstance(result, dg.DoubleForm)
        assert (result.degree_left, result.degree_right) == (1, 1)
        assert result.dim_space == 2


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_matching_doubleforms_preserve_block_form_metadata(operation):
    left = dg.DoubleForm(
        CF((1, 2, 3, 4), dims=(2, 2)), p=1, q=1, dim=2
    )
    right = dg.DoubleForm(
        CF((5, 6, 7, 8), dims=(2, 2)), p=1, q=1, dim=2
    )

    result = operation(left, right)

    assert isinstance(result, dg.DoubleForm)
    assert (result.degree_left, result.degree_right) == (1, 1)
    assert result.dim_space == 2


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_doubleform_and_equivalent_kform_degrade_consistently(operation):
    double_form = dg.DoubleForm(CF((1, 2)), p=1, q=0, dim=2)
    one_form = dg.OneForm(CF((3, 4)))

    for result in (
        operation(double_form, one_form),
        operation(one_form, double_form),
    ):
        assert isinstance(result, dg.OneForm)
        assert not isinstance(result, dg.DoubleForm)
