"""Public form API regressions from the second kforms audit."""
import operator

import numpy as np
import pytest

import ngsdiffgeo as dg
from ngsdiffgeo import ngsdiffgeo as cpp
from ngsdiffgeo import wrappers as wrappers
from ngsolve import CF, Id, IfPos, Norm, x, y
from netgen.libngpy._meshing import NgException
from tests._helpers import assert_l2_close


def _linear_case(name):
    manifold = dg.RiemannianManifold(Id(2))
    a = dg.OneForm(CF((y, x*x)))
    b = dg.OneForm(CF((1+y, 2+x)))
    df = dg.DoubleForm(CF((x, y, 2*y, 3*x), dims=(2, 2)), p=1, q=1, dim=2)
    if name == "scale":
        return a, 2*a
    if name == "add":
        return a, a+b, b
    if name == "neg":
        return a, -a
    if name == "divide":
        return a, a/(1+x)
    if name == "wedge":
        return a, dg.Wedge(a, b)
    if name == "d":
        return a, dg.d(a)
    if name == "star1":
        return a, dg.star(a, manifold)
    if name == "star0":
        a = dg.ScalarField(x+1, dim=2)
        return a, dg.star(a, manifold)
    if name == "star2":
        a = dg.TwoForm(CF((0, x, -x, 0), dims=(2, 2)), dim=2)
        return a, dg.star(a, manifold)
    if name == "double_scale":
        return df, 2*df
    if name == "transpose":
        return df, df.trans
    if name == "sym":
        return df, dg.Sym(df)
    if name == "double_star":
        return df, dg.star(df, manifold)
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["scale", "add", "neg", "divide", "wedge", "d", "star0", "star1", "star2", "double_scale", "transpose", "sym", "double_star"])
@pytest.mark.parametrize("action", ["diff", "replace"])
@pytest.mark.parametrize("variable_kind", ["wrapper", "coefficient"])
def test_form_operations_preserve_symbolic_operands(name, action, variable_kind, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    a, result, *offset = _linear_case(name)
    expected_derivative = result.coef - offset[0].coef if offset else result.coef
    variable = a if variable_kind == "wrapper" else a.coef
    if action == "diff":
        actual = result.Diff(variable, a.coef)
        expected = expected_derivative
    else:
        actual = result.Replace({variable: 2*a.coef})
        expected = result.coef + expected_derivative
    assert_l2_close(actual, expected, mesh, tol=1e-9)


@pytest.mark.parametrize("kind", ["kform", "doubleform"])
def test_raw_matrix_dimension_inference(kind):
    raw = CF(tuple(range(9)), dims=(3, 3))
    result = dg.KForm(raw, k=2) if kind == "kform" else wrappers.as_doubleform(raw, p=1, q=1)
    assert result.dim_space == 3


@pytest.mark.parametrize("kind", ["one", "two", "three", "double"])
def test_conversions_reject_conflicting_metadata(kind):
    if kind == "one":
        a = dg.OneForm(CF((x, y)))
        convert = lambda: dg.KForm(a, k=1, dim=3)
    elif kind == "two":
        a = dg.TwoForm(Id(2), dim=2)
        convert = lambda: wrappers.as_twoform(a, dim=3)
    elif kind == "three":
        a = dg.ThreeForm(CF((0,)*27, dims=(3,)*3), dim=3)
        convert = lambda: wrappers.as_threeform(a, dim=4)
    else:
        a = dg.DoubleForm(Id(2), p=1, q=1, dim=2)
        convert = lambda: wrappers.as_doubleform(a, p=0, q=2, dim=2)
    with pytest.raises((ValueError, TypeError, NgException)):
        convert()


@pytest.mark.parametrize("call", [
    lambda a: dg.KForm(a, k=1.9, dim=2),
    lambda a: dg.KForm(a, k=1, dim=2.9),
    lambda a: dg.KForm(a, k=1, dimension=3),
    lambda a: dg.KForm(a, 1, 2, "ignored"),
    lambda a: dg.KForm(a, 1, k=1, dim=2),
    lambda a: dg.DoubleForm(a, p=1.9, q=0, dim=2),
])
def test_form_factories_reject_invalid_arguments(call):
    with pytest.raises(TypeError):
        call(CF((x, y)))


@pytest.mark.parametrize("operation", [lambda a,b: dg.Wedge(a,b), lambda a,b: a*b, lambda a,b: b*a])
def test_singleton_scalar_times_form_preserves_shape(operation, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    scalar = dg.ScalarField(x+1, dim=1)
    one = dg.OneForm(CF((2,), dims=(1,)))
    result = operation(scalar, one)
    assert isinstance(result, dg.OneForm)
    assert tuple(result.dims) == (1,)
    np.testing.assert_allclose(result(mesh(0.2,0.3)), (2.4,))


def test_singleton_wedge_and_formal_dispatch():
    one = dg.OneForm(CF((2,), dims=(1,)))
    result = dg.Wedge(one, one)
    assert result.degree == 2
    assert tuple(result.dims) == (1,1)
    with pytest.raises(TypeError):
        dg.Wedge(dg.FormalZeroDoubleForm(1,0,1), one)


@pytest.mark.parametrize("operation", [operator.mul, operator.truediv])
@pytest.mark.parametrize("kind", ["one", "two", "three", "generic"])
def test_kform_products_only_accept_scalar_operands(operation, kind):
    rank, dim = {"one":(1,2), "two":(2,2), "three":(3,3), "generic":(4,4)}[kind]
    a = dg.KForm(CF((1,)*(dim**rank), dims=(dim,)*rank), k=rank, dim=dim)
    with pytest.raises(TypeError, match="scalar"):
        operation(a, a)


@pytest.mark.parametrize("operation", [operator.add, operator.mul, dg.Wedge])
def test_scalar_operations_reject_conflicting_dimensions(operation):
    with pytest.raises((ValueError, TypeError, NgException), match="dimension"):
        operation(dg.ScalarField(x,dim=2), dg.ScalarField(y,dim=3))


@pytest.mark.parametrize("operation", [dg.star, dg.inv_star])
def test_formal_star_checks_manifold_dimension(operation):
    with pytest.raises(ValueError, match="dimension"):
        operation(dg.FormalZeroKForm(1,3), dg.RiemannianManifold(Id(2)))


def test_wedge_validates_dimensions_before_overflow():
    a = dg.DoubleForm(Id(2), p=1,q=1,dim=2)
    b = dg.DoubleForm(CF((1,)*27,dims=(3,)*3),p=2,q=1,dim=3)
    with pytest.raises(ValueError, match="dimension"):
        dg.Wedge(a,b)


@pytest.mark.parametrize("case", ["contract_dimension", "contract_null", "d_cov_vb", "delta_cov_dimension", "trace_negative", "project_dimension"])
def test_zero_shortcuts_validate_arguments(case):
    M = dg.RiemannianManifold(Id(2))
    zero = dg.FormalZeroDoubleForm(0,0,3 if "dimension" in case else 2)
    calls = {
        "contract_dimension": lambda: M.ContractSlot(zero, dg.VectorField(CF((1,0)))),
        "contract_null": lambda: M.ContractSlot(zero, None),
        "d_cov_vb": lambda: M.d_cov(zero, vb=999),
        "delta_cov_dimension": lambda: M.delta_cov(zero),
        "trace_negative": lambda: M.Trace(zero, l=-1),
        "project_dimension": lambda: M.ProjectDoubleForm(zero),
    }
    with pytest.raises((TypeError, ValueError, NgException)):
        calls[case]()


@pytest.mark.parametrize("kind", ["alternation", "wedge"])
def test_scalar_nodes_support_complex_ifpos(kind, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    node = cpp.Alternation(x,0,2) if kind == "alternation" else cpp.Wedge(cpp.DoubleForm(x,0,0,2),cpp.DoubleForm(y,0,0,2))
    assert IfPos(node,1j,-1j)(mesh(0.2,0.3)) == pytest.approx(1j)


@pytest.mark.parametrize("method", ["wedge", "star", "inv_star", "d"])
def test_form_methods_return_python_arithmetic_wrappers(method):
    a = dg.OneForm(CF((y,x*x)))
    M = dg.RiemannianManifold(Id(2))
    result = a.wedge(a) if method == "wedge" else a.d() if method == "d" else getattr(a,method)(M)
    assert isinstance(2*result, dg.KForm)


def test_scalar_multiplication_preserves_doubleform_in_both_orders():
    f = dg.ScalarField(x,dim=2)
    a = dg.DoubleForm(Id(2),p=1,q=1,dim=2)
    assert isinstance(f*a, dg.DoubleForm)
    assert isinstance(a*f, dg.DoubleForm)


@pytest.mark.parametrize("slot,canonical", [(0,"left"),(1,"right"),(-1,"both"),("LEFT","left"),("RIGHT","right"),("BOTH","both")])
@pytest.mark.parametrize("operation", [dg.star,dg.inv_star])
def test_star_slot_aliases(slot,canonical,operation,make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    M = dg.RiemannianManifold(Id(2))
    a = dg.DoubleForm(CF((x,y,2*y,3*x),dims=(2,2)),p=1,q=1,dim=2)
    assert_l2_close(operation(a,M,slot=slot),operation(a,M,slot=canonical),mesh,tol=1e-12)


# Spatial differentiation acts on the direction field itself, so it has no
# pointwise matrix Jacobian satisfying this contraction identity. Its full
# directional derivative is covered above.
@pytest.mark.parametrize("name", ["scale", "wedge", "star0", "star1", "star2", "transpose", "sym", "double_star"])
def test_form_jacobian_contracts_to_directional_derivative(name, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    a, result = _linear_case(name)
    jacobian = result.Diff(a)
    assert tuple(jacobian.dims) == tuple(result.dims) + tuple(a.dims)
    contracted = (jacobian.Reshape((result.dim, a.dim)) * a.coef.Reshape((a.dim,))).Reshape(tuple(result.dims))
    assert_l2_close(contracted, result, mesh, tol=1e-9)


@pytest.mark.parametrize("kind", ["wedge", "star", "transpose"])
@pytest.mark.parametrize("scale", [1, 1+2j])
def test_symbolic_form_graphs_compile_and_pickle(kind, scale, make_unit_square_mesh, monkeypatch):
    import pickle
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    mesh = make_unit_square_mesh(maxh=0.7)
    a = dg.OneForm(CF((scale*(1+x), scale*y)))
    b = dg.OneForm(CF((y, 2+x)))
    matrix = dg.DoubleForm(CF((x,y,2*y,3*x),dims=(2,2))*scale,p=1,q=1,dim=2)
    if kind == "wedge":
        result = dg.Wedge(a,b)
        value = scale*((1+x)*(2+x)-y*y)
        expected = CF((0,value,-value,0),dims=(2,2))
    elif kind == "star":
        result = dg.star(a,dg.RiemannianManifold(Id(2)))
        expected = CF((-scale*y,scale*(1+x)))
    else:
        result = matrix.trans
        expected = scale*CF((x,2*y,y,3*x),dims=(2,2))
    for candidate in (pickle.loads(pickle.dumps(result)),
                      result.Compile(realcompile=False),
                      result.Compile(realcompile=True,maxderiv=2,wait=True)):
        assert tuple(candidate.dims) == tuple(expected.dims)
        assert_l2_close(Norm(candidate-expected),CF(0),mesh,tol=1e-10)


def test_scalarfield_still_multiplies_integration_measures(make_unit_square_mesh):
    from ngsolve import Integrate, dx
    mesh = make_unit_square_mesh(maxh=0.7)
    assert Integrate(dg.ScalarField(x,dim=2)*dx,mesh) == pytest.approx(0.5)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("action", ["diff", "replace"])
def test_wedge_dimension_promotion_preserves_scalar_identity(reverse, action, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    scalar = dg.ScalarField(1+x)
    a = dg.OneForm(CF((1+y,x)))
    result = dg.Wedge(a,scalar) if reverse else dg.Wedge(scalar,a)
    if action == "diff":
        actual, expected = result.Diff(scalar,scalar.coef), result
    else:
        actual, expected = result.Replace({scalar:2*scalar.coef}), 2*result
    assert_l2_close(actual,expected,mesh,tol=1e-12)


@pytest.mark.parametrize("kind", ["kform", "doubleform"])
def test_formal_wedge_infers_unknown_scalar_dimension(kind):
    scalar = dg.ScalarField(1+x)
    zero = dg.FormalZeroKForm(1,2) if kind == "kform" else dg.FormalZeroDoubleForm(1,1,2)
    result = dg.Wedge(zero,scalar)
    assert type(result) is type(zero)
    assert result.dim_space == 2


@pytest.mark.parametrize("operation", ["inner", "slot", "slot_free", "delta", "delta_free"])
def test_default_scalar_keeps_supported_manifold_operations(operation, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    scalar = dg.ScalarField(1 + x)
    calls = {
        "inner": lambda: manifold.InnerProduct(scalar, scalar),
        "slot": lambda: manifold.SlotInnerProduct(scalar),
        "slot_free": lambda: dg.slot_inner_product(scalar, manifold),
        "delta": lambda: manifold.delta(scalar),
        "delta_free": lambda: dg.delta(scalar, manifold),
    }
    result = calls[operation]()
    if operation.startswith("delta"):
        assert isinstance(result, dg.FormalZeroKForm)
        assert (result.degree, result.dim_space) == (-1, 0)
    else:
        assert isinstance(result, dg.ScalarField)
        assert result.dim_space == 2
        expected = (1 + x)**2 if operation == "inner" else 1 + x
        assert_l2_close(result, expected, mesh, tol=1e-12)
    assert scalar.dim_space == 0


@pytest.mark.parametrize("components", [(1 + x,), (x, y), (1j * x, y)])
def test_as_oneform_accepts_component_tuples(components, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    result = wrappers.as_oneform(components)
    assert isinstance(result, dg.OneForm)
    assert tuple(result.dims) == (len(components),)
    assert result.dim_space == len(components)
    expected = CF(components, dims=(len(components),))
    assert_l2_close(Norm(result - expected), CF(0), mesh, tol=1e-12)
