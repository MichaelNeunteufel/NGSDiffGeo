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
    cos,
    sin,
    sqrt,
    x,
    y,
    z,
)
import ngsdiffgeo as dg

from tests._helpers import l2_inner


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

    with pytest.raises(Exception):
        dg.TensorField(A, "0")
    with pytest.raises(Exception):
        dg.TensorField(A, "0x")


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
