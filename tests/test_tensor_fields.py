import pytest

from ngsolve import *
from netgen.occ import unit_square
import ngsdiffgeo as dg


def l2_inner(a, b, mesh):
    return sqrt(Integrate(InnerProduct(a - b, a - b), mesh))


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


if __name__ == "__main__":
    pytest.main([__file__])
