import pytest
from netgen.occ import unit_square, unit_cube
from ngsolve import *

import ngsdiffgeo as dg


def l2_error(a, b, mesh):
    return sqrt(Integrate(InnerProduct(a - b, a - b), mesh))


def l2_norm(a, mesh):
    return sqrt(Integrate(InnerProduct(a, a), mesh))


@pytest.mark.parametrize("dim", [2, 3])
def test_star_star_identity(dim):
    mesh = (
        Mesh(unit_square.GenerateMesh(maxh=0.3))
        if dim == 2
        else Mesh(unit_cube.GenerateMesh(maxh=0.35))
    )
    rm = dg.RiemannianManifold(Id(dim))

    if dim == 2:
        dx = dg.OneForm(CF((1, 0)))
        dy = dg.OneForm(CF((0, 1)))
        forms = [
            (dg.ScalarField(CF(1), dim=dim), 0),
            (dx, 1),
            (dy, 1),
            (dg.Wedge(dx, dy), 2),
        ]
    else:
        dx = dg.OneForm(CF((1, 0, 0)))
        dy = dg.OneForm(CF((0, 1, 0)))
        dz = dg.OneForm(CF((0, 0, 1)))
        forms = [
            (dg.ScalarField(CF(1), dim=dim), 0),
            (dx, 1),
            (dy, 1),
            (dz, 1),
            (dg.Wedge(dy, dz), 2),
            (dg.Wedge(dz, dx), 2),
            (dg.Wedge(dx, dy), 2),
            (dg.Wedge(dx, dg.Wedge(dy, dz)), 3),
        ]

    for form, k in forms:
        sign = (-1) ** (k * (dim - k))
        ss = rm.star(rm.star(form))
        assert l2_error(ss, sign * form, mesh) < 1e-10


@pytest.mark.parametrize("dim", [2, 3])
def test_coderivative_degrees_and_zero(dim):
    mesh = (
        Mesh(unit_square.GenerateMesh(maxh=0.3))
        if dim == 2
        else Mesh(unit_cube.GenerateMesh(maxh=0.35))
    )
    rm = dg.RiemannianManifold(Id(dim))

    # δ on scalars returns typed zero
    f = dg.ScalarField(CF(1), dim=dim)
    delta_f = rm.delta(f)
    assert delta_f.degree == 0
    assert l2_norm(delta_f, mesh) < 1e-12

    # δ on constant 1-forms is zero and drops degree
    if dim == 2:
        dx = dg.OneForm(CF((1, 0)))
    else:
        dx = dg.OneForm(CF((1, 0, 0)))
    delta_dx = dg.delta(dx, rm)
    assert delta_dx.degree == 0
    assert l2_norm(delta_dx, mesh) < 1e-12

    # δ on a constant top-form is zero and has degree k-1
    if dim == 2:
        top = dg.Wedge(dx, dg.OneForm(CF((0, 1))))
    else:
        dy = dg.OneForm(CF((0, 1, 0)))
        dz = dg.OneForm(CF((0, 0, 1)))
        top = dg.Wedge(dx, dg.Wedge(dy, dz))
    delta_top = rm.delta(top)
    assert delta_top.degree == top.degree - 1
    assert l2_norm(delta_top, mesh) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
