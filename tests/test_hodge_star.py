import pytest
from netgen.occ import unit_square, unit_cube
from ngsolve import *

import ngsdiffgeo as dg


def l2_error(a, b, mesh):
    return sqrt(Integrate(InnerProduct(a - b, a - b), mesh))


def l2_norm(a, mesh):
    return sqrt(Integrate(InnerProduct(a, a), mesh))


def test_hodge_star_2d_euclidean():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    dim = 2
    rm = dg.RiemannianManifold(Id(dim))

    dx = dg.OneForm(CF((1, 0)))
    dy = dg.OneForm(CF((0, 1)))

    star_dx = dg.star(dx, rm)
    star_dy = dg.star(dy, rm)

    assert l2_error(star_dx, dy, mesh) < 1e-12
    assert l2_error(star_dy, -dx, mesh) < 1e-12

    # star of 1 should be the (oriented) area form matching dx^dy with our normalization
    area = dg.Wedge(dx, dy)
    star_one = dg.star(dg.ScalarField(CF(1), dim=dim), rm)
    assert l2_error(star_one, area, mesh) < 1e-12

    # star of top-form returns scalar
    back_to_scalar = dg.star(star_one, rm)
    assert l2_error(back_to_scalar, dg.ScalarField(CF(1), dim=dim), mesh) < 1e-12

    # star-star identity: (-1)^{k(n-k)} for k=1,n=2 gives -1
    ss = dg.star(star_dx, rm)
    assert l2_error(ss, -dx, mesh) < 1e-12


def test_hodge_star_3d_euclidean():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.35))
    dim = 3
    rm = dg.RiemannianManifold(Id(dim))

    dx = dg.OneForm(CF((1, 0, 0)))
    dy = dg.OneForm(CF((0, 1, 0)))
    dz = dg.OneForm(CF((0, 0, 1)))

    expected_star_dx = dg.Wedge(dy, dz)
    star_dx = dg.star(dx, rm)
    assert l2_error(star_dx, expected_star_dx, mesh) < 1e-12

    # star-star identity with k=1, n=3 gives +1
    ss = dg.star(star_dx, rm)
    assert l2_error(ss, dx, mesh) < 1e-12

    # star of zero preserves typing and degree
    zero_form = dg.OneForm(CF((0, 0, 0)))
    star_zero = dg.star(zero_form, rm)
    assert isinstance(star_zero, dg.KForm)
    assert star_zero.degree == 2
    assert l2_norm(star_zero, mesh) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
