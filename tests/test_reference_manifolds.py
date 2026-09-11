"""Regression tests for closed-form reference manifold data."""

import numpy as np
from netgen.occ import unit_cube, unit_square
from ngsolve import Mesh

import ngsdiffgeo as dg


def _value(cf, point):
    return np.asarray(cf(point), dtype=float)


def test_hyperbolic_h2_first_kind_christoffel_formula():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.8))
    point = mesh(0.2, 0.3)
    reference = dg.RiemannianManifold(dg.HyperbolicH2().metric)

    assert np.allclose(
        _value(dg.HyperbolicH2().chr1, point),
        _value(reference.Christoffel(second_kind=False), point),
        rtol=1e-12,
        atol=1e-12,
    )


def test_hyperbolic_h3_riemann_formula():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.9))
    point = mesh(0.2, 0.3, 0.4)
    model = dg.HyperbolicH3()
    reference = dg.RiemannianManifold(model.metric)

    assert np.allclose(
        _value(model.Riemann, point),
        _value(reference.Riemann, point),
        rtol=2e-9,
        atol=1e-7,
    )
