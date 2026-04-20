import pytest

import ngsdiffgeo as dg
from netgen.occ import unit_cube, unit_square
from ngsolve import Id, Mesh


@pytest.fixture
def make_unit_square_mesh():
    def _make(maxh=0.3):
        return Mesh(unit_square.GenerateMesh(maxh=maxh))

    return _make


@pytest.fixture
def make_unit_cube_mesh():
    def _make(maxh=0.5):
        return Mesh(unit_cube.GenerateMesh(maxh=maxh))

    return _make


@pytest.fixture
def rm_euclidean_2d():
    return dg.RiemannianManifold(Id(2))


@pytest.fixture
def rm_euclidean_3d():
    return dg.RiemannianManifold(Id(3))


@pytest.fixture
def cigar_metric():
    return dg.CigarSoliton().metric


@pytest.fixture
def warped_metric():
    return dg.WarpedProduct().metric
