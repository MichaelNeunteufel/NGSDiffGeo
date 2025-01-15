import pytest



def test_import():
    # you need to import ngsolve before importing the extension
    # such that all runtime dependencies are loaded
    import ngsolve
    import ngsdiffgeo


def test_riemann_mf():
    from ngsdiffgeo import RiemannianManifold
    from netgen.occ import unit_square
    import ngsolve as ng
    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.1))
    g = 1 * ng.Id(2)
    mf = RiemannianManifold(metric=g)
    assert ng.Integrate(mf.dx, mesh) == pytest.approx(1.0)
    assert ng.Integrate(mf.ds, mesh, ng.BND) == pytest.approx(4.0)
