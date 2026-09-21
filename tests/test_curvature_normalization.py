"""The geometric curvature operator is independent of metric representation."""
import ngsolve as ng
import ngsdiffgeo as dg
import pytest
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh


@pytest.fixture(params=[2, 3])
def metric_case(request):
    dim = request.param
    mesh = (MakeStructured2DMesh(False, nx=1, ny=1) if dim == 2 else
            MakeStructured3DMesh(False, nx=1, ny=1, nz=1))
    coords = (ng.x, ng.y, ng.z)
    g = ng.CF(tuple(2 + coords[i]**2 if i == j else 0.2
                    for i in range(dim) for j in range(dim)), dims=(dim, dim))
    fes = ng.HCurlCurl(mesh, order=2)
    gf = ng.GridFunction(fes)
    gf.Set(g)
    assert ng.sqrt(ng.Integrate(ng.InnerProduct(gf-g, gf-g), mesh)) < 1e-12
    return dim, mesh, g, gf


def l2(cf, mesh):
    return ng.sqrt(ng.Integrate(ng.InnerProduct(cf, cf), mesh))


@pytest.mark.parametrize('sign', [False, True])
def test_curvature_metric_representations_and_identities(metric_case, sign):
    dim, mesh, g, gf = metric_case
    cf_manifold = dg.RiemannianManifold(g, change_riemann_sign=sign)
    regge = dg.RiemannianManifold(gf, change_riemann_sign=sign)
    assert l2(cf_manifold.Curvature-regge.Curvature, mesh) < 2e-7
    assert l2(regge.Curvature-gf.Operator('curvature')/ng.Det(gf), mesh) < 1e-12
    for mf in (cf_manifold, regge):
        if dim == 2:
            assert isinstance(mf.Curvature, dg.ScalarField)
            assert l2(mf.Gauss-mf.Curvature, mesh) < 1e-12
            assert l2(mf.Gauss-0.5*mf.Scalar, mesh) < 2e-7
        else:
            assert mf.Curvature.covariant_indices == '00'
            assert l2(mf.Curvature + mf.Raise(mf.Einstein, [0, 1]), mesh) < 2e-7
        assert l2(mf.Curvature.Compile()-mf.Curvature, mesh) < 1e-12


@pytest.mark.parametrize('sign', [False, True])
def test_proxy_curvature_matches_gridfunction(metric_case, sign):
    dim, mesh, g, gf = metric_case
    trial = gf.space.TrialFunction()
    proxy = dg.RiemannianManifold(trial, change_riemann_sign=sign)
    regge = dg.RiemannianManifold(gf, change_riemann_sign=sign)
    assert tuple(proxy.Curvature.dims) == (() if dim == 2 else (dim, dim))
    # Energy supplies all nonlinear native proxy values from the metric state.
    energy = ng.BilinearForm(gf.space)
    energy += ng.Variation(ng.InnerProduct(proxy.Curvature, proxy.Curvature)*ng.dx(bonus_intorder=16))
    expected = ng.Integrate(ng.InnerProduct(regge.Curvature, regge.Curvature), mesh, order=20)
    assert energy.Energy(gf.vec) == pytest.approx(expected, rel=1e-10, abs=1e-12)
    normalized = ng.Integrate(ng.InnerProduct(gf.Operator('curvature'),
                                             gf.Operator('curvature'))/ng.Det(gf)**2, mesh, order=20)
    assert energy.Energy(gf.vec) == pytest.approx(normalized, rel=1e-10, abs=1e-12)
    if dim == 2:
        assert isinstance(proxy.Curvature, dg.ScalarField)
        gauss_energy = ng.BilinearForm(gf.space)
        gauss_energy += ng.Variation(proxy.Gauss**2*ng.dx(bonus_intorder=16))
        assert gauss_energy.Energy(gf.vec) == pytest.approx(normalized, rel=1e-10, abs=1e-12)


def test_inward_sff_matches_outward_paper_convention():
    mesh = MakeStructured3DMesh(False, nx=1, ny=1, nz=1)
    # For g = a(x)^2 I, the left face has outward-paper II_yy = -a'(0) = -1.
    g = (1 + ng.x)**2 * ng.Id(3)
    mf = dg.RiemannianManifold(g, normal_sign=-1)
    left = ng.IfPos(1e-8 - ng.x, 1, 0)
    faces = ng.dx(element_boundary=True)

    assert ng.Integrate(left * (mf.normal[0] - 1)**2 * faces, mesh) < 1e-10
    assert ng.Integrate(left * (mf.SFF.coef[1, 1] + 1)**2 * faces, mesh) < 1e-10
    assert ng.Integrate(left * (mf.MeanCurvature + 2)**2 * faces, mesh) < 1e-10

    normal_flat = mf.Lower(mf.normal)
    normal_square = dg.DoubleForm(
        dg.TensorProduct(normal_flat, normal_flat), p=1, q=1, dim=3
    )
    q_facet = mf.Raise(mf.star(dg.Wedge(normal_square, mf.SFF)), [0, 1])
    einstein_facet = mf.Raise(mf.S(mf.SFF), [0, 1], vb=ng.BND)
    difference = q_facet.coef + einstein_facet.coef
    assert ng.Integrate(ng.InnerProduct(difference, difference) * faces, mesh) < 1e-10
