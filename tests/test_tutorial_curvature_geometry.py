"""Check tutorial geometry terms without running refinement studies or solvers."""
import json
from pathlib import Path

import ngsolve as ngs
import ngsolve.fem as fem
import ngsdiffgeo as dg
import pytest
from ngsolve.meshes import MakeStructured3DMesh


TUTORIAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
SCALAR_CURVATURE_TUTORIAL = (
    TUTORIAL_DIR / "09_distributional_scalar_curvature.ipynb"
)
RIEMANN_CURVATURE_TUTORIAL = (
    TUTORIAL_DIR / "11_distributional_riemann_curvature_tensor.ipynb"
)


@pytest.fixture
def metric_mesh():
    mesh = MakeStructured3DMesh(False, nx=1, ny=1, nz=1)
    g = ngs.CF((2+ngs.x**2, .2, .1, .2, 1.5+ngs.y**2, .3,
                .1, .3, 1.2+ngs.z**2), dims=(3, 3))
    gf = ngs.GridFunction(ngs.HCurlCurl(mesh, order=2))
    gf.Set(g)
    return mesh, g, gf


def tutorial_terms(path, function, mesh, metric):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    source = next(''.join(c['source']) for c in cells
                  if c['cell_type'] == 'code' and 'def '+function+'(' in ''.join(c['source']))
    env = dict(vars(ngs), dg=dg, fem=fem, G_ex=metric, S_ex=0,
               Q_ex=ngs.CF((0,)*9, dims=(3, 3)),
               ComputeHm2Norm=lambda vol, bnd, edge, *args, **kwargs: (vol, bnd, edge))
    exec(compile(source, str(path), 'exec'), env)
    return env[function](mesh, metric, order=2)[0]


def error(value, mesh, measure):
    return ngs.sqrt(ngs.Integrate(ngs.InnerProduct(value, value)*measure, mesh))


def test_riemann_tutorial_edge_angle_uses_metric_orthogonal_plane(metric_mesh):
    mesh, g, gf = metric_mesh
    edge = tutorial_terms(
        RIEMANN_CURVATURE_TUTORIAL, "ComputeCurvature", mesh, g
    )[2]
    tangents = ngs.specialcf.EdgeFaceTangentialVectors(3)
    a, b = tangents[:, 0], tangents[:, 1]
    t = ngs.specialcf.tangential(3, True)
    # Independently project into the metric-orthogonal complement of the edge.
    ag = a - (gf*a*t)/(gf*t*t)*t
    bg = b - (gf*b*t)/(gf*t*t)*t
    theta = ngs.acos(a*b) - ngs.acos((gf*ag*bg)/ngs.sqrt((gf*ag*ag)*(gf*bg*bg)))
    expected = theta/ngs.sqrt(gf*t*t)*ngs.OuterProduct(t, t)
    assert error(edge-expected, mesh, ngs.dx(element_vb=ngs.BBND)) < 1e-11


def test_scalar_tutorial_matches_standalone_geometry(metric_mesh):
    mesh, g, gf = metric_mesh
    vol, bnd, edge = tutorial_terms(
        SCALAR_CURVATURE_TUTORIAL, "ComputeScalarCurvature", mesh, g
    )
    inv = ngs.Inv(gf)
    n = ngs.specialcf.normal(3)
    ng = inv*n/ngs.sqrt(inv*n*n)
    p = inv-ngs.OuterProduct(ng, ng)
    h = ngs.InnerProduct(gf, p*(-gf.Operator('christoffel')*ng)*p)
    tangents = ngs.specialcf.EdgeFaceTangentialVectors(3)
    t = ngs.specialcf.tangential(3, True)
    n1, n2 = ngs.Cross(tangents[:, 0], t), ngs.Cross(tangents[:, 1], t)
    theta = ngs.acos(n1*n2)-ngs.acos((inv*n1*n2)/ngs.sqrt((inv*n1*n1)*(inv*n2*n2)))
    expected = (gf.Operator('scalar')*ngs.sqrt(ngs.Det(gf)),
                2*h*ngs.sqrt(ngs.Cof(gf)*n*n), 2*theta*ngs.sqrt(gf*t*t))
    for actual, ref, measure in zip((vol, bnd, edge), expected,
                                  (ngs.dx, ngs.dx(element_boundary=True), ngs.dx(element_vb=ngs.BBND))):
        assert error(actual-ref, mesh, measure) < 1e-11


def test_riemann_tutorial_facet_matches_tensor_cross(metric_mesh):
    mesh, g, gf = metric_mesh
    facet = tutorial_terms(
        RIEMANN_CURVATURE_TUTORIAL, "ComputeCurvature", mesh, g
    )[1]
    n = -ngs.specialcf.normal(3)
    ng = ngs.Inv(gf)*n/ngs.sqrt(ngs.Inv(gf)*n*n)
    eps = fem.LeviCivitaSymbol(3)
    ref = (ngs.sqrt(ngs.Cof(gf)*n*n)/ngs.Det(gf)
           * fem.Einsum('ikl,jmn,km,ln->ij', eps, eps,
                        ngs.OuterProduct(gf*ng, gf*ng), gf.Operator('christoffel')*ng))
    assert error(facet-ref, mesh, ngs.dx(element_boundary=True)) < 1e-11
