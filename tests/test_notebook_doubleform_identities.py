from math import factorial

import ngsolve
import ngsdiffgeo as dg
import pytest
from netgen.occ import Box, OCCGeometry
from ngsolve import (
    BBND,
    BND,
    CF,
    GridFunction,
    HCurlCurl,
    Id,
    Integrate,
    Mesh,
    TaskManager,
    VOL,
    specialcf,
    x,
    y,
    z,
)

from tests._helpers import l2_error, l2_norm


def _sample_doubleforms():
    alpha = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
    beta = dg.OneForm(CF((0.3 * z * y, x * z**2, y**2)))
    gamma = dg.OneForm(
        CF((0.3 * y * z - x * y, x**2 * z + 0.34 * y**3, -x * y * z))
    )
    delta = dg.OneForm(CF((z + x, y - z, x * y)))

    f = dg.ScalarField(20 * x * y * (1 - y) * (1 - z), dim=3)
    a00 = dg.DoubleForm(f, p=0, q=0, dim=3)
    b11 = dg.DoubleForm(dg.Einsum("i,j->ij", alpha, beta), p=1, q=1, dim=3)
    c11 = dg.DoubleForm(dg.Einsum("i,j->ij", beta, gamma), p=1, q=1, dim=3)
    a11 = dg.DoubleForm(dg.Einsum("i,j->ij", gamma, delta), p=1, q=1, dim=3)
    a22 = dg.Wedge(b11, c11)

    return a00, b11, c11, a11, a22


def _heisenberg_manifold(mesh):
    gf_metric = GridFunction(HCurlCurl(mesh, order=2))
    gf_metric.Set(dg.Heisenberg().metric)
    return dg.RiemannianManifold(
        gf_metric, normal_sign=-1, change_riemann_sign=True
    )


def _warped_product_manifold(mesh):
    gf_metric = GridFunction(HCurlCurl(mesh, order=2))
    gf_metric.Set(dg.WarpedProduct().metric)
    return dg.RiemannianManifold(
        gf_metric, normal_sign=-1, change_riemann_sign=True
    )


def _dev_doubleform(rm, phi):
    dim = 3
    p = phi.degree_left
    total = None
    for k in range(p + 1):
        term = (
            (-1) ** (p - k)
            * factorial(dim - 2 * p + 1)
            / (factorial(dim - p - k + 1) * factorial(p - k))
            * dg.Wedge(rm.Trace(phi, l=p - k), rm.G ** (p - k))
        )
        total = term if total is None else total + term
    return total


def _spherical_doubleform(rm, phi):
    dim = 3
    p = phi.degree_left
    if p == 0:
        chi = dg.ScalarField(0, dim=dim)
    else:
        chi = None
        for k in range(p):
            term = (
                (-1) ** (p - k + 1)
                * factorial(dim - 2 * p + 1)
                / (factorial(dim - p - k + 1) * factorial(p - k))
                * dg.Wedge(rm.Trace(phi, l=p - k), rm.G ** (p - k - 1))
            )
            chi = term if chi is None else chi + term
    return dg.Wedge(chi, rm.G)


def _f_op(rm, sigma, phi):
    return (
        dg.Wedge(rm.Trace(sigma), phi)
        + dg.Wedge(sigma, rm.Trace(phi))
        - rm.Trace(dg.Wedge(sigma, phi))
    )


def _h_t(rm, phi):
    return 0.5 * (
        rm.d_cov(rm.d_cov(phi, slot="right"), slot="left")
        + rm.d_cov(rm.d_cov(phi, slot="left"), slot="right")
    )


def _h_t_star(rm, phi):
    return 0.5 * (
        rm.delta_cov(rm.delta_cov(phi, slot="right"), slot="left")
        + rm.delta_cov(rm.delta_cov(phi, slot="left"), slot="right")
    )


def _distributional_curvature(rm, p, phi):
    if p != 0 and (phi.degree_left != p or phi.degree_right != p):
        raise ValueError("phi must be a (p,p) double form.")
    if p == 0 and not isinstance(phi, dg.ScalarField):
        raise ValueError("For p=0, phi must be a ScalarField.")

    vol_term = (
        rm.SlotInnerProduct(dg.Wedge(rm.Riemann, phi), vb=VOL)
        * rm.VolumeForm(VOL)
        * ngsolve.dx
    )
    bnd_term = (
        rm.SlotInnerProduct(
            dg.Wedge(rm.SFF, rm.ProjectDoubleForm(phi, left="F", right="F")),
            vb=BND,
        )
        * rm.VolumeForm(BND)
        * ngsolve.dx(element_boundary=True)
    )
    bbnd_term = (
        rm.SlotInnerProduct(
            dg.Wedge(
                rm.AngleDefect,
                rm.ProjectDoubleForm(phi, left="E", right="E"),
            ),
            vb=BBND,
        )
        * rm.VolumeForm(BBND)
        * ngsolve.dx(element_vb=BBND)
    )
    return vol_term + bnd_term + bbnd_term


def _edge_angle_defect(rm):
    bbnd_tang = specialcf.EdgeFaceTangentialVectors(3)
    return dg.ScalarField(
        ngsolve.acos(bbnd_tang[:, 0] * bbnd_tang[:, 1])
        - ngsolve.acos(rm.G[rm.EdgeConormal(0), rm.EdgeConormal(1)]),
        dim=3,
    )


def _distributional_scalar_curvature(rm, v):
    theta = _edge_angle_defect(rm)
    return (
        rm.Scalar * v * rm.VolumeForm(VOL) * ngsolve.dx
        + 2
        * rm.MeanCurvature
        * v
        * rm.VolumeForm(BND)
        * ngsolve.dx(element_boundary=True)
        + 2
        * rm.SlotInnerProduct(theta * v, vb=VOL)
        * rm.VolumeForm(BBND)
        * ngsolve.dx(element_vb=BBND)
    )


def _distributional_einstein_tensor(rm, sigma):
    theta = _edge_angle_defect(rm)
    trace_reversed_sff = rm.SFF - rm.MeanCurvature * rm.G_F
    return (
        rm.InnerProduct(rm.Einstein, sigma, forms=True)
        * rm.VolumeForm(VOL)
        * ngsolve.dx
        + rm.InnerProduct(
            trace_reversed_sff,
            rm.ProjectDoubleForm(sigma, left="F", right="F"),
            forms=True,
        )
        * rm.VolumeForm(BND)
        * ngsolve.dx(element_boundary=True)
        - theta
        * rm.Trace(rm.ProjectDoubleForm(sigma, left="E", right="E"), vb=BBND)
        * rm.VolumeForm(BBND)
        * ngsolve.dx(element_vb=BBND)
    )


def test_notebook_curvature_and_boundary_symmetry_identities(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=2)
    rm = _heisenberg_manifold(mesh)
    riem = rm.Riemann
    ricci = rm.Trace(riem)
    scalar = rm.Trace(riem, l=2)
    einstein = ricci - 0.5 * scalar * rm.G

    with TaskManager():
        assert l2_norm(rm.s(riem), mesh, bonus_intorder=3) < 1e-12
        assert l2_norm(rm.s(ricci), mesh, bonus_intorder=3) < 1e-12
        assert l2_norm(rm.s(einstein), mesh, bonus_intorder=3) < 1e-12
        assert l2_norm(rm.s(rm.SFF, vb=BND), mesh, vb=BND, bonus_intorder=3) < 1e-10

        star_riem = rm.star(riem)
        star_formula = 0.5 * scalar * rm.G - dg.Wedge(ricci, rm.G**0)
        assert l2_error(star_riem, star_formula, mesh, bonus_intorder=3) < 1e-12


def test_notebook_trace_free_and_spherical_decomposition(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.8)
    rm = dg.RiemannianManifold(Id(3))
    sample_forms = _sample_doubleforms()

    with TaskManager():
        for phi in (sample_forms[0], sample_forms[1], sample_forms[4]):
            dev_phi = _dev_doubleform(rm, phi)
            assert l2_norm(rm.Trace(dev_phi), mesh, bonus_intorder=3) < 1e-12
            assert (
                l2_error(
                    phi - dev_phi,
                    _spherical_doubleform(rm, phi),
                    mesh,
                    bonus_intorder=3,
                )
                < 1e-12
            )


def test_notebook_f_operator_trace_slot_and_adjoint_identities(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=0.8)
    rm = dg.RiemannianManifold(Id(3))
    _, b11, c11, a11, _ = _sample_doubleforms()

    sigma = c11 + c11.trans
    phi = b11 + b11.trans
    psi = a11 + a11.trans

    with TaskManager():
        f_phi = _f_op(rm, sigma, phi)
        trace_formula = 2 * rm.TraceSigma(phi, sigma) + _f_op(
            rm, sigma, rm.Trace(phi)
        )
        slot_formula = 2 * rm.SlotInnerProduct(rm.TraceSigma(phi, sigma))

        assert (
            l2_error(rm.Trace(f_phi), trace_formula, mesh, bonus_intorder=3)
            < 1e-12
        )
        assert (
            l2_error(rm.SlotInnerProduct(f_phi), slot_formula, mesh, bonus_intorder=3)
            < 1e-12
        )
        assert (
            l2_error(_f_op(rm, rm.G, phi), 2 * phi, mesh, bonus_intorder=3)
            < 1e-12
        )

        left = rm.InnerProduct(f_phi, psi, forms=True)
        right = rm.InnerProduct(phi, _f_op(rm, sigma, psi), forms=True)
        assert l2_error(left, right, mesh, bonus_intorder=3) < 1e-12


def test_notebook_ht_star_matches_star_conjugated_ht(make_unit_cube_mesh):
    mesh = make_unit_cube_mesh(maxh=2)
    rm = _heisenberg_manifold(mesh)
    _, b11, _, _, a22 = _sample_doubleforms()

    with TaskManager():
        for phi in (b11, a22):
            p = phi.degree_left - 1
            q = phi.degree_right - 1
            right = (-1) ** (p + q) * rm.inv_star(_h_t(rm, rm.star(phi)))
            assert (
                l2_error(_h_t_star(rm, phi), right, mesh, bonus_intorder=3)
                < 1e-6
            )


def test_notebook_distributional_curvature_identities():
    shape = Box((0.3, 0.3, 0.3), (1, 1, 1))
    mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=2))
    rm = _warped_product_manifold(mesh)
    v = dg.ScalarField(20 * x * y * (1 - y) * (1 - z), dim=3)
    _, b11, _, _, _ = _sample_doubleforms()

    with TaskManager():
        left = Integrate(_distributional_curvature(rm, 1, dg.Wedge(v, rm.G)), mesh)
        right = Integrate(_distributional_curvature(rm, 0, v), mesh)
        assert left == pytest.approx(right, abs=1e-12)

        left = 2 * Integrate(_distributional_curvature(rm, 0, v), mesh)
        right = Integrate(_distributional_scalar_curvature(rm, v), mesh)
        assert left == pytest.approx(right, abs=1e-12)

        left = -Integrate(_distributional_curvature(rm, 1, b11), mesh)
        right = Integrate(_distributional_einstein_tensor(rm, b11), mesh)
        assert left == pytest.approx(right, abs=1e-12)
