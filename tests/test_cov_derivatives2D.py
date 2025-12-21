import ngsdiffgeo as dg
import pytest
from netgen.occ import unit_square
from ngsolve import *

order = 6
addorder = 6


def CovDerS(s, mesh, gfgamma, cov=True):
    Xgf = GridFunction(L2(mesh, order=order + addorder))
    Xgf.Set(s)
    if cov:
        return Grad(Xgf)
    else:
        return Inv(gfgamma) * Grad(Xgf)


def CovDerV(X, mesh, gfgamma, contra=True):
    Xgf = GridFunction(VectorL2(mesh, order=order + addorder))
    Xgf.Set(X)
    christoffel2 = gfgamma.Operator("christoffel2")
    if contra:
        return Grad(Xgf).trans + christoffel2[X, :, :]
    else:
        return (Grad(Xgf) - christoffel2 * X).trans


def CovDerT(A, mesh, gfgamma, contra=[True, True]):
    chr2 = gfgamma.Operator("christoffel2")
    Xgf = GridFunction(MatrixValued(L2(mesh, order=order + addorder)))
    Xgf.Set(A)
    term = fem.Einsum("ijk->kij", Grad(Xgf))
    for i, con in enumerate(contra):
        str_con = "ila,jk->ijk"
        str_cov = "ial,jk->ijk"
        if con:
            signature = list(str_con)
            signature[2] = str_con[4 + i]
            signature[4 + i] = str_con[1]
            # print("con signature", "".join(signature))
            term = term + fem.Einsum("".join(signature), chr2, A)
        else:
            signature = list(str_cov)
            signature[1] = str_cov[4 + i]
            signature[4 + i] = str_cov[2]
            # print("cov signature", "".join(signature))
            term = term - fem.Einsum("".join(signature), chr2, A)
    return term


def test_cov_der_scal():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    metric = dg.CigarSoliton().metric
    mf = dg.RiemannianManifold(metric=metric)
    gf_metric = GridFunction(HCurlCurl(mesh, order=5))
    gf_metric.Set(metric, dual=True)

    f = CoefficientFunction(x**2 * y - 0.1 * y * x)
    sf = dg.ScalarField(f, dim=2)

    term1 = CovDerS(f, mesh, gf_metric, cov=True)
    term2 = mf.CovDeriv(sf)
    assert Integrate(term1, mesh) == pytest.approx(Integrate(term2, mesh))

    return


def test_cov_der_vec():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.15))
    metric = dg.CigarSoliton().metric
    mf = dg.RiemannianManifold(metric=metric)
    gf_metric = GridFunction(HCurlCurl(mesh, order=5))
    gf_metric.Set(metric, dual=True)

    v = CF((10 * x * y**3 - x**2, y**4 * z * x - y))
    vv = dg.VectorField(v)
    ov = dg.OneForm(v)

    term1 = CovDerV(v, mesh, gf_metric, contra=True)
    term2 = mf.CovDeriv(vv)
    assert sqrt(Integrate(InnerProduct(term1 - term2, term1 - term2), mesh)) < 5e-7

    term1 = CovDerV(v, mesh, gf_metric, contra=False)
    term2 = mf.CovDeriv(ov)
    assert sqrt(Integrate(InnerProduct(term1 - term2, term1 - term2), mesh)) < 5e-7

    return


def test_cov_der_mat():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.15))
    metric = dg.CigarSoliton().metric
    mf = dg.RiemannianManifold(metric=metric)
    gf_metric = GridFunction(HCurlCurl(mesh, order=5))
    gf_metric.Set(metric, dual=True)

    A = CF((10 * x * y**3 - x**2, y**4 * x - y, sin(x * y), cos(x) * y**2), dims=(2, 2))
    Acon = dg.TensorField(A, "00")
    Acov = dg.TensorField(A, "11")
    Amix1 = dg.TensorField(A, "10")
    Amix2 = dg.TensorField(A, "01")

    term1 = CovDerT(A, mesh, gf_metric, contra=[True, True])
    term2 = mf.CovDeriv(Acon)
    assert sqrt(Integrate(InnerProduct(term1 - term2, term1 - term2), mesh)) < 5e-7

    term1 = CovDerT(A, mesh, gf_metric, contra=[False, False])
    term2 = mf.CovDeriv(Acov)
    assert sqrt(Integrate(InnerProduct(term1 - term2, term1 - term2), mesh)) < 5e-7

    term1 = CovDerT(A, mesh, gf_metric, contra=[False, True])
    term2 = mf.CovDeriv(Amix1)
    assert sqrt(Integrate(InnerProduct(term1 - term2, term1 - term2), mesh)) < 5e-7

    term1 = CovDerT(A, mesh, gf_metric, contra=[True, False])
    term2 = mf.CovDeriv(Amix2)
    assert sqrt(Integrate(InnerProduct(term1 - term2, term1 - term2), mesh)) < 5e-7

    return


def test_integration_by_parts_2d():
    metric = dg.CigarSoliton().metric
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.15))
    mf = dg.RiemannianManifold(metric)
    omega_T = dg.ScalarField(mf.VolumeForm(VOL), dim=2)
    omega_S = dg.ScalarField(mf.VolumeForm(BND), dim=2)

    f = dg.ScalarField(CF(x**2 * y - 0.1 * y * x), dim=2)
    g = dg.ScalarField(CF(x * y**2 + 0.1 * y * x - 0.73 * x), dim=2)

    X = dg.VectorField(CF((10 * x * y**3 - x**2, y**4 * x - y)))
    # Y = dg.VectorField(CF((y**2 * x - 0.1 * y * x**2, 10 * x * y**3 + x**2 - y)))

    alpha = dg.OneForm(CF((sin(x * y), cos(x) * y**2)))
    beta = dg.OneForm(
        CF((0.2 * y * x**2 + 0.37 * x**3, 2 * x**2 * y**2 + 0.1 * x**2 - 1.34 * y))
    )

    A = dg.TensorField(
        CF(
            (10 * x * y**3 - x**2, y**4 * x - y, sin(x * y), cos(x) * y**2), dims=(2, 2)
        ),
        "00",
    )
    B = dg.TensorProduct(alpha, X)
    C = dg.TensorProduct(alpha, beta)

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovDeriv(f), X) * omega_T * dx, mesh)
            - Integrate(
                -mf.InnerProduct(mf.CovDiv(X), f) * omega_T * dx
                + f * mf.InnerProduct(X, mf.normal) * omega_S * ds,
                mesh,
            )
        )
        < 1e-8
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovDeriv(X), A) * omega_T * dx, mesh)
            - Integrate(
                -mf.InnerProduct(mf.CovDiv(A), X) * omega_T * dx
                + mf.InnerProduct(dg.TensorProduct(mf.normal, X), A) * omega_S * ds,
                mesh,
            )
        )
        < 1e-8
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovCurl(X), f) * omega_T * dx, mesh)
            - Integrate(
                mf.InnerProduct(X, mf.CovRot(f)) * omega_T * dx
                + f * mf.InnerProduct(X, mf.tangent) * omega_S * ds,
                mesh,
            )
        )
        < 1e-8
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovCurl(C), X) * omega_T * dx, mesh)
            - Integrate(
                mf.InnerProduct(C, mf.CovRot(X)) * omega_T * dx
                + mf.InnerProduct(C, dg.TensorProduct(X, mf.tangent)) * omega_S * ds,
                mesh,
            )
        )
        < 5e-8
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovInc(C, True), f) * omega_T * dx, mesh)
            - 0.5
            * Integrate(
                mf.InnerProduct(mf.CovCurl(C), mf.CovRot(f)) * omega_T * dx
                + f
                * mf.InnerProduct(mf.CovCurl(C), mf.tangent)
                * omega_S
                * dx(element_boundary=True),
                mesh,
            )
            - 0.5
            * Integrate(
                mf.InnerProduct(C, mf.CovRot(mf.CovRot(f))) * omega_T * dx
                + (
                    mf.InnerProduct(C, dg.TensorProduct(mf.CovRot(f), mf.tangent))
                    + f * mf.InnerProduct(mf.CovCurl(C), mf.tangent)
                )
                * omega_S
                * dx(element_boundary=True),
                mesh,
            )
        )
        < 1e-7
    )

    assert (
        abs(
            Integrate(
                mf.InnerProduct(mf.CovDiv(B), mf.CovDeriv(f)) * omega_T * dx, mesh
            )
            - Integrate(
                -mf.InnerProduct(B, mf.CovHesse(f)) * omega_T * dx
                + mf.InnerProduct(B, dg.TensorProduct(mf.normal, mf.CovDeriv(f)))
                * omega_S
                * dx(element_boundary=True),
                mesh,
            )
        )
        < 1e-7
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.Trace(mf.CovHesse(f)), g) * omega_T * dx, mesh)
            - Integrate(
                -mf.InnerProduct(mf.CovDeriv(f), mf.CovDeriv(g)) * omega_T * dx
                + mf.InnerProduct(mf.CovDeriv(f), mf.normal)
                * g
                * omega_S
                * dx(element_boundary=True),
                mesh,
            )
        )
        < 5e-8
    )
    return


def test_integration_by_parts_3d():
    metric = dg.WarpedProduct().metric
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.1))
    mf = dg.RiemannianManifold(metric)
    omega_T = mf.VolumeForm(VOL)
    omega_S = mf.VolumeForm(BND)

    X = dg.VectorField(CF((10 * x * y**3 - x**2 * z, y**4 * x - y * z, z * x * y)))
    Y = dg.VectorField(
        CF(
            (
                y**2 * x * z - 0.1 * y * x**2,
                10 * x * y**3 + x**2 - y * z,
                sin(z) + cos(x * y),
            )
        )
    )

    alpha = dg.OneForm(CF((sin(x * y), cos(x) * y**2, 2 * x * y * z)))
    beta = dg.OneForm(
        CF(
            (
                0.2 * y * x**2 + 0.37 * x**3 * z,
                2 * x**2 * y**2 + 0.1 * x**2 - 1.34 * y * z,
                cos(x * y) + 0.1 * z,
            )
        )
    )

    assert (
        abs(
            Integrate(
                mf.InnerProduct(mf.CovCurl(X), Y) * omega_T * dx(bonus_intorder=1), mesh
            )
            - Integrate(
                mf.InnerProduct(X, mf.CovCurl(Y)) * omega_T * dx(bonus_intorder=1)
                - mf.InnerProduct(mf.Cross(X, mf.normal), Y)
                * omega_S
                * ds(bonus_intorder=1),
                mesh,
            )
        )
        < 1e-8
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovCurl(X), alpha) * omega_T * dx, mesh)
            - Integrate(
                mf.InnerProduct(X, mf.CovCurl(alpha)) * omega_T * dx
                - mf.InnerProduct(mf.Cross(X, mf.normal), alpha) * omega_S * ds,
                mesh,
            )
        )
        < 1e-7
    )

    assert (
        abs(
            Integrate(mf.InnerProduct(mf.CovCurl(alpha), beta) * omega_T * dx, mesh)
            - Integrate(
                mf.InnerProduct(alpha, mf.CovCurl(beta)) * omega_T * dx
                - mf.InnerProduct(mf.Cross(alpha, mf.normal), beta) * omega_S * ds,
                mesh,
            )
        )
        < 1e-8
    )
    return


if __name__ == "__main__":
    pytest.main([__file__])
