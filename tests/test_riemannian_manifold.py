import pytest

from ngsolve import *
from netgen.occ import unit_square, unit_cube
import ngsdiffgeo as dg

xvec = [x, y, z]


def GetDiffOp(name, cf):
    dims = cf.dims
    dim = cf.dims[0]
    if name == "grad":
        if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
            return CF(tuple([cf.Diff(xvec[i]) for i in range(dim)]))
        elif len(cf.dims) == 1:
            return CF(
                tuple(
                    [
                        cf[i].Diff(xvec[j])
                        for i in range(cf.dims[0])
                        for j in range(cf.dims[0])
                    ]
                ),
                dims=(cf.dims[0], cf.dims[0]),
            )
        else:
            return CF(
                tuple(
                    [
                        cf[j, k].Diff(xvec[i])
                        for i in range(cf.dims[0])
                        for j in range(cf.dims[0])
                        for k in range(cf.dims[0])
                    ]
                ),
                dims=(cf.dims[0], cf.dims[0] ** 2),
            )
    elif name == "christoffel":
        cfgrad = GetDiffOp("grad", cf)
        return 0.5 * CF(
            tuple(
                [
                    cfgrad[i, j + dim * k]
                    + cfgrad[j, i + dim * k]
                    - cfgrad[k, i + dim * j]
                ]
                for i in range(dim)
                for j in range(dim)
                for k in range(dim)
            ),
            dims=(dim, dim, dim),
        )
    elif name == "christoffel2":
        chr1 = GetDiffOp("christoffel", cf)
        return CF((CF((chr1), dims=(dim**2, dim)) * Inv(cf)), dims=(dim, dim, dim))
    else:
        raise Exception(
            "In GetDiffOp: Something went wrong: name =",
            name,
            ", order =",
            order,
            ", dim =",
            dim,
            ", dim =",
            dims,
            ", sym =",
            sym,
            ", dev =",
            dev,
            ", vb =",
            vb,
        )


@pytest.mark.parametrize("interpolate", [True, False])
def test_volume_forms_vectors_2D(interpolate):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    cf_metric = dg.CigarSoliton().metric

    if interpolate:
        metric = GridFunction(HCurlCurl(mesh, order=2))
        metric.Set(cf_metric, dual=True)
    else:
        metric = cf_metric
    mf = dg.RiemannianManifold(metric=metric)
    tang = specialcf.tangential(mesh.dim)
    nv = specialcf.normal(2)
    assert sqrt(
        Integrate((mf.VolumeForm(VOL) - sqrt(Det(metric))) ** 2, mesh)
    ) == pytest.approx(0)
    print(type(metric))
    assert sqrt(
        Integrate(
            (mf.VolumeForm(BND) - sqrt(metric[tang, tang])) ** 2
            * dx(element_boundary=True),
            mesh,
        )
    ) == pytest.approx(0)

    assert sqrt(
        Integrate(
            (mf.normal - 1 / sqrt(Inv(metric) * nv * nv) * Inv(metric) * nv) ** 2
            * dx(element_boundary=True),
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.tangent - 1 / sqrt(metric[tang, tang]) * tang) ** 2
            * dx(element_boundary=True),
            mesh,
        )
    ) == pytest.approx(0)

    return


@pytest.mark.parametrize("interpolate", [True, False])
def test_volume_forms_vectors_3D(interpolate):
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.3))
    cf_metric = dg.WarpedProduct().metric

    if interpolate:
        metric = GridFunction(HCurlCurl(mesh, order=2))
        metric.Set(cf_metric, dual=True)
    else:
        metric = cf_metric
    mf = dg.RiemannianManifold(metric=metric)
    nv = specialcf.normal(mesh.dim)
    assert sqrt(
        Integrate((mf.VolumeForm(VOL) - sqrt(Det(metric))) ** 2, mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.VolumeForm(BND) - sqrt(Cof(metric) * nv * nv)) ** 2
            * dx(element_boundary=True),
            mesh,
        )
    ) == pytest.approx(0)
    # TODO: test for BBND

    assert sqrt(
        Integrate(
            (mf.normal - 1 / sqrt(Inv(metric) * nv * nv) * Inv(metric) * nv) ** 2
            * dx(element_boundary=True),
            mesh,
        )
    ) == pytest.approx(0)

    # TODO: test tangent

    return


@pytest.mark.parametrize("interpolate", [True, False])
def test_metric_inverse_derivative2D(interpolate):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    cf_metric = dg.TestMetric(dim=2, order=4)
    if interpolate:
        metric = GridFunction(HCurlCurl(mesh, order=2))
        metric.Set(cf_metric, dual=True)
    else:
        metric = cf_metric
    mf = dg.RiemannianManifold(metric=metric)

    assert sqrt(
        Integrate(InnerProduct(mf.G - metric, mf.G - metric), mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(InnerProduct(mf.G_inv - Inv(metric), mf.G_inv - Inv(metric)), mesh)
    ) == pytest.approx(0)

    g_grad = (
        GetDiffOp("grad", cf=metric) if not interpolate else metric.Operator("grad")
    )
    assert (
        sqrt(Integrate(InnerProduct(mf.G_deriv - g_grad, mf.G_deriv - g_grad), mesh))
        < 1e-10
    )
    chr1 = (
        GetDiffOp(name="christoffel", cf=metric)
        if not interpolate
        else metric.Operator("christoffel")
    )

    assert (
        sqrt(
            Integrate(
                InnerProduct(
                    mf.Christoffel(second_kind=False) - chr1,
                    mf.Christoffel(second_kind=False) - chr1,
                ),
                mesh,
            )
        )
        < 1e-10
    )
    chr2 = (
        GetDiffOp(name="christoffel2", cf=metric)
        if not interpolate
        else metric.Operator("christoffel2")
    )
    assert (
        sqrt(
            Integrate(
                InnerProduct(
                    mf.Christoffel(second_kind=True) - chr2,
                    mf.Christoffel(second_kind=True) - chr2,
                ),
                mesh,
            )
        )
        < 1e-10
    )

    return


@pytest.mark.parametrize("interpolate", [True, False])
def test_metric_inverse_derivative3D(interpolate):
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.3))
    cf_metric = dg.TestMetric(dim=3, order=4)  # WarpedProduct

    if interpolate:
        metric = GridFunction(HCurlCurl(mesh, order=2))
        metric.Set(cf_metric, dual=True)
    else:
        metric = cf_metric

    mf = dg.RiemannianManifold(metric=metric)

    assert sqrt(
        Integrate(InnerProduct(mf.G - metric, mf.G - metric), mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(InnerProduct(mf.G_inv - Inv(metric), mf.G_inv - Inv(metric)), mesh)
    ) == pytest.approx(0)

    g_grad = (
        GetDiffOp("grad", cf=metric) if not interpolate else metric.Operator("grad")
    )
    assert (
        sqrt(Integrate(InnerProduct(mf.G_deriv - g_grad, mf.G_deriv - g_grad), mesh))
        < 1e-10
    )
    chr1 = (
        GetDiffOp(name="christoffel", cf=metric)
        if not interpolate
        else metric.Operator("christoffel")
    )

    assert (
        sqrt(
            Integrate(
                InnerProduct(
                    mf.Christoffel(second_kind=False) - chr1,
                    mf.Christoffel(second_kind=False) - chr1,
                ),
                mesh,
            )
        )
        < 1e-10
    )
    chr2 = (
        GetDiffOp(name="christoffel2", cf=metric)
        if not interpolate
        else metric.Operator("christoffel2")
    )
    assert (
        sqrt(
            Integrate(
                InnerProduct(
                    mf.Christoffel(second_kind=True) - chr2,
                    mf.Christoffel(second_kind=True) - chr2,
                ),
                mesh,
            )
        )
        < 1e-10
    )

    return


@pytest.mark.parametrize("interpolate", [True, False])
def test_inner_product(interpolate):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    cf_metric = dg.CigarSoliton().metric
    if interpolate:
        metric = GridFunction(HCurlCurl(mesh, order=2))
        metric.Set(cf_metric, dual=True)
    else:
        metric = cf_metric
    mf = dg.RiemannianManifold(metric=metric)

    f = CoefficientFunction(x**2 * y - 0.1 * y * x)
    g = CoefficientFunction(x * y**2 + 0.1 * y * x - 0.73 * x)

    fs = dg.ScalarField(f, dim=2)
    gs = dg.ScalarField(g, dim=2)

    v = CF((10 * x * y**3 - x**2, y**4 * x - y))
    w = CF((y**2 * x - 0.1 * y * x**2, 10 * x * y**3 + x**2 - y))

    vv = dg.VectorField(v)
    vo = dg.OneForm(v)
    wv = dg.VectorField(w)
    wo = dg.OneForm(w)

    A = CF((10 * x * y**3 - x**2, y**4 * x - y, sin(x * y), cos(x) * y**2), dims=(2, 2))
    Acon = dg.TensorField(A, "00")
    Acov = dg.TensorField(A, "11")
    Amix1 = dg.TensorField(A, "10")
    Amix2 = dg.TensorField(A, "01")

    B = OuterProduct(v, w)
    Bcon = dg.TensorProduct(vv, wv)
    Bcov = dg.TensorProduct(vo, wo)
    Bmix1 = dg.TensorProduct(vo, wv)
    Bmix2 = dg.TensorProduct(vv, wo)

    assert sqrt(
        Integrate((mf.InnerProduct(fs, gs) - InnerProduct(f, g)) ** 2, mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate((mf.InnerProduct(vv, wo) - InnerProduct(v, w)) ** 2, mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(vv, wv) - metric[w, v]) ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(vo, wo) - Inv(metric)[w, v]) ** 2,
            mesh,
        )
    ) == pytest.approx(0)

    assert sqrt(
        Integrate((mf.InnerProduct(Acon, Bcov) - InnerProduct(A, B)) ** 2, mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Acon, Bcon) - InnerProduct(metric * A * metric, B)) ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (
                mf.InnerProduct(Acov, Bcov)
                - InnerProduct(Inv(metric) * A * Inv(metric), B)
            )
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate((mf.InnerProduct(Amix2, Bmix1) - InnerProduct(A, B)) ** 2, mesh)
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Amix1, Bmix1) - InnerProduct(Inv(metric) * A * metric, B))
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Amix2, Bmix2) - InnerProduct(metric * A * Inv(metric), B))
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Amix2, Bcon) - InnerProduct(metric * A, B)) ** 2, mesh
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Acov, Bmix2) - InnerProduct(A * Inv(metric), B)) ** 2, mesh
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Acon, Bmix1) - InnerProduct(A * metric, B)) ** 2, mesh
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (mf.InnerProduct(Amix1, Bcov) - InnerProduct(Inv(metric) * A, B)) ** 2, mesh
        )
    ) == pytest.approx(0)

    return


@pytest.mark.parametrize("interpolate", [True, False])
def test_contraction(interpolate):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    cf_metric = dg.CigarSoliton().metric

    if interpolate:
        metric = GridFunction(HCurlCurl(mesh, order=2))
        metric.Set(cf_metric, dual=True)
    else:
        metric = cf_metric

    mf = dg.RiemannianManifold(metric=metric)
    v = CF((10 * x * y**3 - x**2, y**4 * x - y))
    w = CF((y**2 * x - 0.1 * y * x**2, 10 * x * y**3 + x**2 - y))

    vv = dg.VectorField(v)
    vo = dg.OneForm(v)
    wv = dg.VectorField(w)
    wo = dg.OneForm(w)

    A = CF((10 * x * y**3 - x**2, y**4 * x - y, sin(x * y), cos(x) * y**2), dims=(2, 2))
    Acon = dg.TensorField(A, "00")
    Acov = dg.TensorField(A, "11")
    Amix1 = dg.TensorField(A, "10")
    Amix2 = dg.TensorField(A, "01")

    B = OuterProduct(v, w)

    assert sqrt(
        Integrate(
            (
                mf.InnerProduct(wv, mf.Contraction(Acon, vv, 0))
                - mf.InnerProduct(dg.TensorProduct(vv, wv), Acon)
            )
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (
                mf.InnerProduct(wv, mf.Contraction(Acon, vv, 1))
                - mf.InnerProduct(dg.TensorProduct(wv, vv), Acon)
            )
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (
                mf.InnerProduct(wv, mf.Contraction(Acov, vv, 0))
                - mf.InnerProduct(dg.TensorProduct(vv, wv), Acov)
            )
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    assert sqrt(
        Integrate(
            (
                mf.InnerProduct(wv, mf.Contraction(Acov, vv, 1))
                - mf.InnerProduct(dg.TensorProduct(wv, vv), Acov)
            )
            ** 2,
            mesh,
        )
    ) == pytest.approx(0)
    return


if __name__ == "__main__":
    pytest.main([__file__])
