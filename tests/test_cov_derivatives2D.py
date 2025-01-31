import pytest
from ngsolve import *
from netgen.occ import unit_square, unit_cube
import ngsdiffgeo as dg

order = 6
addorder = 6

"""
A = CF( (10*x*y**3-x**2, y**4*x-y, 
         sin(x*y), cos(x)*y**2), dims=(2,2) )
s = CF( 10*x*y**3-x**2*cos(x)*y**2)
v = CF( (10*x*y**3-x**2, y**4*z*x-y) )
sigma = Sym( CF( (5+x**2*y**3, sin(x)*cos(y), exp(x-y), 1+y**5), dims=(2,2) ) )
"""

def CovDerS(s, mesh, gfgamma, cov=True):
    Xgf = GridFunction(L2(mesh, order=order+addorder))
    Xgf.Set(s)
    if cov:
        return Grad(Xgf)
    else:
        return Inv(gfgamma)*Grad(Xgf)

def CovDerV(X, mesh, gfgamma, contra=True):
    Xgf = GridFunction(VectorL2(mesh, order=order+addorder))
    Xgf.Set(X)
    christoffel2 = gfgamma.Operator("christoffel2")
    if contra:
        return (Grad(Xgf).trans + christoffel2[X,:,:])
    else:
        return (Grad(Xgf) - christoffel2*X).trans
    
def CovDerT(A, mesh, gfgamma, contra=[True,True]):
    chr2 = gfgamma.Operator("christoffel2")
    Xgf = GridFunction(MatrixValued(L2(mesh, order=order+addorder)))
    Xgf.Set(A)
    term = fem.Einsum("ijk->kij", Grad(Xgf))
    for i, con in enumerate(contra):
        str_con = "ila,jk->ijk"
        str_cov = "ial,jk->ijk"
        if con:
            signature = list(str_con)
            signature[2] = str_con[4+i]
            signature[4+i] = str_con[1]  
            print("con signature", "".join(signature))
            term = term + fem.Einsum("".join(signature), chr2, A)
        else:
            signature = list(str_cov)
            signature[1] = str_cov[4+i]
            signature[4+i] = str_cov[2] 
            print("cov signature", "".join(signature))
            term = term - fem.Einsum("".join(signature), chr2, A)
    return term

def test_cov_der_scal():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    metric = dg.CigarSoliton().metric
    mf = dg.RiemannianManifold(metric=metric)
    gf_metric = GridFunction(HCurlCurl(mesh, order=5))
    gf_metric.Set (metric, dual=True)

    f = CoefficientFunction( x**2*y-0.1*y*x)
    sf = dg.ScalarField(f)

    term1 = CovDerS(f, mesh, gf_metric, cov=True)
    term2 = mf.CovDeriv(sf)
    assert Integrate(term1,mesh) == pytest.approx(Integrate(term2,mesh))

    return

def test_cov_der_vec():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    metric = dg.CigarSoliton().metric
    mf = dg.RiemannianManifold(metric=metric)
    gf_metric = GridFunction(HCurlCurl(mesh, order=5))
    gf_metric.Set (metric, dual=True)

    v = CF( (10*x*y**3-x**2, y**4*z*x-y) )
    vv = dg.VectorField(v)
    ov = dg.OneForm(v)

    term1 = CovDerV(v, mesh, gf_metric, contra=True)
    term2 = mf.CovDeriv(vv)
    assert sqrt(Integrate(InnerProduct(term1-term2,term1-term2),mesh)) < 1e-7
    
    term1 = CovDerV(v, mesh, gf_metric, contra=False)
    term2 = mf.CovDeriv(ov)
    assert sqrt(Integrate(InnerProduct(term1-term2,term1-term2),mesh)) < 1e-7

    return

def test_cov_der_mat():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    metric = dg.CigarSoliton().metric
    mf = dg.RiemannianManifold(metric=metric)
    gf_metric = GridFunction(HCurlCurl(mesh, order=5))
    gf_metric.Set (metric, dual=True)

    A = CF( (10*x*y**3-x**2, y**4*x-y, 
         sin(x*y), cos(x)*y**2), dims=(2,2) )
    Acon = dg.TensorField(A, "00")
    Acov = dg.TensorField(A, "11")
    Amix1 = dg.TensorField(A, "10")
    Amix2 = dg.TensorField(A, "01")

    term1 = CovDerT(A, mesh, gf_metric, contra=[True,True])
    term2 = mf.CovDeriv(Acon)
    assert sqrt(Integrate(InnerProduct(term1-term2,term1-term2),mesh)) < 1e-7
    
    term1 = CovDerT(A, mesh, gf_metric, contra=[False,False])
    term2 = mf.CovDeriv(Acov)
    assert sqrt(Integrate(InnerProduct(term1-term2,term1-term2),mesh)) < 1e-7

    term1 = CovDerT(A, mesh, gf_metric, contra=[False,True])
    term2 = mf.CovDeriv(Amix1)
    assert sqrt(Integrate(InnerProduct(term1-term2,term1-term2),mesh)) < 1e-7

    term1 = CovDerT(A, mesh, gf_metric, contra=[True,False])
    term2 = mf.CovDeriv(Amix2)
    assert sqrt(Integrate(InnerProduct(term1-term2,term1-term2),mesh)) < 1e-7

    return


if __name__ == "__main__":
    test_cov_der_scal()
    test_cov_der_vec()
    test_cov_der_mat()

    print("All tests passed!")