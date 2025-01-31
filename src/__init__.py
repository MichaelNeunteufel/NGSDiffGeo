# It's important to import ngsolve first, to load all shared libs before the add on is loaded
import ngsolve
from ngsolve.fem import Einsum
from .ngsdiffgeo import *

#from .utils import *

class EuclideanMetric:
    """
    Euclidean metric on R^dim.
    """
    def __init__(self, dim=2):
        self.metric = ngsolve.Id(dim)

        # flat manifold
        self.chr1 = ngsolve.CF( (0,)*dim**3, dims=(dim,dim,dim) )
        self.chr2 = ngsolve.CF( (0,)*dim**3, dims=(dim,dim,dim) )
        self.Riemann = ngsolve.CF( (0,)*dim**4, dims=(dim,dim,dim,dim) )
        self.Ricci = ngsolve.CF( (0,)*dim**2, dims=(dim,dim) )
        self.scalar = CF(0)
        self.Einstein = CF( (0,)*dim**2, dims=(dim,dim) )

        return


class Sphere2:
    """
    Standard metric on sphere S^2. x and y are interpreted as angles; x in [0,pi], y in [0,2*pi). Has constant positive curvature.
    """
    def __init__(self):
        # metric tensor
        self.metric = ngsolve.CF( (1,0,
                    0,ngsolve.sin(ngsolve.x)**2), dims=(2,2) )
        # Christoffel symbols of the first kind Gamma_{ijk}=0.5*(d_ig_jk+d_jg_ik-d_kg_ij)
        self.chr1 = ngsolve.CF( (0,0, 0,sin(x)*cos(x),
                                 0,sin(x)*cos(x), -sin(x)*cos(x),0), dims=(2,2,2) )
        # Christoffel symbols of the second kind Gamma_{ij}^k=g^{kl}Gamma_{ijl}
        self.chr2 = ngsolve.CF( (0,0, 0,cos(x)/sin(x),
                                 0,cos(x)/sin(x), -sin(x)*cos(x),0), dims=(2,2,2) )
        # Riemann curvature tensor R_{ijkl}=d_jGamma_{ikl}-d_kGamma_{ijl}+Gamma_{ijm}Gamma_{mkl}-Gamma_{ikm}Gamma_{mjl}
        self.Riemann = ngsolve.CF( (0,0, 0,0,
                                    0,0, 0,0,
                                    0,0, 0,0,
                                    0,0, 0,0), dims=(2,2,2,2) )
        # Ricci curvature tensor R_{ij}=g^{kl}R_{kilj}=-g^{kl}R_{ikjl}
        self.Ricci = None
        # Scalar curvature R=g^{ij}R_{ij}
        self.scalar = None
        # Einstein tensor G_{ij}=R_{ij}-0.5*g_{ij}R
        self.Einstein = CF( (0,0,0,0), dims=(2,2) )
        return

# 
class Sphere3:
    """
    Standard metric on sphere S^3. x, y, and z are interpreted as angles; x in [0,pi], y in [0,pi], z in [0,2*pi). Has constant positive curvature.
    """
    def __init__(self):
        # metric
        self.metric = ngsolve.CF( (1,0,0, 
                       0,ngsolve.sin(ngsolve.x)**2,0,
                       0,0,ngsolve.sin(ngsolve.x)**2*ngsolve.sin(ngsolve.y)**2), dims=(3,3) )
         # Christoffel symbols of the first kind Gamma_{ijk}=0.5*(d_ig_jk+d_jg_ik-d_kg_ij)
         

        return

class PoincareDisk:
    """
    Hyperbolic metric on the Poincare Disk B_1(0)= {(x,y) in R^2 : x^2+y^2 < 1}. Has constant negative curvature.
    """
    def __init__(self):
        self.metric = 4/(1-ngsolve.x**2-ngsolve.y**2)**2 * ngsolve.Id(2)

        self.chr1 = 8/(1-(ngsolve.x**2+ngsolve.y**2))**3 * ngsolve.CF( (ngsolve.x,-ngsolve.y, ngsolve.y, ngsolve.x, ngsolve.y, ngsolve.x, -ngsolve.x, ngsolve.y), dims=(2,2,2) )
        self.chr2 = 2/(1-ngsolve.x**2-ngsolve.y**2)*ngsolve.CF( (ngsolve.x,-ngsolve.y, ngsolve.y, ngsolve.x, ngsolve.y, ngsolve.x, -ngsolve.x, ngsolve.y), dims=(2,2,2) )

        self.Riemann = 16/(1-ngsolve.x**2-ngsolve.y**2)**4*ngsolve.CF( (0,0, 0,0,  0,1, -1,0,
                            0,-1, 1,0, 0,0, 0,0), dims=(2,2,2,2) )
        self.Ricci = -4/(1-ngsolve.x**2-ngsolve.y**2)**2 *ngsolve.Id(2)
        self.scalar = -2
        self.Einstein = ngsolve.CF( (0,0,0,0), dims=(2,2) )
        return

# 
class HyperbolicH2:
    """
    Hyperbolic metric on H2={(x,y) in R^2 : y > 0}. Has constant negative curvature.
    """
    def __init__(self):
        self.metric = 1/ngsolve.y**2 * ngsolve.Id(2)
        self.chr1 = -1/ngsolve.y**3*ngsolve.CF( (0,-1, 1,0,
                                                 1,0, 0, 1), dims=(2,2,2) )
        self.chr2 = -1/ngsolve.y*ngsolve.CF( (0,-1, 1,0,
                                            1,0, 0, 1), dims=(2,2,2))
        self.Riemann = 1/ngsolve.y**4*ngsolve.CF( (0,0, 0,0,  0,1, -1,0,
                            0,-1, 1,0, 0,0, 0,0), dims=(2,2,2,2) )
        self.Ricci = -1/ngsolve.y**2 * ngsolve.Id(2)
        self.scalar = -2
        self.Einstein = ngsolve.CF( (0,0,0,0), dims=(2,2) )
        return
    
# 
class HyperbolicH3:
    """
    Hyperbolic metric on H3={(x,y,z) in R^3 : z > 0}. Has constant negative curvature.
    """
    def __init__(self):
        self.metric = 1/ngsolve.z**2 * ngsolve.Id(3)
        self.chr1 = -1/ngsolve.z**3*ngsolve.CF( (0,0,-1, 0,0,0, 1,0,0,
                                                    0,0,0, 0,0,-1, 0,1,0,
                                                    1,0,0, 0,1,0, 0,0,1), dims=(3,3,3) )
        self.chr2 = -1/ngsolve.z*ngsolve.CF( (0,0,-1, 0,0,0, 1,0,0,
                                                    0,0,0, 0,0,-1, 0,1,0,
                                                    1,0,0, 0,1,0, 0,0,1), dims=(3,3,3) )
        self.Riemann = None
        self.Ricci = -2/ngsolve.z**2 * ngsolve.Id(3)
        self.scalar = -6
        #G_{ij}=R_{ij}-0.5*g_{ij}R
        self.Einstein = 1/ngsolve.z**2 * ngsolve.Id(3)
        return

class Heisenberg:
    """
    Heisenberg metric on R^3. Has non-zero Ricci curvature.
    """
    def __init__(self):
        self.metric = ngsolve.CF( (1,0,0,
                                   0,1+ngsolve.x**2,-ngsolve.x,
                                   0,-ngsolve.x,1), dims=(3,3) )
        self.chr1 = ngsolve.CF( (0,0,0, 0,ngsolve.x,-1/2, 0,-1/2,0,
                                 0,ngsolve.x,-1/2, -ngsolve.x,0,0, 1/2,0,0,
                                 0,-1/2,0, 1/2,0,0, 0,0,0), dims=(3,3,3) )
        self.chr2 = None
        self.Riemann = None
        self.Ricci = ngsolve.CF( (0,0,0, 0,2*ngsolve.x,0, 0,0,0), dims=(3,3) )
        self.scalar = None 
        return

class CigarSoliton:
    """
    Cigar soliton metric on R^2.
    """
    def __init__(self):
        self.metric = 1/(1+ngsolve.x**2+ngsolve.y**2)*ngsolve.Id(2)
        self.chr1 = -1/(1+ngsolve.x**2+ngsolve.y**2)**2*ngsolve.CF( (ngsolve.x,-ngsolve.y, ngsolve.y,ngsolve.x,  ngsolve.y,ngsolve.x, -ngsolve.x,ngsolve.y), dims=(2,2,2))
        self.chr2 = None
        self.Riemann = None
        self.Ricci = None
        self.scalar = None
        return

class WarpedProduct:
    """
    Warped product metric on R^3.
    """
    def __init__(self):
        self.metric = ngsolve.CF( (ngsolve.exp(2*ngsolve.z),0,0, 
                                   0,ngsolve.exp(2*ngsolve.z),0,
                                   0,0,1), dims=(3,3) )
        self.chr1 = ngsolve.exp(2*ngsolve.z)*ngsolve.CF( (0,0,-1, 0,0,0, 1,0,0,
                                                  0,0,0, 0,0,-1, 0,1,0,
                                                  1,0,0, 0,1,0, 0,0,0), dims=(3,3,3) )
        self.chr2 = None
        self.Riemann = None
        self.Ricci = None
        self.scalar = None
        return


def TestMetric(dim,order=4):
    xvec = [ngsolve.x, ngsolve.y, ngsolve.z]
    return 10*ngsolve.Id(dim) + 0.1*ngsolve.CF( tuple( [xvec[i]**order - 3*xvec[j]**order + 5*(xvec[(i+1)%dim]*xvec[(j+2)%dim])**int(order/2) + (4 if i == j else 0) + (1/3*xvec[(i+1)%dim]**2 if i == 0 and j == 0 else 0) for i in range(dim) for j in range(dim)]), dims=(dim,dim) )