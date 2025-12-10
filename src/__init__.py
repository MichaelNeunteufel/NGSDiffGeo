# It's important to import ngsolve first, to load all shared libs before the add on is loaded
import ngsolve
from ngsolve.fem import Einsum
from .ngsdiffgeo import *

from .wrappers import (
    ScalarField,
    OneForm,
    TwoForm,
    ThreeForm,
    KForm,
    VectorField,
    TensorField,
    Wedge,
    d,
    star,
    delta,
    RiemannianManifold,
)
from .manifolds import (
    EuclideanMetric,
    Sphere2,
    Sphere3,
    PoincareDisk,
    HyperbolicH2,
    HyperbolicH3,
    Heisenberg,
    CigarSoliton,
    WarpedProduct,
    TestMetric,
)
