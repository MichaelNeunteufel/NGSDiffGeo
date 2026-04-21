try:
    from ._version import version as __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ngsdiffgeo")
    except Exception:
        __version__ = "0+unknown"


# It's important to import ngsolve first, to load all shared libs before the add on is loaded
import ngsolve
from ngsolve.fem import Einsum
from .ngsdiffgeo import *

from .wrappers import (
    FormalZeroBase,
    FormalZeroKForm,
    FormalZeroDoubleForm,
    ScalarField,
    OneForm,
    TwoForm,
    ThreeForm,
    KForm,
    DoubleForm,
    VectorField,
    TensorField,
    Wedge,
    Sym,
    WedgePower,
    d,
    d_formal,
    star,
    star_formal,
    inv_star,
    inv_star_formal,
    slot_inner_product,
    delta,
    delta_formal,
    is_formal_zero,
    is_formal_zero_kform,
    is_formal_zero_doubleform,
    materialize_zero,
    wedge_formal,
    contract_slot_formal,
    project_doubleform_formal,
    inner_product_formal,
    compare_formal_zero,
    d_cov_formal,
    delta_cov_formal,
    covdiv_formal,
    trace_formal,
    trace_sigma_formal,
    contraction_formal,
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
