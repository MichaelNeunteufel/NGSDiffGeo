from functools import partial

import pytest

import ngsdiffgeo as dg
from ngsolve import (
    BBND,
    BND,
    CF,
    VOL,
    CoefficientFunction,
    Cross,
    Id,
    Integrate,
    Mesh,
    Norm,
    acos,
    dx,
    specialcf,
    sqrt,
    x,
    y,
    z,
)
from ngsolve.fem import Einsum

from tests._helpers import l2_error as _l2_error
from tests._helpers import l2_error_bbnd as _l2_error_bbnd
from tests._helpers import l2_error_bnd as _l2_error_bnd
from tests._helpers import l2_norm as _l2_norm

l2_error = partial(_l2_error, bonus_intorder=3)
l2_norm = partial(_l2_norm, bonus_intorder=3)
l2_error_bnd = partial(_l2_error_bnd, bonus_intorder=3)
l2_error_bbnd = partial(_l2_error_bbnd, bonus_intorder=3)

__all__ = [
    "BBND",
    "BND",
    "CF",
    "VOL",
    "CoefficientFunction",
    "Cross",
    "Einsum",
    "Id",
    "Integrate",
    "Mesh",
    "Norm",
    "acos",
    "dg",
    "dx",
    "l2_error",
    "l2_error_bbnd",
    "l2_error_bnd",
    "l2_norm",
    "pytest",
    "specialcf",
    "sqrt",
    "x",
    "y",
    "z",
]
