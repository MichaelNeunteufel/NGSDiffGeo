"""Cross-module regressions: semantic derivatives, proxy dispatch and adapters."""

import pickle

import pytest
from ngsolve import (
    BBND, BilinearForm, CF, Grad, H1, Id, InnerProduct, Integrate,
    Norm, SymbolicBFI, x, y,
)

from netgen.libngpy._meshing import NgException

import ngsdiffgeo as dg
from ngsdiffgeo import wrappers
from tests._helpers import assert_l2_close, l2_error_bnd


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("wrapper_target", [False, True])
def test_projection_replace_rebuilds_operand(double, wrapper_target, make_unit_square_mesh):
    mesh = make_unit_square_mesh(maxh=0.7)
    manifold = dg.RiemannianManifold(Id(2))
    a = dg.DoubleForm(CF((1+x, 2, 3, 4+y), dims=(2, 2)), p=1, q=1, dim=2)
    project = (lambda t: manifold.ProjectDoubleForm(t, left="F")) if double else (lambda t: manifold.ProjectTensor(t, "F"))
    replacement = dg.DoubleForm(2*a.coef, p=1, q=1, dim=2)
    result = project(a).Replace({a if wrapper_target else a.coef: replacement})
    assert l2_error_bnd(result, project(replacement), mesh) < 1e-12
