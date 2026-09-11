import pickle

import pytest
from netgen.occ import unit_square
from ngsolve import (
    BND,
    CF,
    BilinearForm,
    H1,
    Id,
    InnerProduct,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
    VOL,
    sin,
    x,
    y,
)

import ngsdiffgeo as dg
from tests._helpers import l2_error, l2_error_bnd


@pytest.fixture(scope="module")
def mesh():
    return Mesh(unit_square.GenerateMesh(maxh=0.4))


def _manifold():
    scale = 1 + x**2 + y**2
    metric = CF((scale, 0, 0, scale), dims=(2, 2))
    return dg.RiemannianManifold(metric)


def _one_zero_form():
    alpha = dg.OneForm(CF((sin(x * y) + x**2, x * y + y**2)))
    return dg.DoubleForm(alpha, p=1, q=0, dim=2)


def test_covderiv_graph_compilation_matches_default_and_preserves_metadata(mesh):
    manifold = _manifold()
    tensor = dg.TensorField(
        CF(
            (
                sin(x * y) + x**2,
                x * y + y**2,
                x**2 * y,
                x - y**2,
            ),
            dims=(2, 2),
        ),
        "10",
    )

    default = manifold.CovDerivative(tensor)
    explicit_default = manifold.CovDerivative(tensor, compile_inner=False)
    compiled = manifold.CovDerivative(tensor, compile_inner="graph")

    assert tuple(compiled.dims) == tuple(default.dims)
    assert compiled.covariant_indices == default.covariant_indices == "110"
    assert l2_error(explicit_default, default, mesh) < 1e-13
    assert l2_error(compiled, default, mesh) < 2e-9


@pytest.mark.parametrize(
    ("method_name", "slot", "vb", "expected_degrees"),
    [
        ("d_cov", "left", VOL, (2, 0)),
        ("d_cov", "right", BND, (1, 1)),
        ("delta_cov", "left", VOL, (0, 0)),
        ("delta_cov", "left", BND, (0, 0)),
    ],
)
def test_double_form_graph_compilation_matches_default(
    mesh, method_name, slot, vb, expected_degrees
):
    manifold = _manifold()
    form = _one_zero_form()
    method = getattr(manifold, method_name)

    default = method(form, slot=slot, vb=vb)
    compiled = method(form, slot=slot, vb=vb, compile_inner="graph")

    assert (compiled.degree_left, compiled.degree_right) == expected_degrees
    assert compiled.covariant_indices == default.covariant_indices
    if vb == BND:
        assert l2_error_bnd(compiled, default, mesh) < 2e-9
    else:
        assert l2_error(compiled, default, mesh) < 2e-9


def test_boundary_double_form_graph_keeps_bilinear_simd_enabled(mesh):
    result = _manifold().d_cov(
        _one_zero_form(),
        slot="right",
        vb=BND,
        compile_inner="graph",
    )
    energy = InnerProduct(result, result)
    space = NumberSpace(mesh)
    trial, test = space.TnT()

    values = []
    integrators = []
    for simd in (False, True):
        form = BilinearForm(space)
        integrator = SymbolicBFI(
            energy * trial * test,
            element_vb=BND,
            bonus_intorder=4,
            simd_evaluate=simd,
        )
        form += integrator
        with TaskManager():
            form.Assemble()
        values.append(Norm(form.mat.AsVector()))
        integrators.append(integrator)

    assert values[1] == pytest.approx(values[0], rel=1e-11, abs=1e-11)
    assert integrators[1].simd_evaluate


def test_nested_d_cov_graph_compilation_matches_default_and_pickles(mesh):
    manifold = _manifold()
    first = manifold.d_cov(_one_zero_form(), slot="right")

    default = manifold.d_cov(first, slot="left")
    compiled = manifold.d_cov(
        first,
        slot="left",
        compile_inner="graph",
    )
    restored = pickle.loads(pickle.dumps(compiled))

    assert (compiled.degree_left, compiled.degree_right) == (2, 1)
    assert l2_error(compiled, default, mesh) < 3e-9
    assert l2_error(restored, compiled, mesh) < 3e-9


@pytest.mark.parametrize("compile_inner", [True, 0, "native", "GRAPH"])
def test_covariant_operators_reject_unknown_compile_inner_modes(compile_inner):
    manifold = dg.RiemannianManifold(Id(2))
    form = _one_zero_form()
    tensor = dg.TensorField(CF((x, y)), "0")

    for operation in (
        lambda: manifold.d_cov(form, compile_inner=compile_inner),
        lambda: manifold.delta_cov(form, compile_inner=compile_inner),
        lambda: manifold.CovDerivative(tensor, compile_inner=compile_inner),
    ):
        with pytest.raises(
            ValueError,
            match="compile_inner must be False, None, or 'graph'",
        ):
            operation()


def test_graph_compilation_preserves_formal_zero_metadata():
    manifold = dg.RiemannianManifold(Id(3))
    zero = dg.FormalZeroDoubleForm(-1, 0, 3)

    exterior = manifold.d_cov(
        zero,
        slot="left",
        compile_inner="graph",
    )
    codifferential = manifold.delta_cov(
        zero,
        slot="right",
        compile_inner="graph",
    )

    assert isinstance(exterior, dg.FormalZeroDoubleForm)
    assert (exterior.degree_left, exterior.degree_right) == (0, 0)
    assert isinstance(codifferential, dg.FormalZeroDoubleForm)
    assert (codifferential.degree_left, codifferential.degree_right) == (-1, -1)


def test_graph_compilation_rejects_trial_function_input(mesh):
    trial, _ = H1(mesh, order=1).TnT()
    scalar = dg.ScalarField(trial, dim=2)
    manifold = dg.RiemannianManifold(Id(2))

    with pytest.raises(
        Exception,
        match="inner graph compilation is not supported for trial/test functions",
    ):
        manifold.CovDerivative(scalar, compile_inner="graph")


@pytest.mark.parametrize("compile_inner", [False, "graph"])
def test_covderiv_rejects_null_input(compile_inner):
    manifold = dg.RiemannianManifold(Id(2))

    with pytest.raises(Exception, match="CovDerivative: input must be non-null"):
        manifold.CovDerivative(None, compile_inner=compile_inner)
