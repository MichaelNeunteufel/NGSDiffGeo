import pickle

import pytest

import ngsdiffgeo as dg
from netgen.occ import Circle, OCCGeometry, unit_cube, unit_square
from ngsolve import (
    BND,
    BilinearForm,
    CF,
    Grad,
    GridFunction,
    H1,
    InnerProduct,
    Integrate,
    Id,
    LinearForm,
    Mesh,
    Norm,
    NumberSpace,
    OuterProduct,
    SymbolicBFI,
    SymbolicLFI,
    TaskManager,
    VectorH1,
    dx,
    specialcf,
    sqrt,
    x,
    y,
    z,
)
from ngsolve.meshes import Make1DMesh


def _l2_error(left, right, mesh, *, vb=None, element_boundary=False):
    error = InnerProduct(left - right, left - right)
    if element_boundary:
        return abs(sqrt(Integrate(error * dx(element_boundary=True), mesh)))
    if vb is not None:
        return abs(sqrt(Integrate(error, mesh, vb)))
    return abs(sqrt(Integrate(error, mesh)))


def _linear_form_integral(
    coefficient,
    mesh,
    *,
    simd,
    vb=None,
    element_vb=None,
    bonus_intorder=4,
):
    space = NumberSpace(mesh)
    test = space.TestFunction()
    form = LinearForm(space)
    kwargs = {
        "bonus_intorder": bonus_intorder,
        "simd_evaluate": simd,
    }
    if element_vb is not None:
        kwargs["element_vb"] = element_vb
    if vb is None:
        form += SymbolicLFI(coefficient * test, **kwargs)
    else:
        form += SymbolicLFI(coefficient * test, vb, **kwargs)
    with TaskManager():
        form.Assemble()
    return Norm(form.vec)


def _bilinear_form_integral(
    coefficient,
    mesh,
    *,
    simd,
    element_vb=None,
    bonus_intorder=4,
):
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    form = BilinearForm(space)
    kwargs = {
        "bonus_intorder": bonus_intorder,
        "simd_evaluate": simd,
    }
    if element_vb is not None:
        kwargs["element_vb"] = element_vb
    integrator = SymbolicBFI(coefficient * trial * test, **kwargs)
    form += integrator
    with TaskManager():
        form.Assemble()
    return Norm(form.mat.AsVector()), integrator.simd_evaluate


@pytest.mark.parametrize("dim", [-1, 0, 4])
@pytest.mark.parametrize("coefficient", [CF(0), x])
def test_gradcf_validates_dimension_before_special_cases(dim, coefficient):
    with pytest.raises(Exception, match="only dimensions 1,2,3 supported"):
        dg.GradCF(coefficient, dim)


def test_gradcf_rejects_null_and_invalid_surface_dimension():
    with pytest.raises(Exception, match="input coefficient is null"):
        dg.GradCF(None, 2)
    with pytest.raises(Exception, match="only dimensions 2,3 supported"):
        dg.GradCF(x, 1, surface=True)


def test_hessecf_rejects_invalid_boundary_dimension_before_zero_special_case():
    for coefficient in (CF(0), x):
        with pytest.raises(Exception, match="only dimensions 2,3 supported"):
            dg.HesseCF(coefficient, 1, boundary=True)


def test_numerical_gradient_preserves_nonambient_component_axes():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    coefficient = CF((x**2, x * y, y**2), dims=(3,))
    expected = CF(
        (2 * x, y, 0, 0, x, 2 * y),
        dims=(2, 3),
    )

    gradient = dg.GradCF(coefficient, 2)

    assert tuple(gradient.dims) == (2, 3)
    assert _l2_error(gradient, expected, mesh) < 3e-11


def test_numerical_hessian_preserves_nonambient_component_axes():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    coefficient = CF((x**2, x * y, y**2), dims=(3,))
    expected = CF(
        (
            2,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            2,
        ),
        dims=(2, 2, 3),
    )

    hessian = dg.HesseCF(coefficient, 2)

    assert tuple(hessian.dims) == (2, 2, 3)
    assert _l2_error(hessian, expected, mesh) < 2e-7


@pytest.mark.parametrize("space_kind", ["h1-dim", "vector-h1"])
def test_hessecf_supports_tensor_valued_proxy_with_derivative_axes_first(
    space_kind,
):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    component_count = 3 if space_kind == "h1-dim" else 2
    space = (
        H1(mesh, order=3, dim=component_count)
        if space_kind == "h1-dim"
        else VectorH1(mesh, order=3)
    )
    trial, test = space.TnT()
    hessian = dg.HesseCF(trial, 2)

    assert tuple(hessian.dims) == (2, 2, component_count)
    boundary_hessian = dg.HesseCF(trial, 2, boundary=True)
    assert tuple(boundary_hessian.dims) == (2, 2, component_count)

    polynomials = (
        x**3 + x * y**2,
        x**2 * y + y**3,
        x**2 + x * y + y**2,
    )[:component_count]
    expected_hessians = (
        ((6 * x, 2 * y), (2 * y, 2 * x)),
        ((2 * y, 2 * x), (2 * x, 6 * y)),
        ((2, 1), (1, 2)),
    )[:component_count]

    weighted_hessian = 0
    weighted_expected = 0
    for first_direction in range(2):
        for second_direction in range(2):
            for component in range(component_count):
                weight = 1 + first_direction + 3 * second_direction + 7 * component
                weighted_hessian += (
                    weight
                    * hessian[first_direction, second_direction, component]
                    * test[component]
                )
                weighted_expected += (
                    weight
                    * expected_hessians[component][first_direction][second_direction]
                    * test[component]
                )

    matrix = BilinearForm(space)
    matrix += SymbolicBFI(weighted_hessian, simd_evaluate=False)
    matrix.Assemble()
    gridfunction = GridFunction(space)
    gridfunction.Set(CF(polynomials, dims=(component_count,)))
    actual = gridfunction.vec.CreateVector()
    actual.data = matrix.mat * gridfunction.vec

    reference = LinearForm(space)
    reference += SymbolicLFI(weighted_expected, simd_evaluate=False)
    reference.Assemble()
    actual.data -= reference.vec

    assert Norm(actual) < 2e-10

    boundary_form = BilinearForm(space)
    boundary_form += SymbolicBFI(
        sum(
            boundary_hessian[direction, direction, component]
            * test[component]
            for direction in range(2)
            for component in range(component_count)
        ),
        BND,
        simd_evaluate=False,
    )
    boundary_form.Assemble()


@pytest.mark.parametrize("complex_scale", [1.0, 1.0 + 2.0j])
def test_gradcf_values_compilation_and_pickle(complex_scale):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    coefficient = complex_scale * (x**3 * y + x * y**2)
    expected = CF(
        (coefficient.Diff(x), coefficient.Diff(y)),
        dims=(2,),
    )
    gradient = dg.GradCF(coefficient, 2)

    assert gradient.is_complex == coefficient.is_complex
    assert gradient(mesh(0.2, 0.3)) == pytest.approx(
        expected(mesh(0.2, 0.3)), abs=2e-11
    )
    assert _l2_error(gradient, expected, mesh) < 2e-11

    restored = pickle.loads(pickle.dumps(gradient))
    assert restored.is_complex == gradient.is_complex
    assert _l2_error(restored, gradient, mesh) < 1e-13

    for realcompile in (False, True):
        compiled = gradient.Compile(
            realcompile=realcompile,
            wait=True,
            maxderiv=0,
        )
        assert _l2_error(compiled, expected, mesh) < 2e-11


@pytest.mark.parametrize("realcompile", [False, True])
def test_gradcf_compiled_inner_preserves_values_metadata_and_pickle(
    realcompile,
    monkeypatch,
):
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    first = x**3 * y + x * y**2
    second = x**2 - 2 * x * y + y**3
    coefficient = CF((first, second), dims=(2,))
    expected = CF(
        (
            first.Diff(x),
            second.Diff(x),
            first.Diff(y),
            second.Diff(y),
        ),
        dims=(2, 2),
    )

    compiled_inner = coefficient.Compile(
        realcompile=realcompile,
        wait=True,
        maxderiv=0,
    )
    gradient = dg.GradCF(compiled_inner, 2)

    assert tuple(gradient.dims) == (2, 2)
    assert gradient.is_complex == coefficient.is_complex
    assert _l2_error(gradient, expected, mesh) < 3e-11

    restored = pickle.loads(pickle.dumps(gradient))
    assert tuple(restored.dims) == tuple(gradient.dims)
    assert restored.is_complex == gradient.is_complex
    # Native machine code is not archived. The restored graph can therefore
    # round the cancellation-prone finite-difference stencil differently.
    assert _l2_error(restored, expected, mesh) < 3e-11


@pytest.mark.parametrize("compile_graph", [False, True])
def test_gridfunction_gradient_ignores_original_rule_cache(
    compile_graph,
):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.35))
    space = H1(mesh, order=3)
    gridfunction = GridFunction(space)
    gridfunction.Set(x**2 + x * y + 2 * y**2)
    coefficient = (
        gridfunction.Compile(realcompile=False, maxderiv=0)
        if compile_graph
        else gridfunction
    )
    gradient = dg.GradCF(coefficient, 2)
    expected = Grad(gridfunction)
    difference = gradient - expected
    error = InnerProduct(difference, difference)

    for simd in (False, True):
        assert _linear_form_integral(error, mesh, simd=simd) < 2e-17


def test_unsupported_simd_modes_fall_back_to_scalar_evaluation():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    cases = [
        (
            InnerProduct(
                dg.GradCF((1 + 2j) * (x**2 + x * y), 2),
                dg.GradCF((1 + 2j) * (x**2 + x * y), 2),
            ).real,
            None,
        ),
        (
            InnerProduct(
                dg.GradCF(x**2 + x * y, 2, surface=True),
                dg.GradCF(x**2 + x * y, 2, surface=True),
            ),
            BND,
        ),
    ]

    for energy, vb in cases:
        scalar = _linear_form_integral(
            energy,
            mesh,
            simd=False,
            vb=vb,
        )
        requested_simd = _linear_form_integral(
            energy,
            mesh,
            simd=True,
            vb=vb,
        )
        assert requested_simd == pytest.approx(scalar, rel=2e-13, abs=2e-13)


def _mesh_for_dimension(dim):
    if dim == 1:
        return Make1DMesh(4)
    if dim == 2:
        return Mesh(unit_square.GenerateMesh(maxh=0.45))
    return Mesh(unit_cube.GenerateMesh(maxh=0.8))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_real_volume_simd_gradient_matches_independent_derivative(dim):
    mesh = _mesh_for_dimension(dim)
    coordinates = (x, y, z)
    coefficient = sum(
        (direction + 1) * coordinates[direction] ** 2
        for direction in range(dim)
    )
    expected = CF(
        tuple(
            2 * (direction + 1) * coordinates[direction]
            for direction in range(dim)
        ),
        dims=(dim,),
    )
    gradient = dg.GradCF(
        coefficient.Compile(realcompile=False, maxderiv=0),
        dim,
    )
    difference = gradient - expected
    error = InnerProduct(difference, difference)

    scalar = _linear_form_integral(error, mesh, simd=False)
    vectorized = _linear_form_integral(error, mesh, simd=True)

    assert scalar < 2e-17
    assert vectorized < 2e-17
    assert vectorized == pytest.approx(scalar, rel=2e-12, abs=2e-18)


def test_real_volume_simd_gradient_supports_element_boundary_rules():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    coefficient = (x**2 + x * y + 2 * y**2).Compile(
        realcompile=False,
        maxderiv=0,
    )
    expected = CF((2 * x + y, x + 4 * y), dims=(2,))
    difference = dg.GradCF(coefficient, 2) - expected
    error = InnerProduct(difference, difference)

    scalar, _ = _bilinear_form_integral(
        error,
        mesh,
        simd=False,
        element_vb=BND,
    )
    requested_simd, simd_remained_enabled = _bilinear_form_integral(
        error,
        mesh,
        simd=True,
        element_vb=BND,
    )

    assert scalar < 2e-17
    assert requested_simd < 2e-17
    assert simd_remained_enabled


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_proxy_gradient_matches_ngsolve_gradient_matrix(dim, order):
    mesh = _mesh_for_dimension(dim)
    space = H1(mesh, order=order)
    trial, test = space.TnT()
    gradient = dg.GradCF(trial, dim)
    expected = Grad(trial)

    for component in range(dim):
        form = BilinearForm(space)
        form += SymbolicBFI(
            (gradient[component] - expected[component]) * test,
            simd_evaluate=False,
        )
        form.Assemble()
        assert Norm(form.mat.AsVector()) < 2e-10


def test_complex_proxy_gradient_preserves_metadata_and_matrix():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.45))
    space = H1(mesh, order=2, complex=True)
    trial, test = space.TnT()
    gradient = dg.GradCF(trial, 2)
    expected = Grad(trial)

    assert gradient.is_complex
    for component in range(2):
        form = BilinearForm(space)
        form += SymbolicBFI(
            (gradient[component] - expected[component]) * test,
            simd_evaluate=False,
        )
        form.Assemble()
        assert Norm(form.mat.AsVector()) < 2e-10


def test_vector_proxy_gradient_uses_derivative_first_tensor_ordering():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.45))
    space = VectorH1(mesh, order=2)
    trial, test = space.TnT()
    gradient = dg.GradCF(trial, 2)
    expected = Grad(trial)

    assert tuple(gradient.dims) == (2, 2)
    for direction in range(2):
        for component in range(2):
            form = BilinearForm(space)
            form += SymbolicBFI(
                (
                    gradient[direction, component]
                    - expected[component, direction]
                )
                * test[component],
                simd_evaluate=False,
            )
            form.Assemble()
            assert Norm(form.mat.AsVector()) < 2e-10


def test_gradproxy_name_remains_a_compatibility_alias():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    trial = H1(mesh, order=1).TrialFunction()

    assert isinstance(dg.GradCF(trial, 2), dg.GradProxy)


def test_proxy_gradient_uses_ngsolve_primary_proxy_differentiation():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    space = H1(mesh, order=2)
    trial, test = space.TnT()
    gradient = dg.GradCF(trial, 2)
    differentiated = gradient.Diff(trial, trial)

    for component in range(2):
        form = BilinearForm(space)
        form += SymbolicBFI(
            (differentiated[component] - gradient[component]) * test,
            simd_evaluate=False,
        )
        form.Assemble()
        assert Norm(form.mat.AsVector()) < 2e-10


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("complex_scale", [1.0, 1.0 + 2.0j])
def test_surface_gradient_on_boundary_and_element_boundary(dim, complex_scale):
    geometry = unit_square if dim == 2 else unit_cube
    mesh = Mesh(geometry.GenerateMesh(maxh=0.6 if dim == 2 else 0.8))
    coordinates = (x, y, z)
    coefficient = complex_scale * sum(
        (direction + 1) * coordinates[direction]
        for direction in range(dim)
    )
    ambient_gradient = CF(
        tuple(complex_scale * (direction + 1) for direction in range(dim)),
        dims=(dim,),
    )
    normal = specialcf.normal(dim)
    normal_component = sum(
        normal[direction] * ambient_gradient[direction]
        for direction in range(dim)
    )
    expected = ambient_gradient - normal * normal_component
    gradient = dg.GradCF(coefficient, dim, surface=True)

    assert _l2_error(gradient, expected, mesh, vb=BND) < 3e-10
    assert (
        _l2_error(
            gradient,
            expected,
            mesh,
            element_boundary=True,
        )
        < 3e-10
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_nonlinear_surface_gradient_on_boundary_and_element_boundary(dim):
    geometry = unit_square if dim == 2 else unit_cube
    mesh = Mesh(geometry.GenerateMesh(maxh=0.6 if dim == 2 else 0.8))
    coordinates = (x, y, z)
    coefficient = sum(
        (direction + 1) * coordinates[direction] ** 2
        for direction in range(dim)
    )
    ambient_gradient = CF(
        tuple(
            2 * (direction + 1) * coordinates[direction]
            for direction in range(dim)
        ),
        dims=(dim,),
    )
    normal = specialcf.normal(dim)
    expected = ambient_gradient - normal * InnerProduct(normal, ambient_gradient)
    gradient = dg.GradCF(coefficient, dim, surface=True)

    assert _l2_error(gradient, expected, mesh, vb=BND) < 3e-10
    assert (
        _l2_error(
            gradient,
            expected,
            mesh,
            element_boundary=True,
        )
        < 3e-10
    )


@pytest.mark.parametrize("complex_scale", [1.0, 1.0 + 2.0j])
def test_volume_and_surface_gradients_on_curved_mesh(complex_scale):
    mesh = Mesh(
        OCCGeometry(Circle((0, 0), 1).Face(), dim=2).GenerateMesh(maxh=0.45)
    ).Curve(3)
    coefficient = complex_scale * (x**3 * y + x * y**2)
    ambient_gradient = CF(
        (
            complex_scale * (3 * x**2 * y + y**2),
            complex_scale * (x**3 + 2 * x * y),
        ),
        dims=(2,),
    )

    gradient = dg.GradCF(coefficient, 2)
    assert _l2_error(gradient, ambient_gradient, mesh) < 1e-10

    normal = specialcf.normal(2)
    normal_component = sum(
        normal[direction] * ambient_gradient[direction]
        for direction in range(2)
    )
    expected_surface = ambient_gradient - normal * normal_component
    surface_gradient = dg.GradCF(coefficient, 2, surface=True)
    assert _l2_error(surface_gradient, expected_surface, mesh, vb=BND) < 1e-10
    assert (
        _l2_error(
            surface_gradient,
            expected_surface,
            mesh,
            element_boundary=True,
        )
        < 1e-10
    )


def test_surface_gradient_pickle_preserves_surface_mode():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    gradient = dg.GradCF(x**2 + y**2, 2, surface=True)
    restored = pickle.loads(pickle.dumps(gradient))

    assert tuple(restored.dims) == tuple(gradient.dims)
    assert _l2_error(restored, gradient, mesh, vb=BND) < 1e-13
    assert (
        _l2_error(restored, gradient, mesh, element_boundary=True)
        < 1e-13
    )


def test_pure_boundary_hessian_is_symmetric_and_tangential_on_curved_boundary():
    mesh = Mesh(
        OCCGeometry(Circle((0, 0), 1).Face(), dim=2).GenerateMesh(maxh=0.3)
    ).Curve(4)
    normal = specialcf.normal(2)
    tangent_projection = Id(2) - OuterProduct(normal, normal)
    hessian = dg.HesseCF(x, 2, boundary=True)

    antisymmetric = hessian - hessian.trans
    assert (
        _l2_error(antisymmetric, CF((0,) * 4, dims=(2, 2)), mesh, vb=BND)
        < 1e-9
    )
    assert _l2_error(hessian * normal, CF((0, 0)), mesh, vb=BND) < 1e-9
    assert _l2_error(hessian.trans * normal, CF((0, 0)), mesh, vb=BND) < 1e-9

    projected = tangent_projection * hessian * tangent_projection
    assert _l2_error(hessian, projected, mesh, vb=BND) < 1e-9


def test_gradcf_equivalence_key_includes_surface_mode():
    coefficient = x**2 + y
    first = dg.GradCF(coefficient, 2)
    second = dg.GradCF(coefficient, 2)
    surface = dg.GradCF(coefficient, 2, surface=True)

    assert first._equivalence_key == second._equivalence_key
    assert first._equivalence_key != surface._equivalence_key
