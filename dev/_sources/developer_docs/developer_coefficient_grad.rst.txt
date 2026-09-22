Numerical-gradient developer guide
==================================

This page documents the implementation of ``GradCF`` and ``HesseCF``.


Dispatch and shape
------------------

``GradCF(cf, dim, surface=False)`` always puts the physical derivative axis
first. If ``cf.dims == (d0, ..., dk)``, the result dimensions are
``(dim, d0, ..., dk)``. Tensor-field code relies on this ordering when it adds
the derivative slot to a tensor signature.

The factory selects one of three paths:

* a childless native zero produces a correctly shaped NGSolve ``ZeroCF``. Zero-valued wrappers and intermediate expressions retain their dependencies,
* an expression containing trial or test proxies uses NGSolve's symbolic gradient operators, with a chain-rule fallback for composite expressions,
* every pure coefficient graph is wrapped in
  ``GradCoefficientFunction<dim>``.

Expressions containing both trial and test functions in one ``GradCF`` call
are rejected. Surface gradients of proxy expressions use NGSolve's
``Gradboundary`` operator. The symbolic path retains the surrounding value
graph and normalizes NGSolve's component-first proxy layout to the public
derivative-first convention. Direct scalar proxies retain the native
``GradProxy`` return type. Wrapped proxy expressions retain their original
operand alongside the native evaluator, so differentiation or replacement of
that wrapper commutes with the spatial derivative. Native ``ProxyFunction``
objects are not registered for archives in the supported NGSolve build, so
proxy-bearing expressions cannot be pickled. Replacing the semantic operand
with a pure coefficient rebuilds the derivative without the old proxy evaluator.
The resulting numerical derivative can be archived.

Proxy discovery visits each node in the evaluation DAG once. Spatial chain-rule
collection also visits shared nodes once and treats retained differential
expressions as complete operands rather than descending into their evaluators.
Construction-cost regressions use deterministic visit counts, not timing limits.


Numerical evaluator
-------------------

``GradCoefficientFunction`` evaluates its child at the four reference-space
offsets ``-2h``, ``-h``, ``h``, and ``2h`` with ``h = 1e-4``. It uses the
fourth-order centered stencil

.. math::

 \partial_i f \approx
 \frac{f(x-2h e_i)-8f(x-h e_i)+8f(x+h e_i)-f(x+2h e_i)}{12h}.

Reference derivatives are mapped to physical coordinates by the inverse
Jacobian, so evaluation requires a mapped integration rule.

NGSolve may attach ``ProxyUserData`` containing values cached for the original
integration rule. Those values are invalid at the perturbed stencil points.
Every child evaluation therefore uses ``PushUserData`` to detach the original
cache for the lifetime of the perturbed rule. Omitting this guard can return a
plausible but incorrect zero gradient of a cached ``GridFunction``.


Surface rules
-------------

Surface evaluation supports both NGSolve boundary layouts:

* a boundary-element transformation has an intrinsic ``dim-1`` reference rule. Its mapped Jacobian inverse is the tangential pseudoinverse,
* element-boundary integration maps facet points into the volume reference element before coefficient evaluation. The implementation uses the retained facet number and the inverse reference-facet map to recover intrinsic facet coordinates, perturbs those coordinates, and maps each stencil point into the volume exactly once.

The result still has ``dim`` ambient components. ``surface=True`` changes the
derivative map, not the result shape.


NGSolve coefficient-function contract
--------------------------------------

The custom coefficient node follows the standard NGSolve graph interfaces:

* ``InputCoefficientFunctions`` and ``TraverseTree`` expose the child graph,
* ``Transform`` preserves the surface flag and participates in NGSolve's replacement cache,
* directional ``Diff`` differentiates the child and reconstructs ``GradCF``,
* ``NonZeroPattern`` conservatively copies each child dependency to every physical derivative of that component,
* ``GetCArgs`` contains the child and surface flag, and all three dimensional specializations are registered for polymorphic archiving.

Consequently, graph transformation, pickling, and compilation retain the
component shape, complex-valued state, and surface mode. ``GradProxy`` remains a Python
compatibility alias for NGSolve's native ``ProxyFunction``. It is no longer a
separate NGSDiffGeo coefficient class.


SIMD support
------------

Native SIMD evaluation currently covers real, non-surface
``GradCoefficientFunction`` objects on full-dimensional mapped rules. This
also covers element-boundary rules represented through the volume
transformation. SIMD and scalar evaluation use the same stencil and physical
derivative convention.

The following modes deliberately raise ``ExceptionNOSIMD`` so the enclosing
integrator can retry with scalar evaluation:

* surface gradients,
* complex and automatic-differentiation SIMD values,
* child coefficient graphs that do not support SIMD.

When diagnosing boundary performance, use a SIMD-capable integrator.
NGSolve's ``SymbolicLFI`` currently takes a scalar-only path for
``element_vb != VOL`` before evaluating the coefficient tree, whereas
``SymbolicBFI`` supports SIMD element-boundary rules. The benchmark reports
both requested and still-active SIMD state to distinguish these cases from a
coefficient fallback.


Hessian dispatch
----------------

``HesseCF`` prepends two physical derivative axes. Pure coefficient graphs may
have arbitrary component dimensions and are differentiated by applying
``GradCF`` twice. Direct ``VectorH1`` and ``H1(dim=...)`` proxies are also
supported. ``VectorH1`` Hessians are reshaped and transposed from NGSolve's
component-first operator layout. ``H1(dim=...)`` proxies use an explicit block
Hessian operator because their native additional evaluator is scalar.
Direct H1 gradient proxies and their component expressions use the primary
proxy's native Hessian when differentiated again. Wrapped scalar trial functions
therefore support the Euclidean ``CovHesse`` path as well. Tensor proxy Hessians
continue to use the derivative-first layout described above.
Composite proxy expressions still require NGSolve to expose a native Hessian
operator for the complete expression and otherwise fail with a specific error.

Both ``HesseCF(f, dim)`` and ``GradCF(GradCF(f, dim), dim)`` retain zero-valued
semantic inputs. For example, differentiating either expression with respect to
a zero scalar wrapper in direction ``x*x + y*y`` gives ``2*Id(2)``. A zero value
is not sufficient evidence that an operand is independent of a symbolic target.

Boundary Hessians follow the same ambient-dimension restriction as surface
gradients and therefore require dimension two or three. Both derivative axes
are projected into the boundary tangent space.


Covariant inner-graph compilation
---------------------------------

``RiemannianManifold.CovDeriv``, ``d_cov``, and ``delta_cov`` accept the opt-in
mode ``compile_inner="graph"``. It compiles one shared copy of the input value
graph before constructing the gradient and connection terms. ``False`` and
``None`` retain the default uncompiled path. Trial/test proxy inputs are
rejected in graph mode because compiling them would erase the symbolic finite
element differentiation contract.

Measure the one-time construction cost and repeated assembly cost with::

 python benchmarks/benchmark_covariant_inner.py --depth 5 --iterations 9

The benchmark alternates the default and graph-compiled cases, validates their
assembled values, reports expression-graph statistics and SIMD state, and
estimates the number of repeated assemblies needed to amortize compilation.
It times the volume ``CovDeriv`` core used by ``d_cov`` and ``delta_cov``.
Their additional alternation or contraction is not included in the timing.
Timings are diagnostic and therefore have no fixed CI threshold. A lightweight
smoke test keeps the benchmark interface and correctness check operational.


Maintenance rules
-----------------

Keep the numerical and proxy paths consistent in these contracts:

* the derivative axis is first,
* the reference stencil and step size agree,
* reference derivatives are mapped to physical coordinates,
* scalar and complex mapped rules reject incompatible dimensions before a downcast,
* unsupported SIMD modes use ``ExceptionNOSIMD``, not a generic exception.

If the stencil or step size changes, rerun the polynomial, ``GridFunction``,
curved-mesh, boundary, complex, proxy, compilation, pickling, and SIMD tests.
Absolute performance thresholds do not belong in the correctness test suite.
Record comparative timing in a dedicated benchmark when performance changes.
Run the scalar/SIMD comparison from the repository root with::

 python benchmarks/benchmark_gradcf.py --dim 2 --iterations 10

Use ``--json`` for machine-readable output. The benchmark validates each
assembled value against an analytic-gradient baseline before reporting timing,
so it rejects a fast but incorrect path. Select ``--mode volume`` (the
default), ``--mode element-boundary``, or ``--mode surface-boundary`` to cover
the volume SIMD path, SIMD-capable element-boundary rules, and the expected
scalar fallback for surface gradients, respectively.


Verification
------------

After changing the coefficient implementation, build and install the extension
before running the focused tests::

 cmake --build build
 cmake --install build
 CCACHE_DISABLE=1 python -m pytest -q tests/test_coefficient_grad.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_symbolic_gradcf.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_cross_module_regressions.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_covariant_inner_compilation.py
 ctest --test-dir build --output-on-failure

Performance comparisons belong in ``benchmarks/benchmark_gradcf.py`` and are
not part of the correctness test suite.