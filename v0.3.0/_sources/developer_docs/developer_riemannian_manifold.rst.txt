Riemannian manifolds
====================

This page documents ``src/riemannian_manifold.hpp`` and
``src/riemannian_manifold.cpp``. The component owns metric-dependent tensor
algebra, covariant differential operators, boundary projections, and curvature
dispatch. It builds on :doc:`developer_tensor_fields` for tensor metadata and
semantic expression nodes, on :doc:`developer_coefficient_grad` for physical
derivatives, and on :doc:`developer_kforms` for form representation.


Construction and metric sources
-------------------------------

A manifold is constructed from a non-null square metric of dimension two or
three. ``normal_sign`` must be exactly ``+1`` or ``-1``. Other magnitudes,
zero, and non-finite values are rejected instead of silently scaling or
invalidating the normal. Test-function metrics are rejected. The constructor distinguishes three
metric sources:

* an ordinary coefficient function uses the analytic derivative and curvature
  construction in this component,
* an ``HCurlCurlFESpace`` trial proxy uses the Regge additional proxies, and
* a grid function in ``HCurlCurlFESpace`` uses the corresponding additional
  evaluators.

A direct proxy or grid function from another finite-element space is rejected.
The metric, its inverse, determinant, normalized normals, tangents, conormals,
and projectors are initialized eagerly. Metric derivatives, Christoffel
symbols, curvature tensors, and the second fundamental form are initialized
on first access through ``EnsureCurvature``. Initialization is protected by
``std::call_once`` so concurrent first access constructs one shared cache and
publishes it atomically. Keep this dispatch centralized:
public curvature accessors must not independently reconstruct one of the three
source paths.

The constructor retains whether the metric graph contains a trial proxy in
``has_trial`` for future trial-dependent dispatch. It also retains ``det_g`` as
the unique determinant node used by the volume density and by any future
determinant consumers. Do not reconstruct ``DeterminantCF(g)`` independently.

The cached ``G``, ``G_inv``, ``G_F``, ``G_F_inv``, ``G_E``, and ``G_E_inv``
objects are typed views of the corresponding value graphs. ``normal_sign``
orients the Euclidean normal used during construction. ``change_riemann_sign``
selects the supported Riemann-tensor sign convention and is carried into the
lazy curvature construction.


Intrinsic strata
-----------------

The ``VorB`` argument selects metric data on an intrinsic stratum:

``VOL``
   The full metric ``g``, its inverse, and ``sqrt(det(g))``.

``BND``
   The face-tangential metric and inverse. Covariant derivatives are projected
   in every output slot after differentiation.

``BBND``
   The edge-tangential metric and inverse. It is available to algebraic
   operations such as raising, lowering, traces, and inner products, but not to
   ``CovDerivative``.

``BBBND``
   Only the stored zero-dimensional volume factor is exposed.

Every public operation validates its supported ``VorB`` values before an
identity, scalar, empty-slot, or structural-zero shortcut. ``GetVolumeForm``
validates all four array indices explicitly. Do not index the internal volume
array with an unchecked enum value.


Tensor and form contracts
-------------------------

Inputs must be non-null and belong to the manifold dimension. Generic tensor
fields encode the dimension in each component axis. K-forms and double forms
also carry an explicit ambient dimension, which is authoritative for scalar
rank-zero values. A scalar K-form may use ambient dimension zero to mean
unknown. Supported scalar operations infer the current manifold dimension,
matching the form-component contract. Axis variance is read from ``TensorMeta``.
Operations must change values and metadata together.

``Raise`` and ``Lower`` act on one selected axis using the inverse metric or
metric for the selected stratum. Their vector and one-form specializations are
preserved. Rank-one and rank-two operations use reconstructible symbolic
matrix-product nodes. Higher ranks use reconstructible einsum nodes.
Multi-index overloads apply the same checked single-index operation in the
supplied order.
All metric-backed algebra selects the volume, boundary, or edge metric and its
inverse through the same internal selector. New overloads should use that
selector so validation and cached metric identity cannot drift between
``Raise``, ``Lower``, ``InnerProduct``, and ``Trace``.

Python addition and subtraction require identical axis variance for typed
tensor fields. In particular, a vector field cannot be added to a one-form.
Combining a k-form with a generic tensor of the same covariant shape returns a
generic tensor, because the operation cannot prove that alternation is retained.
Multiplication and division between typed tensor wrappers are scalar scaling,
except for rank-two matrix products whose contracted axes have opposite
variance. That contraction returns a typed rank-two tensor with the two free
axis variances. ``VectorField * VectorField`` is deliberately not an inner
product: the vector objects do not own a metric, so geometric contraction must
be written as ``M.InnerProduct(v, w)``. Multiplication by an explicitly untyped,
non-scalar NGSolve coefficient retains NGSolve's native matrix/vector-product
behavior and returns an untyped coefficient, since no output variance can be
inferred.

``Trace`` contracts two selected tensor axes. Opposite-variance axes contract
directly. Equal-variance axes insert the appropriate metric. A double-form
trace repeatedly contracts the first left and right axes and returns a scalar
only after both degrees reach zero. ``TraceSigma`` raises both axes of its
``(1,1)`` sigma operand and contracts one left/right pair. ``SlotInnerProduct``
is the complete double-form trace, with factorial normalization when
``forms=True``.

``Contraction`` removes one tensor axis by pairing it with a vector. A
contravariant tensor axis inserts the metric before contraction.
``ContractSlot`` maps the double-form left/right slot convention onto that
primitive. ``Transpose`` is an axis permutation and therefore permutes both
component values and covariance metadata.

Tensor inner products pair every axis. Axes with opposite variance contract
directly. Equal-variance pairs insert one metric or inverse metric. With
``forms=True``, k-form and double-form overloads apply the appropriate
factorial normalization.


Semantic expression graphs
--------------------------

Public manifold algebra retains tensor and form wrappers as semantic operands.
This is essential even when an operand currently evaluates to zero. Native
NGSolve einsum, addition, or multiplication may simplify such an expression
and erase the wrapper. A later ``Replace`` or ``Diff`` targeting that wrapper
would then incorrectly remain zero.

Use ``SymbolicEinsumCF``, ``SymbolicSumCF``, ``ScaleCoefficientCF``, and
``PermuteTensorCF`` for public-operand algebra. These helpers keep semantic
operands for transformation and differentiation while delegating numerical
evaluation to efficient native coefficient functions. Do not pass
``GetFullCoefficient()`` to a new native algebra node merely as an
optimization: that accessor intentionally removes the wrapper identity.

``InputCoefficientFunctions`` and ``TraverseTree`` continue to expose the
native evaluation layout required by NGSolve evaluators, zero patterns, and
generated code. Policy checks that need the retained meaning of an expression,
such as rejecting a test proxy hidden by native zero simplification, use
``TraverseSemanticDAG`` instead. Do not change evaluation traversal to expose
semantic operands: the evaluator input order is a separate contract.

Low-rank metric products, inner products, traces, and matrix transposes use
specialized semantic nodes with native evaluators. This preserves the same
``Diff`` and ``Replace`` contract without routing common rank-one and rank-two
operations through a generalized einsum. Numerical evaluators may unwrap one
semantic layer, but recursive unwrapping during construction is forbidden: it
would repeatedly traverse shared descendants and turn linear DAG rebuilding
into quadratic work.

The contract covers musical maps, traces, tensor and slot inner products,
cross products, contractions, transposition, ``S``, ``J``, ``s``, and the
algebra used to assemble differential operators. Structural results that are
mathematically independent of the input, such as a degree-overflow form, may
still return a fresh zero.


Covariant differential operators
--------------------------------

``CovDerivative`` prepends a covariant derivative axis. It differentiates the
semantic input before adding one Christoffel term per component axis: a minus
term for a covariant axis and a plus term for a contravariant axis. A
zero-valued wrapper must take this same path. ``IsZeroCF`` is not sufficient to
prove independence from future differentiation or replacement targets.

``CovDerivative`` and ``CovHessian`` are the canonical public Python names.
``CovDeriv`` and ``CovHesse`` remain compatibility aliases and issue
``DeprecationWarning``. Existing scripts continue to run, while new code and
maintained examples should use the canonical spellings.

Only ``VOL`` and ``BND`` derivatives are supported. Boundary output is
projected after all derivative and connection terms are assembled. Composite
operators such as covariant Hessian, divergence, curl, ``inc``, ``ein``,
Laplacian, symmetric derivative, rotation, and the Lichnerowicz Laplacian reuse
these checked primitives and retain their intermediate semantic graphs.

``compile_inner=False`` preserves the original semantic input throughout the
derivative graph. ``compile_inner=True`` compiles one shared native value graph
before constructing the gradient and connection terms. This mode is a
performance option for evaluation, not a promise that later wrapper-targeted
``Replace`` or ``Diff`` can cross the compiled boundary. Trial and test proxies
are rejected in this mode because compiling their inner graph would violate
NGSolve proxy assembly semantics.

Exterior covariant derivatives and codifferentials of double forms first call
``CovDerivative``. They then permute, alternate, trace, scale, and optionally
project the result. Keep their scaling symbolic so a presently zero derivative
does not lose its source operand.

``CovDerivative`` and ``CovHessian`` are the canonical Python and C++ binding
names. ``CovDeriv`` and ``CovHesse`` remain compatibility aliases. The Python
wrapper emits ``DeprecationWarning`` when they are used. Existing programs
therefore continue to run, while new code and gradual migrations should use
the canonical spellings.


Performance evidence
--------------------

``benchmarks/benchmark_tensor_algebra.py`` checks values and records
construction time, assembly time, and coefficient-tree statistics for
``Raise``, ``Lower``, ``InnerProduct``, and ``Trace``. It also compares the
rank-one and rank-two ``Raise`` matrix-product implementations with the exact
reconstructible einsum alternatives in alternating order. Reports belong in
``benchmarks/results``. Correctness is smoke-tested in CI, but timing thresholds
are intentionally excluded because short assembly timings are host- and
load-sensitive.


Projection modes
----------------

Projection modes are validated before rank-zero or empty-slot shortcuts.
``ProjectTensor`` supports mode zero (identity), one (face tangent), two
(normal contraction of the first axis, with remaining axes face tangent), and
three (edge tangent). ``ProjectDoubleForm`` applies modes independently to the
left and right slots. Its mode two is a pure normal contraction. It additionally
supports mode four, contraction with a supplied conormal. By default, remaining
indices in that slot are edge-projected before this contraction, and
``project_remaining=False`` suppresses that projection. Edge modes require the
edge projector, conormal mode requires an explicit conormal, and contracting an
empty slot is an error.

Projectors are applied according to variance: covariant and contravariant axes
use transposed representations of the same geometric map. Projection nodes
retain their input tensor as a symbolic operand, as documented in
:doc:`developer_tensor_fields`.


Curvature conventions
---------------------

All curvature accessors trigger the same lazy initialization. For ordinary
coefficient-function metrics, curvature is derived from the physical metric
gradient and Christoffel symbols. Regge proxies and grid functions provide
native additional operators. The native Regge ``curvature`` output is divided
by ``det(g)`` so it agrees with the geometric curvature operator used here:
Gauss curvature in two dimensions and the contravariant curvature operator in
three dimensions.

``change_riemann_sign`` changes the Riemann convention consistently across the
ordinary and Regge paths. Tests compare normalized Regge, proxy, and analytic
results. Do not validate one curvature path only against another expression
that reuses the same convention code.


Validation and maintenance
--------------------------

Validate pointers, manifold dimensions, indices, modes, slots, and ``VorB``
values at public boundaries. Validation must precede shortcuts so invalid input
does not accidentally become an identity or zero. New public tensor operations
should have regression coverage for a zero-valued semantic wrapper under both
directional ``Diff`` and ``Replace``.

Private state should represent persistent manifold state only. Temporary proxy
handles and constructor-only orientation arguments remain local. Remove
disabled experimental coefficient-function classes and branches rather than
keeping unreachable alternatives beside the active graph implementation.


Verification
------------

After configuring a developer build and making that extension available to
Python, run the focused C++ and Python contracts first::

   cmake --build build --target test_riemannian_manifold
   ctest --test-dir build -R '^riemannian_manifold$' --output-on-failure
   python -m pytest -q tests/test_riemannian_manifold_contracts.py

Then run the manifold core, covariant-operator, double-form, projection,
compiled-inner-graph, and cross-component regression files. Changes to the
shared symbolic algebra or derivative construction require the full C++ and
Python suites, following :doc:`index`.
