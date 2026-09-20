K-forms and double forms
========================

The public C++ interface lives in ``src/kforms.hpp``. The implementation uses
``src/kforms.cpp``, ``src/kforms_basis.cpp``, and ``src/kforms_bindings.cpp``.
The public Python adapters live in ``src/wrappers.py``. It builds on
:doc:`developer_tensor_fields` for metadata and value forwarding, and on
:doc:`developer_coefficient_grad` for physical derivatives.


Source layout
-------------

``src/kforms.cpp`` deliberately remains one translation unit. Its files below
``src/kforms/`` are textual C++ implementation fragments rather than separate
compilation units. Keeping the hot lookup, construction, and evaluation helpers
in one translation unit preserves cross-fragment inlining in Release builds.
Do not add the fragments as independent CMake sources. Editors should associate
``*.inc`` with C++ for syntax highlighting and symbol navigation.

The fragments are grouped by ownership:

* ``kforms_common.inc``, ``kforms_wrappers.inc``, and ``kforms_dispatch.inc``
  provide shared validation, semantic construction, and stable public dispatch.
* ``kforms_storage_facades.inc``, ``kforms_compact_facade.inc``, and
  ``kforms_symbolic_operations.inc`` own compact representation and semantic
  operation reconstruction.
* ``kforms_dense_nodes.inc`` and ``kforms_hodge_helpers.inc`` contain the exact
  dense/reference nodes and boundary or codimension Hodge helpers.
* ``kforms_metric.inc`` and ``kforms_trace_hodge.inc`` own induced metrics,
  inner products, trace, and compact Hodge evaluation.
* ``kforms_compact_algebra.inc`` owns compact wedge tables and evaluator nodes.
  ``kforms_algebra_operations.inc`` owns the representation-specific algebra
  factories, arithmetic, and slot operations.
* ``kforms_compact_exterior.inc`` owns the compact exterior-derivative table and
  evaluator, while ``kforms_exterior_factories.inc`` owns compact and dense
  exterior-derivative factories.
* ``kforms_cache_diagnostics.inc`` contains only cross-family cache-container
  accounting for tests and benchmarks. It is included at its original
  definition point so the organizational split does not change Release code
  layout.

``src/kforms_basis.hpp`` and ``src/kforms_basis.cpp`` own canonical form bases.
``src/kforms_internal.hpp`` is the narrow interface used by other native
components, while ``src/kforms_detail.hpp`` is a private implementation and
testing seam. ``src/kforms_diagnostics.hpp`` declares the unstable internal
benchmark diagnostics. ``src/kforms_bindings.cpp`` is the central Python-binding
entry point for the form component.


Representation and construction
-------------------------------

A K-form records a degree and an ambient dimension. A double form records two
degrees, with all left component axes preceding all right component axes.
Every axis is covariant. Constructors validate shape, degree, and ambient
dimension. They preserve the supplied values without checking or applying
alternation. In particular, a two-form constructor can contain nonzero
diagonal entries. Algorithms using exterior-algebra identities require
alternating input.

Canonical factories reuse compatible wrappers and remove metadata-only wrapper
layers when retyping. ``GetFullCoefficient()``, and Python ``.coef``, always
expose the full public tensor shape and values. They never expose a vector of
independent alternating components. Proven alternating results may nevertheless
keep only independent components behind an internal facade. Dense constructor
inputs are not compacted or projected automatically, so a non-alternating
coefficient retains all of its supplied values.

Keep the following limits distinct:

* ambient dimensions for form algebra: one through four
* full permutation or alternating-block length: at most four
* ordinary concrete form rank: at most eight
* physical gradients and nontrivial exterior derivatives: dimensions one through three, as required by ``GradCF``
* the tensor signature limit also bounds higher-rank shaped double-form zeros.

Scalar forms can carry unknown ambient dimension zero. Known dimensions must
match the manifold. Scalar inner products, slot inner products, and the scalar
codifferential accept the default unknown dimension, without mutating the input.
Inner-product results carry the manifold dimension. The scalar codifferential
retains the existing formal-zero dimension metadata. Hodge operations require
known ambient dimensions. Python wedge dispatch can infer a scalar operand's
dimension from the other operand. Component count one
does not imply scalar shape: a one-form in dimension one has dimensions
``(1,)``, while a scalar has dimensions ``()``.

Formal zeros with unknown dimension zero remain valid inputs to subsequent
formal operations. In particular, applying ``delta`` repeatedly to a default
scalar continues to decrease the formal degree without changing its unknown
dimension. A known, conflicting formal-zero dimension is still rejected.


Python arithmetic and dispatch
------------------------------

K-form and double-form multiplication and division accept scalar operands.
``alpha * beta`` for positive-degree K-forms raises ``TypeError``. Choose
``Wedge`` for an exterior product or an explicit ``InnerProduct`` for a
contraction. Plain tensor/matrix arithmetic is available through ``.coef``.
Scalar multiplication in either operand order preserves the form type and
component shape, including ``(1,)`` in ambient dimension one. Scalar fields
also retain NGSolve integration-measure syntax such as ``f * dx``.

The shared K-form arithmetic uses semantic wrappers as operands. It does not
strip them to ``.coef`` before building the graph. Addition, subtraction,
scaling, division, negation, symmetrization, wedge, Hodge operations, and exterior
differentiation retain zero-valued
operand wrappers too, so ``Diff`` and ``Replace`` preserve the operation rather
than treating its current zero value as permanent. K-form ``wedge``, ``d``, ``star``
and ``inv_star`` methods return the same Python form wrappers as the free
functions. Double-form methods and ``trans`` follow the same rule.
The native K-form and double-form classes receive these method surfaces from
one explicit operation list per type. Add a new method to that list and its
reflection test instead of maintaining separate assignment loops.

Explicit degrees and dimensions must be integers. Conversion helpers validate
requested metadata before reusing an existing Python wrapper. Ambient
inference uses ``dim_space`` or tensor axes. The flat coefficient ``dim`` is
not an ambient dimension. Component tuples are converted to a coefficient
before inference, including singleton and complex one-forms.
Known, conflicting operand dimensions are rejected.
Manifold operations validate dimensions and operation arguments before formal
or empty-slot zero shortcuts. Negative *formal degrees* still propagate.

Hodge slot aliases are normalized before calling C++: ``0``/``left``,
``1``/``right``, and ``-1``/``both``. String names are case-insensitive.


Operator conventions
--------------------

``AlternationCF`` is the unnormalized signed sum over all permutations.
Its rank-zero and rank-one cases are identities. Repeated-index outputs vanish.

``BlockAlternationByPermutationCF`` applies the same unnormalized operation
within one contiguous block, leaving the other axes in place. Shape, covariance,
dimension, and block bounds are validated even for an identity block of length
zero or one.

K-form wedge uses signed shuffles of the two input blocks. Double-form wedge
uses independent shuffles of the left and right blocks and returns the combined
left block followed by the combined right block. The exterior derivative is
``AlternationCF(GradCF(a), k+1, dim) / k!``. The gradient axis comes first.
These conventions must agree with independent component formulas, not merely
with another operator using the same permutation tables.

The Hodge operations validate their input and ambient dimension in C++ before
taking zero shortcuts. The intrinsic dimension is the ambient dimension minus
zero, one, or two for ``VOL``, ``BND``, or ``BBND``. Component axes still
use ambient coordinates on boundaries and edges. A single-slot double-form
star complements only its selected degree, including when the input is zero.
Only selected degrees are checked against the intrinsic dimension. For ambient
dimension two and ``BBND``, a selected degree-zero slot is an identity operation:
single-slot stars return the original operand, preserving differentiation and
replacement targets. Forward and inverse Python Hodge entrypoints share dispatch.
The method and free-function ``delta`` likewise use one implementation.

Python formal-zero objects retain degree information for operations outside
the concrete form range, including negative degrees. They are distinct from
full-shaped NGSolve zero coefficients. Preserve the existing dispatch and
materialization policies when changing a C++ zero shortcut.


Coefficient-function contracts
------------------------------

The semantic wrappers inherit evaluation, transformation, differentiation, and
archive behavior from the tensor-field component. Their ``Rewrap`` methods
retain degree and ambient-dimension metadata. Scalar Jacobians retain their
semantic wrapper. Tensor-valued Jacobians append variable axes and are untyped.
Differentiation with respect to the wrapper itself returns the supplied
direction or identity directly.

Native operation nodes, including alternation, compact arithmetic, wedge,
trace, induced metric, and Hodge maps, share the following contract. They:

* propagate child domains and elementwise-constant state
* implement both nonzero-pattern overloads, including first and second derivative dependencies
* expose their children through traversal and ``InputCoefficientFunctions``
* preserve the same child identities in ``Transform``, ``Diff``, and
  ``DiffJacobi``
* include operation parameters and child keys in their equivalence keys
* reconstruct through registered archive constructors.

Form operations use the shared symbolic-expression infrastructure described in
:doc:`developer_tensor_fields` to preserve operands through differentiation,
replacement, and serialization. ``SymbolicEinsumCF`` handles tensor products,
permutations, and Hodge contractions. ``SymbolicSumCF`` and
``ScaleCoefficientCF`` preserve signed sums and scalar factors.

Replacement reuses auxiliary subgraphs without replacement targets, so untouched
native inverse-metric nodes need no ``Transform`` support. Replacing inside
such native nodes still depends on their NGSolve support. Shared subgraphs
are visited once, including semantic operands removed by native simplification.

Do not replace these expressions with native ``+`` or ``*`` where a zero-valued
semantic operand must survive: native simplification may discard the operand
or return it directly, losing a factor. Hodge and double-wedge construction
likewise retain their operator graphs instead of returning a fresh zero solely
because an operand is zero. Degree-overflow zeros remain structural shortcuts.
``GradCF`` and ``HesseCF`` preserve zero-valued wrappers and dependent
intermediate expressions. A childless native zero can still be replaced with a
correctly shaped derivative zero.

Jacobians use directional derivatives of the semantic operands, including when
native evaluation simplifies the result to a childless zero. A zero-valued form
wrapper remains a differentiation target. Symbolic Jacobian columns also retain
zero-valued intermediate derivatives for subsequent mixed differentiation or
replacement. Native ``ZeroCF`` itself retains
NGSolve's constant-zero differentiation semantics.

Double wedge's children are the semantic operand wrappers. Replacing or
differentiating an operand wrapper must work as well as doing so with its
underlying value coefficient. Bypassing the wrapper when rebuilding the graph
breaks this contract.

Nonzero patterns conservatively propagate dependencies. They identify
structurally impossible repeated-index entries but need not detect every
cancellation. ``IsZeroCF()`` is likewise a structural test, not a numerical
test of whether an expression happens to evaluate to zero. Double wedge caches
its complete structural-zero result during construction and recomputes it when
reconstructed. Querying a shared wedge graph does not recursively expand both
operand branches.

``SymbolicEinsumCF`` evaluates nonzero patterns once per dependency in its
evaluation DAG, using the precomputed-child overloads. The cache lives only for
one pattern query, so changing proxy components cannot reuse stale patterns.
This also bounds pattern work when native einsum reconstructs a shared graph.


Evaluation, compilation, and archives
-------------------------------------

Full and block alternation reuse the shared signed-permutation cache when
building component lookup tables. Nodes do not store their own permutation
copies. Component tables remain local because they depend on shape and block
position. Separate classes and archive constructors are retained.

Form bases and the tables for compact expansion, trace, wedge, exterior
derivative, induced metrics, and Hodge maps are immutable and initialized with
``std::call_once``. They are keyed only by dimension and degrees. Evaluator
nodes retain a shared table handle rather than copying row, sign, permutation,
or metric-index vectors. When a new cache family is added, include both its
fixed container and owned dynamic storage in the private benchmark accounting.

Interpreted evaluation and generated code use the same signed lookup tables.
The nodes generate their assignments directly, using the tensor
component's shared declaration helper for real, complex, SIMD, and derivative
scalar types. Full alternation does not embed an evaluation-fallback pointer
in generated code. Rank-zero alternation and scalar double wedge also support
the native scalar point-evaluation overload, including use as conditions in
complex-valued ``IfPos`` expressions.

Generated signed sums must initialize the first term explicitly. Applying
``operator-=`` to an empty ``CodeExpr`` loses a leading minus sign. Use the
shared signed-accumulation helper for every generated contraction or
permutation sum, and cover a leading-negative case with
``realcompile=True``.

``Compile(realcompile=False)`` tests graph compilation. ``realcompile=True``
also invokes native code generation. Compilation preserves values and shapes
but returns an untyped NGSolve compiled coefficient. Tests must exercise the
internal operators, not just a semantic wrapper around a simple coefficient.

Archive constructor arguments contain the children and defining operation
parameters. Derived dimensions and lookup tables are reconstructed from those
arguments. When modifying constructor state, update pickle tests in the same
change. Metadata stored in Python instance attributes is not authoritative.


Verification
------------

After configuring a developer build and making that extension available to
Python, run::

 cmake --build build --target cpp_tests
 ctest --test-dir build --output-on-failure
 CCACHE_DISABLE=1 python -m pytest -q tests/test_kforms.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_symbolic_einsum_regressions.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_zero_form_symbolics.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_wrappers_kforms_*.py
 CCACHE_DISABLE=1 python -m pytest -q tests/test_wrappers_hodge_star_delta.py

``test_kforms.cpp`` checks domain/constant propagation, equivalence keys,
nonzero-pattern overloads, direct code generation, validation, and native
scalar/SIMD evaluation. Traversal and pattern counters check shared-graph
replacement without a timing threshold. Its tests join the existing ``cpp_tests`` aggregate
on platforms supporting addon-linked C++ tests.

``test_kforms.py`` includes independent component references, compilation and
pickle checks, operand replacement/Jacobians, zero-star metadata, public
dimension validation, and subprocess crash regressions for null manifolds.
``test_wrappers_kforms_regressions.py`` covers public Python arithmetic,
factory validation, singleton dispatch, null shortcuts, slot aliases,
method return types, operand identities, contracted Jacobians, and compiled
and pickled operation graphs.
``test_symbolic_einsum_regressions.py`` checks nested replacements before and
after pickling and zero-form Jacobians against independent basis directions.
``test_zero_form_symbolics.py`` checks zero operands under scaling, division,
negation, scalar and positive-degree wedges, Hodge operations, exterior
differentiation, and composition against explicit component formulas. It also
checks repeated scalar codifferentials and conflicting formal-zero dimensions.
Spatial differentiation is tested with nonconstant directions, since it does
not have a pointwise matrix Jacobian.
The existing tensor-wrapper, covariant-operator, formal-zero, and
``TaskManager`` tests provide downstream coverage. Run the full suites after
changes to child identity, code generation, archives, or dispatch.

Performance comparisons belong in dedicated benchmarks. Compact storage must
be compared against this corrected dense implementation and independent
component references. Component-count reductions alone are not timing results.
