Tensor fields
=============

This page documents the tensor metadata and coefficient-function wrappers in
``src/tensor_fields.hpp`` and ``src/tensor_fields.cpp``. The corresponding
``TensorField`` and ``VectorField`` adapters in ``src/wrappers.py`` are included
where they define the Python-facing behavior.

K-forms and double forms derive from the tensor wrapper, but their degrees,
ambient dimensions, alternation, and wedge operations belong to the separate
k-form component documentation.


Role and representation
-----------------------

``TensorFieldCoefficientFunction`` wraps an NGSolve ``CoefficientFunction``
without changing its component values. The wrapped coefficient remains the
full-shaped value graph. The tensor wrapper adds ordered variance metadata so
geometric operations can distinguish covariant and contravariant axes.

``TensorMeta`` is the authoritative metadata representation. Character ``i``
of the public covariance string and bit ``i`` of its internal mask describe
component axis ``i``: ``"1"`` means covariant and ``"0"`` contravariant. Tensor
labels use the same axis order. The available einsum alphabet limits the rank
to ``MAX_SIGNATURE_LABELS`` (currently 52).

The metadata helpers preserve that ordering when they append, prepend, erase,
or concatenate axes. Operations needing an additional einsum label must obtain
it through ``FreshLabel`` so the rank limit is checked consistently.


Wrapper invariants
------------------

Every ``TensorFieldCoefficientFunction`` satisfies these invariants after
construction:

* the wrapped coefficient is non-null,
* the metadata rank equals ``coef->Dimensions().Size()``, and
* all component axes have the same dimension.

Rank-zero tensor fields are valid. ``VectorFieldCoefficientFunction`` is the
rank-one specialization with fixed covariance string ``"0"``.

The C++ object owns the authoritative metadata. Python exposes
``covariant_indices`` and the full-shaped value graph as read-only properties.
Python code should not duplicate this state in instance attributes.


Canonical wrapping
------------------

Construct C++ wrappers through ``TensorFieldCF`` and ``VectorFieldCF``. These
factories centralize validation and prevent repeated retyping from growing the
expression graph.

``TensorFieldCF`` reuses an existing tensor wrapper when its ``TensorMeta``
matches. Otherwise it removes tensor-wrapper layers until reaching the
underlying value graph and constructs a wrapper with the requested metadata.
This also means that explicit generic retyping can intentionally discard a
derived semantic type, such as a form, while preserving its full component
values.

``VectorFieldCF`` similarly reuses an existing vector wrapper, removes other
tensor-wrapper layers, and then verifies that the value graph has rank one.

On the Python side, arithmetic implemented by ``TensorField`` and
``VectorField`` operates on ``.coef`` and calls the corresponding ``as_*``
adapter to restore metadata. ``as_tensorfield`` also dispatches the signatures
``""``, ``"0"``, and ``"1"`` to the scalar-field, vector-field, and one-form
specializations. An explicitly requested covariance string is validated and
applied even when the input is already a Python tensor wrapper. Only an omitted
or matching string permits immediate reuse. Numeric scaling accepts complex
scalars in either operand order. Derived objects define their additional
behavior in their own components.


Value-graph behavior
--------------------

The tensor wrapper forwards evaluation, domain checks, derivative evaluation,
zero detection, and nonzero patterns to the wrapped coefficient. Generated
code declares an output with the same shape and real or complex scalar type,
then copies the generated input values into it.

``GetFullCoefficient`` is the C++ accessor for the full-shaped semantic value
graph. Python ``.coef`` exposes the same graph. Canonical conversions use this
accessor when removing metadata-only wrapper layers. Operations whose
operands must remain targets for ``Diff`` or ``Replace`` instead retain those
semantic operand wrappers in their expression graphs.

``SymbolicExpressionCoefficientFunction`` in ``src/symbolic_expression.*``
shares evaluation, transformation, nonzero-pattern caching, and Jacobian
construction between tensor algebra and proxy derivatives. It retains semantic
operands independently of the native evaluation graph. Evaluation, child
discovery, nonzero patterns, and code generation must use a consistent graph
layout that keeps native proxy leaves visible to NGSolve.
``SymbolicEinsumCF`` retains its original operands for symbolic operations while
removing metadata-only wrapper layers from the native evaluation inputs.
Projection uses this same helper: its archive retains the original tensor
operand, and ``Replace`` reconstructs the contraction from that operand.

``Transform`` transforms the value graph and reconstructs the semantic wrapper
through the virtual ``Rewrap`` hook. It also participates in NGSolve's
replacement and transformation caches. Derived tensor types override only
``Rewrap`` when the base transformation behavior is sufficient.

Directional ``Diff`` also uses ``Rewrap``, except when the wrapper itself is the
differentiation variable, in which case NGSolve's direction is returned
directly.

For Jacobian differentiation:

* a scalar-valued variable appends no component axis, so ``Rewrap`` preserves the original semantic wrapper,
* a tensor-valued variable appends axes whose variance is not defined by the original wrapper, so the Jacobian is an untyped ``CoefficientFunction``, and
* differentiating a wrapper with respect to itself returns NGSolve's identity coefficient function directly.

``Compile`` preserves shape, scalar type, and values, but returns NGSolve's
compiled coefficient-function type. Code that subsequently needs tensor
metadata must wrap the compiled coefficient explicitly.


Tensor operations
-----------------

The operations implemented in ``tensor_fields.cpp`` preserve axis order and
metadata as follows:

``TensorProduct``
   Concatenates the left axes with the right axes, concatenates their
   ``TensorMeta`` objects in the same order, and uses disjoint einsum labels.
   The combined rank may not exceed ``MAX_SIGNATURE_LABELS``.

``PermuteTensorCF``
   Validates that the supplied order is a permutation of all axes, then applies
   the same permutation to component values and the covariance string.

``SymbolicEinsumCF``
   Delegates evaluation and code generation to native einsum, and reconstructs
   transformations and derivatives from the original signature and operands.
   Tensor products, permutations, and the contractions used by form Hodge
   operations use this helper. Its constructor arguments reconstruct the
   native evaluator on archive input. See :doc:`developer_kforms` for the
   operand-identity contract and native auxiliary-node limitations.

``SymbolicSumCF`` and ``ScaleCoefficientCF``
   Preserve operands when constructing sums and scalar products, including
   operands whose current value is zero. Form operations use these helpers where native
   zero simplification would otherwise erase an operand or a scalar factor.

``ApplyProjectorToIndex``
   Contracts a projector with one selected component axis. It uses a fresh
   einsum label while retaining the original output label, slot position, and
   covariance metadata.

``IsVectorField`` and ``IsOneForm``
   Inspect ``TensorMeta`` only: they recognize rank-one covariance masks zero
   and one, respectively. They do not require a particular dynamic subclass.


Equivalence and serialization
-----------------------------

The equivalence key contains the dynamic description, covariance string, and
wrapped value-graph key. Derived semantic types with additional metadata must
override the key calculation and include that state.

``TensorFieldCoefficientFunction::GetCArgs`` archives the full coefficient and
covariance string. ``VectorFieldCoefficientFunction::GetCArgs`` archives the
full coefficient because its variance is fixed. Both dynamic types are
registered with their immediate archive-aware base, and both Python bindings
provide ``NGSPickle``.

When constructor state changes, update ``GetCArgs`` and the pickle round-trip
tests in the same change. Python instance attributes are not a substitute for
restoring C++ metadata.


Extending tensor fields
-----------------------

For a new class derived from ``TensorFieldCoefficientFunction``:

#. Validate its shape and semantic metadata during construction.
#. Override ``Rewrap`` and delegate reconstruction to a canonical factory.
#. Include all additional metadata in its equivalence key and ``GetCArgs``.
#. Register the type for polymorphic archiving and add ``NGSPickle`` when the
   concrete type is exposed to Python.
#. Test construction, replacement, directional differentiation, scalar and
   tensor-valued Jacobians, compilation, pickling, and wrapper flattening.


Component verification
----------------------

After building and installing the extension as described in :doc:`index`, run
the tensor metadata and forwarding tests::

   ctest --test-dir build \
       -R '^(tensor_meta|tensor_field_forwarding)$' \
       --output-on-failure

The current Python wrapper regression file also contains downstream form and
manifold cases because those types derive from or consume tensor fields::

   python -m pytest -q tests/test_wrappers_tensor_fields.py

Changes to the base wrapper, generated code, or serialization can affect every
derived form type. For those changes, follow the full-suite guidance in
:doc:`index`.


Zero-valued Jacobians
---------------------

Jacobian axes are the result axes followed by the variable axes. Construction
retains every directional column in a symbolic Jacobian node, even when native
stacking simplifies its evaluation to zero. This is needed for mixed derivatives
and subsequent replacement: for zero scalar wrappers ``a`` and ``b``,
``Wedge(a, b).Diff(a).Diff(b, CF(1))`` must still equal one. Column dependencies
survive archive round trips. A childless native ``ZeroCF`` remains a constant.
Its differentiation behavior is not changed.
