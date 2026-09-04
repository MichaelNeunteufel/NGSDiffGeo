Developer guide
===============

This guide is for contributors who modify NGSDiffGeo's C++ implementation,
Python bindings, or wrapper behavior. User-facing definitions and examples
belong in the tutorials. Installation instructions are in :doc:`../install`.

Component pages document stable implementation contracts: ownership,
invariants, representation choices, extension points, and behavior that is not
obvious from an individual function. Function signatures and local algorithmic
details remain in the source.


Documentation policy
--------------------

Developer pages should:

* identify the source files and responsibilities owned by the component,
* record only the cross-component relationships needed to use or extend it,
* describe behavior and invariants rather than reproduce implementation steps,
* keep mathematical definitions, derivations, and executable examples in the
  tutorials or technical report, and
* update the relevant tests and documentation when a documented contract
  changes.


Verification strategy
---------------------

Use the smallest test set that covers the change, then expand according to its
impact:

* documentation-only changes require a warning-free Sphinx build,
* component-local changes require that component's C++ and Python tests, and
* changes to shared base classes, serialization, generated code, or wrapper
  dispatch require the dependent tests and normally the full suites.

Build and install the current extension before running Python tests. Package
builds exclude the C++ test executables, so configure a developer build with
``-DBUILD_TESTING=ON`` when they are needed. The source installation options
are described in :doc:`../install`. With ``build`` as the CMake build
directory, the common commands are::

   cmake --build build
   cmake --install build
   ctest --test-dir build --output-on-failure
   python -m pytest -q

The normal documentation configuration executes every listed notebook. For a
fast documentation check that parses and renders notebooks without running
their cells, use this command from the repository root::

   sphinx-build -W --keep-going -b html \
       -D nbsphinx_execute=never docs docs/_build/html-fast

Use the regular ``make -C docs html`` build when notebook execution itself must
be verified.


Components
----------

.. toctree::
   :maxdepth: 1

   developer_tensor_fields
   developer_coefficient_grad
