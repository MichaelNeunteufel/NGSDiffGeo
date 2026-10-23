Installation
------------

NGSDiffGeo is distributed as a standard Python package and can be installed either

* from `PyPI <https://pypi.org/project/ngsdiffgeo/>`_ (recommended), or
* from source (advanced / developer setup).

The package requires `NGSolve <https://ngsolve.org/>`_ 6.2.2607 or newer.
When installing from PyPI, ``pip`` will also install ``ngsolve`` automatically
*if* a compatible wheel is available for your platform. For details on NGSolve
installation options (pip, conda, or building from source), see
`the official NGSolve installation guide <https://ngsolve.org/installation.html>`_.

For fixes listed under ``Unreleased`` in the changelog, build from a source
checkout containing the fix.


Install from PyPI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing NGSDiffGeo into a dedicated virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate # Windows (PowerShell)

   python -m pip install --upgrade pip
   python -m pip install ngsdiffgeo

If your platform does not provide a prebuilt ``ngsolve`` wheel, install NGSolve first (see the link above),
then install NGSDiffGeo via ``pip`` as shown.


Build from source (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A source build is useful if you want to develop NGSDiffGeo or need to compile against a local NGSolve build.

The build uses the NGSolve installation visible to its Python interpreter.
Build and run NGSDiffGeo in the same environment. Reconfigure and rebuild
NGSDiffGeo after changing the NGSolve version or installation.

Prerequisites
^^^^^^^^^^^^^

1. A working NGSolve installation (pip-installed or self-compiled). See
   `NGSolve installation <https://ngsolve.org/installation.html>`_.
2. A C++ toolchain and CMake (Linux/macOS: GCC/Clang; Windows: Visual Studio).
3. Standard Python build tooling (``pip``). Build requirements are declared in ``pyproject.toml``.

Clone the repository:

.. code-block:: bash

   git clone https://github.com/MichaelNeunteufel/NGSDiffGeo.git
   cd NGSDiffGeo


Option A: Build with CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir -p build
   cd build
   cmake ..
   make install # Linux/macOS
   # cmake --build . --config Release --target install # Windows

If CMake cannot locate your NGSolve installation, you may need to set ``CMAKE_PREFIX_PATH`` to the directory
containing NGSolve/Netgen CMake package configuration files (platform-specific).


Option B: Build and install with pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This uses the ``pyproject.toml`` build configuration and is typically the most convenient for development.

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install --no-build-isolation .

For editable development installs:

.. code-block:: bash

   python -m pip install --no-build-isolation -e .


Development snapshots (TestPyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For testers and early adopters, development snapshots are published to TestPyPI. These builds can change
frequently and may be unstable.

To install from TestPyPI (while still resolving dependencies from PyPI):

.. code-block:: bash

   python -m pip install -i https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple \
     ngsdiffgeo