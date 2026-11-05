Installation
------------

NGSDiffGeo is distributed as a standard Python package and can be installed either

* from `PyPI <https://pypi.org/project/ngsdiffgeo/>`_ (recommended), or
* from source (advanced / developer setup).

NGSDiffGeo requires `NGSolve <https://ngsolve.org/>`_
6.2.2608.post166.dev0 or newer. For NGSolve 6.2.2607 or the stable
6.2.2608 release, use NGSDiffGeo v0.3.1.

``pip`` installs the package's Python dependencies automatically. For
NGSolve installation options and supported platforms, see
`the official NGSolve installation guide <https://ngsolve.org/installation.html>`_.


Install from PyPI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing NGSDiffGeo into a dedicated virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .\.venv\Scripts\Activate.ps1 # Windows (PowerShell)

   python -m pip install --upgrade pip
   python -m pip install ngsdiffgeo

If no compatible NGSolve wheel is available for your platform, install a
compatible NGSolve build first, following the guide linked above. Then build
NGSDiffGeo from source against that installation as described below.


Build from source (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build from source to develop NGSDiffGeo or compile it against an existing
NGSolve installation.

The build uses the NGSolve installation visible to its Python interpreter.
Build and run NGSDiffGeo in the same environment. Reconfigure and rebuild
NGSDiffGeo after changing the NGSolve version or installation.

Prerequisites
^^^^^^^^^^^^^

1. A compatible NGSolve installation that can be imported by the Python
   interpreter used for the build.
2. A C++ toolchain and CMake (Linux/macOS: GCC/Clang; Windows: Visual Studio).
3. Python build tools for the chosen installation method. The pip build
   requirements are listed in ``[build-system].requires`` in ``pyproject.toml``;
   the CMake installation also uses ``pybind11_stubgen`` to generate type stubs.

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

To select a specific Python installation, configure CMake with
``cmake .. -DPython3_EXECUTABLE=/path/to/python``. That interpreter must be
able to import NGSolve.


Option B: Build and install with pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These commands use the ``pyproject.toml`` build configuration and the NGSolve
installation in the active Python environment. ``--no-build-isolation``
requires the build dependencies listed in ``[build-system].requires`` to be
installed in that environment before running pip.

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install --no-build-isolation .

For editable development installs:

.. code-block:: bash

   python -m pip install --no-build-isolation -e .

If pip does not recognize a compatible local NGSolve build from its package
metadata, add ``--no-deps`` to either command. This skips installation of all
package dependencies, so use it only when those dependencies are already
installed.


Development snapshots (TestPyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For testers and early adopters, development snapshots are published to TestPyPI. These builds can change
frequently and may be unstable.

To install from TestPyPI (while still resolving dependencies from PyPI):

.. code-block:: bash

   python -m pip install -i https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple \
     ngsdiffgeo
