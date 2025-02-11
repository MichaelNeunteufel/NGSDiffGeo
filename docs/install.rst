Installation
-----------------

If you DON'T have a NGSolve installation you can install NGSDiffGeo using pip

.. code-block:: bash

    pip install ngsdiffgeo

or using the GitHub repository

.. code-block:: bash

    python -m pip install  git+https://github.com/MichaelNeunteufel/NGSDiffGeo.git

**Alternative** needed for self-compiled NGSolve

If you have a working NGSolve installation, you need to clone the GitHub repository, and then you can build it from source using CMake

.. code-block:: bash

    git clone https://github.com/MichaelNeunteufel/NGSDiffGeo.git
    cd NGSDiffGeo
    mkdir build && cd build
    cmake ..
    make install

or using pip 

.. code-block:: bash
    
    python -m pip install scikit-build-core pybind11_stubgen toml
    git clone https://github.com/MichaelNeunteufel/NGSDiffGeo.git
    cd NGSDiffGeo
    python -m pip install --no-build-isolation .

