Installation
-----------------

The package will soon be available on PyPI, but for now you can install it from the GitHub repository if you have a self-compiled NGSolve.

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

