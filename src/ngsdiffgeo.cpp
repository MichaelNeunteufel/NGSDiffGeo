#include "riemannian_manifold.hpp"

PYBIND11_MODULE(ngsdiffgeo, m)
{
    cout << "Loading ngsdiffgeo" << endl;

    ExportRiemannianManifold(m);
}