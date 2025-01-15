#ifndef RIEMANNIAN_MANIFOLD
#define RIEMANNIAN_MANIFOLD

#include <fem.hpp>

namespace ngfem
{

    /*
     TODO
    */
    class RiemannianManifold
    {
        int dim;
        shared_ptr<CoefficientFunction> g;
        shared_ptr<CoefficientFunction> g_inv;
        shared_ptr<CoefficientFunction> vol[4];

    public:
        RiemannianManifold(shared_ptr<CoefficientFunction> _g);

        shared_ptr<CoefficientFunction> GetMetric() const;

        shared_ptr<CoefficientFunction> GetVolumeForm(VorB vb) const;
    };
}

#include <python_ngstd.hpp>
void ExportRiemannianManifold(py::module m);

#endif // RIEMANNIAN_MANIFOLD
