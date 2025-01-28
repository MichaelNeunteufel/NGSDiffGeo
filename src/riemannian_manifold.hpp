#ifndef RIEMANNIAN_MANIFOLD
#define RIEMANNIAN_MANIFOLD

#include <fem.hpp>

namespace ngfem
{

    class TensorFieldCoefficientFunction;
    class VectorFieldCoefficientFunction;

    class RiemannianManifold
    {
        int dim;
        bool has_trial;
        bool is_regge;

        // metric tensor, its inverse, derivative, and volume forms
        shared_ptr<CoefficientFunction> g;
        shared_ptr<CoefficientFunction> g_inv;
        shared_ptr<CoefficientFunction> g_deriv;
        shared_ptr<CoefficientFunction> vol[4];

        // Christoffel symbols of first and second kind
        shared_ptr<CoefficientFunction> chr1;
        shared_ptr<CoefficientFunction> chr2;

        // Euclidean and g-normalized normal and tangent vectors
        shared_ptr<CoefficientFunction> nv;
        shared_ptr<CoefficientFunction> tv;
        shared_ptr<CoefficientFunction> g_nv;
        shared_ptr<CoefficientFunction> g_tv;

    public:
        RiemannianManifold(shared_ptr<CoefficientFunction> _g);

        shared_ptr<CoefficientFunction> GetMetric() const;

        shared_ptr<CoefficientFunction> GetMetricInverse() const;

        shared_ptr<CoefficientFunction> GetVolumeForm(VorB vb) const;

        shared_ptr<CoefficientFunction> GetChristoffelSymbol(bool second_kind) const;

        shared_ptr<CoefficientFunction> GetNV() const;
        shared_ptr<CoefficientFunction> GetEdgeTangent(bool consistent) const;

        // ------- Tensor operations --------
        shared_ptr<CoefficientFunction> IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const;

        shared_ptr<CoefficientFunction> Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const;

        // ------- Covariant differential operators --------
        shared_ptr<CoefficientFunction> CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovCurl(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        // ------- Trace and contraction --------
        shared_ptr<CoefficientFunction> Trace(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index1 = 0, size_t index2 = 1) const;

        shared_ptr<CoefficientFunction> Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot = 0) const;
    };
}

#include <python_ngstd.hpp>
void ExportRiemannianManifold(py::module m);

#endif // RIEMANNIAN_MANIFOLD
