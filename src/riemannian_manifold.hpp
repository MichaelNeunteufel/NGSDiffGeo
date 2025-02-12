#ifndef RIEMANNIAN_MANIFOLD
#define RIEMANNIAN_MANIFOLD

#include <fem.hpp>

namespace ngfem
{

    class TensorFieldCoefficientFunction;
    class VectorFieldCoefficientFunction;

    /**
     * @class RiemannianManifold
     * @brief Represents a Riemannian manifold with various geometric and differential properties.
     *
     * This class encapsulates the properties and operations related to a Riemannian manifold,
     * including metric tensors, Christoffel symbols, curvature tensors, and various covariant
     * differential operators.
     */
    class RiemannianManifold
    {
        // Dimension of the manifold
        int dim;
        bool has_trial;
        bool is_regge;
        bool is_proxy;
        shared_ptr<ProxyFunction> regge_proxy;
        shared_ptr<ngcomp::FESpace> regge_space;

        // metric tensor, its inverse, derivative, and volume forms
        shared_ptr<CoefficientFunction> g;
        shared_ptr<CoefficientFunction> g_inv;
        shared_ptr<CoefficientFunction> g_deriv;
        shared_ptr<CoefficientFunction> vol[4];

        // Christoffel symbols of first and second kind
        shared_ptr<CoefficientFunction> chr1;
        shared_ptr<CoefficientFunction> chr2;

        // curvature quantities
        shared_ptr<CoefficientFunction> Riemann;
        shared_ptr<CoefficientFunction> Curvature;
        shared_ptr<CoefficientFunction> Ricci;
        shared_ptr<CoefficientFunction> Einstein;
        shared_ptr<CoefficientFunction> Scalar;

        // Euclidean and g-normalized normal and tangent vectors
        shared_ptr<CoefficientFunction> nv;
        shared_ptr<CoefficientFunction> tv;
        shared_ptr<CoefficientFunction> g_nv;
        shared_ptr<CoefficientFunction> g_tv;

    public:
        /**
         * @fn RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g)
         * @brief Constructor for RiemannianManifold.
         * @param _g The metric tensor.
         */
        RiemannianManifold(shared_ptr<CoefficientFunction> _g);

        // ------- Metric tensor and related quantities --------
        shared_ptr<CoefficientFunction> GetMetric() const;

        shared_ptr<CoefficientFunction> GetMetricInverse() const;

        shared_ptr<CoefficientFunction> GetVolumeForm(VorB vb) const;

        // -------  musical isomorphisms -------
        shared_ptr<CoefficientFunction> Raise(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index = 0) const;

        shared_ptr<CoefficientFunction> Lower(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index = 0) const;

        shared_ptr<CoefficientFunction> GetLeviCivitaSymbol(bool covariant) const;

        shared_ptr<CoefficientFunction> GetMetricDerivative() const;

        shared_ptr<CoefficientFunction> GetChristoffelSymbol(bool second_kind) const;

        // Full 4th order Riemann curvature tensor
        shared_ptr<CoefficientFunction> GetRiemannCurvatureTensor() const;

        // Ricci tensor, symmetric dim x dim matrix
        shared_ptr<CoefficientFunction> GetRicciTensor() const;

        // Einstein tensor, 0 for dim < 3, otherwise symmetric dim x dim matrix
        shared_ptr<CoefficientFunction> GetEinsteinTensor() const;

        // Scalar curvature (twice contracted Riemann tensor; trace of Ricci tensor)
        shared_ptr<CoefficientFunction> GetScalarCurvature() const;

        // Curvature operator
        // 2D -> scalar Gauss curvature, 3D -> 3x3 symmetric curvature operator
        shared_ptr<CoefficientFunction> GetCurvatureOperator() const;

        // ------- Normal and tangent vectors --------
        shared_ptr<CoefficientFunction> GetNV() const;
        shared_ptr<CoefficientFunction> GetEdgeTangent() const;

        // ------- Tensor operations --------
        shared_ptr<CoefficientFunction> IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const;

        shared_ptr<CoefficientFunction> Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const;

        // ------- Covariant differential operators --------
        shared_ptr<CoefficientFunction> CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovHessian(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovCurl(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovInc(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<CoefficientFunction> CovRot(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        // ------- Algebraic operations --------
        shared_ptr<CoefficientFunction> Trace(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index1 = 0, size_t index2 = 1) const;

        shared_ptr<CoefficientFunction> Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot = 0) const;

        shared_ptr<CoefficientFunction> Transpose(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1 = 0, size_t index2 = 1) const;
    };
}

#include <python_ngstd.hpp>
void ExportRiemannianManifold(py::module m);

#endif // RIEMANNIAN_MANIFOLD
