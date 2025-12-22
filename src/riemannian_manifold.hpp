#ifndef RIEMANNIAN_MANIFOLD
#define RIEMANNIAN_MANIFOLD

// #include <fem.hpp>
#include <coefficient.hpp>
#include <fespace.hpp>
#include <symbolicintegrator.hpp>

namespace ngfem
{

    class TensorFieldCoefficientFunction;
    class VectorFieldCoefficientFunction;
    class ScalarFieldCoefficientFunction;
    class KFormCoefficientFunction;

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
        shared_ptr<CoefficientFunction> det_g;
        shared_ptr<CoefficientFunction> g_F;
        shared_ptr<CoefficientFunction> g_F_inv;
        shared_ptr<CoefficientFunction> g_E;
        shared_ptr<CoefficientFunction> g_E_inv;
        shared_ptr<CoefficientFunction> g_deriv;
        shared_ptr<CoefficientFunction> vol[4];

        // Christoffel symbols of first and second kind
        shared_ptr<CoefficientFunction> chr1;
        shared_ptr<CoefficientFunction> chr2;

        // curvature quantities
        shared_ptr<TensorFieldCoefficientFunction> Riemann;
        shared_ptr<TensorFieldCoefficientFunction> Curvature;
        shared_ptr<TensorFieldCoefficientFunction> Ricci;
        shared_ptr<TensorFieldCoefficientFunction> Einstein;
        shared_ptr<TensorFieldCoefficientFunction> Scalar;

        shared_ptr<TensorFieldCoefficientFunction> SFF;

        // Euclidean and g-normalized normal and tangent vectors
        shared_ptr<CoefficientFunction> nv;
        shared_ptr<CoefficientFunction> tv;
        shared_ptr<VectorFieldCoefficientFunction> g_nv;
        shared_ptr<VectorFieldCoefficientFunction> g_tv;

        shared_ptr<CoefficientFunction> P_n;
        shared_ptr<CoefficientFunction> P_F;

        mutable shared_ptr<TensorFieldCoefficientFunction> levi_civita_cov;
        mutable shared_ptr<TensorFieldCoefficientFunction> levi_civita_contra;

    public:
        /**
         * @fn RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g)
         * @brief Constructor for RiemannianManifold.
         * @param _g The metric tensor.
         */
        RiemannianManifold(shared_ptr<CoefficientFunction> _g);

        int GetDimension() const { return dim; }

        // ------- Metric tensor and related quantities --------
        shared_ptr<CoefficientFunction> GetMetric() const;

        shared_ptr<CoefficientFunction> GetMetricInverse() const;

        shared_ptr<CoefficientFunction> GetVolumeForm(VorB vb) const;
        int Dimension() const { return dim; }

        // -------  musical isomorphisms -------
        shared_ptr<TensorFieldCoefficientFunction> Raise(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index = 0) const;

        shared_ptr<TensorFieldCoefficientFunction> Lower(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index = 0) const;

        shared_ptr<TensorFieldCoefficientFunction> GetLeviCivitaSymbol(bool covariant) const;

        shared_ptr<CoefficientFunction> GetMetricDerivative() const;

        shared_ptr<CoefficientFunction> GetChristoffelSymbol(bool second_kind) const;

        // Full 4th order Riemann curvature tensor
        shared_ptr<TensorFieldCoefficientFunction> GetRiemannCurvatureTensor() const;

        // Ricci tensor, symmetric dim x dim matrix
        shared_ptr<TensorFieldCoefficientFunction> GetRicciTensor() const;

        // Einstein tensor, 0 for dim < 3, otherwise symmetric dim x dim matrix
        shared_ptr<TensorFieldCoefficientFunction> GetEinsteinTensor() const;

        // Scalar curvature (twice contracted Riemann tensor; trace of Ricci tensor)
        shared_ptr<ScalarFieldCoefficientFunction> GetScalarCurvature() const;

        // Gauss curvature in 2D
        shared_ptr<ScalarFieldCoefficientFunction> GetGaussCurvature() const;

        // Curvature operator
        // 2D -> scalar Gauss curvature, 3D -> 3x3 symmetric curvature operator
        shared_ptr<TensorFieldCoefficientFunction> GetCurvatureOperator() const;

        // ------- second fundamental form --------
        shared_ptr<TensorFieldCoefficientFunction> GetSecondFundamentalForm() const;
        shared_ptr<ScalarFieldCoefficientFunction> GetGeodesicCurvature() const;
        shared_ptr<ScalarFieldCoefficientFunction> GetMeanCurvature() const;

        // ------- Normal and tangent vectors --------
        shared_ptr<VectorFieldCoefficientFunction> GetNV() const;
        shared_ptr<VectorFieldCoefficientFunction> GetEdgeTangent() const;

        // ------- Tensor operations --------
        shared_ptr<ScalarFieldCoefficientFunction> IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2, VorB vb = VOL) const;

        shared_ptr<TensorFieldCoefficientFunction> Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const;

        // ------- Forms --------
        shared_ptr<KFormCoefficientFunction> MakeKForm(shared_ptr<CoefficientFunction> cf, int k) const;
        shared_ptr<KFormCoefficientFunction> Star(shared_ptr<KFormCoefficientFunction> a, VorB vb = VOL) const;
        shared_ptr<KFormCoefficientFunction> Coderivative(shared_ptr<KFormCoefficientFunction> a) const;

        // ------- Covariant differential operators --------
        shared_ptr<TensorFieldCoefficientFunction> CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb = VOL) const;

        shared_ptr<TensorFieldCoefficientFunction> CovHessian(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<TensorFieldCoefficientFunction> CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb = VOL) const;

        shared_ptr<TensorFieldCoefficientFunction> CovCurl(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<TensorFieldCoefficientFunction> CovInc(shared_ptr<TensorFieldCoefficientFunction> c1, bool matrix = false) const;

        shared_ptr<TensorFieldCoefficientFunction> CovEin(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<TensorFieldCoefficientFunction> CovLaplace(shared_ptr<TensorFieldCoefficientFunction> c1) const;
        shared_ptr<TensorFieldCoefficientFunction> CovDef(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<TensorFieldCoefficientFunction> CovRot(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        shared_ptr<TensorFieldCoefficientFunction> LichnerowiczLaplacian(shared_ptr<TensorFieldCoefficientFunction> c1) const;

        // ------- Algebraic operations --------
        shared_ptr<TensorFieldCoefficientFunction> Trace(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index1 = 0, size_t index2 = 1, VorB vb = VOL) const;

        shared_ptr<TensorFieldCoefficientFunction> Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot = 0) const;

        shared_ptr<TensorFieldCoefficientFunction> Transpose(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1 = 0, size_t index2 = 1) const;

        shared_ptr<TensorFieldCoefficientFunction> S_op(shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb = VOL) const;
        shared_ptr<TensorFieldCoefficientFunction> J_op(shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb = VOL) const;
    };
}

#include <python_ngstd.hpp>
void ExportRiemannianManifold(py::module m);

#endif // RIEMANNIAN_MANIFOLD
