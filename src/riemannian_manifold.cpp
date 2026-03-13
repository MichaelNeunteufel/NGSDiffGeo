#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"
#include "kforms.hpp"

#include <coefficient_stdmath.hpp>
#include <python_comp.hpp>
// #include <fem.hpp>
#include <integratorcf.hpp>
#include <hcurlcurlfespace.hpp>
#include <algorithm>
#include <cctype>
#include <vector>

namespace ngfem
{
    namespace
    {
        inline int DeltaSign(int n, int k)
        {
            int exponent = n * (k + 1) + 1;
            return (exponent % 2 == 0) ? 1 : -1;
        }

        char FreshLabel(std::string_view used)
        {
            for (char c : SIGNATURE)
                if (used.find(c) == std::string_view::npos)
                    return c;
            throw Exception("s_op (double-form): not enough signature labels available");
        }

        struct CurvatureSources
        {
            shared_ptr<CoefficientFunction> g_deriv;
            shared_ptr<CoefficientFunction> chr1;
            shared_ptr<CoefficientFunction> chr2;
            shared_ptr<DoubleFormCoefficientFunction> Riemann;
            shared_ptr<TensorFieldCoefficientFunction> Curvature;
            shared_ptr<DoubleFormCoefficientFunction> Ricci;
            shared_ptr<DoubleFormCoefficientFunction> Einstein;
            shared_ptr<ScalarFieldCoefficientFunction> Scalar;
        };

        CurvatureSources FromProxy(shared_ptr<ProxyFunction> g_proxy, int dim, bool change_riemann_sign = false)
        {
            CurvatureSources src;
            src.g_deriv = g_proxy->GetAdditionalProxy("grad");
            src.chr1 = g_proxy->GetAdditionalProxy("christoffel");
            src.chr2 = g_proxy->GetAdditionalProxy("christoffel2");
            if (change_riemann_sign)
                src.Riemann = DoubleFormCF(EinsumCF("ijkl->ijlk", {g_proxy->GetAdditionalProxy("Riemann")}), 2, 2, dim);
            else
                src.Riemann = DoubleFormCF(g_proxy->GetAdditionalProxy("Riemann"), 2, 2, dim);
            src.Curvature = TensorFieldCF(g_proxy->GetAdditionalProxy("curvature"), "00");
            src.Ricci = DoubleFormCF(g_proxy->GetAdditionalProxy("Ricci"), 1, 1, dim);
            src.Einstein = DoubleFormCF(g_proxy->GetAdditionalProxy("Einstein"), 1, 1, dim);
            src.Scalar = ScalarFieldCF(g_proxy->GetAdditionalProxy("scalar"), dim);
            if (!src.g_deriv || !src.chr1 || !src.chr2 || !src.Riemann || !src.Curvature || !src.Ricci || !src.Einstein || !src.Scalar)
                throw Exception("In RMF: Could not load all additional proxy functions");
            return src;
        }

        CurvatureSources FromRegge(shared_ptr<ngcomp::GridFunction> gf, shared_ptr<ngcomp::FESpace> regge_space, int dim, bool change_riemann_sign = false)
        {
            CurvatureSources src;
            auto diffop_grad = regge_space->GetAdditionalEvaluators()["grad"];
            auto diffop_chr1 = regge_space->GetAdditionalEvaluators()["christoffel"];
            auto diffop_chr2 = regge_space->GetAdditionalEvaluators()["christoffel2"];
            auto diffop_Riemann = regge_space->GetAdditionalEvaluators()["Riemann"];
            auto diffop_curvature = regge_space->GetAdditionalEvaluators()["curvature"];
            auto diffop_Ricci = regge_space->GetAdditionalEvaluators()["Ricci"];
            auto diffop_Einstein = regge_space->GetAdditionalEvaluators()["Einstein"];
            auto diffop_scalar = regge_space->GetAdditionalEvaluators()["scalar"];

            if (!diffop_grad || !diffop_chr1 || !diffop_chr2 || !diffop_Riemann || !diffop_curvature || !diffop_Ricci || !diffop_Einstein || !diffop_scalar || !gf)
                throw Exception("In RMF: Could not load all additional evaluators");

            src.g_deriv = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_grad);
            src.g_deriv->SetDimensions(diffop_grad->Dimensions());

            src.chr1 = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_chr1);
            src.chr1->SetDimensions(diffop_chr1->Dimensions());

            src.chr2 = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_chr2);
            src.chr2->SetDimensions(diffop_chr2->Dimensions());

            auto Riemann_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_Riemann);
            Riemann_gf->SetDimensions(diffop_Riemann->Dimensions());
            if (change_riemann_sign)
                src.Riemann = DoubleFormCF(EinsumCF("ijkl->ijlk", {Riemann_gf}), 2, 2, dim);
            else
                src.Riemann = DoubleFormCF(Riemann_gf, 2, 2, dim);

            auto Curvature_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_curvature);
            Curvature_gf->SetDimensions(diffop_curvature->Dimensions());
            if (dim == 2)
                src.Curvature = ScalarFieldCF(Curvature_gf, dim);
            else
                src.Curvature = TensorFieldCF(Curvature_gf, "00");

            auto Ricci_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_Ricci);
            Ricci_gf->SetDimensions(diffop_Ricci->Dimensions());
            src.Ricci = DoubleFormCF(Ricci_gf, 1, 1, dim);

            auto Einstein_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_Einstein);
            Einstein_gf->SetDimensions(diffop_Einstein->Dimensions());
            src.Einstein = DoubleFormCF(Einstein_gf, 1, 1, dim);

            src.Scalar = ScalarFieldCF(make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_scalar), dim);
            return src;
        }

        CurvatureSources FromCF(RiemannianManifold &M, shared_ptr<CoefficientFunction> g, shared_ptr<CoefficientFunction> g_inv, bool change_riemann_sign = false)
        {
            CurvatureSources src;
            int dim = M.Dimension();
            src.g_deriv = GradCF(g, dim);
            Array<shared_ptr<CoefficientFunction>> values(dim * dim * dim);

            for (auto i : Range(dim))
                for (auto j : Range(dim))
                    for (auto k : Range(dim))
                    {
                        values[i * dim * dim + j * dim + k] = 0.5 * (MakeComponentCoefficientFunction(src.g_deriv, i * dim * dim + j + dim * k) + MakeComponentCoefficientFunction(src.g_deriv, j * dim * dim + i + dim * k) - MakeComponentCoefficientFunction(src.g_deriv, k * dim * dim + i + dim * j));
                    }
            auto tmp = MakeVectorialCoefficientFunction(std::move(values));
            src.chr1 = tmp->Reshape(Array<int>({dim, dim, dim}));
            src.chr2 = EinsumCF("ijl,lk->ijk", {src.chr1, g_inv});

            auto chr1_grad = GradCF(src.chr1, dim);
            auto lin_part = chr1_grad - EinsumCF("ijkl->jikl", {chr1_grad});
            auto non_lin_part = EinsumCF("jlm,ikm->ijkl", {src.chr1, src.chr2}) - EinsumCF("ilm,jkm->ijkl", {src.chr2, src.chr1});
            if (change_riemann_sign)
                src.Riemann = DoubleFormCF(EinsumCF("ijkl->ijlk", {lin_part + non_lin_part}), 2, 2, dim);
            else
                src.Riemann = DoubleFormCF(lin_part + non_lin_part, 2, 2, dim);
            auto LeviCivita = M.GetLeviCivitaSymbol(false);
            string signature = SIGNATURE.substr(0, 4) + "," + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(0, 2) + "," + SIGNATURE.substr(2 + dim, dim - 2) + SIGNATURE.substr(2, 2) + "->" + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(2 + dim, dim - 2);
            double sign = change_riemann_sign ? -1.0 : 1.0;
            if (dim == 2)
                src.Curvature = ScalarFieldCF(-sign * 0.25 * EinsumCF(signature, {src.Riemann, LeviCivita, LeviCivita}), dim);
            else
                src.Curvature = TensorFieldCF(-sign * 0.25 * EinsumCF(signature, {src.Riemann, LeviCivita, LeviCivita}), "00");
            src.Ricci = change_riemann_sign ? DoubleFormCF(M.Trace(src.Riemann, 0, 2), 1, 1, dim)
                                            : DoubleFormCF(M.Trace(src.Riemann, 0, 3), 1, 1, dim);
            src.Scalar = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(M.Trace(src.Ricci, 0, 1));
            src.Einstein = DoubleFormCF(src.Ricci - 0.5 * src.Scalar * g, 1, 1, dim);
            return src;
        }

        enum class CovOp
        {
            Exterior,
            Codifferential
        };

        shared_ptr<DoubleFormCoefficientFunction> CovExteriorOrCodiff(const RiemannianManifold &M,
                                                                      shared_ptr<DoubleFormCoefficientFunction> tf,
                                                                      int slot,
                                                                      CovOp op,
                                                                      VorB vb)
        {
            const char *name = nullptr;
            if (op == CovOp::Exterior)
                name = (slot == 0) ? "CovExteriorDerivative1" : "CovExteriorDerivative2";
            else
                name = (slot == 0) ? "CovCodifferential1" : "CovCodifferential2";

            if (!tf)
                throw Exception(string(name) + ": input must be non-null");
            if (vb != VOL && vb != BND)
                throw Exception(string(name) + ": only implemented for vb=VOL and vb=BND.");
            if (tf->DimensionOfSpace() != M.Dimension())
                throw Exception(string(name) + ": double-form dimension does not match manifold dimension");

            int p = tf->LeftDegree();
            int q = tf->RightDegree();
            int dim = M.Dimension();
            int surface_dim = (vb == BND) ? dim - 1 : dim;

            if (slot != 0 && slot != 1)
                throw Exception(string(name) + ": slot must be 0/1 or 'left'/'right'");

            auto cov_der = M.CovDerivative(tf, vb);

            auto project_if_needed = [&](shared_ptr<DoubleFormCoefficientFunction> df)
            {
                if (vb != BND || df->Meta().rank == 0)
                    return df;
                auto projected = M.ProjectTensorToEuclideanTangent(static_pointer_cast<TensorFieldCoefficientFunction>(df));
                return DoubleFormCF(projected, df->LeftDegree(), df->RightDegree(), dim);
            };

            if (op == CovOp::Exterior)
            {
                if (slot == 0 && p + 1 > surface_dim)
                    return ZeroDoubleForm(p + 1, q, dim);
                if (slot == 1 && q + 1 > surface_dim)
                    return ZeroDoubleForm(p, q + 1, dim);

                int total = p + q + 1;

                if (slot == 0)
                {
                    auto alt = BlockAlternationByPermutationCF(cov_der, total, 0, p + 1);
                    double scale = 1.0 / double(Factorial(p));
                    auto out = scale * alt;
                    return project_if_needed(DoubleFormCF(out, p + 1, q, dim));
                }

                std::vector<int> order;
                order.reserve(size_t(total));
                for (int i = 0; i < p; ++i)
                    order.push_back(1 + i);
                order.push_back(0);
                for (int i = 0; i < q; ++i)
                    order.push_back(1 + p + i);

                auto reordered = PermuteTensorCF(cov_der, order);
                auto alt = BlockAlternationByPermutationCF(reordered, total, p, q + 1);
                double scale = 1.0 / double(Factorial(q));
                auto out = scale * alt;
                return project_if_needed(DoubleFormCF(out, p, q + 1, dim));
            }

            if (slot == 0 && p == 0)
                return ZeroDoubleForm(0, q, dim);
            if (slot == 1 && q == 0)
                return ZeroDoubleForm(p, 0, dim);

            if (slot == 0)
            {
                auto traced = M.Trace(cov_der, 0, 1, vb);
                auto out = (-1.0) * traced;
                return project_if_needed(DoubleFormCF(out, p - 1, q, dim));
            }

            auto traced = M.Trace(cov_der, 0, size_t(p + 1), vb);
            auto out = (-1.0) * traced;
            return project_if_needed(DoubleFormCF(out, p, q - 1, dim));
        }

    } // namespace

    using namespace std;
    RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g, double normal_sign_, bool change_riemann_sign_)
        : has_trial(false), is_regge(false), is_proxy(false), normal_sign(normal_sign_), change_riemann_sign(change_riemann_sign_), regge_proxy(nullptr), regge_space(nullptr), g(_g)
    {
        if (_g->Dimensions().Size() != 2 || _g->Dimensions()[0] != _g->Dimensions()[1])
            throw Exception("In RMF: input must be a square matrix");

        // check if _g involves a trial function
        _g->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          if (auto proxy = dynamic_cast<ProxyFunction*> (&nodecf))
            {
              if (proxy->IsTestFunction())
                  throw Exception("In RMF: test function not allowed");
              else
                  has_trial = true;
            } });

        // check if _g itself is a Regge trial function
        if (dynamic_pointer_cast<ProxyFunction>(g))
        {
            // cout << "In RMF: g is a ProxyFunction" << endl;
            regge_proxy = dynamic_pointer_cast<ProxyFunction>(g);
            is_proxy = true;
            is_regge = true;
            if (regge_proxy->GetFESpace()->GetClassName().find(string("HCurlCurlFESpace")) == string::npos)
                throw Exception("In RMF: ProxyFunction must be from HCurlCurlFESpace");
            regge_space = regge_proxy->GetFESpace();
        }
        else if (auto gf = dynamic_pointer_cast<ngcomp::GridFunction>(g))
        {
            // cout << "In RMF: g is a GridFunction from space " << gf->GetFESpace()->GetClassName() << endl;

            is_regge = true;
            if (gf->GetFESpace()->GetClassName().find(string("HCurlCurlFESpace")) == string::npos)
                throw Exception("In RMF: GridFunction must be from HCurlCurlFESpace");
            regge_space = gf->GetFESpace();
        }

        dim = g->Dimensions()[0];

        if (dim != 2 && dim != 3)
            throw Exception("In RMF: only 2D and 3D manifolds are supported");

        g_inv = InverseCF(g);

        // Volume forms on VOL, BND, BBND, and BBBND
        det_g = DeterminantCF(g);
        vol[VOL] = sqrt(det_g);
        auto one_cf = make_shared<ConstantCoefficientFunction>(1.0);
        tv = TangentialVectorCF(dim, false);
        nv = normal_sign * NormalVectorCF(dim);
        auto nv_mat = nv->Reshape(Array<int>({dim, 1}));
        P_n = nv_mat * TransposeCF(nv_mat);
        P_F = IdentityCF(dim) - P_n;

        vol[BND] = one_cf;
        vol[BBND] = one_cf;
        vol[BBBND] = one_cf;
        g_E = one_cf;
        g_E_inv = one_cf;

        if (dim == 2)
        {
            auto g_tv_norm = InnerProduct(g * tv, tv);
            vol[BND] = sqrt(g_tv_norm);
            g_tv = VectorFieldCF(1 / vol[BND] * tv);

            // more efficient?
            // auto tv_mat = tv->Reshape(Array<int>({dim, 1}));
            // auto P_F = tv_mat * TransposeCF(tv_mat);
            // g_F = InnerProduct(g * tv, tv) * P_F;
            // g_F_inv = 1/InnerProduct(g * tv, tv) * P_F;
            shared_ptr<OneFormCoefficientFunction> g_nv_lower = OneFormCF(vol[VOL] / vol[BND] * nv);
            g_nv = dynamic_pointer_cast<VectorFieldCoefficientFunction>(Raise(g_nv_lower));

            // g_F = P_F * g * P_F;
            // g_F_inv = P_F * InverseCF(g_F + P_n) * P_F;
            g_F = g - TensorProduct(g_nv_lower, g_nv_lower);
            g_F_inv = g_inv - TensorProduct(g_nv, g_nv);
            g_E = one_cf;
            g_E_inv = one_cf;

            // dim 2,2
            auto v_tang_vec = VertexTangentialVectorsCF(2);
            cnv[0] = MakeSubTensorCoefficientFunction(v_tang_vec, 0, {2}, {2});
            cnv[1] = MakeSubTensorCoefficientFunction(v_tang_vec, 1, {2}, {2});
            g_cnv[0] = VectorFieldCF(1 / sqrt(InnerProduct(g * cnv[0], cnv[0])) * cnv[0]);
            g_cnv[1] = VectorFieldCF(1 / sqrt(InnerProduct(g * cnv[1], cnv[1])) * cnv[1]);
        }
        else if (dim == 3)
        {
            vol[BND] = sqrt(InnerProduct(CofactorCF(g) * nv, nv));
            auto g_tv_norm = InnerProduct(g * tv, tv);
            vol[BBND] = sqrt(g_tv_norm);

            g_tv = VectorFieldCF(1 / vol[BBND] * tv);

            shared_ptr<OneFormCoefficientFunction> g_nv_lower = OneFormCF(vol[VOL] / vol[BND] * nv);
            g_nv = dynamic_pointer_cast<VectorFieldCoefficientFunction>(Raise(g_nv_lower));

            // g_F = P_F * g * P_F;
            // g_F_inv = P_F * InverseCF(g_F + P_n) * P_F;
            g_F = g - TensorProduct(g_nv_lower, g_nv_lower);
            g_F_inv = g_inv - TensorProduct(g_nv, g_nv);

            auto ef_tang_vec = EdgeFaceTangentialVectorsCF(3);
            cnv[0] = MakeSubTensorCoefficientFunction(ef_tang_vec, 0, {3}, {2});
            cnv[1] = MakeSubTensorCoefficientFunction(ef_tang_vec, 1, {3}, {2});

            for (int i = 0; i < 2; ++i)
            {
                auto n_euc = CrossProduct(TangentialVectorCF(dim, true), cnv[i]);
                auto g_n = VectorFieldCF(1 / sqrt(InnerProduct(g_inv * n_euc, n_euc)) * (g_inv * n_euc));
                g_nv_BBND[i] = g_n;

                auto proj_tv = InnerProduct(g * g_tv, cnv[i]);
                auto proj_n = InnerProduct(g * g_n, cnv[i]);
                auto cnv_g = cnv[i] - g_tv * proj_tv - g_n * proj_n;
                auto cnv_g_norm = sqrt(InnerProduct(g * cnv_g, cnv_g));
                g_cnv[i] = VectorFieldCF(1 / cnv_g_norm * cnv_g);
            }

            // auto tv_mat = tv->Reshape(Array<int>({dim, 1}));
            // auto P_E = tv_mat * TransposeCF(tv_mat);
            // g_E = g_tv_norm * P_E;
            // g_E_inv = 1 / g_tv_norm * P_E;

            shared_ptr<OneFormCoefficientFunction> g_tv_lower = OneFormCF(g * g_tv);
            g_E = TensorProduct(g_tv_lower, g_tv_lower);
            g_E_inv = TensorProduct(g_tv, g_tv);
            P_E_g = TensorProduct(g_tv, g_tv_lower);
        }

        P_F_g = IdentityCF(dim) - TensorProduct(g_nv, Lower(g_nv));

        CurvatureSources sources;
        if (is_regge)
        {
            if (is_proxy)
            {
                auto g_proxy = dynamic_pointer_cast<ProxyFunction>(g);
                sources = FromProxy(g_proxy, dim, change_riemann_sign);
            }
            else
            {
                auto gf = dynamic_pointer_cast<ngcomp::GridFunction>(g);
                sources = FromRegge(gf, regge_space, dim, change_riemann_sign);
            }
        }
        else
        {
            sources = FromCF(*this, g, g_inv, change_riemann_sign);
        }

        g_deriv = sources.g_deriv;
        chr1 = sources.chr1;
        chr2 = sources.chr2;
        Riemann = sources.Riemann;
        Curvature = sources.Curvature;
        Ricci = sources.Ricci;
        Einstein = sources.Einstein;
        Scalar = sources.Scalar;
        SFF = DoubleFormCF(EinsumCF("ia,ijk,k,jb->ab", {P_F_g, chr1->Reshape(Array<int>({dim, dim, dim})), g_nv, P_F_g}), 1, 1, dim);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetric() const
    {
        return DoubleFormCF(g, 1, 1, dim);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricF() const
    {
        return DoubleFormCF(g_F, 1, 1, dim);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricFInverse() const
    {
        return TensorFieldCF(g_F_inv, "00");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricE() const
    {
        return DoubleFormCF(g_E, 1, 1, dim);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricEInverse() const
    {
        return TensorFieldCF(g_E_inv, "00");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricInverse() const
    {
        return TensorFieldCF(g_inv, "00");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetVolumeForm(VorB vb) const
    {
        return vol[int(vb)];
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Raise(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index, VorB vb) const
    {
        if (tf->Dimensions().Size() <= index)
            throw Exception(ToString("Raise: Dimension of tf = ") + ToString(tf->Dimensions().Size()) + "<= index = " + ToString(index));

        shared_ptr<CoefficientFunction> metric_inv;
        switch (vb)
        {
        case VOL:
            metric_inv = g_inv;
            break;
        case BND:
            metric_inv = g_F_inv;
            break;
        case BBND:
            metric_inv = g_E_inv;
            break;
        default:
            throw Exception("Raise: only implemented for VOL, BND, and BBND");
        }

        auto m = tf->Meta();
        if (!m.Covariant(index))
            throw Exception("Raise: index is already contravariant");

        std::string sig = tf->GetSignature();
        char a = sig[index];
        char b = m.FreshLabel();

        std::string lhs = sig;
        lhs[index] = b;

        std::string eins = lhs + "," + std::string(1, a) + std::string(1, b) + "->" + sig;

        auto mout = m.WithCovariant(index, false);
        auto out_cf = EinsumCF(eins, {tf->GetCoefficients(), metric_inv});

        // if tf is a OneFormCoefficientFunction, return a VectorFieldCoefficientFunction
        if (dynamic_pointer_cast<OneFormCoefficientFunction>(tf))
            return VectorFieldCF(out_cf);

        return TensorFieldCF(out_cf, mout);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Raise(shared_ptr<TensorFieldCoefficientFunction> tf, const std::vector<size_t> &indices, VorB vb) const
    {
        if (indices.size() > tf->Dimensions().Size())
            throw Exception(ToString("Raise: number of indices = ") + ToString(indices.size()) + " exceeds tensor dimension = " + ToString(tf->Dimensions().Size()));

        shared_ptr<TensorFieldCoefficientFunction> out = tf;
        for (auto index : indices)
            out = Raise(out, index, vb);
        return out;
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Lower(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index, VorB vb) const
    {
        if (tf->Dimensions().Size() <= index)
            throw Exception(ToString("Lower: Dimension of tf = ") + ToString(tf->Dimensions().Size()) + "<= index = " + ToString(index));

        shared_ptr<CoefficientFunction> metric;
        switch (vb)
        {
        case VOL:
            metric = g;
            break;
        case BND:
            metric = g_F;
            break;
        case BBND:
            metric = g_E;
            break;
        default:
            throw Exception("Lower: only implemented for VOL, BND, and BBND");
        }

        auto m = tf->Meta();
        if (index >= m.rank)
            throw Exception("Lower: index out of range");
        if (m.Covariant(index))
            throw Exception("Lower: index is already covariant");

        std::string sig = tf->GetSignature();
        char a = sig[index];
        char b = m.FreshLabel();

        std::string lhs = sig;
        lhs[index] = b;

        std::string eins = lhs + "," + std::string(1, a) + std::string(1, b) + "->" + sig;

        auto mout = m.WithCovariant(index, true);
        auto out_cf = EinsumCF(eins, {tf->GetCoefficients(), metric});

        // if tf is a VectorFieldCoefficientFunction, return a OneFormCoefficientFunction
        if (dynamic_pointer_cast<VectorFieldCoefficientFunction>(tf))
            return OneFormCF(out_cf);

        return TensorFieldCF(out_cf, mout);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Lower(shared_ptr<TensorFieldCoefficientFunction> tf, const std::vector<size_t> &indices, VorB vb) const
    {
        if (indices.size() > tf->Dimensions().Size())
            throw Exception(ToString("Lower: number of indices = ") + ToString(indices.size()) + " exceeds tensor dimension = " + ToString(tf->Dimensions().Size()));

        shared_ptr<TensorFieldCoefficientFunction> out = tf;
        for (auto index : indices)
            out = Lower(out, index, vb);
        return out;
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetLeviCivitaSymbol(bool covariant) const
    {
        if (covariant)
        {
            if (!levi_civita_cov)
            {
                auto levi_civita_symbol = LeviCivitaCF(dim);
                levi_civita_cov = TensorFieldCF(GetVolumeForm(VOL) * levi_civita_symbol, string(dim, '1'));
            }
            return levi_civita_cov;
        }

        if (!levi_civita_contra)
        {
            auto levi_civita_symbol = LeviCivitaCF(dim);
            levi_civita_contra = TensorFieldCF(make_shared<ConstantCoefficientFunction>(1.0) / GetVolumeForm(VOL) * levi_civita_symbol, string(dim, '0'));
        }
        return levi_civita_contra;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricDerivative() const
    {
        return g_deriv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetChristoffelSymbol(bool second_kind) const
    {
        return second_kind ? chr2 : chr1;
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::GetRiemannCurvatureTensor() const
    {
        return Riemann;
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetCurvatureOperator() const
    {
        return Curvature;
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::GetRicciTensor() const
    {
        return Ricci;
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::GetEinsteinTensor() const
    {
        return Einstein;
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::GetScalarCurvature() const
    {
        return ScalarFieldCF(Scalar, dim);
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::GetGaussCurvature() const
    {
        if (dim != 2)
            throw Exception("In RMF: Gauss curvature only available in 2D");
        return ScalarFieldCF(1 / det_g * Curvature, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::GetSecondFundamentalForm() const
    {
        return SFF;
    }
    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::GetGeodesicCurvature() const
    {
        if (dim != 2)
            throw Exception("In RMF: Geodesic curvature only available in 2D");
        return ScalarFieldCF(InnerProduct(SFF * g_tv, g_tv), dim);
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::GetMeanCurvature() const
    {
        return dynamic_pointer_cast<ScalarFieldCoefficientFunction>(this->Trace(GetSecondFundamentalForm(), 0, 1, BND));
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::ProjectTensorToEuclideanTangent(shared_ptr<TensorFieldCoefficientFunction> tf) const
    {
        if (!tf)
            throw Exception("ProjectTensorToEuclideanTangent: input must be non-null");
        auto m = tf->Meta();
        if (m.rank == 0)
            return tf;

        auto current = tf;
        const auto &cov_ind = tf->GetCovariantIndices();
        auto P_F_g_T = TransposeCF(P_F_g);
        for (size_t i = 0; i < m.rank; ++i)
        {
            auto proj = (cov_ind[i] == '1') ? P_F_g : P_F_g_T;
            current = ApplyProjectorToIndex(current, proj, i);
        }
        return current;
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::ProjectTensor(shared_ptr<TensorFieldCoefficientFunction> tf, int mode) const
    {
        if (!tf)
            throw Exception("ProjectTensor: input must be non-null");
        auto m = tf->Meta();
        if (m.rank == 0)
        {
            if (mode == 2)
                throw Exception("ProjectTensor: cannot take normal component of rank-0 tensor");
            return tf;
        }

        if (mode == 0)
            return tf;

        auto current = tf;
        const auto &cov_ind = tf->GetCovariantIndices();
        auto P_F_g_T = TransposeCF(P_F_g);
        auto edge_proj = P_E_g;
        auto P_E_g_T = edge_proj ? TransposeCF(edge_proj) : nullptr;

        if (mode == 1)
        {
            for (size_t i = 0; i < m.rank; ++i)
            {
                auto proj = (cov_ind[i] == '1') ? P_F_g : P_F_g_T;
                current = ApplyProjectorToIndex(current, proj, i);
            }
            return current;
        }

        if (mode == 2)
        {
            for (size_t i = 1; i < m.rank; ++i)
            {
                auto proj = (cov_ind[i] == '1') ? P_F_g : P_F_g_T;
                current = ApplyProjectorToIndex(current, proj, i);
            }
            return Contraction(current, g_nv, 0);
        }

        if (mode == 3)
        {
            if (!edge_proj)
                throw Exception("ProjectTensor: edge projector not available");
            for (size_t i = 0; i < m.rank; ++i)
            {
                auto proj = (cov_ind[i] == '1') ? edge_proj : P_E_g_T;
                current = ApplyProjectorToIndex(current, proj, i);
            }
            return current;
        }

        throw Exception("ProjectTensor: mode must be 0 (none), 1 (tangent), 2 (normal), or 3 (edge)");
    }

    shared_ptr<VectorFieldCoefficientFunction> RiemannianManifold::GetNV() const
    {
        return g_nv;
    }

    shared_ptr<VectorFieldCoefficientFunction> RiemannianManifold::GetEdgeTangent() const
    {
        return g_tv;
    }

    shared_ptr<VectorFieldCoefficientFunction> RiemannianManifold::GetEdgeNormal(int i) const
    {
        if (i < 0 || i > 1)
            throw Exception("GetEdgeNormal: index must be 0 or 1");
        if (!g_nv_BBND[i])
            throw Exception("GetEdgeNormal: normal not available");
        return g_nv_BBND[i];
    }

    shared_ptr<VectorFieldCoefficientFunction> RiemannianManifold::GetEdgeConormal(int i) const
    {
        if (i < 0 || i > 1)
            throw Exception("GetEdgeConormal: index must be 0 or 1");
        if (!g_cnv[i])
            throw Exception("GetEdgeConormal: conormal not available");
        return g_cnv[i];
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2, VorB vb, bool forms) const
    {
        string cov_ind1 = c1->GetCovariantIndices();
        string cov_ind2 = c2->GetCovariantIndices();

        if (cov_ind1.size() != cov_ind2.size())
            throw Exception("IP: size of c1 and c2 must match");
        if (c1->Dimensions().Size() && c1->Dimensions()[0] != c2->Dimensions()[0])
            throw Exception("IP: dimensions of c1 and c2 must match");

        if (c1->Dimensions().Size() && c1->Dimensions()[0] != dim)
            throw Exception(ToString("IP: dimensions of c1 and c2 must be ") + ToString(dim) + ". Received " + ToString(c1->Dimensions()[0]));

        shared_ptr<CoefficientFunction> metric;
        shared_ptr<CoefficientFunction> metric_inv;

        switch (vb)
        {
        case VOL:
            metric = g;
            metric_inv = g_inv;
            break;
        case BND:
            metric = g_F;
            metric_inv = g_F_inv;
            break;
        case BBND:
            metric = g_E;
            metric_inv = g_E_inv;
            break;
        default:
            throw Exception("IP: VorB must be VOL, BND, or BBND");
            break;
        }

        // create boolean array with true if cov_ind1 and ind_cov2 coincide at the position
        Array<bool> same_index(cov_ind1.size());
        Array<size_t> position_same_index;
        for (size_t i = 0; i < cov_ind1.size(); i++)
        {
            same_index[i] = cov_ind1[i] == cov_ind2[i];
            if (same_index[i])
                position_same_index.Append(i);
        }

        char new_char = 'a';
        char new_char_g = 'A';

        string signature_c1 = "";
        string signature_c2 = "";
        string raise_lower_signatures;

        for (size_t i = 0; i < cov_ind1.size(); i++)
        {
            signature_c1 += new_char;
            if (same_index[i])
            {
                raise_lower_signatures += "," + ToString(new_char++) + new_char_g;
                signature_c2 += char(new_char_g++);
            }
            else
            {
                signature_c2 += char(new_char++);
            }
        }

        Array<shared_ptr<CoefficientFunction>> cfs(2 + position_same_index.Size());
        cfs[0] = c1;
        cfs[1] = c2;
        for (size_t i = 0; i < position_same_index.Size(); i++)
        {
            cfs[2 + i] = cov_ind1[position_same_index[i]] == '1' ? metric_inv : metric;
        }
        auto result = ScalarFieldCF(EinsumCF(signature_c1 + "," + signature_c2 + raise_lower_signatures, cfs), dim);
        if (!forms)
            return result;

        auto k1 = dynamic_pointer_cast<KFormCoefficientFunction>(c1);
        auto k2 = dynamic_pointer_cast<KFormCoefficientFunction>(c2);
        if (!k1 || !k2)
            throw Exception("IP: forms=true requires k-forms or double-forms");
        if (k1->Degree() != k2->Degree())
            throw Exception("IP: form degrees must match");

        double scale = 1.0 / double(Factorial(k1->Degree()));
        return ScalarFieldCF(scale * result->GetCoefficients(), dim);
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::IP(shared_ptr<DoubleFormCoefficientFunction> c1, shared_ptr<DoubleFormCoefficientFunction> c2, VorB vb, bool forms) const
    {
        if (!c1 || !c2)
            throw Exception("IP: inputs must be non-null");
        if (c1->DimensionOfSpace() != dim || c2->DimensionOfSpace() != dim)
            throw Exception("IP: double-form dimension does not match manifold dimension");
        if (c1->LeftDegree() != c2->LeftDegree() || c1->RightDegree() != c2->RightDegree())
            throw Exception("IP: double-form degrees must match");

        auto tf1 = static_pointer_cast<TensorFieldCoefficientFunction>(c1);
        auto tf2 = static_pointer_cast<TensorFieldCoefficientFunction>(c2);
        auto result = IP(tf1, tf2, vb, false);
        if (!forms)
            return result;

        double scale = 1.0 / double(Factorial(c1->LeftDegree()) * Factorial(c1->RightDegree()));
        return ScalarFieldCF(scale * result->GetCoefficients(), dim);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const
    {
        if (!c1 || !c2)
            throw Exception("Cross: inputs must be non-null");
        string cov_ind1 = c1->GetCovariantIndices();
        string cov_ind2 = c2->GetCovariantIndices();
        if (cov_ind1.size() != 1 || cov_ind2.size() != 1)
            throw Exception("Cross: only available for vector fields and 1-forms yet.");
        if (c1->Dimensions()[0] != 3 || c2->Dimensions()[0] != 3)
        {
            throw Exception("Cross: only available for 3D yet.");
        }

        if (cov_ind1 == cov_ind2)
        {
            if (cov_ind1[0] == '1')
            {
                // both 1-forms
                return VectorFieldCF(EinsumCF("ijk,j,k->i", {GetLeviCivitaSymbol(false), c1, c2}));
            }
            else
            {
                // both vector-fields
                return VectorFieldCF(EinsumCF("ai,ijk,j,k->a", {g_inv, GetLeviCivitaSymbol(true), c1, c2}));
            }
        }
        if (cov_ind1[0] == '1')
        {
            // c1 1-form, c2 vector field
            return VectorFieldCF(EinsumCF("ijk,j,kl,l->i", {GetLeviCivitaSymbol(false), c1, g, c2}));
        }
        else
        {
            // c1 vector field, c2 1-form
            return VectorFieldCF(EinsumCF("ijk,jl,l,k->i", {g_inv, GetLeviCivitaSymbol(false), g, c1, c2}));
        }
    }

    shared_ptr<KFormCoefficientFunction> RiemannianManifold::MakeKForm(shared_ptr<CoefficientFunction> cf, int k) const
    {
        return KFormCF(cf, k, dim);
    }

    shared_ptr<KFormCoefficientFunction> RiemannianManifold::Star(shared_ptr<KFormCoefficientFunction> a, VorB vb) const
    {
        if (!a)
            throw Exception("Star: input must be non-null");
        if (a->DimensionOfSpace() != dim)
            throw Exception("Star: form dimension does not match manifold dimension");
        return HodgeStar(a, *this, vb);
    }

    shared_ptr<KFormCoefficientFunction> RiemannianManifold::InvStar(shared_ptr<KFormCoefficientFunction> a, VorB vb) const
    {
        if (!a)
            throw Exception("InvStar: input must be non-null");
        if (a->DimensionOfSpace() != dim)
            throw Exception("InvStar: form dimension does not match manifold dimension");
        return InverseHodgeStar(a, *this, vb);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::InvStar(shared_ptr<DoubleFormCoefficientFunction> a, VorB vb) const
    {
        if (!a)
            throw Exception("InvStar: input must be non-null");
        if (a->DimensionOfSpace() != dim)
            throw Exception("InvStar: double-form dimension does not match manifold dimension");
        return InverseHodgeStar(a, *this, vb);
    }

    shared_ptr<KFormCoefficientFunction> RiemannianManifold::Coderivative(shared_ptr<KFormCoefficientFunction> a) const
    {
        if (!a)
            throw Exception("Coderivative: input must be non-null");
        if (a->DimensionOfSpace() != dim)
            throw Exception("Coderivative: form dimension does not match manifold dimension");

        int k = a->Degree();
        if (k == 0)
            return ZeroKForm(0, dim);

        auto first_star = Star(a);
        auto d_first_star = ExteriorDerivative(first_star);
        auto second_star = Star(d_first_star);

        int sign = DeltaSign(dim, k);
        if (sign == 1)
            return second_star;

        auto signed_cf = (-1.0) * second_star->GetCoefficients();
        return KFormCF(signed_cf, k - 1, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::CovExteriorDerivative1(shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb) const
    {
        return CovExteriorOrCodiff(*this, tf, 0, CovOp::Exterior, vb);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::CovExteriorDerivative2(shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb) const
    {
        return CovExteriorOrCodiff(*this, tf, 1, CovOp::Exterior, vb);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::CovCodifferential1(shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb) const
    {
        return CovExteriorOrCodiff(*this, tf, 0, CovOp::Codifferential, vb);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::CovCodifferential2(shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb) const
    {
        return CovExteriorOrCodiff(*this, tf, 1, CovOp::Codifferential, vb);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb) const
    {
        if (vb != VOL && vb != BND)
            throw Exception("CovDerivative: only implemented for vb=VOL and vb=BND.");

        shared_ptr<TensorFieldCoefficientFunction> result;

        // scalar field
        if (c1->Dimensions().Size() == 0)
        {
            result = OneFormCF(GradCF(c1, dim));
        }

        // vector field
        else if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
        {
            result = TensorFieldCF(GradCF(vf->GetCoefficients(), dim) + EinsumCF("ikj,k->ij", {chr2, vf->GetCoefficients()}), "10");
        }

        // one-form field
        else if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
        {
            result = TensorFieldCF(GradCF(of->GetCoefficients(), dim) - EinsumCF("ijk,k->ij", {chr2, of->GetCoefficients()}), "11");
        }

        // General tensor field
        else
        {
            string signature = c1->GetSignature();
            string tmp_signature = signature;
            string cov_ind = c1->GetCovariantIndices();
            char new_char = FreshLabel(signature);

            auto result_cf = GradCF(c1->GetCoefficients(), dim);
            for (size_t i = 0; i < signature.size(); i++)
            {
                tmp_signature = signature;
                tmp_signature[i] = FreshLabel(signature + std::string(1, new_char));
                if (cov_ind[i] == '1')
                {
                    // covariant
                    string einsum_signature = ToString(new_char) + signature[i] + tmp_signature[i] + "," + tmp_signature + "->" + new_char + signature;
                    result_cf = result_cf - EinsumCF(einsum_signature, {chr2, c1->GetCoefficients()});
                }
                else
                {
                    // contravariant
                    string einsum_signature = ToString(new_char) + tmp_signature[i] + signature[i] + "," + tmp_signature + "->" + new_char + signature;
                    result_cf = result_cf + EinsumCF(einsum_signature, {chr2, c1->GetCoefficients()});
                }
            }
            result = TensorFieldCF(result_cf, "1" + cov_ind);
        }

        if (vb == BND)
        {
            const auto &cov_ind = result->GetCovariantIndices();
            auto P_F_g_T = TransposeCF(P_F_g);
            for (size_t i = 0; i < cov_ind.size(); ++i)
            {
                if (cov_ind[i] == '1')
                    result = ApplyProjectorToIndex(result, P_F_g, i);
                else
                    result = ApplyProjectorToIndex(result, P_F_g_T, i);
            }
        }

        return result;
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovHessian(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (auto sf = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(c1))
            return CovDerivative(OneFormCF(GradCF(sf, dim)));
        else
            return CovDerivative(CovDerivative(c1));
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb) const
    {
        if (c1->Dimensions().Size() == 0)
            throw Exception("CovDivergence: TensorField must have at least one index");

        return this->Trace(this->CovDerivative(c1, vb), 0, 1, vb);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::CovDivergence(shared_ptr<DoubleFormCoefficientFunction> c1, int slot, VorB vb) const
    {
        if (!c1)
            throw Exception("CovDivergence: input must be non-null");
        if (c1->DimensionOfSpace() != dim)
            throw Exception("CovDivergence: double-form dimension does not match manifold dimension");
        if (slot != 0 && slot != 1)
            throw Exception("CovDivergence: slot must be 0/1 or 'left'/'right'");

        int p = c1->LeftDegree();
        int q = c1->RightDegree();
        if (slot == 0 && p == 0)
            throw Exception("CovDivergence: left slot degree is zero");
        if (slot == 1 && q == 0)
            throw Exception("CovDivergence: right slot degree is zero");

        auto cov = CovDerivative(static_pointer_cast<TensorFieldCoefficientFunction>(c1), vb);
        size_t trace_idx = (slot == 0) ? 1 : size_t(p + 1);
        auto tr = Trace(cov, 0, trace_idx, vb);

        int out_p = (slot == 0) ? p - 1 : p;
        int out_q = (slot == 1) ? q - 1 : q;
        return DoubleFormCF(tr->GetCoefficients(), out_p, out_q, dim);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovCurl(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (c1->Dimensions().Size() == 0)
            throw Exception("CovCurl: called with scalar field");
        // if (c1->Dimensions()[0] != 3)
        //     throw Exception("CovCurl: only available in 3D yet");

        if (dim == 3)
        {
            if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
            {
                return VectorFieldCF(EinsumCF("ijk,jk->i", {GetLeviCivitaSymbol(false), GradCF(c1, dim)}));
            }
            else if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
            {
                return VectorFieldCF(EinsumCF("ijk,jk->i", {GetLeviCivitaSymbol(false), GradCF(Lower(c1), dim)}));
            }
            else if (c1->Dimensions().Size() == 2 && c1->GetCovariantIndices() == "11")
            {
                // Row-wise curl for covariant (2,0)-tensors:
                // (Curl T)_{i}^p = eps^{pab} \nabla_a T_{ib}
                auto cov_der = CovDerivative(c1);
                return TensorFieldCF(EinsumCF("pab,aib->ip", {GetLeviCivitaSymbol(false), cov_der}), "10");
            }
            else
                throw Exception("CovCurl: only available for vector fields, 1-forms, and covariant (2,0)-tensors yet. Invoked with signature " + c1->GetSignature() + " and covariant indices " + c1->GetCovariantIndices());
        }
        else if (dim == 2)
        {
            // throw Exception("CovCurl: only available in 2D yet");

            if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
            {
                return ScalarFieldCF(EinsumCF("ij,ij->", {GetLeviCivitaSymbol(false), GradCF(c1, dim)}), dim);
            }
            else if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
            {
                return ScalarFieldCF(EinsumCF("ij,ij->", {GetLeviCivitaSymbol(false), GradCF(Lower(c1), dim)}), dim);
            }
            else if (c1->Dimensions().Size() == 2 && c1->GetCovariantIndices() == "11")
            {
                return OneFormCF(EinsumCF("jk,jik->i", {GetLeviCivitaSymbol(false), GradCF(c1, dim) - EinsumCF("jim,mk->jik", {chr2, c1->GetCoefficients()})}));
            }
            else
                throw Exception("CovCurl: only available for vector fields, 1-forms, and (2,0)-tensors yet. Invoked with signature " + c1->GetSignature() + " and covariant indices " + c1->GetCovariantIndices());
        }
        else
            throw Exception("CovCurl: only available in 2D and 3D yet");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovInc(shared_ptr<TensorFieldCoefficientFunction> c1, bool matrix) const
    {
        if (c1->Dimensions().Size() < 2)
            throw Exception("CovInc: called with scalar, vector, or 1-form field");
        if (matrix)
        {
            if (dim == 3)
            {
                return CovCurl(Transpose(c1));
            }
            else if (dim == 2)
            {
                return CovCurl(CovCurl(c1));
            }
            else
                throw Exception("CovInc: not implemented for dim = " + ToString(dim) + " yet");
        }
        shared_ptr<CoefficientFunction> cov_hesse = CovHessian(c1);
        shared_ptr<CoefficientFunction> p_cov_hesse = make_shared<ConstantCoefficientFunction>(0.25) * (cov_hesse - EinsumCF("ijkl->kjil", {cov_hesse}) - EinsumCF("ijkl->ilkj", {cov_hesse}) + EinsumCF("ijkl->klij", {cov_hesse}));

        return TensorFieldCF(-EinsumCF("ijkl->ikjl", {p_cov_hesse}), "1111");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovEin(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        auto J_c1 = J_op(c1);
        auto lap_term = CovLaplace(J_c1);
        auto def_term = CovDef(CovDivergence(J_c1));

        return TensorFieldCF(J_op(def_term) - 0.5 * lap_term, "11");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovLaplace(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        return CovDivergence(CovDerivative(c1));
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovDef(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (c1->GetCovariantIndices().size() != 1 || c1->GetCovariantIndices()[0] != '1')
            throw Exception("CovDef: Only implemented for 1-forms");
        auto cov_der = CovDerivative(c1);
        return TensorFieldCF(0.5 * (Transpose(cov_der, 0, 1) + cov_der), "11");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovRot(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (dim != 2)
            throw Exception("CovRot: only available in 2D");

        if (auto sf = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(c1))
        {
            return VectorFieldCF(EinsumCF("ij,j->i", {GetLeviCivitaSymbol(false), GradCF(sf, dim)}));
        }
        else if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
        {
            return TensorFieldCF(EinsumCF("jk,ki->ij", {GetLeviCivitaSymbol(false), GradCF(vf, dim)}) + EinsumCF("jq,qki,k->ij", {GetLeviCivitaSymbol(false), chr2, vf}), "00");
        }
        else if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
        {
            return CovRot(VectorFieldCF(Raise(of)));
        }
        else
            throw Exception("CovRot: only available for scalar or vector fields. Invoked with signature " + c1->GetSignature() + " and covariant indices " + c1->GetCovariantIndices());
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::LichnerowiczLaplacian(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        return TensorFieldCF(CovLaplace(c1) + 2 * EinsumCF("ikjl,kl->ij", {GetRiemannCurvatureTensor(), g_inv * c1 * g_inv}) - GetRicciTensor() * g_inv * c1 - TransposeCF(GetRicciTensor() * g_inv * c1), "11");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Trace(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2, VorB vb) const
    {
        if (index1 == index2)
            throw Exception("Trace: indices must be different");

        shared_ptr<CoefficientFunction> metric;
        shared_ptr<CoefficientFunction> metric_inv;

        switch (vb)
        {
        case VOL:
            metric = g;
            metric_inv = g_inv;
            break;
        case BND:
            metric = g_F;
            metric_inv = g_F_inv;
            break;
        case BBND:
            metric = g_E;
            metric_inv = g_E_inv;
            break;
        default:
            throw Exception("Trace: VorB must be VOL, BND, or BBND");
            break;
        }

        auto m = tf->Meta();

        if (index1 == index2)
            throw Exception("Trace: indices must be different");
        if (std::max(index1, index2) >= m.rank)
            throw Exception("Trace: index out of range");

        auto mout = m.Erased2(index1, index2);

        std::string sig = tf->GetSignature();
        std::string sigout = Erase2Labels(sig, index1, index2);

        shared_ptr<CoefficientFunction> result;

        bool cov1 = m.Covariant(index1);
        bool cov2 = m.Covariant(index2);

        if (cov1 != cov2)
        {
            std::string sigmod = sig;
            sigmod[index2] = sigmod[index1];

            std::string eins = sigmod + "->" + sigout;
            result = EinsumCF(eins, {tf->GetCoefficients()});
        }
        else
        {
            char fresh = m.FreshLabel();
            std::string sigmod = sig;
            sigmod[index2] = fresh;

            std::string metric_idx;
            metric_idx.reserve(2);
            metric_idx.push_back(fresh);
            metric_idx.push_back(sig[index1]);

            std::string eins = sigmod + "," + metric_idx + "->" + sigout;
            result = EinsumCF(eins, {tf->GetCoefficients(), cov1 ? metric_inv : metric});
        }

        return mout.rank ? TensorFieldCF(result, mout.CovString())
                         : ScalarFieldCF(result, dim);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Trace(shared_ptr<DoubleFormCoefficientFunction> tf, size_t l, VorB vb) const
    {
        if (!tf)
            throw Exception("Trace: input must be non-null");
        if (tf->DimensionOfSpace() != dim)
            throw Exception("Trace: double-form dimension does not match manifold dimension");

        int p = tf->LeftDegree();
        int q = tf->RightDegree();
        if (l == 0)
            return tf;

        if (l > size_t(p) || l > size_t(q))
        {
            Array<int> dims;
            auto zero_cf = ZeroCF(dims);
            return ScalarFieldCF(zero_cf, dim);
        }

        auto current = tf;
        for (size_t i = 0; i < l; ++i)
        {
            int lp = current->LeftDegree();
            int lq = current->RightDegree();
            auto traced = Trace(static_pointer_cast<TensorFieldCoefficientFunction>(current), 0, size_t(lp), vb);
            current = DoubleFormCF(traced, lp - 1, lq - 1, dim);
        }

        if (current->LeftDegree() == 0 && current->RightDegree() == 0)
            return ScalarFieldCF(current->GetCoefficients(), dim);

        return current;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::TraceSigma(shared_ptr<DoubleFormCoefficientFunction> tf, shared_ptr<DoubleFormCoefficientFunction> sigma, VorB vb) const
    {
        if (!tf || !sigma)
            throw Exception("TraceSigma: inputs must be non-null");
        if (tf->DimensionOfSpace() != dim)
            throw Exception("TraceSigma: double-form dimension does not match manifold dimension");
        if (sigma->DimensionOfSpace() != dim)
            throw Exception("TraceSigma: sigma dimension does not match manifold dimension");

        if (sigma->LeftDegree() != 1 || sigma->RightDegree() != 1)
            throw Exception("TraceSigma: sigma must be a (1,1) double form");

        int p = tf->LeftDegree();
        int q = tf->RightDegree();
        if (p == 0 || q == 0)
        {
            Array<int> dims;
            auto zero_cf = ZeroCF(dims);
            return ScalarFieldCF(zero_cf, dim);
        }

        shared_ptr<CoefficientFunction> metric_inv;
        switch (vb)
        {
        case VOL:
            metric_inv = g_inv;
            break;
        case BND:
            metric_inv = g_F_inv;
            break;
        case BBND:
            metric_inv = g_E_inv;
            break;
        default:
            throw Exception("TraceSigma: VorB must be VOL, BND, or BBND");
        }

        auto raise_with_metric = [&](shared_ptr<TensorFieldCoefficientFunction> tf_in, size_t index)
        {
            if (tf_in->Dimensions().Size() <= index)
                throw Exception("TraceSigma: sigma index out of range");

            auto m = tf_in->Meta();
            if (!m.Covariant(index))
                throw Exception("TraceSigma: sigma indices must be covariant");

            std::string sig = tf_in->GetSignature();
            char a = sig[index];
            char b = m.FreshLabel();
            std::string lhs = sig;
            lhs[index] = b;

            std::string eins = lhs + "," + std::string(1, a) + std::string(1, b) + "->" + sig;
            auto mout = m.WithCovariant(index, false);
            auto out_cf = EinsumCF(eins, {tf_in->GetCoefficients(), metric_inv});
            return TensorFieldCF(out_cf, mout);
        };

        auto sigma_tf = static_pointer_cast<TensorFieldCoefficientFunction>(sigma);
        auto sigma_raised = raise_with_metric(sigma_tf, 0);
        sigma_raised = raise_with_metric(sigma_raised, 1);

        std::string sig = tf->GetSignature();
        std::string sig_out = Erase2Labels(sig, 0, size_t(p));

        std::string sigma_sig;
        sigma_sig.push_back(sig[0]);
        sigma_sig.push_back(sig[size_t(p)]);

        std::string eins = sig + "," + sigma_sig + "->" + sig_out;
        auto out_cf = EinsumCF(eins, {tf->GetCoefficients(), sigma_raised});

        if (sig_out.empty())
            return ScalarFieldCF(out_cf, dim);
        return DoubleFormCF(out_cf, p - 1, q - 1, dim);
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::SlotInnerProduct(shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb, bool forms) const
    {
        if (!tf)
            throw Exception("SlotInnerProduct: input must be non-null");
        if (tf->DimensionOfSpace() != dim)
            throw Exception("SlotInnerProduct: double-form dimension does not match manifold dimension");

        int p = tf->LeftDegree();
        int q = tf->RightDegree();
        if (p != q)
            throw Exception("SlotInnerProduct: double-form degrees must match");

        if (p == 0)
            return ScalarFieldCF(tf->GetCoefficients(), dim);

        auto res = Trace(tf, size_t(p), vb);
        if (auto sf = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(res))
        {
            if (!forms)
                return sf;
            double scale = 1.0 / double(Factorial(p));
            return ScalarFieldCF(scale * sf->GetCoefficients(), dim);
        }
        auto out = ScalarFieldCF(res, dim);
        if (!forms)
            return out;
        double scale = 1.0 / double(Factorial(p));
        return ScalarFieldCF(scale * out->GetCoefficients(), dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::ProjectDoubleForm(shared_ptr<DoubleFormCoefficientFunction> tf, int left_mode, int right_mode,
                                                                                    shared_ptr<VectorFieldCoefficientFunction> normal,
                                                                                    shared_ptr<VectorFieldCoefficientFunction> conormal) const
    {
        if (!tf)
            throw Exception("ProjectDoubleForm: input must be non-null");
        if (tf->DimensionOfSpace() != dim)
            throw Exception("ProjectDoubleForm: double-form dimension does not match manifold dimension");
        if (normal && normal->Dimensions().Size() && normal->Dimensions()[0] != dim)
            throw Exception("ProjectDoubleForm: normal dimension does not match manifold dimension");
        if (conormal && conormal->Dimensions().Size() && conormal->Dimensions()[0] != dim)
            throw Exception("ProjectDoubleForm: conormal dimension does not match manifold dimension");

        int p = tf->LeftDegree();
        int q = tf->RightDegree();

        auto current = static_pointer_cast<TensorFieldCoefficientFunction>(tf);
        auto proj = P_F_g;
        auto n = g_nv;
        auto edge_proj = P_E_g;

        if (normal)
        {
            n = normal;
            proj = IdentityCF(dim) - TensorProduct(n, Lower(n));
        }

        bool needs_edge = (left_mode == 3 || left_mode == 4 || right_mode == 3 || right_mode == 4);
        if (needs_edge && !edge_proj)
            throw Exception("ProjectDoubleForm: edge tangent projector not available");

        bool needs_conormal = (left_mode == 4 || right_mode == 4);
        if (needs_conormal && !conormal)
            throw Exception("ProjectDoubleForm: conormal must be provided for mode 'm'");

        auto apply_slot = [&](int slot, int mode)
        {
            if (mode == 0)
                return;

            int count = (slot == 0) ? p : q;
            if (count == 0)
            {
                if (mode == 2 || mode == 4)
                    throw Exception("ProjectDoubleForm: cannot contract empty slot");
                return;
            }

            size_t start = (slot == 0) ? 0 : size_t(p);

            if (mode == 2)
            {
                for (size_t i = start + 1; i < start + size_t(count); ++i)
                    current = ApplyProjectorToIndex(current, proj, i);
                current = Contraction(current, n, start);
                if (slot == 0)
                    --p;
                else
                    --q;
                return;
            }

            if (mode == 1)
            {
                for (size_t i = start; i < start + size_t(count); ++i)
                    current = ApplyProjectorToIndex(current, proj, i);
                return;
            }

            if (mode == 3)
            {
                for (size_t i = start; i < start + size_t(count); ++i)
                    current = ApplyProjectorToIndex(current, edge_proj, i);
                return;
            }

            if (mode == 4)
            {
                for (size_t i = start + 1; i < start + size_t(count); ++i)
                    current = ApplyProjectorToIndex(current, edge_proj, i);
                current = Contraction(current, conormal, start);
                if (slot == 0)
                    --p;
                else
                    --q;
                return;
            }

            throw Exception("ProjectDoubleForm: mode must be 0 (none), 1 (F/tangent), 2 (n/normal), 3 (E/edge), or 4 (m/conormal)");
        };

        apply_slot(0, left_mode);
        apply_slot(1, right_mode);

        return DoubleFormCF(current, p, q, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::ContractSlot(shared_ptr<DoubleFormCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, int slot) const
    {
        if (!tf || !vf)
            throw Exception("ContractSlot: inputs must be non-null");
        if (tf->DimensionOfSpace() != dim)
            throw Exception("ContractSlot: double-form dimension does not match manifold dimension");

        int p = tf->LeftDegree();
        int q = tf->RightDegree();
        size_t index = 0;

        if (slot == 0)
        {
            if (p == 0)
                throw Exception("ContractSlot: left slot degree is zero");
            index = 0;
        }
        else if (slot == 1)
        {
            if (q == 0)
                throw Exception("ContractSlot: right slot degree is zero");
            index = size_t(p);
        }
        else
        {
            throw Exception("ContractSlot: slot must be 0/1 or 'left'/'right'");
        }

        auto out = Contraction(static_pointer_cast<TensorFieldCoefficientFunction>(tf), vf, index);
        return DoubleFormCF(out, slot == 0 ? p - 1 : p, slot == 1 ? q - 1 : q, dim);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot) const
    {

        auto m = tf->Meta();
        if (slot >= m.rank)
            throw Exception("Contraction: slot out of range");

        char a = m.Label(slot);
        std::string sig = tf->GetSignature();
        std::string sig_out = EraseLabel(sig, slot);

        shared_ptr<CoefficientFunction> out_cf;

        if (m.Covariant(slot))
        {
            std::string eins = sig + "," + std::string(1, a) + "->" + sig_out;
            out_cf = EinsumCF(eins, {tf->GetCoefficients(), vf->GetCoefficients()});
        }
        else
        {
            char b = m.FreshLabel();
            std::string eins = sig + "," + std::string(1, a) + std::string(1, b) + "," + std::string(1, b) + "->" + sig_out;
            out_cf = EinsumCF(eins, {tf->GetCoefficients(), g, vf->GetCoefficients()});
        }

        return m.Erased(slot).rank ? TensorFieldCF(out_cf, m.Erased(slot).CovString()) : ScalarFieldCF(out_cf, dim);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Transpose(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2) const
    {
        if (index1 == index2)
            throw Exception("Transpose: indices must be different");
        if (tf->Dimensions().Size() <= max(index1, index2))
            throw Exception("Transpose: index out of range");

        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();
        string signature_result = signature;
        swap(signature_result[index1], signature_result[index2]);
        swap(cov_ind[index1], cov_ind[index2]);
        return TensorFieldCF(EinsumCF(signature + "->" + signature_result, {tf->GetCoefficients()}), cov_ind);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::S_op(shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb) const
    {
        if (tf->Dimensions().Size() != 2 || tf->Dimensions()[0] != tf->Dimensions()[1])
            throw Exception("S_op: only available for 2-tensors");
        if (tf->GetCovariantIndices() != "11")
            throw Exception("S_op: currently only implemented for (2,0)-tensors!");
        switch (vb)
        {
        case VOL:
            return TensorFieldCF(Transpose(tf) - this->Trace(tf, 0, 1, VOL) * g, "11");
        case BND:
            return TensorFieldCF(P_F_g * Transpose(tf) * P_F_g - this->Trace(tf, 0, 1, BND) * g_F, "11");
        default:
            throw Exception("S_op: Only implemented for VOL and BND");
        }
    }

    shared_ptr<DoubleFormCoefficientFunction> RiemannianManifold::s_op(shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb) const
    {
        if (!tf)
            throw Exception("s_op: input must be non-null");
        if (tf->DimensionOfSpace() != dim)
            throw Exception("s_op: double-form dimension does not match manifold dimension");

        int p = tf->LeftDegree();
        int q = tf->RightDegree();
        if (q < 1)
            throw Exception("s_op: q must be at least 1");

        shared_ptr<CoefficientFunction> mix;
        switch (vb)
        {
        case VOL:
            mix = IdentityCF(dim);
            break;
        case BND:
            mix = TransposeCF(P_F_g);
            break;
        default:
            throw Exception("s_op: only implemented for VOL and BND");
        }

        std::string sig = tf->GetSignature();
        std::string left_sig = sig.substr(0, size_t(p));
        std::string right_sig = sig.substr(size_t(p), size_t(q));
        char new_left = FreshLabel(sig);

        std::string out_sig;
        out_sig.reserve(size_t(p + q));
        out_sig.push_back(new_left);
        out_sig += left_sig;
        if (q > 1)
            out_sig += right_sig.substr(1);

        std::string gsig;
        gsig.reserve(2);
        gsig.push_back(new_left);
        gsig.push_back(right_sig[0]);

        std::string eins = gsig + "," + sig + "->" + out_sig;
        auto contracted = EinsumCF(eins, {mix, tf->GetCoefficients()});
        auto alt = BlockAlternationByPermutationCF(contracted, p + q, 0, p + 1);

        double scale = 1.0 / double(Factorial(p));
        auto out = scale * alt;
        return DoubleFormCF(out, p + 1, q - 1, dim);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::J_op(shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb) const
    {
        if (tf->Dimensions().Size() != 2 || tf->Dimensions()[0] != tf->Dimensions()[1])
            throw Exception("J_op: only available for 2-tensors");
        if (tf->GetCovariantIndices() != "11")
            throw Exception("J_op: currently only implemented for (2,0)-tensors!");
        switch (vb)
        {
        case VOL:
            return TensorFieldCF(Transpose(tf) - 0.5 * this->Trace(tf, 0, 1, VOL) * g, "11");
        case BND:
            return TensorFieldCF(P_F_g * Transpose(tf) * P_F_g - 0.5 * this->Trace(tf, 0, 1, BND) * g_F, "11");
        default:
            throw Exception("J_op: Only implemented for VOL and BND");
        }
    }

}

void ExportRiemannianManifold(py::module m)
{
    using namespace ngfem;

    auto parse_slot = [](py::object slot, const std::string &name) -> int
    {
        if (py::isinstance<py::str>(slot))
        {
            std::string s = py::cast<std::string>(slot);
            if (s == "left" || s == "0")
                return 0;
            if (s == "right" || s == "1")
                return 1;
            throw Exception(name + ": slot must be 'left'/'right' or 0/1");
        }
        if (py::isinstance<py::int_>(slot))
            return py::cast<int>(slot);
        throw Exception(name + ": slot must be 'left'/'right' or 0/1");
    };

    auto parse_proj = [](py::object mode, const std::string &name) -> int
    {
        if (py::isinstance<py::str>(mode))
        {
            std::string s = py::cast<std::string>(mode);
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            if (s == "f" || s == "tangent" || s == "tan" || s == "1")
                return 1;
            if (s == "n" || s == "normal" || s == "2")
                return 2;
            if (s == "e" || s == "edge" || s == "3")
                return 3;
            if (s == "m" || s == "conormal" || s == "4")
                return 4;
            if (s == "none" || s == "0")
                return 0;
            throw Exception(name + ": mode must be 'F'/'tangent', 'n'/'normal', 'E'/'edge', 'm'/'conormal', or 'none'");
        }
        if (py::isinstance<py::int_>(mode))
            return py::cast<int>(mode);
        throw Exception(name + ": mode must be string or int");
    };

    py::class_<RiemannianManifold, shared_ptr<RiemannianManifold>>(m, "RiemannianManifold")
        .def(py::init<shared_ptr<CoefficientFunction>, double, bool>(), "constructor", py::arg("metric"), py::arg("normal_sign") = 1.0, py::arg("change_riemann_sign") = false)
        .def("VolumeForm", &RiemannianManifold::GetVolumeForm, "return the volume form of given dimension", py::arg("vb"))
        .def_property_readonly("dim", &RiemannianManifold::GetDimension, "return the manifold dimension")
        .def_property_readonly("G", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetric(); }, "return the metric tensor")
        .def_property_readonly("G_F", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricF(); }, "return the metric tensor g_F")
        .def_property_readonly("G_F_inv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricFInverse(); }, "return the inverse of the metric tensor g_F")
        .def_property_readonly("G_E", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricE(); }, "return the metric tensor g_E")
        .def_property_readonly("G_E_inv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricEInverse(); }, "return the inverse of the metric tensor g_E")
        .def_property_readonly("G_inv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricInverse(); }, "return the inverse of the metric tensor")
        .def_property_readonly("normal", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetNV(); }, "return the normal vector")
        .def_property_readonly("tangent", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetEdgeTangent(); }, "return the tangent vector")
        .def("EdgeNormal", [](shared_ptr<RiemannianManifold> self, int i)
             { return self->GetEdgeNormal(i); }, "return the edge normal (i=0 or 1)", py::arg("i"))
        .def("EdgeConormal", [](shared_ptr<RiemannianManifold> self, int i)
             { return self->GetEdgeConormal(i); }, "return the edge conormal (i=0 or 1)", py::arg("i"))
        .def_property_readonly("G_deriv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricDerivative(); }, "return the derivative of the metric tensor")
        .def("Christoffel", [](shared_ptr<RiemannianManifold> self, bool second_kind)
             { return self->GetChristoffelSymbol(second_kind); }, "return the Christoffel symbol of the first or second kind", py::arg("second_kind") = false)
        .def("LeviCivitaSymbol", [](shared_ptr<RiemannianManifold> self, bool covariant)
             { return self->GetLeviCivitaSymbol(covariant); }, "return the Levi-Civita symbol", py::arg("covariant") = false)
        .def("Raise", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, size_t index, VorB vb)
             { return self->Raise(tf, index, vb); }, "Raise a tensor index using the manifold metric", py::arg("tf"), py::arg("index") = 0, py::arg("vb") = VOL)
        .def("Raise", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, const std::vector<size_t> &indices, VorB vb)
             { return self->Raise(tf, indices, vb); }, "Raise tensor indices using the manifold metric", py::arg("tf"), py::arg("indices"), py::arg("vb") = VOL)
        .def("Lower", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, size_t index, VorB vb)
             { return self->Lower(tf, index, vb); }, "Lower a tensor index using the manifold metric", py::arg("tf"), py::arg("index") = 0, py::arg("vb") = VOL)
        .def("Lower", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, const std::vector<size_t> &indices, VorB vb)
             { return self->Lower(tf, indices, vb); }, "Lower tensor indices using the manifold metric", py::arg("tf"), py::arg("indices"), py::arg("vb") = VOL)
        .def_property_readonly("Riemann", &RiemannianManifold::GetRiemannCurvatureTensor, "return the Riemann curvature tensor")
        .def_property_readonly("Curvature", &RiemannianManifold::GetCurvatureOperator, "return the curvature operator")
        .def_property_readonly("Gauss", &RiemannianManifold::GetGaussCurvature, "return the Gauss curvature in 2D")
        .def_property_readonly("Ricci", &RiemannianManifold::GetRicciTensor, "return the Ricci tensor")
        .def_property_readonly("Einstein", &RiemannianManifold::GetEinsteinTensor, "return the Einstein tensor")
        .def_property_readonly("Scalar", &RiemannianManifold::GetScalarCurvature, "return the scalar curvature")
        .def_property_readonly("SFF", &RiemannianManifold::GetSecondFundamentalForm, "return the second fundamental form")
        .def_property_readonly("GeodesicCurvature", &RiemannianManifold::GetGeodesicCurvature, "return the geodesic curvature")
        .def_property_readonly("MeanCurvature", &RiemannianManifold::GetMeanCurvature, "return the mean curvature")
        .def("KForm", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> cf, int k)
             { return self->MakeKForm(cf, k); }, "Wrap a CoefficientFunction as a k-form using the manifold dimension", py::arg("cf"), py::arg("k"))
        .def("star", [](shared_ptr<RiemannianManifold> self, shared_ptr<KFormCoefficientFunction> a, VorB vb)
             { return self->Star(a, vb); }, "Hodge star of a k-form using the manifold metric", py::arg("a"), py::arg("vb") = VOL)
        .def("inv_star", [](shared_ptr<RiemannianManifold> self, shared_ptr<KFormCoefficientFunction> a, VorB vb)
             { return self->InvStar(a, vb); }, "Inverse Hodge star of a k-form using the manifold metric", py::arg("a"), py::arg("vb") = VOL)
        .def("inv_star", [](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> a, VorB vb)
             { return self->InvStar(a, vb); }, "Inverse Hodge star of a double-form using the manifold metric", py::arg("a"), py::arg("vb") = VOL)
        .def("delta", [](shared_ptr<RiemannianManifold> self, shared_ptr<KFormCoefficientFunction> a)
             { return self->Coderivative(a); }, "Exterior coderivative of a k-form using the manifold metric", py::arg("a"))
        .def("d_cov", [parse_slot](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, py::object slot, VorB vb)
             {
                 shared_ptr<DoubleFormCoefficientFunction> df;
                 if (auto dform = dynamic_pointer_cast<DoubleFormCoefficientFunction>(tf))
                     df = dform;
                 else if (tf && tf->Dimensions().Size() == 0)
                     df = DoubleFormCF(tf, 0, 0, self->Dimension());
                 else
                     throw Exception("d_cov: expected DoubleForm or scalar field");
                 int slot_id = parse_slot(slot, "d_cov");
                 if (slot_id == 0)
                     return self->CovExteriorDerivative1(df, vb);
                 if (slot_id == 1)
                     return self->CovExteriorDerivative2(df, vb);
                 throw Exception("d_cov: slot must be 0/1 or 'left'/'right'"); }, "Exterior covariant derivative of a double-form", py::arg("tf"), py::arg("slot") = "left", py::arg("vb") = VOL)
        .def("delta_cov", [parse_slot](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, py::object slot, VorB vb)
             {
                 shared_ptr<DoubleFormCoefficientFunction> df;
                 if (auto dform = dynamic_pointer_cast<DoubleFormCoefficientFunction>(tf))
                     df = dform;
                 else if (tf && tf->Dimensions().Size() == 0)
                     df = DoubleFormCF(tf, 0, 0, self->Dimension());
                 else
                     throw Exception("delta_cov: expected DoubleForm or scalar field");
                 int slot_id = parse_slot(slot, "delta_cov");
                 if (slot_id == 0)
                     return self->CovCodifferential1(df, vb);
                 if (slot_id == 1)
                     return self->CovCodifferential2(df, vb);
                 throw Exception("delta_cov: slot must be 0/1 or 'left'/'right'"); }, "Exterior covariant codifferential of a double-form", py::arg("tf"), py::arg("slot") = "left", py::arg("vb") = VOL)
        .def("ProjectDoubleForm", [parse_proj](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, py::object left, py::object right, py::object normal, py::object conormal)
             {
                 int left_mode = parse_proj(left, "ProjectDoubleForm");
                 int right_mode = parse_proj(right, "ProjectDoubleForm");
                 shared_ptr<VectorFieldCoefficientFunction> n = nullptr;
                 shared_ptr<VectorFieldCoefficientFunction> cn = nullptr;
                 if (!normal.is_none())
                     n = py::cast<shared_ptr<VectorFieldCoefficientFunction>>(normal);
                 if (!conormal.is_none())
                     cn = py::cast<shared_ptr<VectorFieldCoefficientFunction>>(conormal);
                 return self->ProjectDoubleForm(tf, left_mode, right_mode, n, cn); }, "Project a double-form in left/right slots onto tangent or normal components (normal reduces slot degree by one). Optional normal/conormal override the defaults for boundary/edge projections.", py::arg("tf"), py::arg("left") = "none", py::arg("right") = "none", py::arg("normal") = py::none(), py::arg("conormal") = py::none())
        .def("ProjectTensor", [parse_proj](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, py::object mode)
             {
                 int proj_mode = parse_proj(mode, "ProjectTensor");
                 return self->ProjectTensor(tf, proj_mode); }, "Project a tensor field onto tangent or normal components (normal contracts the first index)", py::arg("tf"), py::arg("mode") = "none")
        .def("ContractSlot", [parse_slot](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, py::object slot)
             {
                 int slot_id = parse_slot(slot, "ContractSlot");
                 return self->ContractSlot(tf, vf, slot_id); }, "Contract a double-form with a vector in the left or right slot, reducing the corresponding degree", py::arg("tf"), py::arg("vf"), py::arg("slot") = "left")
        .def("InnerProduct", [](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf1, shared_ptr<DoubleFormCoefficientFunction> tf2, VorB vb, bool forms)
             { return self->IP(tf1, tf2, vb, forms); }, "InnerProduct of two DoubleForms", py::arg("tf1"), py::arg("tf2"), py::arg("vb") = VOL, py::arg("forms") = false)
        .def("InnerProduct", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf1, shared_ptr<TensorFieldCoefficientFunction> tf2, VorB vb, bool forms)
             { return self->IP(tf1, tf2, vb, forms); }, "InnerProduct of two TensorFields", py::arg("tf1"), py::arg("tf2"), py::arg("vb") = VOL, py::arg("forms") = false)
        .def("Cross", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf1, shared_ptr<TensorFieldCoefficientFunction> tf2)
             { return self->Cross(tf1, tf2); }, "Cross product in 3D of two vector fields, 1-forms, or both mixed. Returns the resulting vector-field.", py::arg("tf1"), py::arg("tf2"))
        .def("CovDeriv", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb)
             { return self->CovDerivative(tf, vb); }, "Covariant derivative of a TensorField", py::arg("tf"), py::arg("vb") = VOL)
        .def("CovHesse", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->CovHessian(tf); }, "Covariant Hessian of a TensorField.", py::arg("tf"))
        .def("CovCurl", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->CovCurl(tf); }, "Covariant curl of a TensorField in 3D", py::arg("tf"))
        .def("CovInc", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, bool matrix = false)
             { return self->CovInc(tf, matrix); }, "Covariant inc of a TensorField. If matrix=True a scalar in 2D and matrix in 3D is returned", py::arg("tf"), py::arg("matrix") = false)
        .def("CovEin", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->CovEin(tf); }, "Covariant ein of a TensorField in 2D or 3D", py::arg("tf"))
        .def("CovLaplace", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->CovLaplace(tf); }, "Covariant Laplace of a TensorField ", py::arg("tf"))
        .def("LichnerowiczLaplacian", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->LichnerowiczLaplacian(tf); }, "Lichnerowicz Laplacian of a TensorField", py::arg("tf"))
        .def("CovDef", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->CovDef(tf); }, "Covariant Def (symmetric derivative) of a 1-form", py::arg("tf"))
        .def("CovRot", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf)
             { return self->CovRot(tf); }, "Covariant rot of a TensorField of maximal order 1 in 2D. Returns a contravariant tensor field.", py::arg("tf"))
        .def("CovDiv", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb)
             { return self->CovDivergence(tf, vb); }, "Covariant divergence of a TensorField", py::arg("tf"), py::arg("vb") = VOL)
        .def("CovDiv", [parse_slot](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, py::object slot, VorB vb)
             {
                 int slot_id = parse_slot(slot, "CovDiv");
                 if (slot_id != 0 && slot_id != 1)
                     throw Exception("CovDiv: slot must be 0/1 or 'left'/'right'");
                 return self->CovDivergence(tf, slot_id, vb); }, "Covariant divergence of a DoubleForm in a selected slot", py::arg("tf"), py::arg("slot") = "left", py::arg("vb") = VOL)
        .def("Trace", [](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, size_t l, VorB vb)
             { return self->Trace(tf, l, vb); }, "Trace of DoubleForm: contract first l left/right slots (l=0 returns input).", py::arg("tf"), py::arg("l") = 1, py::arg("vb") = VOL)
        .def("TraceSigma", [](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, shared_ptr<DoubleFormCoefficientFunction> sigma, VorB vb)
             { return self->TraceSigma(tf, sigma, vb); }, "Trace of DoubleForm contracting first left/right slots with (1,1) sigma.", py::arg("tf"), py::arg("sigma"), py::arg("vb") = VOL)
        .def("SlotInnerProduct", [](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb, bool forms)
             { return self->SlotInnerProduct(tf, vb, forms); }, "Inner product of left/right slots for (p,p) DoubleForm", py::arg("tf"), py::arg("vb") = VOL, py::arg("forms") = true)
        .def("Trace", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb, size_t index1, size_t index2)
             { return self->Trace(tf, index1, index2, vb); }, "Trace of TensorField in two indices (l is only supported for DoubleForm).", py::arg("tf"), py::arg("vb") = VOL, py::arg("index1") = 0, py::arg("index2") = 1)
        .def("Contraction", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot)
             { return self->Contraction(tf, vf, slot); }, "Contraction of TensorField with a VectorField at given slot. Default slot is the first.", py::arg("tf"), py::arg("vf"), py::arg("slot") = 0)
        .def("Transpose", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2)
             { return self->Transpose(tf, index1, index2); }, "Transpose of TensorField for given indices. Default indices are first and second.", py::arg("tf"), py::arg("index1") = 0, py::arg("index2") = 1)
        .def("S", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb)
             { return self->S_op(tf, vb); }, "S operator subtracting the trace.", py::arg("tf"), py::arg("vb") = VOL)
        .def("s", [](shared_ptr<RiemannianManifold> self, shared_ptr<DoubleFormCoefficientFunction> tf, VorB vb)
             { return self->s_op(tf, vb); }, "s operator for double forms: (p,q) -> (p+1,q-1)", py::arg("tf"), py::arg("vb") = VOL)
        .def("J", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb)
             { return self->J_op(tf, vb); }, "J operator subtracting half the trace.", py::arg("tf"), py::arg("vb") = VOL);
}
