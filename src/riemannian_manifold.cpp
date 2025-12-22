#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"
#include "kforms.hpp"

#include <coefficient_stdmath.hpp>
#include <python_comp.hpp>
// #include <fem.hpp>
#include <integratorcf.hpp>
#include <hcurlcurlfespace.hpp>

namespace ngfem
{
    namespace
    {
        inline int DeltaSign(int n, int k)
        {
            int exponent = n * (k + 1) + 1;
            return (exponent % 2 == 0) ? 1 : -1;
        }
    } // namespace

    using namespace std;
    RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g)
        : has_trial(false), is_regge(false), is_proxy(false), regge_proxy(nullptr), regge_space(nullptr), g(_g)
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
        nv = NormalVectorCF(dim);
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
            vol[BND] = sqrt(InnerProduct(g * tv, tv));
            g_tv = VectorFieldCF(1 / vol[BND] * tv);

            // more efficient?
            // auto tv_mat = tv->Reshape(Array<int>({dim, 1}));
            // auto P_F = tv_mat * TransposeCF(tv_mat);
            // g_F = InnerProduct(g * tv, tv) * P_F;
            // g_F_inv = 1/InnerProduct(g * tv, tv) * P_F;

            g_F = P_F * g * P_F;
            g_F_inv = P_F * InverseCF(g_F + P_n) * P_F;
            g_E = one_cf;
            g_E_inv = one_cf;
        }
        else if (dim == 3)
        {
            vol[BND] = sqrt(InnerProduct(CofactorCF(g) * nv, nv));
            vol[BBND] = sqrt(InnerProduct(g * tv, tv));

            g_tv = VectorFieldCF(1 / vol[BBND] * tv);

            g_F = P_F * g * P_F;

            g_F_inv = P_F * InverseCF(g_F + P_n) * P_F;

            auto tv_mat = tv->Reshape(Array<int>({dim, 1}));
            auto P_E = tv_mat * TransposeCF(tv_mat);
            g_E = InnerProduct(g * tv, tv) * P_E;
            g_E_inv = 1 / InnerProduct(g * tv, tv) * P_E;
        }

        g_nv = VectorFieldCF(vol[VOL] / vol[BND] * g_inv * nv);

        if (is_regge)
        {
            if (is_proxy)
            {
                // cout << "is_proxy" << endl;

                auto g_proxy = dynamic_pointer_cast<ProxyFunction>(g);
                g_deriv = g_proxy->GetAdditionalProxy("grad");
                chr1 = g_proxy->GetAdditionalProxy("christoffel");
                chr2 = g_proxy->GetAdditionalProxy("christoffel2");
                Riemann = TensorFieldCF(g_proxy->GetAdditionalProxy("Riemann"), "1111");
                Curvature = TensorFieldCF(g_proxy->GetAdditionalProxy("curvature"), "00");
                Ricci = TensorFieldCF(g_proxy->GetAdditionalProxy("Ricci"), "11");
                Einstein = TensorFieldCF(g_proxy->GetAdditionalProxy("Einstein"), "11");
                Scalar = ScalarFieldCF(g_proxy->GetAdditionalProxy("scalar"), dim);
                if (!g_deriv || !chr1 || !chr2 || !Riemann || !Curvature || !Ricci || !Einstein || !Scalar)
                    throw Exception("In RMF: Could not load all additional proxy functions");
                SFF = TensorFieldCF(EinsumCF("ijk,k->ij", {chr1->Reshape(Array<int>({dim, dim, dim})), g_nv}), "11");
            }
            else
            {
                // cout << "not proxy" << endl;
                auto diffop_grad = regge_space->GetAdditionalEvaluators()["grad"];
                auto diffop_chr1 = regge_space->GetAdditionalEvaluators()["christoffel"];
                auto diffop_chr2 = regge_space->GetAdditionalEvaluators()["christoffel2"];
                auto diffop_Riemann = regge_space->GetAdditionalEvaluators()["Riemann"];
                auto diffop_curvature = regge_space->GetAdditionalEvaluators()["curvature"];
                auto diffop_Ricci = regge_space->GetAdditionalEvaluators()["Ricci"];
                auto diffop_Einstein = regge_space->GetAdditionalEvaluators()["Einstein"];
                auto diffop_scalar = regge_space->GetAdditionalEvaluators()["scalar"];
                shared_ptr<ngcomp::GridFunction> gf = dynamic_pointer_cast<ngcomp::GridFunction>(g);

                if (!diffop_grad || !diffop_chr1 || !diffop_chr2 || !diffop_Riemann || !diffop_curvature || !diffop_Ricci || !diffop_Einstein || !diffop_scalar || !gf)
                    throw Exception("In RMF: Could not load all additional evaluators");

                g_deriv = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_grad);
                g_deriv->SetDimensions(diffop_grad->Dimensions());

                chr1 = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_chr1);
                chr1->SetDimensions(diffop_chr1->Dimensions());

                chr2 = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_chr2);
                chr2->SetDimensions(diffop_chr2->Dimensions());

                auto Riemann_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_Riemann);
                Riemann_gf->SetDimensions(diffop_Riemann->Dimensions());
                Riemann = TensorFieldCF(Riemann_gf, "1111");

                auto Curvature_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_curvature);
                Curvature_gf->SetDimensions(diffop_curvature->Dimensions());

                if (dim == 2)
                    Curvature = ScalarFieldCF(Curvature_gf, dim);
                else
                    Curvature = TensorFieldCF(Curvature_gf, "00");

                auto Ricci_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_Ricci);
                Ricci_gf->SetDimensions(diffop_Ricci->Dimensions());
                Ricci = TensorFieldCF(Ricci_gf, "11");
                auto Einstein_gf = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_Einstein);
                Einstein_gf->SetDimensions(diffop_Einstein->Dimensions());
                Einstein = TensorFieldCF(Einstein_gf, "11");
                Scalar = ScalarFieldCF(make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop_scalar), dim);
                SFF = TensorFieldCF(EinsumCF("ijk,k->ij", {chr1->Reshape(Array<int>({dim, dim, dim})), g_nv}), "11");
            }
        }
        else
        {
            // cout << "is CF" << endl;
            g_deriv = GradCF(g, dim);
            Array<shared_ptr<CoefficientFunction>> values(dim * dim * dim);

            for (auto i : Range(dim))
                for (auto j : Range(dim))
                    for (auto k : Range(dim))
                    {
                        values[i * dim * dim + j * dim + k] = 0.5 * (MakeComponentCoefficientFunction(g_deriv, i * dim * dim + j + dim * k) + MakeComponentCoefficientFunction(g_deriv, j * dim * dim + i + dim * k) - MakeComponentCoefficientFunction(g_deriv, k * dim * dim + i + dim * j));
                    }
            auto tmp = MakeVectorialCoefficientFunction(std::move(values));
            chr1 = tmp->Reshape(Array<int>({dim, dim, dim}));
            chr2 = EinsumCF("ijl,lk->ijk", {chr1, g_inv});

            auto chr1_grad = GradCF(chr1, dim);
            auto lin_part = EinsumCF("ijkl->kilj", {chr1_grad}) - EinsumCF("ijkl->likj", {chr1_grad});
            auto non_lin_part = EinsumCF("ijm,klm->iklj", {chr1, chr2}) - EinsumCF("ijm,klm->ilkj", {chr2, chr1});
            Riemann = TensorFieldCF(lin_part + non_lin_part, "1111");
            auto LeviCivita = GetLeviCivitaSymbol(false);
            string signature = SIGNATURE.substr(0, 4) + "," + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(0, 2) + "," + SIGNATURE.substr(2 + dim, dim - 2) + SIGNATURE.substr(2, 2) + "->" + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(2 + dim, dim - 2);
            if (dim == 2)
                Curvature = ScalarFieldCF(1 / 4 * EinsumCF(signature, {Riemann, LeviCivita, LeviCivita}), dim);
            else
                Curvature = TensorFieldCF(1 / 4 * EinsumCF(signature, {Riemann, LeviCivita, LeviCivita}), "00");
            Ricci = Trace(Riemann, 0, 2);
            Scalar = Trace(Ricci, 0, 1);
            Einstein = TensorFieldCF(Ricci - 0.5 * Scalar * g, "11");
        }
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetric() const
    {
        return TensorFieldCF(g, "11");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricInverse() const
    {
        return TensorFieldCF(g_inv, "00");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetVolumeForm(VorB vb) const
    {
        return vol[int(vb)];
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Raise(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index) const
    {
        if (tf->Dimensions().Size() <= index)
            throw Exception(ToString("Raise: Dimension of tf = ") + ToString(tf->Dimensions().Size()) + "<= index = " + ToString(index));

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
        auto out_cf = EinsumCF(eins, {tf, g_inv});

        // if tf is a OneFormCoefficientFunction, return a VectorFieldCoefficientFunction
        if (dynamic_pointer_cast<OneFormCoefficientFunction>(tf))
            return VectorFieldCF(out_cf);

        return TensorFieldCF(out_cf, mout);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Lower(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index) const
    {
        if (tf->Dimensions().Size() <= index)
            throw Exception(ToString("Lower: Dimension of tf = ") + ToString(tf->Dimensions().Size()) + "<= index = " + ToString(index));

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
        auto out_cf = EinsumCF(eins, {tf, g});

        // if tf is a VectorFieldCoefficientFunction, return a OneFormCoefficientFunction
        if (dynamic_pointer_cast<VectorFieldCoefficientFunction>(tf))
            return OneFormCF(out_cf);

        return TensorFieldCF(out_cf, mout);
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

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetRiemannCurvatureTensor() const
    {
        return TensorFieldCF(Riemann, "1111");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetCurvatureOperator() const
    {
        if (dim == 2)
            return ScalarFieldCF(Curvature, dim);
        else
            return TensorFieldCF(Curvature, "00");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetRicciTensor() const
    {
        return TensorFieldCF(Ricci, "11");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetEinsteinTensor() const
    {
        return TensorFieldCF(Einstein, "11");
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

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::GetSecondFundamentalForm() const
    {
        return TensorFieldCF(SFF, "11");
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

    shared_ptr<VectorFieldCoefficientFunction> RiemannianManifold::GetNV() const
    {
        return g_nv;
    }

    shared_ptr<VectorFieldCoefficientFunction> RiemannianManifold::GetEdgeTangent() const
    {
        return g_tv;
    }

    shared_ptr<ScalarFieldCoefficientFunction> RiemannianManifold::IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2, VorB vb) const
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
        return ScalarFieldCF(EinsumCF(signature_c1 + "," + signature_c2 + raise_lower_signatures, cfs), dim);
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const
    {
        string cov_ind1 = c1->GetCovariantIndices();
        string cov_ind2 = c2->GetCovariantIndices();
        if (cov_ind1.size() != 1 || cov_ind2.size() != 1)
            throw Exception("Cross: only available for vector fields and 1-forms yet.");
        if (c1->Dimensions()[0] != 3)
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

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb) const
    {
        if (vb != VOL)
            throw Exception("CovDerivative: only implemented for vb=VOL yet.");

        // scalar field
        if (c1->Dimensions().Size() == 0)
        {
            return OneFormCF(GradCF(c1, dim));
        }

        // vector field
        if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
        {
            return TensorFieldCF(GradCF(vf->GetCoefficients(), dim) + EinsumCF("ikj,k->ij", {chr2, vf->GetCoefficients()}), "10");
        }

        // one-form field
        if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
        {
            return TensorFieldCF(GradCF(of->GetCoefficients(), dim) - EinsumCF("ijk,k->ij", {chr2, of->GetCoefficients()}), "11");
        }

        // General tensor field
        string signature = c1->GetSignature();
        string tmp_signature = c1->GetSignature();
        string cov_ind = c1->GetCovariantIndices();
        char new_char = SIGNATURE[signature.size()];

        auto result = GradCF(c1->GetCoefficients(), dim);
        for (size_t i = 0; i < signature.size(); i++)
        {
            tmp_signature = c1->GetSignature();
            tmp_signature[i] = SIGNATURE[signature.size() + 1];
            if (cov_ind[i] == '1')
            {
                // covariant
                string einsum_signature = ToString(new_char) + signature[i] + tmp_signature[i] + "," + tmp_signature + "->" + new_char + signature;
                result = result - EinsumCF(einsum_signature, {chr2, c1->GetCoefficients()});
            }
            else
            {
                // contravariant
                string einsum_signature = ToString(new_char) + tmp_signature[i] + signature[i] + "," + tmp_signature + "->" + new_char + signature;
                result = result + EinsumCF(einsum_signature, {chr2, c1->GetCoefficients()});
            }
        }

        return TensorFieldCF(result, "1" + cov_ind);
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

        return this->Trace(this->CovDerivative(c1), 0, 1, vb);
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
            else
                throw Exception("CovCurl: only available for vector fields and 1-forms yet");
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
        return TensorFieldCF(CovLaplace(c1) - 2 * EinsumCF("ikjl,lk->ij", {GetRiemannCurvatureTensor(), g_inv * c1 * g_inv}) - GetRicciTensor() * g_inv * c1 - TransposeCF(GetRicciTensor() * g_inv * c1), "11");
    }

    shared_ptr<TensorFieldCoefficientFunction> RiemannianManifold::Trace(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2, VorB vb) const
    {
        if (index1 == index2)
            throw Exception("Trace: indices must be different");
        if (tf->Dimensions().Size() <= max(index1, index2))
            throw Exception("Trace: index out of range");

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
            result = EinsumCF(eins, {tf});
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
            result = EinsumCF(eins, {tf, cov1 ? metric_inv : metric});
        }

        return mout.rank ? TensorFieldCF(result, mout.CovString())
                         : ScalarFieldCF(result, dim);
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
            out_cf = EinsumCF(eins, {tf, vf});
        }
        else
        {
            char b = m.FreshLabel();
            std::string eins = sig + "," + std::string(1, a) + std::string(1, b) + "," + std::string(1, b) + "->" + sig_out;
            out_cf = EinsumCF(eins, {tf, g, vf});
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
        return TensorFieldCF(EinsumCF(signature + "->" + signature_result, {tf}), cov_ind);
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
            return TensorFieldCF(tf - this->Trace(tf, 0, 1, VOL) * g, "11");
        case BND:
            return TensorFieldCF(P_F * tf * P_F - this->Trace(tf, 0, 1, BND) * g_F, "11");
        default:
            throw Exception("S_op: Only implemented for VOL and BND");
        }
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
            return TensorFieldCF(tf - 0.5 * this->Trace(tf, 0, 1, VOL) * g, "11");
        case BND:
            return TensorFieldCF(P_F * tf * P_F - 0.5 * this->Trace(tf, 0, 1, BND) * g_F, "11");
        default:
            throw Exception("J_op: Only implemented for VOL and BND");
        }
    }

}

void ExportRiemannianManifold(py::module m)
{
    using namespace ngfem;

    py::class_<RiemannianManifold, shared_ptr<RiemannianManifold>>(m, "RiemannianManifold")
        .def(py::init<shared_ptr<CoefficientFunction>>(), "constructor", py::arg("metric"))
        .def("VolumeForm", &RiemannianManifold::GetVolumeForm, "return the volume form of given dimension", py::arg("vb"))
        .def_property_readonly("dim", &RiemannianManifold::GetDimension, "return the manifold dimension")
        .def_property_readonly("G", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetric(); }, "return the metric tensor")
        .def_property_readonly("G_inv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricInverse(); }, "return the inverse of the metric tensor")
        .def_property_readonly("normal", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetNV(); }, "return the normal vector")
        .def_property_readonly("tangent", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetEdgeTangent(); }, "return the tangent vector")
        .def_property_readonly("G_deriv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricDerivative(); }, "return the derivative of the metric tensor")
        .def("Christoffel", [](shared_ptr<RiemannianManifold> self, bool second_kind)
             { return self->GetChristoffelSymbol(second_kind); }, "return the Christoffel symbol of the first or second kind", py::arg("second_kind") = false)
        .def("LeviCivitaSymbol", [](shared_ptr<RiemannianManifold> self, bool covariant)
             { return self->GetLeviCivitaSymbol(covariant); }, "return the Levi-Civita symbol", py::arg("covariant") = false)
        .def("Raise", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, size_t index)
             { return self->Raise(tf, index); }, "Raise a tensor index using the manifold metric", py::arg("tf"), py::arg("index") = 0)
        .def("Lower", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, size_t index)
             { return self->Lower(tf, index); }, "Lower a tensor index using the manifold metric", py::arg("tf"), py::arg("index") = 0)
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
        .def("delta", [](shared_ptr<RiemannianManifold> self, shared_ptr<KFormCoefficientFunction> a)
             { return self->Coderivative(a); }, "Exterior coderivative of a k-form using the manifold metric", py::arg("a"))
        .def("InnerProduct", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf1, shared_ptr<TensorFieldCoefficientFunction> tf2, VorB vb)
             { return self->IP(tf1, tf2, vb); }, "InnerProduct of two TensorFields", py::arg("tf1"), py::arg("tf2"), py::arg("vb") = VOL)
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
        .def("Trace", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb, size_t index1, size_t index2)
             { return self->Trace(tf, index1, index2, vb); }, "Trace of TensorField in two indices. Default are the first two.", py::arg("tf"), py::arg("vb") = VOL, py::arg("index1") = 0, py::arg("index2") = 1)
        .def("Contraction", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot)
             { return self->Contraction(tf, vf, slot); }, "Contraction of TensorField with a VectorField at given slot. Default slot is the first.", py::arg("tf"), py::arg("vf"), py::arg("slot") = 0)
        .def("Transpose", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2)
             { return self->Transpose(tf, index1, index2); }, "Transpose of TensorField for given indices. Default indices are first and second.", py::arg("tf"), py::arg("index1") = 0, py::arg("index2") = 1)
        .def("S", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb)
             { return self->S_op(tf, vb); }, "S operator subtracting the trace.", py::arg("tf"), py::arg("vb") = VOL)
        .def("J", [](shared_ptr<RiemannianManifold> self, shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb)
             { return self->J_op(tf, vb); }, "J operator subtracting half the trace.", py::arg("tf"), py::arg("vb") = VOL);
}
