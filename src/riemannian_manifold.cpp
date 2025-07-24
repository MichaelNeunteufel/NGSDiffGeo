#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"

#include <coefficient_stdmath.hpp>
#include <python_comp.hpp>
// #include <fem.hpp>
#include <integratorcf.hpp>
#include <hcurlcurlfespace.hpp>

namespace ngfem
{
    using namespace std;
    RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g)
        : is_regge(false), is_proxy(false), regge_proxy(nullptr), regge_space(nullptr), g(_g)
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
            regge_proxy = dynamic_pointer_cast<ProxyFunction>(g);
            regge_space = dynamic_pointer_cast<ngcomp::HCurlCurlFESpace>(regge_proxy->GetFESpace());
            is_proxy = true;
            is_regge = bool(regge_space); //"HCurlCurlFESpace" == regge_space->GetClassName();
        }
        else if (auto gf = dynamic_pointer_cast<ngcomp::GridFunction>(g))
        {
            regge_space = dynamic_pointer_cast<ngcomp::HCurlCurlFESpace>(gf->GetFESpace());
            is_regge = bool(regge_space);
        }

        dim = g->Dimensions()[0];

        if (dim != 2 && dim != 3)
            throw Exception("In RMF: only 2D and 3D manifolds are supported");

        g_inv = InverseCF(g);

        // Volume forms on VOL, BND, BBND, and BBBND
        vol[VOL] = sqrt(DeterminantCF(g));
        auto one_cf = make_shared<ConstantCoefficientFunction>(1.0);
        tv = TangentialVectorCF(dim, false);
        nv = NormalVectorCF(dim);
        auto nv_mat = nv->Reshape(Array<int>({dim, 1}));
        P_n = nv_mat * TransposeCF(nv_mat);
        P_F = IdentityCF(dim) - P_n;

        vol[BND] = one_cf;
        vol[BBND] = one_cf;
        vol[BBBND] = one_cf;
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
            g_E_inv = 1/InnerProduct(g * tv, tv) * P_E;
        }

        g_nv = VectorFieldCF(vol[VOL] / vol[BND] * g_inv * nv);

        if (is_regge)
        {
            if (is_proxy)
            {
                auto g_proxy = dynamic_pointer_cast<ProxyFunction>(g);
                g_deriv = g_proxy->GetAdditionalProxy("grad");
                chr1 = g_proxy->GetAdditionalProxy("christoffel");
                chr2 = g_proxy->GetAdditionalProxy("christoffel2");
                Riemann = g_proxy->GetAdditionalProxy("Riemann");
                Curvature = g_proxy->GetAdditionalProxy("curvature");
                Ricci = g_proxy->GetAdditionalProxy("Ricci");
                Einstein = g_proxy->GetAdditionalProxy("Einstein");
                Scalar = g_proxy->GetAdditionalProxy("scalar");
                SFF = EinsumCF("ijk,k->ij", {chr1->Reshape(Array<int>({dim, dim, dim})),g_nv});
            }
            else
            {
                auto diffop = regge_space->GetAdditionalEvaluators()["grad"];
                shared_ptr<ngcomp::GridFunction> gf = dynamic_pointer_cast<ngcomp::GridFunction>(g);

                g_deriv = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                g_deriv->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["christoffel"];
                chr1 = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                chr1->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["christoffel2"];
                chr2 = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                chr2->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["Riemann"];
                Riemann = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                Riemann->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["curvature"];
                Curvature = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                Curvature->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["Ricci"];
                Ricci = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                Ricci->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["Einstein"];
                Einstein = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                Einstein->SetDimensions(diffop->Dimensions());

                diffop = regge_space->GetAdditionalEvaluators()["scalar"];
                Scalar = make_shared<ngcomp::GridFunctionCoefficientFunction>(gf, diffop);
                Scalar->SetDimensions(diffop->Dimensions());

                SFF = EinsumCF("ijk,k->ij", {chr1->Reshape(Array<int>({dim, dim, dim})),g_nv});
            }
        }
        else
        {
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

            auto lin_part = EinsumCF("ijkl->kilj", {GradCF(chr1, dim)}) - EinsumCF("ijkl->likj", {GradCF(chr1, dim)});
            auto non_lin_part = EinsumCF("ijm,klm->iklj", {chr1, chr2}) - EinsumCF("ijm,klm->ilkj", {chr2, chr1});
            Riemann = TensorFieldCF(lin_part + non_lin_part, "1111");
            auto LeviCivita = GetLeviCivitaSymbol(false);
            string signature = SIGNATURE.substr(0, 4) + "," + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(0, 2) + "," + SIGNATURE.substr(2 + dim, dim - 2) + SIGNATURE.substr(2, 2) + "->" + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(2 + dim, dim - 2);
            Curvature = TensorFieldCF(1 / 4 * EinsumCF(signature, {Riemann, LeviCivita, LeviCivita}), "00");
            Ricci = Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(Riemann), 0, 2);
            Scalar = Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(Ricci), 0, 1);
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

    shared_ptr<CoefficientFunction> RiemannianManifold::Raise(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index) const
    {
        if (c1->Dimensions().Size() <= index)
            throw Exception(ToString("Raise: Dimension of c1 = ") + ToString(c1->Dimensions().Size()) + "<= index = " + ToString(index));
        if (c1->GetCovariantIndices()[index] == '0')
            throw Exception(ToString("Raise: TensorField at index= ") + ToString(index) + " is already contravariant");

        string einsum_signature = c1->GetSignature();
        char new_char = SIGNATURE[einsum_signature.size()];
        einsum_signature[index] = new_char;

        einsum_signature += "," + ToString(c1->GetSignature()[index]) + ToString(new_char) + "->" + c1->GetSignature();
        string cov_ind = c1->GetCovariantIndices();
        cov_ind[index] = '0';
        // cout << "Raise: signature = " + einsum_signature + " cov ind = " + cov_ind << endl;
        return TensorFieldCF(EinsumCF(einsum_signature, {c1, g_inv}), cov_ind);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Lower(shared_ptr<TensorFieldCoefficientFunction> c1, size_t index) const
    {
        if (c1->Dimensions().Size() <= index)
            throw Exception(ToString("Lower: Dimension of c1 = ") + ToString(c1->Dimensions().Size()) + "<= index = " + ToString(index));
        if (c1->GetCovariantIndices()[index] == '1')
            throw Exception(ToString("Lower: TensorField at index= ") + ToString(index) + " is already covariant");

        string einsum_signature = c1->GetSignature();
        char new_char = SIGNATURE[einsum_signature.size()];
        einsum_signature[index] = new_char;

        einsum_signature += "," + ToString(c1->GetSignature()[index]) + ToString(new_char) + "->" + c1->GetSignature();
        string cov_ind = c1->GetCovariantIndices();
        cov_ind[index] = '1';
        // cout << "Lower: signature = " + einsum_signature + " cov ind = " + cov_ind << endl;
        return TensorFieldCF(EinsumCF(einsum_signature, {c1, g}), cov_ind);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetLeviCivitaSymbol(bool covariant) const
    {
        auto levi_civita_symbol = LeviCivitaCF(dim);

        return covariant ? TensorFieldCF(GetVolumeForm(VOL) * levi_civita_symbol, string(dim, '1')) : TensorFieldCF(make_shared<ConstantCoefficientFunction>(1.0) / GetVolumeForm(VOL) * levi_civita_symbol, string(dim, '0'));
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricDerivative() const
    {
        return g_deriv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetChristoffelSymbol(bool second_kind) const
    {
        return second_kind ? chr2 : chr1;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetRiemannCurvatureTensor() const
    {
        return TensorFieldCF(Riemann,"1111");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetCurvatureOperator() const
    {
        return TensorFieldCF(Curvature,"00");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetRicciTensor() const
    {
        return TensorFieldCF(Ricci,"11");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetEinsteinTensor() const
    {
        return TensorFieldCF(Einstein, "11");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetScalarCurvature() const
    {
        return Scalar;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetGaussCurvature() const
    {
        if (dim != 2)
            throw Exception("In RMF: Gauss curvature only available in 2D");
        return ScalarFieldCF(1/DeterminantCF(g)*Curvature);
    }


    shared_ptr<CoefficientFunction> RiemannianManifold::GetSecondFundamentalForm() const
    {
        return TensorFieldCF(SFF, "11");
    }
    shared_ptr<CoefficientFunction> RiemannianManifold::GetGeodesicCurvature() const
    {
        if (dim != 2)
            throw Exception("In RMF: Geodesic curvature only available in 2D");
        return ScalarFieldCF(InnerProduct(SFF*g_tv,g_tv));
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMeanCurvature() const
    {
        return this->Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(GetSecondFundamentalForm()), 0, 1, BND);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetNV() const
    {
        return g_nv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetEdgeTangent() const
    {
        return g_tv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2, VorB vb) const
    {
        string cov_ind1 = c1->GetCovariantIndices();
        string cov_ind2 = c2->GetCovariantIndices();
        // cout << "cov_indices = " << cov_ind1 << ", " << cov_ind2 << endl;

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
        for (int i = 0; i < cov_ind1.size(); i++)
        {
            same_index[i] = cov_ind1[i] == cov_ind2[i];
            if (same_index[i])
                position_same_index.Append(i);
        }
        // cout << "same_index = " << same_index << endl;

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
        // cout << signature_c1 + "," + signature_c2 + raise_lower_signatures << endl;
        return EinsumCF(signature_c1 + "," + signature_c2 + raise_lower_signatures, cfs);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const
    {
        string cov_ind1 = c1->GetCovariantIndices();
        string cov_ind2 = c2->GetCovariantIndices();
        if (cov_ind1.size() != 1 || cov_ind2.size() != 1)
            throw Exception("Cross: only available for vector fields and 1-forms yet.");
        if (c1->Dimensions()[0] != 3)
        {
            // cout << "dim = " << c1->Dimensions()[0] << endl;
            throw Exception("Cross: only available for 3D yet.");
        }

        if (cov_ind1 == cov_ind2)
        {
            if (cov_ind1[0] == '1')
            {
                // both 1-forms
                // cout << "A" << endl;
                return VectorFieldCF(EinsumCF("ijk,j,k->i", {GetLeviCivitaSymbol(false), c1, c2}));
            }
            else
            {
                // both vector-fields
                // cout << "B" << endl;
                return VectorFieldCF(EinsumCF("ai,ijk,j,k->a", {g_inv, GetLeviCivitaSymbol(true), c1, c2}));
            }
        }
        if (cov_ind1[0] == '1')
        {
            // c1 1-form, c2 vector field
            // cout << "C" << endl;
            return VectorFieldCF(EinsumCF("ijk,j,kl,l->i", {GetLeviCivitaSymbol(false), c1, g, c2}));
        }
        else
        {
            // c1 vector field, c2 1-form
            // cout << "D" << endl;
            return VectorFieldCF(EinsumCF("ijk,jl,l,k->i", {g_inv, GetLeviCivitaSymbol(false), g, c1, c2}));
        }
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb) const
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
        // cout << "signature = " << signature << ", cov_ind = " << cov_ind << endl;
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
                // cout << "cov einsum_signature = " << einsum_signature << endl;
                result = result - EinsumCF(einsum_signature, {chr2, c1->GetCoefficients()});
            }
            else
            {
                // contravariant
                string einsum_signature = ToString(new_char) + tmp_signature[i] + signature[i] + "," + tmp_signature + "->" + new_char + signature;
                // cout << "con einsum_signature = " << einsum_signature << endl;
                result = result + EinsumCF(einsum_signature, {chr2, c1->GetCoefficients()});
            }
        }

        return TensorFieldCF(result, "1" + cov_ind);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovHessian(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (auto sf = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(c1))
            return CovDerivative(dynamic_pointer_cast<OneFormCoefficientFunction>(OneFormCF(GradCF(sf, dim))));
        else
            return CovDerivative(dynamic_pointer_cast<TensorFieldCoefficientFunction>(CovDerivative(c1)));
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1, VorB vb) const
    {
        // cout << "CovDivergence" << endl;
        if (c1->Dimensions().Size() == 0)
            throw Exception("CovDivergence: TensorField must have at least one index");

        return this->Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(this->CovDerivative(c1)), 0, 1, vb);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovCurl(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (c1->Dimensions().Size() == 0)
            throw Exception("CovCurl: called with scalar field");
        // if (c1->Dimensions()[0] != 3)
        //     throw Exception("CovCurl: only available in 3D yet");

        if (dim == 3)
        {
            if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
            {
                // cout << "of" << endl;
                return VectorFieldCF(EinsumCF("ijk,jk->i", {GetLeviCivitaSymbol(false), GradCF(c1, dim)}));
            }
            else if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
            {
                // cout << "vf" << endl;
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
                // cout << "of" << endl;
                return ScalarFieldCF(EinsumCF("ij,ij->", {GetLeviCivitaSymbol(false), GradCF(c1, dim)}));
            }
            else if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
            {
                // cout << "vf" << endl;
                return ScalarFieldCF(EinsumCF("ij,ij->", {GetLeviCivitaSymbol(false), GradCF(Lower(c1), dim)}));
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

    shared_ptr<CoefficientFunction> RiemannianManifold::CovInc(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (c1->Dimensions().Size() < 2)
            throw Exception("CovInc: called with scalar, vector, or 1-form field");

        shared_ptr<CoefficientFunction> cov_hesse = CovHessian(c1);
        shared_ptr<CoefficientFunction> p_cov_hesse = make_shared<ConstantCoefficientFunction>(0.25)*(cov_hesse-EinsumCF("ijkl->kjil",{cov_hesse})-EinsumCF("ijkl->ilkj",{cov_hesse})+EinsumCF("ijkl->klij",{cov_hesse}));
        
        return TensorFieldCF(-EinsumCF("ijkl->ikjl",{p_cov_hesse}),"1111");
        
        // if (dim == 3)
        // {
        //     return CovCurl(dynamic_pointer_cast<TensorFieldCoefficientFunction>(Transpose(dynamic_pointer_cast<TensorFieldCoefficientFunction>(c1))));
        // }
        // else if (dim == 2)
        // {
        //     return CovCurl(dynamic_pointer_cast<TensorFieldCoefficientFunction>(CovCurl(c1)));
        // }
        // else
        //     throw Exception("CovInc: not implemented for dim = " + ToString(dim) + " yet");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovEin(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        auto J_c1 = J_op(c1); 
        auto lap_term = CovLaplace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(J_c1));
        auto def_term = CovDef(dynamic_pointer_cast<TensorFieldCoefficientFunction>(CovDivergence(dynamic_pointer_cast<TensorFieldCoefficientFunction>(J_c1))));

        return TensorFieldCF( J_op(dynamic_pointer_cast<TensorFieldCoefficientFunction>(def_term))-0.5*lap_term, "11");
        // auto der_J_c1 = CovDerivative(dynamic_pointer_cast<TensorFieldCoefficientFunction>(J_c1));
        // auto term0 = Transpose(dynamic_pointer_cast<TensorFieldCoefficientFunction>(der_J_c1),0,1);
        // auto term1 =  0.5*J_op(dynamic_pointer_cast<TensorFieldCoefficientFunction>(TensorFieldCF(der_J_c1+term0,"11")));
        // auto term2 =  - 0.5*CovDivergence(dynamic_pointer_cast<TensorFieldCoefficientFunction>(CovDerivative(dynamic_pointer_cast<TensorFieldCoefficientFunction>(J_c1))));
        // return TensorFieldCF(term1+term2, "11");

    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovLaplace(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {

        return CovDivergence(dynamic_pointer_cast<TensorFieldCoefficientFunction>(CovDerivative(c1)));
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovDef(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        if (c1->GetCovariantIndices().size() != 1 || c1->GetCovariantIndices()[0] != '1' )
            throw Exception("CovDef: Only implemented for 1-forms");
        auto cov_der = CovDerivative(c1);
        return TensorFieldCF(0.5*(Transpose(dynamic_pointer_cast<TensorFieldCoefficientFunction>(cov_der),0,1)+cov_der), "11");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovRot(shared_ptr<TensorFieldCoefficientFunction> c1) const
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
            return CovRot(dynamic_pointer_cast<VectorFieldCoefficientFunction>(VectorFieldCF(Raise(of))));
        }
        else
            throw Exception("CovRot: only available for scalar or vector fields. Invoked with signature " + c1->GetSignature() + " and covariant indices " + c1->GetCovariantIndices());
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::LichnerowiczLaplacian(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        return TensorFieldCF(CovLaplace(c1) -2*EinsumCF("ikjl,lk->ij", {GetRiemannCurvatureTensor(), g_inv*c1*g_inv}) - GetRicciTensor()*g_inv*c1-TransposeCF(GetRicciTensor()*g_inv*c1), "11");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Trace(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2, VorB vb) const
    {
        // cout << "Trace" << endl;
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

        if (index1 > index2)
            swap(index1, index2);

        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();

        // cout << "cov ind = " << cov_ind << ", signature " << signature << endl;

        string cov_ind_result = cov_ind;
        cov_ind_result = cov_ind_result.size() > 2 ? cov_ind_result.erase(index2, 1).erase(index1, 1) : "";

        shared_ptr<CoefficientFunction> result;

        if (cov_ind[index1] != cov_ind[index2])
        {
            signature[index2] = signature[index1];
            string signature_result = signature;
            signature_result = signature_result.size() > 2 ? signature_result.erase(index2, 1).erase(index1, 1) : "";
            // return TensorFieldCF(EinsumCF(signature + "->" + signature_result, {tf}), cov_ind_result);
            result = EinsumCF(signature + "->" + signature_result, {tf});
        }
        else
        {
            signature[index2] = SIGNATURE[signature.size()];
            string raise_lower_signature = ToString(signature[index2]) + signature[index1];
            string signature_result = signature;
            signature_result = signature_result.size() > 2 ? signature_result.erase(index2, 1).erase(index1, 1) : "";

            if (cov_ind[index1] == '1')
                // return TensorFieldCF(EinsumCF(signature + "," + raise_lower_signature + "->" + signature_result, {tf, g_inv}), cov_ind_result);
                result = EinsumCF(signature + "," + raise_lower_signature + "->" + signature_result, {tf, metric_inv});
            else
                // return TensorFieldCF(EinsumCF(signature + "," + raise_lower_signature + "->" + signature_result, {tf, g}), cov_ind_result);
                result = EinsumCF(signature + "," + raise_lower_signature + "->" + signature_result, {tf, metric});
        }

        if (cov_ind_result.size())
            return TensorFieldCF(result, cov_ind_result);
        else
            return ScalarFieldCF(result);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot) const
    {
        // cout << "Contraction, slot = " << slot << endl;
        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();

        if (slot >= signature.size())
            throw Exception("Contraction: slot out of range");

        // TODO: Check if after contraction it is a tensorfield or a scalarfield
        shared_ptr<CoefficientFunction> result;

        if (cov_ind[slot] == '1')
        {
            string einsum_signature = signature + "," + signature[slot] + "->" + signature.erase(slot, 1);
            // cout << einsum_signature << endl;
            // return TensorFieldCF(EinsumCF(einsum_signature, {tf, vf}), cov_ind.erase(slot, 1));
            result = EinsumCF(einsum_signature, {tf, vf});
        }
        else
        {
            char new_char = SIGNATURE[signature.size()];
            string einsum_signature = signature + "," + signature[slot] + new_char + "," + new_char + "->" + signature.erase(slot, 1);
            // cout << einsum_signature << endl;
            // return TensorFieldCF(EinsumCF(einsum_signature, {tf, g, vf}), cov_ind.erase(slot, 1));
            result = EinsumCF(einsum_signature, {tf, g, vf});
        }

        if (cov_ind.size() > 1)
            return TensorFieldCF(result, cov_ind.erase(slot, 1));
        else
            return ScalarFieldCF(result);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Transpose(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2) const
    {
        // cout << "Transpose" << endl;
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


    shared_ptr<CoefficientFunction> RiemannianManifold::S_op(shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb) const
    {
        if (tf->Dimensions().Size() !=2 && tf->Dimensions()[0] != tf->Dimensions()[1])
            throw Exception("S_op: only available for 2-tensors");
        if (tf->GetCovariantIndices() != "11")
            throw Exception("S_op: currently only implemented for (2,0)-tensors!");
        switch(vb)
        {
            case VOL:
                return TensorFieldCF(tf - this->Trace(tf, 0, 1, VOL)*g,"11");
            case BND:
                return TensorFieldCF(P_F*tf*P_F - this->Trace(tf, 0, 1, BND)*g_F, "11");
            default:
                throw Exception("S_op: Only implemented for VOL and BND");
        }
        
    }
    
    shared_ptr<CoefficientFunction> RiemannianManifold::J_op(shared_ptr<TensorFieldCoefficientFunction> tf, VorB vb) const
    {
        if (tf->Dimensions().Size() !=2 && tf->Dimensions()[0] != tf->Dimensions()[1])
            throw Exception("J_op: only available for 2-tensors");
        if (tf->GetCovariantIndices() != "11")
            throw Exception("J_op: currently only implemented for (2,0)-tensors!");
        switch(vb)
        {
            case VOL:
                return TensorFieldCF(tf - 0.5*this->Trace(tf, 0, 1, VOL)*g, "11");
            case BND:
                return TensorFieldCF(P_F*tf*P_F - 0.5*this->Trace(tf, 0, 1, BND)*g_F, "11");
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
        //.def("GetMetric", &RiemannianManifold::GetMetric, "return the metric")
        .def("VolumeForm", &RiemannianManifold::GetVolumeForm, "return the volume form of given dimension", py::arg("vb"))
        // .def_property_readonly("dx", [](shared_ptr<RiemannianManifold> self)
        //                        { return self->GetVolumeForm(VOL); }, "return the volume form")
        // .def_property_readonly("ds", [](shared_ptr<RiemannianManifold> self)
        //                        { return self->GetVolumeForm(BND); }, "return the boundary (co-dimension 1) volume form")
        // .def_property_readonly("dl", [](shared_ptr<RiemannianManifold> self)
        //                        { return self->GetVolumeForm(BBND); }, "return the edge (co-dimension 2) volume form")
        // .def("dx", [](shared_ptr<RiemannianManifold> self, optional<variant<ngcomp::Region, string>> definedon, bool element_boundary, VorB element_vb, bool skeleton, int bonus_intorder, std::map<ELEMENT_TYPE, IntegrationRule> intrules, shared_ptr<ngcomp::GridFunction> deformation, shared_ptr<BitArray> definedonelements)
        //      {
        //    if (element_boundary) element_vb = BND;
        //    auto dx = DifferentialSymbol(VOL, element_vb, skeleton, /* defon, */ bonus_intorder);
        //    if (definedon)
        //      {
        //        if (auto definedon_region = get_if<ngcomp::Region>(&*definedon); definedon_region)
        //          {
        //            dx.definedon = definedon_region->Mask();
        //            if (VorB(*definedon_region) != VOL)
        //             throw Exception("dx: definedon must be a volume region");
        //            dx.vb = VorB(*definedon_region);
        //          }
        //        if (auto definedon_string = get_if<string>(&*definedon); definedon_string)
        //          dx.definedon = *definedon_string;
        //      }
        //    dx.deformation = deformation;
        //    dx.definedonelements = definedonelements;
        //    for (auto both : intrules)
        //      dx.userdefined_intrules[both.first] =
        //        make_shared<IntegrationRule> (both.second.Copy());
        //    return self->GetVolumeForm(VOL)*dx; }, py::arg("definedon") = nullptr, py::arg("element_boundary") = false, py::arg("element_vb") = VOL, py::arg("skeleton") = false, py::arg("bonus_intorder") = 0, py::arg("intrules") = std::map<ELEMENT_TYPE, IntegrationRule>{}, py::arg("deformation") = nullptr, py::arg("definedonelements") = nullptr)
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
        .def_property_readonly("Riemann", &RiemannianManifold::GetRiemannCurvatureTensor, "return the Riemann curvature tensor")
        .def_property_readonly("Curvature", &RiemannianManifold::GetCurvatureOperator, "return the curvature operator")
        .def_property_readonly("Gauss", &RiemannianManifold::GetGaussCurvature, "return the Gauss curvature in 2D")
        .def_property_readonly("Ricci", &RiemannianManifold::GetRicciTensor, "return the Ricci tensor")
        .def_property_readonly("Einstein", &RiemannianManifold::GetEinsteinTensor, "return the Einstein tensor")
        .def_property_readonly("Scalar", &RiemannianManifold::GetScalarCurvature, "return the scalar curvature")
        .def_property_readonly("SFF", &RiemannianManifold::GetSecondFundamentalForm, "return the second fundamental form")
        .def_property_readonly("GeodesicCurvature", &RiemannianManifold::GetGeodesicCurvature, "return the geodesic curvature")
        .def_property_readonly("MeanCurvature", &RiemannianManifold::GetMeanCurvature, "return the mean curvature")
        .def("InnerProduct", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf1, shared_ptr<CoefficientFunction> tf2, VorB vb)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf1) || !dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf2))
            throw Exception("InnerProduct: input must be a TensorFieldCoefficientFunction");
        return self->IP(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf1), dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf2), vb); }, "InnerProduct of two TensorFields", py::arg("tf1"), py::arg("tf2"), py::arg("vb") = VOL)
        .def("Cross", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf1, shared_ptr<CoefficientFunction> tf2)
             {
            if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf1) || !dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf2))
            throw Exception("InnerProduct: input must be a TensorFieldCoefficientFunction");
            return self->Cross(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf1), dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf2)); }, "Cross product in 3D of two vector fields, 1-forms, or both mixed. Returns the resulting vector-field.", py::arg("tf1"), py::arg("tf2"))
        .def("CovDeriv", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, VorB vb)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovDeriv: input must be a TensorFieldCoefficientFunction");
        return self->CovDerivative(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf), vb); }, "Covariant derivative of a TensorField", py::arg("tf"), py::arg("vb")=VOL)
        .def("CovHesse", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovHesse: input must be a TensorFieldCoefficientFunction");
        return self->CovHessian(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant Hessian of a TensorField.", py::arg("tf"))
        .def("CovCurl", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovCurl: input must be a TensorFieldCoefficientFunction");
        return self->CovCurl(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant curl of a TensorField in 3D", py::arg("tf"))
        .def("CovInc", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovInc: input must be a TensorFieldCoefficientFunction");
        return self->CovInc(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant inc of a TensorField in 2D or 3D", py::arg("tf"))
        .def("CovEin", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovEin: input must be a TensorFieldCoefficientFunction");
        return self->CovEin(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant ein of a TensorField in 2D or 3D", py::arg("tf"))
        .def("CovLaplace", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovLaplace: input must be a TensorFieldCoefficientFunction");
        return self->CovLaplace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant Laplace of a TensorField ", py::arg("tf"))
        .def("LichnerowiczLaplacian", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("LichnerowiczLaplacian: input must be a TensorFieldCoefficientFunction");
        return self->LichnerowiczLaplacian(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Lichnerowicz Laplacian of a TensorField", py::arg("tf"))
        .def("CovDef", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovDef: input must be a TensorFieldCoefficientFunction");
        return self->CovDef(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant Def (symmetric derivative) of a 1-form", py::arg("tf"))
        .def("CovRot", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovRot: input must be a TensorFieldCoefficientFunction");
        return self->CovRot(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant rot of a TensorField of maximal order 1 in 2D. Returns a contravariant tensor field.", py::arg("tf"))
        .def("CovDiv", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, VorB vb)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovDiv: input must be a TensorFieldCoefficientFunction");
        return self->CovDivergence(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf), vb); }, "Covariant divergence of a TensorField", py::arg("tf"), py::arg("vb")=VOL)
        .def("Trace", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, VorB vb, size_t index1, size_t index2)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("Trace: input must be a TensorFieldCoefficientFunction");
        return self->Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),index1,index2,vb); }, "Trace of TensorField in two indices. Default are the first two.", py::arg("tf"), py::arg("vb") = VOL, py::arg("index1") = 0, py::arg("index2") = 1)
        .def("Contraction", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, shared_ptr<CoefficientFunction> vf, size_t slot)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf) || !dynamic_pointer_cast<VectorFieldCoefficientFunction>(vf))
            throw Exception("Contraction: input must be a TensorFieldCoefficientFunction and a VectorField");
        return self->Contraction(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),dynamic_pointer_cast<VectorFieldCoefficientFunction>(vf),slot); }, "Contraction of TensorField with a VectorField at given slot. Default slot is the first.", py::arg("tf"), py::arg("vf"), py::arg("slot") = 0)
        .def("Transpose", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, size_t index1, size_t index2)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("Transpose: input must be a TensorFieldCoefficientFunction");
        return self->Transpose(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),index1,index2); }, "Transpose of TensorField for given indices. Default indices are first and second.", py::arg("tf"), py::arg("index1") = 0, py::arg("index2") = 1)
        .def("S", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, VorB vb)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("Transpose: input must be a TensorFieldCoefficientFunction");
        return self->S_op(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),vb); }, "S operator subtracting the trace.", py::arg("tf"), py::arg("vb") = VOL)
        .def("J", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, VorB vb)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("Transpose: input must be a TensorFieldCoefficientFunction");
        return self->J_op(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),vb); }, "J operator subtracting half the trace.", py::arg("tf"), py::arg("vb") = VOL);
}
