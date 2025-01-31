#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"

#include <coefficient_stdmath.hpp>
#include <python_comp.hpp>
#include <fem.hpp>
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

        vol[BND] = one_cf;
        vol[BBND] = one_cf;
        vol[BBBND] = one_cf;
        if (dim == 2)
        {
            vol[BND] = sqrt(InnerProduct(g * tv, tv));
            g_tv = VectorFieldCF(1 / vol[BND] * tv);
        }
        else if (dim == 3)
        {
            vol[BND] = sqrt(InnerProduct(CofactorCF(g) * nv, nv));
            vol[BBND] = sqrt(InnerProduct(g * tv, tv));

            g_tv = VectorFieldCF(1 / vol[BBND] * tv);
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
            auto non_lin_part = EinsumCF("ijm,mkl->iklj", {chr1, chr2}) - EinsumCF("ijm,mkl->ilkj", {chr2, chr1});
            Riemann = TensorFieldCF(lin_part + non_lin_part, "1111");
            auto LeviCivita = GetLeviCivitaSymbol(false);
            string signature = SIGNATURE.substr(0, 4) + "," + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(0, 2) + "," + SIGNATURE.substr(2 + dim, dim - 2) + SIGNATURE.substr(2, 2) + "->" + SIGNATURE.substr(4, dim - 2) + SIGNATURE.substr(2 + dim, dim - 2);
            Curvature = TensorFieldCF(1 / 4 * EinsumCF(signature, {Riemann, LeviCivita, LeviCivita}), "00");
            Ricci = Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(Riemann), 0, 2);
            Scalar = Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(Ricci), 0, 1);
            Einstein = Ricci - 0.5 * Scalar * g;
        }
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetric() const
    {
        return g;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetricInverse() const
    {
        return g_inv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetVolumeForm(VorB vb) const
    {
        return vol[int(vb)];
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
        return Riemann;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetCurvatureOperator() const
    {
        return Curvature;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetRicciTensor() const
    {
        return Ricci;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetEinsteinTensor() const
    {
        return Einstein;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetScalarCurvature() const
    {
        return Scalar;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetNV() const
    {
        return g_nv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetEdgeTangent(bool consistent) const
    {
        return g_tv;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::IP(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const
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
            cfs[2 + i] = cov_ind1[position_same_index[i]] == '1' ? g_inv : g;
        }
        // cout << signature_c1 + "," + signature_c2 + raise_lower_signatures << endl;
        return EinsumCF(signature_c1 + "," + signature_c2 + raise_lower_signatures, cfs);
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Cross(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2) const
    {
        // only available in 3D (maybe 2D)
        throw Exception("Cross: not implemented yet.");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovDerivative(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
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

    shared_ptr<CoefficientFunction> RiemannianManifold::CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        // cout << "CovDivergence" << endl;
        if (c1->Dimensions().Size() == 0)
            throw Exception("CovDivergence: TensorField must have at least one index");

        // if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
        // {
        //     // Vector field
        //     return EinsumCF("ijj", {CovDerivative(vf)});
        // }
        // else
        // {
        // General case
        return this->Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(this->CovDerivative(c1)), 0, 1);
        //}
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovCurl(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
        throw Exception("CovCurl: not implemented yet");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Trace(shared_ptr<TensorFieldCoefficientFunction> tf, size_t index1, size_t index2) const
    {
        // cout << "Trace" << endl;
        if (index1 == index2)
            throw Exception("Trace: indices must be different");
        if (tf->Dimensions().Size() <= max(index1, index2))
            throw Exception("Trace: index out of range");

        if (index1 > index2)
            swap(index1, index2);

        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();

        // cout << "cov ind = " << cov_ind << ", signature " << signature << endl;

        string cov_ind_result = cov_ind;
        cov_ind_result = cov_ind_result.size() > 2 ? cov_ind_result.erase(index2, 1).erase(index1, 1) : "";

        if (cov_ind[index1] != cov_ind[index2])
        {
            signature[index2] = signature[index1];
            string signature_result = signature;
            signature_result = signature_result.size() > 2 ? signature_result.erase(index2, 1).erase(index1, 1) : "";
            return TensorFieldCF(EinsumCF(signature + "->" + signature_result, {tf}), cov_ind_result);
        }
        else
        {
            signature[index2] = SIGNATURE[signature.size()];
            string raise_lower_signature = ToString(signature[index2]) + signature[index1];
            string signature_result = signature;
            signature_result = signature_result.size() > 2 ? signature_result.erase(index2, 1).erase(index1, 1) : "";

            if (cov_ind[index1] == '1')
                return TensorFieldCF(EinsumCF(signature + "," + raise_lower_signature + "->" + signature_result, {tf, g_inv}), cov_ind_result);
            else
                return TensorFieldCF(EinsumCF(signature + "," + raise_lower_signature + "->" + signature_result, {tf, g}), cov_ind_result);
        }
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot) const
    {
        // cout << "Contraction, slot = " << slot << endl;
        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();

        if (cov_ind[slot] == '1')
        {
            string einsum_signature = signature + "," + signature[slot] + "->" + signature.erase(slot, 1);
            // cout << einsum_signature << endl;
            return TensorFieldCF(EinsumCF(einsum_signature, {tf, vf}), cov_ind.erase(slot, 1));
        }
        else
        {
            char new_char = SIGNATURE[signature.size()];
            string einsum_signature = signature + "," + signature[slot] + new_char + "," + new_char + "->" + signature.erase(slot, 1);
            // cout << einsum_signature << endl;
            return TensorFieldCF(EinsumCF(einsum_signature, {tf, g, vf}), cov_ind.erase(slot, 1));
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
        .def("tangent", [](shared_ptr<RiemannianManifold> self, bool consistent)
             { return self->GetEdgeTangent(consistent); }, "return the tangent vector", py::arg("consistent") = false)
        .def_property_readonly("G_deriv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricDerivative(); }, "return the derivative of the metric tensor")
        .def("Christoffel", [](shared_ptr<RiemannianManifold> self, bool second_kind)
             { return self->GetChristoffelSymbol(second_kind); }, "return the Christoffel symbol of the first or second kind", py::arg("second_kind") = false)
        .def("LeviCivitaSymbol", [](shared_ptr<RiemannianManifold> self, bool covariant)
             { return self->GetLeviCivitaSymbol(covariant); }, "return the Levi-Civita symbol", py::arg("covariant") = false)
        .def("Riemann", &RiemannianManifold::GetRiemannCurvatureTensor, "return the Riemann curvature tensor")
        .def("Curvature", &RiemannianManifold::GetCurvatureOperator, "return the curvature operator")
        .def("Ricci", &RiemannianManifold::GetRicciTensor, "return the Ricci tensor")
        .def("Einstein", &RiemannianManifold::GetEinsteinTensor, "return the Einstein tensor")
        .def("Scalar", &RiemannianManifold::GetScalarCurvature, "return the scalar curvature")
        .def("InnerProduct", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf1, shared_ptr<CoefficientFunction> tf2)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf1) || !dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf2))
            throw Exception("InnerProduct: input must be a TensorFieldCoefficientFunction");
        return self->IP(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf1), dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf2)); }, "InnerProduct of two TensorFields", py::arg("tf1"), py::arg("tf2"))
        .def("CovDeriv", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovDeriv: input must be a TensorFieldCoefficientFunction");
        return self->CovDerivative(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant derivative of a TensorField", py::arg("tf"))
        .def("CovCurl", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovCurl: input must be a TensorFieldCoefficientFunction");
        return self->CovCurl(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant curl of a TensorField in 3D", py::arg("tf"))
        .def("CovDiv", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("CovDiv: input must be a TensorFieldCoefficientFunction");
        return self->CovDivergence(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant divergence of a TensorField", py::arg("tf"))
        .def("Trace", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, size_t index1, size_t index2)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("Trace: input must be a TensorFieldCoefficientFunction");
        return self->Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),index1,index2); }, "Trace of TensorField in two indices. Default are the first two.", py::arg("tf"), py::arg("index1") = 0, py::arg("index2") = 1)
        .def("Contraction", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, shared_ptr<CoefficientFunction> vf, size_t slot)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf) || !dynamic_pointer_cast<VectorFieldCoefficientFunction>(vf))
            throw Exception("Contraction: input must be a TensorFieldCoefficientFunction and a VectorField");
        return self->Contraction(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),dynamic_pointer_cast<VectorFieldCoefficientFunction>(vf),slot); }, "Contraction of TensorField with a VectorField at given slot. Default slot is the first.", py::arg("tf"), py::arg("vf"), py::arg("slot") = 0);
}
