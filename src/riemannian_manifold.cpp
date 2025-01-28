#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"

#include <coefficient_stdmath.hpp>
#include <python_comp.hpp>

namespace ngfem
{
    using namespace std;
    RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g)
        : is_regge(false), g(_g)
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

        // // check if _g itself is a Regge trial function
        if (auto proxy = dynamic_cast<ProxyFunction *>(_g.get()))
        {
            is_regge = "HCurlCurlFESpace" == proxy->GetFESpace()->GetClassName();
        }
        else
            is_regge = false;

        dim = _g->Dimensions()[0];
        g_inv = InverseCF(g);

        g_deriv = GradCF(g, dim);

        Array<shared_ptr<CoefficientFunction>> values(dim * dim * dim);

        for (auto i : Range(dim))
            for (auto j : Range(dim))
                for (auto k : Range(dim))
                {
                    values[i * dim * dim + j * dim + k] = 0.5 * (MakeComponentCoefficientFunction(g_deriv, i * dim * dim + j * dim + k) + MakeComponentCoefficientFunction(g_deriv, j * dim * dim + i * dim + k) - MakeComponentCoefficientFunction(g_deriv, k * dim * dim + i * dim + j));
                }
        auto tmp = MakeVectorialCoefficientFunction(std::move(values));
        chr1 = tmp->Reshape(Array<int>({dim, dim, dim}));
        chr2 = EinsumCF("ijl,lk->ijk", {chr1, g_inv});

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
            g_tv = 1 / vol[BND] * tv;
        }
        else if (dim == 3)
        {
            vol[BND] = sqrt(InnerProduct(CofactorCF(g) * nv, nv));
            vol[BBND] = sqrt(InnerProduct(g * tv, tv));

            g_tv = 1 / vol[BBND] * tv;
        }

        g_nv = vol[VOL] / vol[BND] * g_inv * nv;
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

    shared_ptr<CoefficientFunction> RiemannianManifold::GetChristoffelSymbol(bool second_kind) const
    {
        return second_kind ? chr2 : chr1;
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

        if (cov_ind1.size() != cov_ind2.size())
            throw Exception("IP: dimensions of c1 and c2 must match");

        // create boolean array with true if cov_ind1 and ind_cov2 coincide at the position
        Array<bool> same_index(cov_ind1.size());
        Array<size_t> position_same_index;
        for (int i = 0; i < cov_ind1.size(); i++)
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
            cfs[2 + i] = bool(cov_ind1[position_same_index[i]]) ? g_inv : g;
        }
        cout << signature_c1 + "," + signature_c2 + raise_lower_signatures << endl;
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
            return GradCF(c1, dim);
        }

        // vector field
        if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(c1))
        {
            return TensorFieldCF(GradCF(vf->GetCoefficients(), dim) + EinsumCF("ikj,k->ij", {chr2, vf->GetCoefficients()}), "10");
        }

        // one-form field
        if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(c1))
        {
            return TensorFieldCF(GradCF(of->GetCoefficients(), dim) - EinsumCF("ik,k->ij", {chr2, of->GetCoefficients()}), "11");
        }

        // General tensor field
        throw Exception("CovDerivative: so far only implemented for a scalar, vector, or one-form field");
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::CovDivergence(shared_ptr<TensorFieldCoefficientFunction> c1) const
    {
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
        if (index1 == index2)
            throw Exception("Trace: indices must be different");
        if (tf->Dimensions().Size() <= max(index1, index2))
            throw Exception("Trace: index out of range");

        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();
        if (cov_ind[index1] != cov_ind[index2])
        {
            signature[index2] = signature[index1];
            return TensorFieldCF(EinsumCF(signature + "->" + signature.erase(index1, 1).erase(index2, 1), {tf}), cov_ind.erase(index1, 1).erase(index2, 1));
        }
        else
        {
            signature[index2] = SIGNATURE[signature.size()];
            string raise_lower_signature = ToString(signature[index2]) + signature[index1];
            if (cov_ind[index1])
            {
                return TensorFieldCF(EinsumCF(signature + "," + raise_lower_signature + "->" + signature.erase(index1, 1).erase(index2, 1), {tf, g_inv}), cov_ind.erase(index1, 1).erase(index2, 1));
            }
            else
            {
                return TensorFieldCF(EinsumCF(signature + "," + raise_lower_signature + "->" + signature.erase(index1, 1).erase(index2, 1), {tf, g}), cov_ind.erase(index1, 1).erase(index2, 1));
            }
        }
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::Contraction(shared_ptr<TensorFieldCoefficientFunction> tf, shared_ptr<VectorFieldCoefficientFunction> vf, size_t slot) const
    {
        auto cov_ind = tf->GetCovariantIndices();
        string signature = tf->GetSignature();
        if (cov_ind[slot])
            return TensorFieldCF(EinsumCF(signature + "," + signature[slot] + "->" + signature.erase(slot, 1), {tf, vf}), cov_ind.erase(slot, 1));
        else
        {
            char new_char = SIGNATURE[signature.size()];
            return TensorFieldCF(EinsumCF(signature + "," + signature[slot] + new_char + "," + new_char + "->" + signature.erase(slot, 1), {tf, g, vf}), cov_ind.erase(slot, 1));
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
        //                        { return self->GetVolumeForm(BND); }, "return the boundary volume form")
        .def_property_readonly("G", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetric(); }, "return the metric tensor")
        .def_property_readonly("G_inv", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetMetricInverse(); }, "return the inverse of the metric tensor")
        .def("ChristoffelSymbol", [](shared_ptr<RiemannianManifold> self, bool second_kind)
             { return self->GetChristoffelSymbol(second_kind); }, "return the Christoffel symbol of the first or second kind", py::arg("second_kind") = false)
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
        return self->CovDerivative(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf)); }, "Covariant divergence of a TensorField", py::arg("tf"))
        .def("Trace", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, size_t index1, size_t index2)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf))
            throw Exception("Trace: input must be a TensorFieldCoefficientFunction");
        return self->Trace(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),index1,index2); }, "Trace of TensorField in two indices. Default are the first two.", py::arg("tf"), py::arg("index1") = 0, py::arg("index2") = 1)
        .def("Contraction", [](shared_ptr<RiemannianManifold> self, shared_ptr<CoefficientFunction> tf, shared_ptr<CoefficientFunction> vf, size_t slot)
             {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf) || !dynamic_pointer_cast<VectorFieldCoefficientFunction>(vf))
            throw Exception("Contraction: input must be a TensorFieldCoefficientFunction and a VectorField");
        return self->Contraction(dynamic_pointer_cast<TensorFieldCoefficientFunction>(tf),dynamic_pointer_cast<VectorFieldCoefficientFunction>(vf)); }, "Contraction of TensorField with a VectorField at given slot. Default slot is the first.", py::arg("tf"), py::arg("vf"), py::arg("slot") = 0);
}
