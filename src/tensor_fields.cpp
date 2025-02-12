#include "tensor_fields.hpp"

#include <tensorcoefficient.hpp>

namespace ngfem
{
    shared_ptr<CoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                  const string &covariant_indices)
    {
        if (cf->IsZeroCF())
            return cf;
        return make_shared<TensorFieldCoefficientFunction>(cf, covariant_indices);
    }

    shared_ptr<CoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf)
    {
        if (cf->Dimensions().Size() != 1)
            throw Exception("VectorFieldCF: input must be a vector-valued CoefficientFunction");
        if (cf->IsZeroCF())
            return cf;
        return make_shared<VectorFieldCoefficientFunction>(cf);
    }

    shared_ptr<CoefficientFunction> OneFormCF(const shared_ptr<CoefficientFunction> &cf)
    {
        if (cf->Dimensions().Size() != 1)
            throw Exception("OneFormCF: input must be a vector-valued CoefficientFunction");
        if (cf->IsZeroCF())
            return cf;
        return make_shared<OneFormCoefficientFunction>(cf);
    }

    shared_ptr<CoefficientFunction> ScalarFieldCF(const shared_ptr<CoefficientFunction> &cf)
    {
        if (cf->Dimensions().Size() != 0)
            throw Exception("ScalarFieldCF: input must be a scalar CoefficientFunction");
        if (cf->IsZeroCF())
            return cf;
        return make_shared<ScalarFieldCoefficientFunction>(cf);
    }

    shared_ptr<CoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2)
    {
        string cov_ind = c1->GetCovariantIndices() + c2->GetCovariantIndices();

        if (cov_ind.size() > SIGNATURE.size())
            throw Exception("TensorProduct: Overflow: c1 and c2 have > 52 indices together");

        string signature = SIGNATURE.substr(0, c1->GetCovariantIndices().size());
        signature += ",";
        signature += SIGNATURE.substr(c1->GetCovariantIndices().size(), c2->GetCovariantIndices().size());
        signature += "->";
        signature += SIGNATURE.substr(0, cov_ind.size());

        return TensorFieldCF(EinsumCF(signature, {c1, c2}), cov_ind);
    }
}

void ExportTensorFields(py::module m)
{
    using namespace ngfem;
    using namespace ngfem::tensor_internal;

    m.def("TensorField", [](shared_ptr<CoefficientFunction> cf, string cov_indices)
          {
            for (auto i : Range(cov_indices.size()))
                if (cov_indices[i] != '0' && cov_indices[i] != '1')
                    throw Exception("TensorField: covariant_indices must be a string of 0s and 1s");

            return TensorFieldCF(cf, cov_indices); }, "Create a TensorField from a CoefficientFunction and a list of covariant indices");
    m.def("VectorField", [](shared_ptr<CoefficientFunction> c1)
          { return VectorFieldCF(c1); }, "Create a VectorField from a CoefficientFunction");
    m.def("OneForm", [](shared_ptr<CoefficientFunction> c1)
          { return OneFormCF(c1); }, "Create a OneForm from a CoefficientFunction");
    m.def("ScalarField", [](shared_ptr<CoefficientFunction> c1)
          { return ScalarFieldCF(c1); }, "Create a ScalarField from a CoefficientFunction");
    m.def("TensorProduct", [](shared_ptr<CoefficientFunction> c1, shared_ptr<CoefficientFunction> c2)
          {
        if (!dynamic_pointer_cast<TensorFieldCoefficientFunction>(c1) || !dynamic_pointer_cast<TensorFieldCoefficientFunction>(c2))
        throw Exception("TensorProduct: both inputs must be TensorFields");
        return TensorProduct(dynamic_pointer_cast<TensorFieldCoefficientFunction>(c1), dynamic_pointer_cast<TensorFieldCoefficientFunction>(c2)); }, "Build tensor product from two TensorFields");
}