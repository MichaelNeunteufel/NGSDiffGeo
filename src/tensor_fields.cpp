#include "tensor_fields.hpp"

#include <tensorcoefficient.hpp>

namespace ngfem
{

  bool IsVectorField(const TensorFieldCoefficientFunction &t)
  {
    auto m = t.Meta();
    return m.rank == 1 && m.covmask == 0;
  }

  bool IsOneForm(const TensorFieldCoefficientFunction &t)
  {
    auto m = t.Meta();
    return m.rank == 1 && m.covmask == 1;
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                           const string &covariant_indices)
  {
    if (!cf)
      throw Exception("TensorFieldCF: input coefficient is null");
    auto meta = TensorMeta::FromCovString(covariant_indices);
    if (auto tf = dynamic_pointer_cast<TensorFieldCoefficientFunction>(cf))
      if (tf->Meta() == meta)
        return tf;
    return make_shared<TensorFieldCoefficientFunction>(cf, meta);
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                           const TensorMeta &meta)
  {
    if (!cf)
      throw Exception("TensorFieldCF: input coefficient is null");
    if (auto tf = dynamic_pointer_cast<TensorFieldCoefficientFunction>(cf))
      if (tf->Meta() == meta)
        return tf;
    auto result = make_shared<TensorFieldCoefficientFunction>(cf, meta);
    return result;
  }

  shared_ptr<VectorFieldCoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf)
  {
    if (cf->Dimensions().Size() != 1)
      throw Exception("VectorFieldCF: input must be a vector-valued CoefficientFunction");
    if (cf->Dimension() != cf->Dimensions()[0])
      throw Exception("VectorFieldCF: dimension metadata mismatch");
    // if (cf->IsZeroCF())
    //     return cf;
    if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(cf))
      return vf;
    return make_shared<VectorFieldCoefficientFunction>(cf);
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2)
  {
    auto m1 = c1->Meta();
    auto m2 = c2->Meta();
    auto mout = m1.Concatenated(m2);

    auto make_eins = [](int r1, int r2) -> std::string
    {
      const std::string sig1 = SIGNATURE.substr(0, r1);
      const std::string sig2 = SIGNATURE.substr(r1, r2);
      const std::string sigout = SIGNATURE.substr(0, r1 + r2);
      return sig1 + "," + sig2 + "->" + sigout;
    };
    static std::array<std::array<std::string, 53>, 53> eins_cache;
    if (eins_cache[m1.rank][m2.rank].empty())
      eins_cache[m1.rank][m2.rank] = make_eins(m1.rank, m2.rank);
    const std::string &eins = eins_cache[m1.rank][m2.rank];

    auto out_cf = EinsumCF(eins, {c1, c2});
    return TensorFieldCF(out_cf, mout);
  }
}

void CheckCovariantIndices(const std::string &s)
{
  for (char c : s)
    if (c != '0' && c != '1')
      throw ngstd::Exception("covariant_indices must be a string of 0s and 1s");
}

void ExportTensorFields(py::module m)
{
  using namespace ngfem;
  using std::shared_ptr;
  using std::string;

  // TensorField
  py::class_<TensorFieldCoefficientFunction,
             CoefficientFunction,
             shared_ptr<TensorFieldCoefficientFunction>>(m, "TensorField")
      .def(py::init([](shared_ptr<CoefficientFunction> cf, string cov_indices)
                    {
      CheckCovariantIndices(cov_indices);
     return std::static_pointer_cast<TensorFieldCoefficientFunction>(TensorFieldCF(cf, cov_indices)); }),
           py::arg("cf"), py::arg("covariant_indices"))

      .def_property_readonly("covariant_indices",
                             &TensorFieldCoefficientFunction::GetCovariantIndices)
      .def_property_readonly("coef",
                             &TensorFieldCoefficientFunction::GetCoefficients)

      .def_static("from_cf", [](shared_ptr<CoefficientFunction> cf, string cov_indices)
                  {
      CheckCovariantIndices(cov_indices);
      return TensorFieldCF(cf, cov_indices); }, py::arg("cf"), py::arg("covariant_indices"));

  // VectorField
  py::class_<VectorFieldCoefficientFunction,
             TensorFieldCoefficientFunction,
             shared_ptr<VectorFieldCoefficientFunction>>(m, "VectorField")
      .def(py::init([](shared_ptr<CoefficientFunction> cf)
                    { return VectorFieldCF(cf); }),
           py::arg("cf"))
      .def_static("from_cf", [](shared_ptr<CoefficientFunction> cf)
                  { return VectorFieldCF(cf); }, py::arg("cf"));

  m.def("MakeTensorField", [](shared_ptr<CoefficientFunction> cf, string cov_indices)
        {
        CheckCovariantIndices(cov_indices);
        return TensorFieldCF(cf, cov_indices); });

  m.def("MakeVectorField", [](shared_ptr<CoefficientFunction> cf)
        { return VectorFieldCF(cf); });

  m.def("TensorProduct", [](shared_ptr<TensorFieldCoefficientFunction> a, shared_ptr<TensorFieldCoefficientFunction> b)
        { return TensorProduct(a, b); }, py::arg("a"), py::arg("b"));
}
