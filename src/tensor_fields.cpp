#include "tensor_fields.hpp"

#include <core/register_archive.hpp>
#include <tensorcoefficient.hpp>

namespace ngfem
{
  namespace
  {
    template <typename T>
    shared_ptr<T> RequireNonNull(shared_ptr<T> ptr, const char *name)
    {
      if (!ptr)
        throw Exception(std::string(name) + ": input coefficient is null");
      return ptr;
    }

    std::string MakeTensorProductSignature(size_t r1, size_t r2)
    {
      if (r1 > MAX_SIGNATURE_LABELS || r2 > MAX_SIGNATURE_LABELS - r1)
        throw Exception("TensorProduct: combined tensor rank exceeds " +
                        ToString(MAX_SIGNATURE_LABELS));
      const std::string sig1 = SIGNATURE.substr(0, r1);
      const std::string sig2 = SIGNATURE.substr(r1, r2);
      const std::string sigout = SIGNATURE.substr(0, r1 + r2);
      return sig1 + "," + sig2 + "->" + sigout;
    }
  } // namespace

  bool IsVectorField(const TensorFieldCoefficientFunction &t)
  {
    auto m = t.Meta();
    return m.Rank() == 1 && m.CovarianceMask() == 0;
  }

  bool IsOneForm(const TensorFieldCoefficientFunction &t)
  {
    auto m = t.Meta();
    return m.Rank() == 1 && m.CovarianceMask() == 1;
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                           const string &covariant_indices)
  {
    auto checked_cf = RequireNonNull(cf, "TensorFieldCF");
    auto meta = TensorMeta::FromCovString(covariant_indices);
    return TensorFieldCF(checked_cf, meta);
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                           const TensorMeta &meta)
  {
    auto checked_cf = RequireNonNull(cf, "TensorFieldCF");
    // Tensor wrappers do not alter values. Remove incompatible metadata layers
    // instead of retaining a wrapper-only chain in the expression graph.
    while (auto tf = dynamic_pointer_cast<TensorFieldCoefficientFunction>(checked_cf))
    {
      if (tf->Meta() == meta)
        return tf;
      checked_cf = tf->GetFullCoefficient();
    }
    return make_shared<TensorFieldCoefficientFunction>(checked_cf, meta);
  }

  shared_ptr<VectorFieldCoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf)
  {
    auto checked_cf = RequireNonNull(cf, "VectorFieldCF");
    while (auto tf = dynamic_pointer_cast<TensorFieldCoefficientFunction>(checked_cf))
    {
      if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(tf))
        return vf;
      checked_cf = tf->GetFullCoefficient();
    }
    if (checked_cf->Dimensions().Size() != 1)
      throw Exception("VectorFieldCF: input must be a vector-valued CoefficientFunction");
    return make_shared<VectorFieldCoefficientFunction>(checked_cf);
  }

  shared_ptr<TensorFieldCoefficientFunction> PermuteTensorCF(shared_ptr<TensorFieldCoefficientFunction> tf,
                                                             const std::vector<int> &order)
  {
    tf = RequireNonNull(std::move(tf), "PermuteTensorCF");
    int rank = int(order.size());
    if (int(tf->Dimensions().Size()) != rank)
      throw Exception("PermuteTensorCF: rank mismatch");

    std::vector<char> seen(size_t(rank), 0);

    std::string sig = tf->GetSignature();
    std::string cov = tf->GetCovariantIndices();
    std::string out_sig(size_t(rank), 'a');
    std::string out_cov(size_t(rank), '1');

    for (int i = 0; i < rank; ++i)
    {
      int oi = order[size_t(i)];
      if (oi < 0 || oi >= rank)
        throw Exception("PermuteTensorCF: permutation index out of range");
      if (seen[size_t(oi)])
        throw Exception("PermuteTensorCF: order must be a permutation");
      seen[size_t(oi)] = 1;
      out_sig[size_t(i)] = sig[size_t(oi)];
      out_cov[size_t(i)] = cov[size_t(oi)];
    }

    auto out_cf = EinsumCF(sig + "->" + out_sig, {tf->GetFullCoefficient()});
    return TensorFieldCF(out_cf, out_cov);
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2)
  {
    c1 = RequireNonNull(std::move(c1), "TensorProduct");
    c2 = RequireNonNull(std::move(c2), "TensorProduct");

    auto m1 = c1->Meta();
    auto m2 = c2->Meta();
    auto mout = m1.Concatenated(m2);

    const auto eins = MakeTensorProductSignature(m1.Rank(), m2.Rank());

    auto out_cf = EinsumCF(eins, {c1->GetFullCoefficient(), c2->GetFullCoefficient()});
    return TensorFieldCF(out_cf, mout);
  }

  shared_ptr<TensorFieldCoefficientFunction> ApplyProjectorToIndex(shared_ptr<TensorFieldCoefficientFunction> tf,
                                                                   shared_ptr<CoefficientFunction> proj,
                                                                   size_t index)
  {
    if (!tf || !proj)
      throw Exception("ApplyProjectorToIndex: inputs must be non-null");
    std::string sig = tf->GetSignature();
    if (index >= sig.size())
      throw Exception("ApplyProjectorToIndex: index out of range");

    char old_label = sig[index];
    char new_label = tf->Meta().FreshLabel();
    std::string sigmod = sig;
    sigmod[index] = new_label;

    // Contract one projector axis with the selected tensor axis while retaining
    // the original label and slot order in the output.
    std::string eins = ToString(new_label) + old_label + "," + sigmod + "->" + sig;

    auto result = EinsumCF(eins, {proj, tf->GetFullCoefficient()});
    return TensorFieldCF(result, tf->GetCovariantIndices());
  }

  static ngcore::RegisterClassForArchive<TensorFieldCoefficientFunction, CoefficientFunction>
      reg_tensor_field_cf;
  static ngcore::RegisterClassForArchive<VectorFieldCoefficientFunction, TensorFieldCoefficientFunction>
      reg_vector_field_cf;
}

void ExportTensorFields(py::module m)
{
  using namespace ngfem;
  using std::shared_ptr;
  using std::string;

  auto warn_deprecated = [](const char *old_name, const char *replacement)
  {
    const std::string message = std::string(old_name) +
                                " is deprecated; use " + replacement;
    if (PyErr_WarnEx(PyExc_DeprecationWarning, message.c_str(), 2) < 0)
      throw py::error_already_set();
  };

  // TensorField
  py::class_<TensorFieldCoefficientFunction,
             CoefficientFunction,
             shared_ptr<TensorFieldCoefficientFunction>>(
      m, "TensorField",
      "Coefficient-function wrapper carrying tensor-axis variance metadata.")
      .def(py::init([](shared_ptr<CoefficientFunction> cf, string cov_indices)
                    { return TensorFieldCF(cf, cov_indices); }),
           py::arg("cf"), py::arg("covariant_indices"),
           "Wrap cf as a tensor field. Use '1' for covariant and '0' for contravariant axes.")

      .def_property_readonly("covariant_indices",
                             &TensorFieldCoefficientFunction::GetCovariantIndices,
                             "Variance string with one '0' or '1' per tensor axis.")
      .def_property_readonly("coef",
                             &TensorFieldCoefficientFunction::GetFullCoefficient,
                             "The wrapped full-shaped NGSolve coefficient function.")

      .def_static("from_cf", [warn_deprecated](shared_ptr<CoefficientFunction> cf, string cov_indices)
                  {
      warn_deprecated("TensorField.from_cf", "TensorField(cf, covariant_indices=...)");
      return TensorFieldCF(cf, cov_indices); }, py::arg("cf"), py::arg("covariant_indices"), "Deprecated: construct TensorField(cf, covariant_indices=...) directly.")
      .def(NGSPickle<TensorFieldCoefficientFunction>());

  // VectorField
  py::class_<VectorFieldCoefficientFunction,
             TensorFieldCoefficientFunction,
             shared_ptr<VectorFieldCoefficientFunction>>(
      m, "VectorField", "Contravariant rank-one tensor field.")
      .def(py::init([](shared_ptr<CoefficientFunction> cf)
                    { return VectorFieldCF(cf); }),
           py::arg("cf"), "Wrap a vector-valued coefficient function as a VectorField.")
      .def_static("from_cf", [warn_deprecated](shared_ptr<CoefficientFunction> cf)
                  {
      warn_deprecated("VectorField.from_cf", "VectorField(cf)");
      return VectorFieldCF(cf); }, py::arg("cf"), "Deprecated: construct VectorField(cf) directly.")
      .def(NGSPickle<VectorFieldCoefficientFunction>());

  m.def("MakeTensorField", [warn_deprecated](shared_ptr<CoefficientFunction> cf, string cov_indices)
        {
        warn_deprecated("MakeTensorField", "TensorField(cf, covariant_indices=...)");
        return TensorFieldCF(cf, cov_indices); }, py::arg("cf"), py::arg("covariant_indices"), "Deprecated: construct TensorField(cf, covariant_indices=...) directly.");

  m.def("MakeVectorField", [warn_deprecated](shared_ptr<CoefficientFunction> cf)
        {
        warn_deprecated("MakeVectorField", "VectorField(cf)");
        return VectorFieldCF(cf); }, py::arg("cf"), "Deprecated: construct VectorField(cf) directly.");

  m.def("TensorProduct", [](shared_ptr<TensorFieldCoefficientFunction> a, shared_ptr<TensorFieldCoefficientFunction> b)
        { return TensorProduct(a, b); }, py::arg("a"), py::arg("b"), "Return the tensor product, with axes and variance metadata of a followed by those of b.");
}
