#include "tensor_fields.hpp"

#include <tensorcoefficient.hpp>

#include <array>
#include <mutex>

namespace ngfem
{
  namespace
  {
    using EinsumCache = std::array<std::array<std::string, 53>, 53>;

    std::string MakeEinsumSignature(int r1, int r2)
    {
      const std::string sig1 = SIGNATURE.substr(0, r1);
      const std::string sig2 = SIGNATURE.substr(r1, r2);
      const std::string sigout = SIGNATURE.substr(0, r1 + r2);
      return sig1 + "," + sig2 + "->" + sigout;
    }

    EinsumCache &GetEinsumCache()
    {
      static EinsumCache cache;
      static std::once_flag flag;
      std::call_once(flag, [&]()
                     {
        for (int r1 = 0; r1 <= 52; ++r1)
          for (int r2 = 0; r2 <= 52; ++r2)
            cache[size_t(r1)][size_t(r2)] = MakeEinsumSignature(r1, r2); });
      return cache;
    }
  } // namespace

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
    if (cf->Dimension() != (size_t)(cf->Dimensions()[0]))
      throw Exception("VectorFieldCF: dimension metadata mismatch");
    // if (cf->IsZeroCF())
    //     return cf;
    if (auto vf = dynamic_pointer_cast<VectorFieldCoefficientFunction>(cf))
      return vf;
    return make_shared<VectorFieldCoefficientFunction>(cf);
  }

  shared_ptr<TensorFieldCoefficientFunction> PermuteTensorCF(shared_ptr<TensorFieldCoefficientFunction> tf,
                                                             const std::vector<int> &order)
  {
    int rank = int(order.size());
    if (int(tf->Dimensions().Size()) != rank)
      throw Exception("PermuteTensorCF: rank mismatch");

    std::string sig = tf->GetSignature();
    std::string cov = tf->GetCovariantIndices();
    std::string out_sig(size_t(rank), 'a');
    std::string out_cov(size_t(rank), '1');

    for (int i = 0; i < rank; ++i)
    {
      if (order[size_t(i)] < 0 || order[size_t(i)] >= rank)
        throw Exception("PermuteTensorCF: permutation index out of range");
      out_sig[size_t(i)] = sig[size_t(order[size_t(i)])];
      out_cov[size_t(i)] = cov[size_t(order[size_t(i)])];
    }

    auto out_cf = EinsumCF(sig + "->" + out_sig, {tf->GetCoefficients()});
    return TensorFieldCF(out_cf, out_cov);
  }

  int Factorial(int n)
  {
    static constexpr std::array<int, 13> table = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};
    if (n < 0)
      throw Exception("Factorial: n must be non-negative");
    if (n < int(table.size()))
      return table[size_t(n)];

    int f = table.back();
    for (int i = int(table.size()); i <= n; ++i)
      f *= i;
    return f;
  }

  shared_ptr<TensorFieldCoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2)
  {
    auto m1 = c1->Meta();
    auto m2 = c2->Meta();
    auto mout = m1.Concatenated(m2);

    const auto &eins = GetEinsumCache()[m1.rank][m2.rank];

    auto out_cf = EinsumCF(eins, {c1->GetCoefficients(), c2->GetCoefficients()});
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

    std::string eins = ToString(new_label) + old_label + "," + sigmod + "->" + sig;

    auto result = EinsumCF(eins, {proj, tf});
    return TensorFieldCF(result, tf->GetCovariantIndices());
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
