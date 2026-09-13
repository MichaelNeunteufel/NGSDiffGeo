#include "kforms_bindings.hpp"

#include "kforms_detail.hpp"
#include "kforms_diagnostics.hpp"
#include "riemannian_manifold.hpp"

#include <string>
#include <utility>

namespace
{
    template <typename T>
    std::shared_ptr<T> RequireBindingNonNull(
        std::shared_ptr<T> ptr, const char *name)
    {
        if (!ptr)
            throw ngfem::Exception(
                std::string(name) + ": input coefficient is null");
        return ptr;
    }

    int ParseDoubleFormSlot(const std::string &slot)
    {
        if (slot == "both" || slot == "all")
            return -1;
        if (slot == "left" || slot == "0")
            return 0;
        if (slot == "right" || slot == "1")
            return 1;
        throw ngfem::Exception(
            "slot must be 'left', 'right', or 'both'/'all'");
    }
}

void ExportKForms(py::module m)
{
    using namespace ngfem;

    m.attr("_MAX_FORM_RANK") = MAX_FORM_RANK;
    m.attr("_MAX_SPACE_DIM") = MAX_SPACE_DIM;

    auto warn_deprecated = [](const char *old_name, const char *replacement)
    {
        const std::string message = std::string(old_name) +
                                    " is deprecated; use " + replacement;
        if (PyErr_WarnEx(PyExc_DeprecationWarning, message.c_str(), 2) < 0)
            throw py::error_already_set();
    };

    ExportAlternationBinding(m);

    py::class_<KFormCoefficientFunction,
               TensorFieldCoefficientFunction,
               shared_ptr<KFormCoefficientFunction>>(m, "KForm",
                                                     "Fully covariant rank-k wrapper.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int k, int dim)
                      { return KFormCF(cf, k, dim); }),
             py::arg("cf"), py::arg("k"), py::arg("dim"))
        .def_property_readonly("degree", &KFormCoefficientFunction::Degree)
        .def_property_readonly("dim_space", &KFormCoefficientFunction::DimensionOfSpace)
        .def("wedge", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b)
             { return Wedge(a, b); }, py::arg("b"))
        .def("d", [](shared_ptr<KFormCoefficientFunction> a)
             { return ExteriorDerivative(a); })
        .def("star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
             { return HodgeStar(a, *RequireBindingNonNull(M, "HodgeStar"), vb); }, py::arg("M"), py::arg("vb") = VOL)
        .def("inv_star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
             { return InverseHodgeStar(a, *RequireBindingNonNull(M, "InverseHodgeStar"), vb); }, py::arg("M"), py::arg("vb") = VOL)
        .def(NGSPickle<KFormCoefficientFunction>());

    py::class_<DoubleFormCoefficientFunction,
               TensorFieldCoefficientFunction,
               shared_ptr<DoubleFormCoefficientFunction>>(m, "DoubleForm",
                                                          "Covariant tensor with left/right form degrees. Construction does not alternate entries.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
                      { return DoubleFormCF(cf, p, q, dim); }),
             py::arg("cf"), py::arg("p"), py::arg("q"), py::arg("dim"))
        .def_property_readonly("degree_left", &DoubleFormCoefficientFunction::LeftDegree)
        .def_property_readonly("degree_right", &DoubleFormCoefficientFunction::RightDegree)
        .def_property_readonly("dim_space", &DoubleFormCoefficientFunction::DimensionOfSpace)
        .def_property_readonly("is_zero", [](shared_ptr<DoubleFormCoefficientFunction> a)
                               { return a->IsZeroCF(); })
        .def("wedge", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
             { return Wedge(a, b); }, py::arg("b"))
        .def("star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
             { return HodgeStar(a, *RequireBindingNonNull(M, "HodgeStar"), vb, ParseDoubleFormSlot(slot)); }, py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both")
        .def("inv_star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
             { return InverseHodgeStar(a, *RequireBindingNonNull(M, "InverseHodgeStar"), vb, ParseDoubleFormSlot(slot)); }, py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both")
        .def_property_readonly("trans", [](shared_ptr<DoubleFormCoefficientFunction> a)
                               { return SwapDoubleFormSlots(a); })
        .def(NGSPickle<DoubleFormCoefficientFunction>());

    py::class_<ScalarFieldCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<ScalarFieldCoefficientFunction>>(m, "ScalarField", "Scalar zero-form with optional ambient-dimension metadata.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int dim)
                      { return ScalarFieldCF(cf, dim); }),
             py::arg("cf"), py::arg("dim"))
        .def_static("from_cf", [warn_deprecated](shared_ptr<CoefficientFunction> cf, int dim)
                    {
                        warn_deprecated("ScalarField.from_cf", "ScalarField(cf, dim=...)");
                        return ScalarFieldCF(cf, dim); }, py::arg("cf"), py::arg("dim"))
        .def(NGSPickle<ScalarFieldCoefficientFunction>());

    py::class_<OneFormCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<OneFormCoefficientFunction>>(m, "OneForm", "Covariant one-form with ambient dimension inferred from its vector shape.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf)
                      { return OneFormCF(cf); }),
             py::arg("cf"))
        .def_static("from_cf", [warn_deprecated](shared_ptr<CoefficientFunction> cf)
                    {
                        warn_deprecated("OneForm.from_cf", "OneForm(cf)");
                        return OneFormCF(cf); }, py::arg("cf"))
        .def(NGSPickle<OneFormCoefficientFunction>());

    py::class_<TwoFormCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<TwoFormCoefficientFunction>>(m, "TwoForm", "Rank-two form wrapper. Input alternation is not checked.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int dim)
                      { return TwoFormCF(cf, dim); }),
             py::arg("cf"), py::arg("dim") = -1)
        .def_static("from_cf", [warn_deprecated](shared_ptr<CoefficientFunction> cf, int dim)
                    {
                        warn_deprecated("TwoForm.from_cf", "TwoForm(cf, dim=...)");
                        return TwoFormCF(cf, dim); }, py::arg("cf"), py::arg("dim") = -1)
        .def(NGSPickle<TwoFormCoefficientFunction>());

    py::class_<ThreeFormCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<ThreeFormCoefficientFunction>>(m, "ThreeForm", "Rank-three form wrapper. Input alternation is not checked.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int dim)
                      { return ThreeFormCF(cf, dim); }),
             py::arg("cf"), py::arg("dim") = -1)
        .def_static("from_cf", [warn_deprecated](shared_ptr<CoefficientFunction> cf, int dim)
                    {
                        warn_deprecated("ThreeForm.from_cf", "ThreeForm(cf, dim=...)");
                        return ThreeFormCF(cf, dim); }, py::arg("cf"), py::arg("dim") = -1)
        .def(NGSPickle<ThreeFormCoefficientFunction>());

    m.def("Wedge", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b)
          { return Wedge(a, b); }, py::arg("a"), py::arg("b"));
    m.def("Wedge", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
          { return Wedge(a, b); }, py::arg("a"), py::arg("b"));
    m.def("_AddKForms", &kforms_internal::AddKForms,
          py::arg("a"), py::arg("b"), py::arg("subtract") = false);
    m.def("_ScaleKForm", &kforms_internal::ScaleKForm,
          py::arg("form"), py::arg("scalar"));
    m.def("_ScaleKFormConstant", [](shared_ptr<KFormCoefficientFunction> form, double scalar)
          { return kforms_internal::ScaleKForm(
                std::move(form), ConstantCF(scalar)); }, py::arg("form"), py::arg("scalar"));
    m.def("_AddDoubleForms", &kforms_internal::AddDoubleForms,
          py::arg("a"), py::arg("b"), py::arg("subtract") = false);
    m.def("_ScaleDoubleForm", &kforms_internal::ScaleDoubleForm,
          py::arg("form"), py::arg("scalar"));
    m.def("_ScaleDoubleFormConstant", [](shared_ptr<DoubleFormCoefficientFunction> form, double scalar)
          { return kforms_internal::ScaleDoubleForm(
                std::move(form), ConstantCF(scalar)); }, py::arg("form"), py::arg("scalar"));
    m.def("_WedgeDenseKForms", &kforms_internal::WedgeDenseKForms,
          py::arg("a"), py::arg("b"));
    m.def("_WedgeDenseDoubleForms",
          &kforms_internal::WedgeDenseDoubleForms,
          py::arg("a"), py::arg("b"));
    m.def("_ExteriorDerivativeDenseKForm",
          &kforms_internal::ExteriorDerivativeDenseKForm,
          py::arg("form"));
    m.def("_FormBasisStorageBytesLowerBound", [](int dim, int degree)
          { return kforms_internal::FormBasisStorageBytesLowerBound(dim, degree); }, py::arg("dim"), py::arg("degree"));
    m.def("_CompactDoubleFormExpansionTableStorageBytesLowerBound", [](int dim, int left_degree, int right_degree)
          { return kforms_internal::
                CompactDoubleFormExpansionTableStorageBytesLowerBound(
                    dim, left_degree, right_degree); }, py::arg("dim"), py::arg("left_degree"), py::arg("right_degree"));
    m.def("_CompactWedgeTableStorageBytesLowerBound", [](int dim, int p, int q, int r, int s)
          { return kforms_internal::CompactWedgeTableStorageBytesLowerBound(
                dim, p, q, r, s); }, py::arg("dim"), py::arg("p"), py::arg("q"), py::arg("r"), py::arg("s"));
    m.def("_CompactExteriorDerivativeTableStorageBytesLowerBound", [](int dim, int degree)
          { return kforms_internal::
                CompactExteriorDerivativeTableStorageBytesLowerBound(
                    dim, degree); }, py::arg("dim"), py::arg("degree"));
    m.def("_InducedFormMetricTableStorageBytesLowerBound", [](int dim, int degree)
          { return kforms_internal::InducedFormMetricTableStorageBytesLowerBound(
                dim, degree); }, py::arg("dim"), py::arg("degree"));
    m.def("_CompactHodgeMapTableStorageBytesLowerBound", [](int dim, int degree)
          { return kforms_internal::CompactHodgeMapTableStorageBytesLowerBound(
                dim, degree); }, py::arg("dim"), py::arg("degree"));
    m.def("_CompactDoubleTraceTableStorageBytesLowerBound", [](int dim, int left_degree, int right_degree)
          { return kforms_internal::CompactDoubleTraceTableStorageBytesLowerBound(
                dim, left_degree, right_degree); }, py::arg("dim"), py::arg("left_degree"), py::arg("right_degree"));
    m.def("_CompactFormCacheContainerStorageBytes",
          &kforms_internal::CompactFormCacheContainerStorageBytes);
    m.def("_CompactFormCacheContainerStorageBreakdown", []
          {
        const auto bytes =
            kforms_internal::CompactFormCacheContainerStorageBreakdown();
        py::dict result;
        result["form_basis"] = bytes[0];
        result["double_form_expansion"] = bytes[1];
        result["double_trace"] = bytes[2];
        result["wedge"] = bytes[3];
        result["exterior_derivative"] = bytes[4];
        result["induced_metric"] = bytes[5];
        result["hodge_map"] = bytes[6];
        return result; });

    m.def("d", [](shared_ptr<KFormCoefficientFunction> a)
          { return ExteriorDerivative(a); }, py::arg("a"));
    m.def("star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
          { return RequireBindingNonNull(M, "Star")->Star(a, vb); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL);
    m.def("inv_star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
          { return InverseHodgeStar(a, *RequireBindingNonNull(M, "InverseHodgeStar"), vb); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL);
    m.def("star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
          { return HodgeStar(a, *RequireBindingNonNull(M, "HodgeStar"), vb, ParseDoubleFormSlot(slot)); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both");
    m.def("inv_star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
          { return InverseHodgeStar(a, *RequireBindingNonNull(M, "InverseHodgeStar"), vb, ParseDoubleFormSlot(slot)); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both");
    m.def("slot_inner_product", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, bool forms)
          { return SlotInnerProduct(a, *RequireBindingNonNull(M, "SlotInnerProduct"), vb, forms); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("forms") = true);

    m.def("delta", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M)
          { return RequireBindingNonNull(M, "Coderivative")->Coderivative(a); }, py::arg("a"), py::arg("M"));
}
