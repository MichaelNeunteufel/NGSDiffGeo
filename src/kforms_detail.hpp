#ifndef NGSDIFFGEO_KFORMS_DETAIL_HPP
#define NGSDIFFGEO_KFORMS_DETAIL_HPP

#include "kforms_basis.hpp"
#include "kforms_internal.hpp"

#include <optional>

// Private implementation and testing seam. This is not part of the supported
// public C++ interface. Cross-module consumers should use kforms_internal.hpp
namespace ngfem::kforms_internal
{
    struct CompactKFormView
    {
        shared_ptr<CoefficientFunction> independent;
        int degree;
        int dim;
    };

    std::optional<CompactKFormView> TryGetCompactView(
        const KFormCoefficientFunction &form);

    struct CompactDoubleFormView
    {
        shared_ptr<CoefficientFunction> independent;
        int left_degree;
        int right_degree;
        int dim;
    };

    std::optional<CompactDoubleFormView> TryGetCompactView(
        const DoubleFormCoefficientFunction &form);

    shared_ptr<KFormCoefficientFunction> KFormFromIndependentCF(
        shared_ptr<CoefficientFunction> independent_value,
        int degree, int dim);

    shared_ptr<DoubleFormCoefficientFunction> DoubleFormFromIndependentCF(
        shared_ptr<CoefficientFunction> independent_value,
        int left_degree, int right_degree, int dim);

    shared_ptr<DoubleFormCoefficientFunction> AddDoubleForms(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right,
        bool subtract = false);

    shared_ptr<DoubleFormCoefficientFunction> ScaleDoubleForm(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> scalar);

    shared_ptr<DoubleFormCoefficientFunction> WedgeDenseDoubleForms(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right);

    /// Full-shaped fused evaluator used by compact double-wedge operation
    /// nodes. It consumes independent inputs without materializing the final
    /// independent output as an additional evaluation-graph node
    shared_ptr<CoefficientFunction> MakeCompactDoubleWedgeFullEvaluator(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right);

    shared_ptr<DoubleFormCoefficientFunction> SwapCompactDoubleFormSlots(
        shared_ptr<DoubleFormCoefficientFunction> form);

    shared_ptr<DoubleFormCoefficientFunction> SwapDenseDoubleFormSlots(
        shared_ptr<DoubleFormCoefficientFunction> form);

    shared_ptr<KFormCoefficientFunction> AddKForms(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right,
        bool subtract = false);

    shared_ptr<KFormCoefficientFunction> ScaleKForm(
        shared_ptr<KFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> scalar);

    /// Existing exact full-tensor implementation, retained as the explicit
    /// fallback and as an independent reference for compact-path tests
    shared_ptr<KFormCoefficientFunction> WedgeDenseKForms(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right);

    shared_ptr<KFormCoefficientFunction> ExteriorDerivativeDenseKForm(
        shared_ptr<KFormCoefficientFunction> form);
}

#endif
