#ifndef NGSDIFFGEO_KFORMS_INTERNAL_HPP
#define NGSDIFFGEO_KFORMS_INTERNAL_HPP

#include "kforms.hpp"

// Narrow cross-module interface used by the Riemannian-manifold
// implementation. Private compact-form machinery belongs in kforms_detail.hpp
namespace ngfem::kforms_internal
{
    /// Return a normalized form inner product using independent components,
    /// or nullptr when either operand is not proven compact
    shared_ptr<CoefficientFunction> CompactKFormInnerProduct(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right,
        shared_ptr<CoefficientFunction> inverse_metric);

    /// Double-form analogue with factored induced metrics for both slots
    shared_ptr<CoefficientFunction> CompactDoubleFormInnerProduct(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right,
        shared_ptr<CoefficientFunction> inverse_metric);

    /// Contract the first axis of each alternating block against a supplied
    /// inverse metric. Proven compact inputs retain independent-component
    /// storage. Other inputs use the exact full-tensor fallback
    shared_ptr<DoubleFormCoefficientFunction> TraceDoubleFormWithMetric(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric);

}

#endif
