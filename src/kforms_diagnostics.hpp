#ifndef NGSDIFFGEO_KFORMS_DIAGNOSTICS_HPP
#define NGSDIFFGEO_KFORMS_DIAGNOSTICS_HPP

#include <array>
#include <cstddef>

namespace ngfem::kforms_internal
{
    /// Internal benchmark and testing diagnostics for compact-form storage.
    ///
    /// This is not a stable public API. The lower-bound queries may materialize
    /// the corresponding lazy cache entry, but are otherwise outside normal
    /// coefficient construction and evaluation paths.
    size_t FormBasisStorageBytesLowerBound(int dim, int degree);
    size_t CompactDoubleFormExpansionTableStorageBytesLowerBound(
        int dim, int left_degree, int right_degree);
    size_t CompactWedgeTableStorageBytesLowerBound(
        int dim, int p, int q, int r, int s);
    size_t CompactExteriorDerivativeTableStorageBytesLowerBound(
        int dim, int degree);
    size_t InducedFormMetricTableStorageBytesLowerBound(
        int dim, int degree);
    size_t CompactHodgeMapTableStorageBytesLowerBound(
        int dim, int degree);
    size_t CompactDoubleTraceTableStorageBytesLowerBound(
        int dim, int left_degree, int right_degree);

    std::array<size_t, 7> CompactFormCacheContainerStorageBreakdown();
    size_t CompactFormCacheContainerStorageBytes();
}

#endif
