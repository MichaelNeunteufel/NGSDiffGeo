// Implementation fragment included only by kforms_compact_exterior.inc.
// Aggregate benchmark/testing accounting that needs every compact table type.

// Static container accounting spans every compact table family. It is
// kept here, after all table types are visible in this translation unit.
std::array<size_t, 7> CompactFormCacheContainerStorageBreakdown()
{
    constexpr size_t extent = MAX_SPACE_DIM + 1;
    constexpr size_t basis_entries = extent * extent;
    constexpr size_t expansion_entries =
        extent * extent * extent;
    constexpr size_t trace_entries =
        extent * extent * extent;
    constexpr size_t wedge_entries =
        extent * extent * extent * extent * extent;
    constexpr size_t exterior_entries = extent * extent;
    constexpr size_t induced_metric_entries = extent * extent;
    constexpr size_t hodge_map_entries = extent * extent;
    return {
        basis_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<const FormBasis>)),
        expansion_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<
                    const CompactDoubleFormExpansionTable>)),
        trace_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<const CompactDoubleTraceTable>)),
        wedge_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<const CompactWedgeTable>)),
        exterior_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<
                    const CompactExteriorDerivativeTable>)),
        induced_metric_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<const InducedFormMetricTable>)),
        hodge_map_entries *
            (sizeof(std::once_flag) +
             sizeof(shared_ptr<const CompactHodgeMapTable>))};
}

size_t CompactFormCacheContainerStorageBytes()
{
    const auto breakdown =
        CompactFormCacheContainerStorageBreakdown();
    return std::accumulate(
        breakdown.begin(), breakdown.end(), size_t(0));
}
