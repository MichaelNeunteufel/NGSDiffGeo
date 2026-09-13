#ifndef NGSDIFFGEO_KFORMS_BASIS_HPP
#define NGSDIFFGEO_KFORMS_BASIS_HPP

#include "kforms.hpp"

#include <unordered_map>
#include <vector>

namespace ngfem::kforms_internal
{
    int DenseFormComponentCount(int dim, int degree);

    struct DenseFormBasisEntry
    {
        int independent_index = -1;
        int sign = 0;
    };

    struct FormBasisShuffle
    {
        std::vector<int> left_positions;
        std::vector<int> right_positions;
        int sign = 1;
    };

    /// Immutable canonical component and shuffle tables for one (dim, degree)
    class FormBasis
    {
        int dim;
        int degree;
        std::vector<std::vector<int>> canonical_indices;
        std::unordered_map<int, int> canonical_inverse;
        std::vector<DenseFormBasisEntry> dense_entries;
        std::vector<std::vector<FormBasisShuffle>> shuffles;

    public:
        FormBasis(int adim, int adegree);

        int DimensionOfSpace() const { return dim; }
        int Degree() const { return degree; }
        size_t IndependentSize() const { return canonical_indices.size(); }
        size_t DenseSize() const { return dense_entries.size(); }
        const std::vector<DenseFormBasisEntry> &DenseEntries() const
        {
            return dense_entries;
        }
        const std::vector<int> &CanonicalIndex(size_t index) const;
        int CompactIndex(const std::vector<int> &canonical_index) const;
        DenseFormBasisEntry DenseEntry(size_t index) const;
        const std::vector<FormBasisShuffle> &Shuffles(int left_degree) const;
        /// Lower bound for this object and its owned container payloads.
        /// Allocator bookkeeping and unordered-map node links are excluded
        size_t OwnedStorageBytesLowerBound() const;
    };

    shared_ptr<const FormBasis> GetFormBasis(int dim, int degree);
}

#endif
