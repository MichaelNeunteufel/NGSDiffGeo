#include "kforms_basis.hpp"
#include "kforms_diagnostics.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <mutex>
#include <utility>

namespace ngfem::kforms_internal
{
    int DenseFormComponentCount(int dim, int degree)
    {
        if (dim < 1 || dim > MAX_SPACE_DIM || degree < 0 || degree > dim)
            throw Exception("FormBasis: require 1 <= dim <= " +
                            ToString(MAX_SPACE_DIM) + " and 0 <= degree <= dim");
        size_t value = 1;
        for (int i = 0; i < degree; ++i)
        {
            if (value > size_t(std::numeric_limits<int>::max()) / size_t(dim))
                throw Exception("FormBasis: full component count exceeds int");
            value *= size_t(dim);
        }
        return int(value);
    }

    FormBasis::FormBasis(int adim, int adegree)
        : dim(adim), degree(adegree)
    {
        const int dense_size = DenseFormComponentCount(dim, degree);
        dense_entries.resize(size_t(dense_size));

        std::vector<int> current(static_cast<size_t>(degree));
        function<void(int, int)> enumerate = [&](int slot, int first)
        {
            if (slot == degree)
            {
                int encoded = 0;
                int stride = 1;
                for (int axis : current)
                {
                    encoded += axis * stride;
                    stride *= dim;
                }
                canonical_inverse[encoded] = int(canonical_indices.size());
                canonical_indices.push_back(current);
                return;
            }
            for (int axis = first; axis <= dim - (degree - slot); ++axis)
            {
                current[size_t(slot)] = axis;
                enumerate(slot + 1, axis + 1);
            }
        };
        enumerate(0, 0);

        std::vector<int> axes(static_cast<size_t>(degree));
        for (int dense = 0; dense < dense_size; ++dense)
        {
            int remainder = dense;
            bool repeated = false;
            int inversions = 0;
            for (int slot = 0; slot < degree; ++slot)
            {
                axes[size_t(slot)] = remainder % dim;
                remainder /= dim;
                for (int previous = 0; previous < slot; ++previous)
                {
                    repeated |= axes[size_t(previous)] == axes[size_t(slot)];
                    inversions += axes[size_t(previous)] > axes[size_t(slot)];
                }
            }
            if (repeated)
                continue;

            auto sorted = axes;
            std::sort(sorted.begin(), sorted.end());
            int encoded = 0;
            int stride = 1;
            for (int axis : sorted)
            {
                encoded += axis * stride;
                stride *= dim;
            }
            dense_entries[size_t(dense)] = {
                canonical_inverse.at(encoded), inversions % 2 ? -1 : 1};
        }

        shuffles.resize(size_t(degree + 1));
        for (int left_degree = 0; left_degree <= degree; ++left_degree)
        {
            std::vector<int> selected;
            function<void(int)> enumerate_subsets = [&](int first)
            {
                if (int(selected.size()) == left_degree)
                {
                    FormBasisShuffle shuffle;
                    shuffle.left_positions = selected;
                    int inversions = 0;
                    size_t selected_index = 0;
                    for (int position = 0; position < degree; ++position)
                    {
                        if (selected_index < selected.size() &&
                            selected[selected_index] == position)
                        {
                            inversions += position - int(selected_index);
                            ++selected_index;
                        }
                        else
                            shuffle.right_positions.push_back(position);
                    }
                    shuffle.sign = inversions % 2 ? -1 : 1;
                    shuffles[size_t(left_degree)].push_back(std::move(shuffle));
                    return;
                }
                const int missing = left_degree - int(selected.size());
                for (int position = first; position <= degree - missing; ++position)
                {
                    selected.push_back(position);
                    enumerate_subsets(position + 1);
                    selected.pop_back();
                }
            };
            enumerate_subsets(0);
        }
    }

    const std::vector<int> &FormBasis::CanonicalIndex(size_t index) const
    {
        if (index >= canonical_indices.size())
            throw Exception("FormBasis: canonical index out of range");
        return canonical_indices[index];
    }

    int FormBasis::CompactIndex(
        const std::vector<int> &canonical_index) const
    {
        if (canonical_index.size() != size_t(degree))
            throw Exception("FormBasis: canonical index has the wrong degree");
        int encoded = 0;
        int stride = 1;
        for (int slot = 0; slot < degree; ++slot)
        {
            const int axis = canonical_index[size_t(slot)];
            if (axis < 0 || axis >= dim ||
                (slot > 0 && canonical_index[size_t(slot - 1)] >= axis))
                throw Exception("FormBasis: index is not canonical");
            encoded += axis * stride;
            stride *= dim;
        }
        return canonical_inverse.at(encoded);
    }

    DenseFormBasisEntry FormBasis::DenseEntry(size_t index) const
    {
        if (index >= dense_entries.size())
            throw Exception("FormBasis: dense index out of range");
        return dense_entries[index];
    }

    const std::vector<FormBasisShuffle> &FormBasis::Shuffles(
        int left_degree) const
    {
        if (left_degree < 0 || left_degree > degree)
            throw Exception("FormBasis: shuffle degree out of range");
        return shuffles[size_t(left_degree)];
    }

    size_t FormBasis::OwnedStorageBytesLowerBound() const
    {
        size_t bytes = sizeof(FormBasis) +
                       canonical_indices.capacity() *
                           sizeof(std::vector<int>) +
                       canonical_inverse.size() *
                           sizeof(std::pair<const int, int>) +
                       canonical_inverse.bucket_count() * sizeof(void *) +
                       dense_entries.capacity() * sizeof(DenseFormBasisEntry) +
                       shuffles.capacity() *
                           sizeof(std::vector<FormBasisShuffle>);
        for (const auto &index : canonical_indices)
            bytes += index.capacity() * sizeof(int);
        for (const auto &family : shuffles)
        {
            bytes += family.capacity() * sizeof(FormBasisShuffle);
            for (const auto &shuffle : family)
                bytes += (shuffle.left_positions.capacity() +
                          shuffle.right_positions.capacity()) *
                         sizeof(int);
        }
        return bytes;
    }

    shared_ptr<const FormBasis> GetFormBasis(int dim, int degree)
    {
        DenseFormComponentCount(dim, degree);
        constexpr size_t size = MAX_SPACE_DIM + 1;
        static std::array<std::array<std::once_flag, size>, size> flags;
        static std::array<std::array<shared_ptr<const FormBasis>, size>, size>
            bases;
        std::call_once(flags[size_t(dim)][size_t(degree)], [&]
                       { bases[size_t(dim)][size_t(degree)] =
                             make_shared<const FormBasis>(dim, degree); });
        return bases[size_t(dim)][size_t(degree)];
    }

    size_t FormBasisStorageBytesLowerBound(int dim, int degree)
    {
        return GetFormBasis(dim, degree)->OwnedStorageBytesLowerBound();
    }

}
