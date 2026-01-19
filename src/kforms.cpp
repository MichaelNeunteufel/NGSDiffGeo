#include "kforms.hpp"
#include "riemannian_manifold.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <mutex>
#include <vector>

namespace ngfem
{
    namespace
    {
        const std::vector<std::array<int, 4>> &GeneratePermutations(int rank)
        {
            if (rank < 0 || rank > 4)
                throw Exception("GeneratePermutations: rank must be in [0, 4]");

            static std::array<std::vector<std::array<int, 4>>, 5> cached_perms = []()
            {
                std::array<std::vector<std::array<int, 4>>, 5> result;
                for (int r = 0; r <= 4; ++r)
                {
                    std::vector<int> current(r);
                    std::iota(current.begin(), current.end(), 0);

                    do
                    {
                        std::array<int, 4> p = {0, 1, 2, 3};
                        for (int i = 0; i < r; ++i)
                            p[i] = current[i];
                        result[r].push_back(p);
                    } while (std::next_permutation(current.begin(), current.end()));
                }
                return result;
            }();

            return cached_perms[rank];
        }

        int PermutationSign(const std::array<int, 4> &perm, int rank)
        {
            switch (rank)
            {
            case 0:
            case 1:
                return 1;
            case 2:
                return (perm[0] > perm[1]) ? -1 : 1;
            case 3:
            {
                int inv = 0;
                if (perm[0] > perm[1])
                    inv++;
                if (perm[0] > perm[2])
                    inv++;
                if (perm[1] > perm[2])
                    inv++;
                return (inv & 1) ? -1 : 1;
            }
            case 4:
            {
                int inv = 0;
                if (perm[0] > perm[1])
                    inv++;
                if (perm[0] > perm[2])
                    inv++;
                if (perm[0] > perm[3])
                    inv++;
                if (perm[1] > perm[2])
                    inv++;
                if (perm[1] > perm[3])
                    inv++;
                if (perm[2] > perm[3])
                    inv++;
                return (inv & 1) ? -1 : 1;
            }
            default:
            {
                int inversions = 0;
                for (int i = 0; i < rank; ++i)
                    for (int j = i + 1; j < rank; ++j)
                        if (perm[i] > perm[j])
                            inversions++;
                return (inversions % 2 == 0) ? 1 : -1;
            }
            }
        }

        struct PermOrderCacheEntry
        {
            bool ready = false;
            std::vector<std::vector<int>> orders;
        };

        const std::vector<std::vector<int>> &GetBlockPermutationOrders(int rank_total, int block_start, int block_len,
                                                                        const std::vector<std::array<int, 4>> &perms)
        {
            static std::mutex cache_mutex;
            static std::array<std::array<std::array<PermOrderCacheEntry, 5>, 9>, 9> cache;

            std::lock_guard<std::mutex> lock(cache_mutex);
            auto &entry = cache[size_t(rank_total)][size_t(block_start)][size_t(block_len)];
            if (!entry.ready)
            {
                entry.orders.resize(perms.size());
                for (size_t p = 0; p < perms.size(); ++p)
                {
                    auto &order = entry.orders[p];
                    order.resize(size_t(rank_total));
                    for (int i = 0; i < rank_total; ++i)
                        order[size_t(i)] = i;
                    for (int i = 0; i < block_len; ++i)
                        order[size_t(block_start + i)] = block_start + perms[p][size_t(i)];
                }
                entry.ready = true;
            }
            return entry.orders;
        }

        std::string FreshSignature(std::string_view used, int count)
        {
            if (count < 0)
                throw Exception("FreshSignature: count must be non-negative");
            if (count == 0)
                return std::string();
            std::string out;
            out.reserve(size_t(count));
            for (char c : SIGNATURE)
            {
                if (used.find(c) == std::string_view::npos)
                {
                    out.push_back(c);
                    if (int(out.size()) == count)
                        break;
                }
            }
            if (int(out.size()) != count)
                throw Exception("FreshSignature: not enough signature labels available");
            return out;
        }

        shared_ptr<CoefficientFunction> BlockHodgeStar(shared_ptr<TensorFieldCoefficientFunction> tf, int block_start, int block_len, int n, const RiemannianManifold &M)
        {
            if (block_len < 0 || block_start < 0)
                throw Exception("BlockHodgeStar: invalid block parameters");
            if (block_len > n)
                throw Exception("BlockHodgeStar: block degree exceeds dimension");

            shared_ptr<TensorFieldCoefficientFunction> raised = tf;
            for (int i = 0; i < block_len; ++i)
                raised = M.Raise(raised, size_t(block_start + i));

            auto eps = M.GetLeviCivitaSymbol(true);
            std::string sig = raised->GetSignature();
            if (block_start + block_len > int(sig.size()))
                throw Exception("BlockHodgeStar: block range out of bounds");
            if (sig.empty())
                return raised * eps;

            std::string pre = sig.substr(0, size_t(block_start));
            std::string block = sig.substr(size_t(block_start), size_t(block_len));
            std::string post = sig.substr(size_t(block_start + block_len));
            std::string new_block = FreshSignature(sig, n - block_len);

            std::string eps_sig = block + new_block;
            std::string out_sig = pre + new_block + post;

            std::string eins = sig + "," + eps_sig + "->" + out_sig;
            auto contracted = EinsumCF(eins, {raised, eps});
            double scale = 1.0 / double(Factorial(block_len));
            return scale * contracted;
        }

    } // namespace

    shared_ptr<CoefficientFunction> BlockAlternationByPermutationCF(shared_ptr<CoefficientFunction> T, int rank_total, int block_start, int block_len)
    {
        if (!T)
            throw Exception("BlockAlternationByPermutationCF: input coefficient is null");
        if (rank_total < 0 || rank_total > 8)
            throw Exception("BlockAlternationByPermutationCF: only ranks 0-8 supported");
        if (block_len < 0 || block_len > 4)
            throw Exception("BlockAlternationByPermutationCF: block size must be in [0, 4]");
        if (block_start < 0 || block_start + block_len > rank_total)
            throw Exception("BlockAlternationByPermutationCF: block range out of bounds");
        if (block_len <= 1)
            return T;

        shared_ptr<TensorFieldCoefficientFunction> tf;
        if (auto ttf = dynamic_pointer_cast<TensorFieldCoefficientFunction>(T))
        {
            tf = ttf;
            for (char c : tf->GetCovariantIndices())
                if (c != '1')
                    throw Exception("BlockAlternationByPermutationCF: only covariant tensors are supported");
        }
        else
            tf = TensorFieldCF(T, std::string(size_t(rank_total), '1'));

        if (int(tf->Dimensions().Size()) != rank_total)
            throw Exception("BlockAlternationByPermutationCF: tensor rank mismatch");

        const auto &perms = GeneratePermutations(block_len);
        if ((int)perms.size() != Factorial(block_len))
            throw Exception("BlockAlternationByPermutationCF: permutation generation broken");

        shared_ptr<CoefficientFunction> accum;
        const auto &orders = GetBlockPermutationOrders(rank_total, block_start, block_len, perms);
        for (size_t p = 0; p < perms.size(); ++p)
        {
            shared_ptr<CoefficientFunction> term = PermuteTensorCF(tf, orders[p]);
            int sign = PermutationSign(perms[p], block_len);
            if (sign == -1)
                term = (-1.0) * term;
            accum = accum ? (accum + term) : term;
        }
        return accum;
    }

    KFormCoefficientFunction::KFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, uint8_t ak, uint8_t adim)
        : TensorFieldCoefficientFunction(ac1, std::string(size_t(ak), '1')), degree(ak), dim(adim)
    {
        if (!((adim >= 1 && adim <= 4) || (adim == 0 && ak == 0)))
            throw Exception("KFormCF: dim must be in {1,2,3,4} (or 0 for scalar forms)");
        if (ak > 8)
            throw Exception("KFormCF: only ranks up to 8 are supported");

        const auto &dims = ac1->Dimensions();
        if (dims.Size() != degree)
            throw Exception("KFormCF: underlying coefficient must have rank " + ToString(int(degree)));
        for (auto d : dims)
            if (dim > 0 && d != dim)
            {
                throw Exception("KFormCF: tensor dimensions must all equal dim. dim = " + ToString(int(dim)) + ", but found dimension " + ToString(int(d)));
            }

        if (dim > 0 && degree > dim && !ac1->IsZeroCF())
            throw Exception("KFormCF: degree exceeds dimension (only zero forms allowed in that case)");

        auto meta = Meta();
        if (meta.rank != degree)
            throw Exception("KFormCF: rank mismatch");
        uint64_t expected_covmask = (degree == 0) ? 0 : ((uint64_t(1) << degree) - 1);
        if (meta.covmask != expected_covmask)
            throw Exception("KFormCF: k-forms must be fully covariant");
    }

    shared_ptr<CoefficientFunction>
    KFormCoefficientFunction::Transform(CoefficientFunction::T_Transform &transformation) const
    {
        auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        if (transformation.cache.count(thisptr))
            return transformation.cache[thisptr];
        if (transformation.replace.count(thisptr))
            return transformation.replace[thisptr];
        auto newcf = KFormCF(GetCoefficients()->Transform(transformation), degree, dim);
        transformation.cache[thisptr] = newcf;
        return newcf;
    }

    shared_ptr<CoefficientFunction> KFormCoefficientFunction::Diff(const CoefficientFunction *var,
                                                                   shared_ptr<CoefficientFunction> dir) const
    {
        if (this == var)
            return dir;
        return KFormCF(GetCoefficients()->Diff(var, dir), degree, dim);
    }

    shared_ptr<CoefficientFunction> KFormCoefficientFunction::DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const
    {
        auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        if (cache.find(thisptr) != cache.end())
            return cache[thisptr];

        if (this == var)
            return IdentityCF(this->Dimensions());

        auto res = KFormCF(GetCoefficients()->DiffJacobi(var, cache), degree, dim);
        cache[thisptr] = res;
        return res;
    }

    DoubleFormCoefficientFunction::DoubleFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, uint8_t ap, uint8_t aq, uint8_t adim)
        : TensorFieldCoefficientFunction(ac1, std::string(size_t(ap + aq), '1')), degree_left(ap), degree_right(aq), dim(adim)
    {
        if (!((adim >= 1 && adim <= 4) || (adim == 0 && ap == 0 && aq == 0)))
            throw Exception("DoubleFormCF: dim must be in {1,2,3,4} (or 0 for scalar forms)");
        if (ap + aq > 8)
            throw Exception("DoubleFormCF: only ranks up to 8 are supported");

        const auto &dims = ac1->Dimensions();
        if (dims.Size() != ap + aq)
            throw Exception("DoubleFormCF: underlying coefficient must have rank " + ToString(int(ap + aq)));
        for (auto d : dims)
            if (dim > 0 && d != dim)
            {
                throw Exception("DoubleFormCF: tensor dimensions must all equal dim. dim = " + ToString(int(dim)) + ", but found dimension " + ToString(int(d)));
            }

        if (dim > 0 && (ap > dim || aq > dim) && !ac1->IsZeroCF())
            throw Exception("DoubleFormCF: degree exceeds dimension (only zero forms allowed in that case)");

        auto meta = Meta();
        if (meta.rank != ap + aq)
            throw Exception("DoubleFormCF: rank mismatch");
        uint64_t expected_covmask = (meta.rank == 0) ? 0 : ((uint64_t(1) << meta.rank) - 1);
        if (meta.covmask != expected_covmask)
            throw Exception("DoubleFormCF: double-forms must be fully covariant");
    }

    shared_ptr<CoefficientFunction>
    DoubleFormCoefficientFunction::Transform(CoefficientFunction::T_Transform &transformation) const
    {
        auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        if (transformation.cache.count(thisptr))
            return transformation.cache[thisptr];
        if (transformation.replace.count(thisptr))
            return transformation.replace[thisptr];
        auto newcf = DoubleFormCF(GetCoefficients()->Transform(transformation), degree_left, degree_right, dim);
        transformation.cache[thisptr] = newcf;
        return newcf;
    }

    shared_ptr<CoefficientFunction> DoubleFormCoefficientFunction::Diff(const CoefficientFunction *var,
                                                                        shared_ptr<CoefficientFunction> dir) const
    {
        if (this == var)
            return dir;
        return DoubleFormCF(GetCoefficients()->Diff(var, dir), degree_left, degree_right, dim);
    }

    shared_ptr<CoefficientFunction> DoubleFormCoefficientFunction::DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const
    {
        auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        if (cache.find(thisptr) != cache.end())
            return cache[thisptr];

        if (this == var)
            return IdentityCF(this->Dimensions());

        auto res = DoubleFormCF(GetCoefficients()->DiffJacobi(var, cache), degree_left, degree_right, dim);
        cache[thisptr] = res;
        return res;
    }

    ScalarFieldCoefficientFunction::ScalarFieldCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
        : KFormCoefficientFunction(cf, 0, uint8_t(dim))
    {
        if (cf->Dimensions().Size() != 0)
            throw Exception("ScalarFieldCF: input must be scalar");
    }

    OneFormCoefficientFunction::OneFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
        : KFormCoefficientFunction(cf, 1, uint8_t(dim))
    {
        if (cf->Dimensions().Size() != 1)
            throw Exception("OneFormCF: input must be vector-valued");
    }

    TwoFormCoefficientFunction::TwoFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
        : KFormCoefficientFunction(cf, 2, uint8_t(dim))
    {
        if (cf->Dimensions().Size() != 2)
            throw Exception("TwoFormCF: input must be rank-2");
    }

    ThreeFormCoefficientFunction::ThreeFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
        : KFormCoefficientFunction(cf, 3, uint8_t(dim))
    {
        if (cf->Dimensions().Size() != 3)
            throw Exception("ThreeFormCF: input must be rank-3");
    }

    class AlternationCoefficientFunction : public T_CoefficientFunction<AlternationCoefficientFunction>
    {
        shared_ptr<CoefficientFunction> c1;
        int rank;
        int dim;
        std::vector<std::array<int, 4>> perms;
        std::vector<int> signs;
        std::vector<int> valid_indices;
        std::vector<int> lin_table;

    public:
        AlternationCoefficientFunction(shared_ptr<CoefficientFunction> ac1, int arank, int adim)
            : T_CoefficientFunction<AlternationCoefficientFunction>(ac1->Dimension(), ac1->IsComplex()), c1(ac1), rank(arank), dim(adim)
        {
            if (rank < 0 || rank > 4)
                throw Exception("AlternationCF: only ranks 0-4 supported");
            if (dim < 1 || dim > 4)
                throw Exception("AlternationCF: dim must be in {1,2,3,4}");
            if (rank > dim)
                throw Exception("AlternationCF: rank exceeds dimension");

            if (c1->Dimensions().Size() != size_t(rank))
                throw Exception("AlternationCF: tensor rank mismatch");
            for (auto d : c1->Dimensions())
                if (d != dim)
                    throw Exception("AlternationCF: tensor dimensions must equal dim");

            this->SetDimensions(c1->Dimensions());

            perms = GeneratePermutations(rank);
            if ((int)perms.size() != Factorial(rank))
                throw Exception("AlternationCF: permutation generation broken");

            signs.resize(perms.size());
            for (size_t i = 0; i < perms.size(); ++i)
                signs[i] = PermutationSign(perms[i], rank);

            int comp_dim = this->Dimension();
            valid_indices.reserve(comp_dim);
            lin_table.clear();
            lin_table.reserve(size_t(comp_dim) * perms.size());

            std::array<int, 4> multi = {0, 0, 0, 0};
            for (int idx = 0; idx < comp_dim; ++idx)
            {
                int rem = idx;
                bool repeated = false;
                for (int j = 0; j < rank; ++j)
                {
                    multi[j] = rem % dim;
                    rem /= dim;
                }
                for (int a = 0; a < rank && !repeated; ++a)
                    for (int b = a + 1; b < rank; ++b)
                        if (multi[a] == multi[b])
                        {
                            repeated = true;
                            break;
                        }

                if (repeated)
                    continue;

                valid_indices.push_back(idx);
                for (size_t p = 0; p < perms.size(); ++p)
                {
                    int lin = 0;
                    int stride = 1;
                    for (int j = 0; j < rank; ++j)
                    {
                        lin += multi[perms[p][j]] * stride;
                        stride *= dim;
                    }
                    lin_table.push_back(lin);
                }
            }
        }

        virtual string GetDescription() const override
        {
            return "AlternationCF";
        }

        int Rank() const { return rank; }
        int DimSpace() const { return dim; }

        auto GetCArgs() const { return tuple{c1}; }

        void DoArchive(Archive &ar) override
        {
        }
        virtual void TraverseTree(const function<void(CoefficientFunction &)> &func) override
        {
            c1->TraverseTree(func);
            func(*this);
        }

        virtual Array<shared_ptr<CoefficientFunction>> InputCoefficientFunctions() const override
        {
            return Array<shared_ptr<CoefficientFunction>>({c1});
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            Vector<AutoDiffDiff<1, NonZero>> input(values.Size());
            c1->NonZeroPattern(ud, input);

            values = AutoDiffDiff<1, NonZero>(false);
            for (size_t vi = 0; vi < valid_indices.size(); ++vi)
            {
                int idx = valid_indices[vi];
                auto accum = AutoDiffDiff<1, NonZero>(false);
                const size_t base = vi * perms.size();
                for (size_t p = 0; p < perms.size(); ++p)
                    accum = accum + input(lin_table[base + p]);
                values(idx) = accum;
            }
        }

        shared_ptr<CoefficientFunction>
        Transform(CoefficientFunction::T_Transform &transformation) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (transformation.cache.count(thisptr))
                return transformation.cache[thisptr];
            if (transformation.replace.count(thisptr))
                return transformation.replace[thisptr];
            auto newcf = AlternationCF(c1->Transform(transformation), rank, dim);
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        using T_CoefficientFunction<AlternationCoefficientFunction>::Evaluate;

        virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            throw Exception("AlternationCF:: scalar evaluate called");
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir, BareSliceMatrix<T, ORD> values) const
        {
            int comp_dim = this->Dimension();

            Array<T> temp(comp_dim * mir.Size());
            FlatMatrix<T, ORD> input(comp_dim, mir.Size(), temp.Data());
            c1->Evaluate(mir, input);

            values = T(0);
            for (size_t ip = 0; ip < mir.Size(); ++ip)
            {
                for (size_t vi = 0; vi < valid_indices.size(); ++vi)
                {
                    int idx = valid_indices[vi];
                    T accum = T(0);
                    const size_t base = vi * perms.size();
                    for (size_t p = 0; p < perms.size(); ++p)
                    {
                        int lin = lin_table[base + p];
                        accum += T(signs[p]) * input(lin, ip);
                    }
                    values(idx, ip) = accum;
                }
            }
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            this->T_Evaluate(ir, values);
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return AlternationCF(c1->Diff(var, dir), rank, dim);
        }

        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            auto res = AlternationCF(c1->DiffJacobi(var, cache), rank, dim);
            cache[thisptr] = res;
            return res;
        }

        virtual bool IsZeroCF() const override { return c1->IsZeroCF(); }
    };

    shared_ptr<CoefficientFunction> AlternationCF(shared_ptr<CoefficientFunction> T, int rank, int dim)
    {
        return make_shared<AlternationCoefficientFunction>(T, rank, dim);
    }

    shared_ptr<KFormCoefficientFunction> KFormCF(shared_ptr<CoefficientFunction> cf, int k, int dim)
    {
        if (k < 0)
            throw Exception("KFormCF: degree must be non-negative");
        if (k > 8)
            throw Exception("KFormCF: only ranks up to 8 are supported");

        auto deduce_dim = [&]()
        {
            if (dim > 0)
                return dim;
            if (cf->Dimensions().Size() > 0)
                return int(cf->Dimensions()[0]);
            return 0;
        };

        int used_dim = deduce_dim();
        if (k == 0 && used_dim <= 0)
            throw Exception("KFormCF: dim must be provided for scalar forms");

        if (k == 0)
        {
            if (auto sf = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(cf))
                return sf;
            return make_shared<ScalarFieldCoefficientFunction>(cf, used_dim);
        }
        if (k == 1)
        {
            if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(cf))
                return of;
            return make_shared<OneFormCoefficientFunction>(cf, used_dim);
        }
        if (k == 2)
        {
            if (auto tf = dynamic_pointer_cast<TwoFormCoefficientFunction>(cf))
                return tf;
            return make_shared<TwoFormCoefficientFunction>(cf, used_dim);
        }
        if (k == 3)
        {
            if (auto tf = dynamic_pointer_cast<ThreeFormCoefficientFunction>(cf))
                return tf;
            return make_shared<ThreeFormCoefficientFunction>(cf, used_dim);
        }

        return make_shared<KFormCoefficientFunction>(cf, uint8_t(k), uint8_t(used_dim));
    }

    shared_ptr<DoubleFormCoefficientFunction> DoubleFormCF(shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
    {
        if (p < 0 || q < 0)
            throw Exception("DoubleFormCF: degrees must be non-negative");
        if (p + q > 8)
            throw Exception("DoubleFormCF: only ranks up to 8 are supported");

        auto deduce_dim = [&]()
        {
            if (dim > 0)
                return dim;
            if (cf->Dimensions().Size() > 0)
                return int(cf->Dimensions()[0]);
            return 0;
        };

        int used_dim = deduce_dim();
        if (p + q == 0 && used_dim <= 0)
            throw Exception("DoubleFormCF: dim must be provided for scalar forms");

        if (auto df = dynamic_pointer_cast<DoubleFormCoefficientFunction>(cf))
        {
            if (df->LeftDegree() == p && df->RightDegree() == q && (used_dim <= 0 || df->DimensionOfSpace() == used_dim))
                return df;
        }

        return make_shared<DoubleFormCoefficientFunction>(cf, uint8_t(p), uint8_t(q), uint8_t(used_dim));
    }

    shared_ptr<ScalarFieldCoefficientFunction> ScalarFieldCF(shared_ptr<CoefficientFunction> cf, int dim)
    {
        if (dim <= 0)
            throw Exception("ScalarFieldCF: dim must be provided for scalar forms");
        int used_dim = dim;
        if (cf->Dimension() > 1)
            throw Exception("ScalarFieldCF: input coefficient must be scalar valued");

        if (auto sf = dynamic_pointer_cast<ScalarFieldCoefficientFunction>(cf))
            return sf;
        return make_shared<ScalarFieldCoefficientFunction>(cf, used_dim);
    }

    shared_ptr<OneFormCoefficientFunction> OneFormCF(shared_ptr<CoefficientFunction> cf)
    {
        if (cf->Dimensions().Size() != 1)
            throw Exception("OneFormCF: input coefficient must be vector valued");

        int used_dim = cf->Dimension();
        if (auto of = dynamic_pointer_cast<OneFormCoefficientFunction>(cf))
            return of;
        return make_shared<OneFormCoefficientFunction>(cf, used_dim);
    }

    shared_ptr<TwoFormCoefficientFunction> TwoFormCF(shared_ptr<CoefficientFunction> cf, int dim)
    {
        int used_dim = dim;
        if (used_dim <= 0)
            used_dim = (cf->Dimensions().Size() > 0) ? int(cf->Dimensions()[0]) : 1;
        if (auto tf = dynamic_pointer_cast<TwoFormCoefficientFunction>(cf))
            return tf;
        return make_shared<TwoFormCoefficientFunction>(cf, used_dim);
    }

    shared_ptr<ThreeFormCoefficientFunction> ThreeFormCF(shared_ptr<CoefficientFunction> cf, int dim)
    {
        int used_dim = dim;
        if (used_dim <= 0)
            used_dim = (cf->Dimensions().Size() > 0) ? int(cf->Dimensions()[0]) : 1;
        if (auto tf = dynamic_pointer_cast<ThreeFormCoefficientFunction>(cf))
            return tf;
        return make_shared<ThreeFormCoefficientFunction>(cf, used_dim);
    }

    shared_ptr<KFormCoefficientFunction> ZeroKForm(int k, int dim)
    {
        Array<int> dims;
        for (int i = 0; i < k; ++i)
            dims.Append(dim);
        auto zero_cf = ZeroCF(dims);
        return KFormCF(zero_cf, k, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> ZeroDoubleForm(int p, int q, int dim)
    {
        Array<int> dims;
        for (int i = 0; i < p + q; ++i)
            dims.Append(dim);
        auto zero_cf = ZeroCF(dims);
        return DoubleFormCF(zero_cf, p, q, dim);
    }

    shared_ptr<KFormCoefficientFunction> Wedge(shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b)
    {
        if (a->DimensionOfSpace() != b->DimensionOfSpace())
            throw Exception("Wedge: input forms must have the same dimension of space");
        int dim = a->DimensionOfSpace();
        int k = a->Degree();
        int l = b->Degree();
        if (k + l > dim)
            return ZeroKForm(k + l, dim);
        if (k == 0 || l == 0)
            return KFormCF(a->GetCoefficients() * b->GetCoefficients(), k + l, dim);

        auto T = TensorProduct(a, b);
        auto alt = AlternationCF(T, k + l, dim);
        double scale = 1.0 / double(Factorial(k) * Factorial(l));
        auto out = scale * alt;
        return KFormCF(out, k + l, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> Wedge(shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
    {
        if (a->DimensionOfSpace() != b->DimensionOfSpace())
            throw Exception("Wedge: input double-forms must have the same dimension of space");
        int dim = a->DimensionOfSpace();
        int p = a->LeftDegree();
        int q = a->RightDegree();
        int r = b->LeftDegree();
        int s = b->RightDegree();

        if (p + r > dim || q + s > dim)
            return ZeroDoubleForm(p + r, q + s, dim);

        auto T = TensorProduct(a, b);
        int total = p + q + r + s;
        std::vector<int> order;
        order.reserve(total);
        for (int i = 0; i < p; ++i)
            order.push_back(i);
        for (int i = 0; i < r; ++i)
            order.push_back(p + q + i);
        for (int i = 0; i < q; ++i)
            order.push_back(p + i);
        for (int i = 0; i < s; ++i)
            order.push_back(p + q + r + i);

        auto reordered = PermuteTensorCF(T, order);
        shared_ptr<CoefficientFunction> alt_left = reordered;
        if (p + r > 1)
            alt_left = BlockAlternationByPermutationCF(reordered, total, 0, p + r);

        shared_ptr<CoefficientFunction> alt_both = alt_left;
        if (q + s > 1)
            alt_both = BlockAlternationByPermutationCF(alt_left, total, p + r, q + s);

        double scale = 1.0 / double(Factorial(p) * Factorial(r) * Factorial(q) * Factorial(s));
        auto out = scale * alt_both;
        return DoubleFormCF(out, p + r, q + s, dim);
    }

    shared_ptr<KFormCoefficientFunction> ExteriorDerivative(shared_ptr<KFormCoefficientFunction> a)
    {
        int dim = a->DimensionOfSpace();
        int k = a->Degree();
        if (k + 1 > dim)
            return ZeroKForm(k + 1, dim);

        auto G = GradCF(a->GetCoefficients(), dim);
        auto alt = AlternationCF(G, k + 1, dim);

        double scale = 1.0 / double(Factorial(k));
        auto out = scale * alt;
        return KFormCF(out, k + 1, dim);
    }

    shared_ptr<KFormCoefficientFunction> HodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb)
    {
        int ambient_dim = M.Dimension();
        if (vb == BBND)
            throw Exception("HodgeStar: not implemented for BBND (codimension-2) yet");

        int n = (vb == VOL) ? ambient_dim : ambient_dim - 1;
        int k = a->Degree();
        if (k > n)
            throw Exception("HodgeStar: form degree exceeds manifold dimension");

        if (a->IsZeroCF())
            return ZeroKForm(n - k, vb == VOL ? n : ambient_dim);

        // Boundary star via ambient star followed by contraction with the unit normal:
        //   *star_bnd(alpha) = i_nu (star_vol(alpha))*
        if (vb == BND)
        {
            auto star_vol = HodgeStar(a, M, VOL);
            auto normal = M.GetNV();

            auto contracted = M.Contraction(star_vol, normal); // reduce degree by 1
            return KFormCF(contracted->GetCoefficients(), n - k, ambient_dim);
        }

        shared_ptr<TensorFieldCoefficientFunction> raised = a;
        for (int i = 0; i < k; ++i)
            raised = M.Raise(raised, i);

        auto eps = M.GetLeviCivitaSymbol(true);

        std::string alpha_sig = raised->GetSignature(); // length k
        std::string eps_sig = alpha_sig + SIGNATURE.substr(k, n - k);
        std::string out_sig = SIGNATURE.substr(k, n - k); // empty if n==k

        std::string eins;
        Array<shared_ptr<CoefficientFunction>> args;
        if (alpha_sig.empty())
        {
            eins = eps_sig + "->" + out_sig;
            args = {a * eps};
        }
        else
        {
            eins = alpha_sig + "," + eps_sig + "->" + out_sig;
            args = {raised, eps};
        }

        auto contracted = EinsumCF(eins, args);
        auto scaled = 1 / double(Factorial(k)) * contracted;

        return KFormCF(scaled, n - k, n);
    }

    shared_ptr<KFormCoefficientFunction> InverseHodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb)
    {
        int n = (vb == VOL) ? M.Dimension() : M.Dimension() - 1;
        int k = a->Degree();
        int exponent = k * (n - k);
        int sign = (exponent % 2 == 0) ? 1 : -1;

        auto star = HodgeStar(a, M, vb);
        if (sign == 1)
            return star;
        return KFormCF((-1.0) * star->GetCoefficients(), n - k, vb == VOL ? n : M.Dimension());
    }

    shared_ptr<DoubleFormCoefficientFunction> HodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb)
    {
        int ambient_dim = M.Dimension();
        if (vb == BBND)
            throw Exception("HodgeStar (double-form): not implemented for BBND (codimension-2) yet");

        int n = (vb == VOL) ? ambient_dim : ambient_dim - 1;
        int p = a->LeftDegree();
        int q = a->RightDegree();
        if (p > n || q > n)
            throw Exception("HodgeStar (double-form): form degree exceeds manifold dimension");

        if (a->IsZeroCF())
            return ZeroDoubleForm(n - p, n - q, vb == VOL ? n : ambient_dim);

        if (vb == BND)
        {
            auto star_vol = HodgeStar(a, M, VOL);
            auto normal = M.GetNV();

            int left_deg = star_vol->LeftDegree();
            auto contracted_left = M.Contraction(star_vol, normal, 0);
            auto contracted_right = M.Contraction(contracted_left, normal, size_t(left_deg - 1));
            return DoubleFormCF(contracted_right, n - p, n - q, ambient_dim);
        }

        auto left_star = BlockHodgeStar(a, 0, p, n, M);
        auto left_tf = TensorFieldCF(left_star, std::string(size_t(n - p + q), '1'));
        auto right_star = BlockHodgeStar(left_tf, n - p, q, n, M);

        return DoubleFormCF(right_star, n - p, n - q, n);
    }

    shared_ptr<DoubleFormCoefficientFunction> InverseHodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb)
    {
        if (vb == BBND)
            throw Exception("InverseHodgeStar (double-form): not implemented for BBND (codimension-2) yet");

        int n = (vb == VOL) ? M.Dimension() : M.Dimension() - 1;
        int p = a->LeftDegree();
        int q = a->RightDegree();
        int exponent = p * (n - p) + q * (n - q);
        int sign = (exponent % 2 == 0) ? 1 : -1;

        auto star = HodgeStar(a, M, vb);
        if (sign == 1)
            return star;
        return DoubleFormCF((-1.0) * star->GetCoefficients(), n - p, n - q, vb == VOL ? n : M.Dimension());
    }

    shared_ptr<ScalarFieldCoefficientFunction> SlotInnerProduct(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb, bool forms)
    {
        return M.SlotInnerProduct(a, vb, forms);
    }

    shared_ptr<DoubleFormCoefficientFunction> SwapDoubleFormSlots(shared_ptr<DoubleFormCoefficientFunction> a)
    {
        int p = a->LeftDegree();
        int q = a->RightDegree();
        int dim = a->DimensionOfSpace();
        int total = p + q;

        std::vector<int> order;
        order.reserve(total);
        for (int i = 0; i < q; ++i)
            order.push_back(p + i);
        for (int i = 0; i < p; ++i)
            order.push_back(i);

        auto reordered = PermuteTensorCF(a, order);
        return DoubleFormCF(reordered, q, p, dim);
    }
}

void ExportKForms(py::module m)
{
    using namespace ngfem;

    py::class_<AlternationCoefficientFunction,
               CoefficientFunction,
               shared_ptr<AlternationCoefficientFunction>>(m, "Alternation")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int rank, int dim)
                      {
                          auto base = AlternationCF(cf, rank, dim);
                          auto casted = dynamic_pointer_cast<AlternationCoefficientFunction>(base);
                          if (!casted)
                              throw Exception("Alternation pybind: returned object is not AlternationCoefficientFunction");
                          return casted; }),
             py::arg("cf"), py::arg("rank"), py::arg("dim"))
        .def_property_readonly("rank", &AlternationCoefficientFunction::Rank)
        .def_property_readonly("dim", &AlternationCoefficientFunction::DimSpace);

    py::class_<KFormCoefficientFunction,
               TensorFieldCoefficientFunction,
               shared_ptr<KFormCoefficientFunction>>(m, "KForm")
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
             { return HodgeStar(a, *M, vb); }, py::arg("M"), py::arg("vb") = VOL)
        .def("inv_star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
             { return InverseHodgeStar(a, *M, vb); }, py::arg("M"), py::arg("vb") = VOL)
        .def_property_readonly("coef", &KFormCoefficientFunction::GetCoefficients);

    py::class_<DoubleFormCoefficientFunction,
               TensorFieldCoefficientFunction,
               shared_ptr<DoubleFormCoefficientFunction>>(m, "DoubleForm")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
                      { return DoubleFormCF(cf, p, q, dim); }),
             py::arg("cf"), py::arg("p"), py::arg("q"), py::arg("dim"))
        .def_property_readonly("degree_left", &DoubleFormCoefficientFunction::LeftDegree)
        .def_property_readonly("degree_right", &DoubleFormCoefficientFunction::RightDegree)
        .def_property_readonly("dim_space", &DoubleFormCoefficientFunction::DimensionOfSpace)
        .def("wedge", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
             { return Wedge(a, b); }, py::arg("b"))
        .def("star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
             { return HodgeStar(a, *M, vb); }, py::arg("M"), py::arg("vb") = VOL)
        .def("inv_star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
             { return InverseHodgeStar(a, *M, vb); }, py::arg("M"), py::arg("vb") = VOL)
        .def_property_readonly("trans", [](shared_ptr<DoubleFormCoefficientFunction> a)
                               { return SwapDoubleFormSlots(a); })
        .def_property_readonly("coef", &DoubleFormCoefficientFunction::GetCoefficients);

    py::class_<ScalarFieldCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<ScalarFieldCoefficientFunction>>(m, "ScalarField")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int dim)
                      { return ScalarFieldCF(cf, dim); }),
             py::arg("cf"), py::arg("dim"))
        .def_static("from_cf", [](shared_ptr<CoefficientFunction> cf, int dim)
                    { return ScalarFieldCF(cf, dim); }, py::arg("cf"), py::arg("dim"));

    py::class_<OneFormCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<OneFormCoefficientFunction>>(m, "OneForm")
        .def(py::init([](shared_ptr<CoefficientFunction> cf)
                      { return OneFormCF(cf); }),
             py::arg("cf"))
        .def_static("from_cf", [](shared_ptr<CoefficientFunction> cf)
                    { return OneFormCF(cf); }, py::arg("cf"));

    py::class_<TwoFormCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<TwoFormCoefficientFunction>>(m, "TwoForm")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int dim)
                      { return TwoFormCF(cf, dim); }),
             py::arg("cf"), py::arg("dim") = -1)
        .def_static("from_cf", [](shared_ptr<CoefficientFunction> cf, int dim)
                    { return TwoFormCF(cf, dim); }, py::arg("cf"), py::arg("dim") = -1);

    py::class_<ThreeFormCoefficientFunction,
               KFormCoefficientFunction,
               shared_ptr<ThreeFormCoefficientFunction>>(m, "ThreeForm")
        .def(py::init([](shared_ptr<CoefficientFunction> cf, int dim)
                      { return ThreeFormCF(cf, dim); }),
             py::arg("cf"), py::arg("dim") = -1)
        .def_static("from_cf", [](shared_ptr<CoefficientFunction> cf, int dim)
                    { return ThreeFormCF(cf, dim); }, py::arg("cf"), py::arg("dim") = -1);

    m.def("Wedge", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b)
          { return Wedge(a, b); }, py::arg("a"), py::arg("b"));
    m.def("Wedge", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
          { return Wedge(a, b); }, py::arg("a"), py::arg("b"));

    m.def("d", [](shared_ptr<KFormCoefficientFunction> a)
          { return ExteriorDerivative(a); }, py::arg("a"));
    m.def("star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
          { return M->Star(a, vb); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL);
    m.def("inv_star", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
          { return InverseHodgeStar(a, *M, vb); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL);
    m.def("star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
          { return HodgeStar(a, *M, vb); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL);
    m.def("inv_star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb)
          { return InverseHodgeStar(a, *M, vb); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL);
    m.def("slot_inner_product", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, bool forms)
          { return SlotInnerProduct(a, *M, vb, forms); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("forms") = true);

    m.def("delta", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M)
          { return M->Coderivative(a); }, py::arg("a"), py::arg("M"));
}
