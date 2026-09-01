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
        std::string GeneratedCoefficientType(const Code &code, bool is_complex)
        {
            std::string type = is_complex ? "Complex" : "double";
            if (code.is_simd)
                type = "SIMD<" + type + ">";
            if (code.deriv == 1)
                type = "AutoDiff<1," + type + ">";
            if (code.deriv == 2)
                type = "AutoDiffDiff<1," + type + ">";
            return type;
        }

        void DeclareGeneratedCoefficient(Code &code, int index,
                                         FlatArray<int> dims, bool is_complex)
        {
            // Code::Declare is not exported by the NGSolve DLL on Windows.
            // Generate the equivalent declaration locally so addon wheels link.
            const std::string type = GeneratedCoefficientType(code, is_complex);

            if (code_uses_tensors)
            {
                code.body += "Tens<" + type;
                for (int dim : dims)
                    code.body += ',' + ToLiteral(dim);
                code.body += "> var_" + ToLiteral(index) + ";\n";
                return;
            }

            size_t component_count = 1;
            for (int dim : dims)
                component_count *= size_t(dim);
            for (size_t component = 0; component < component_count; ++component)
                code.body += Var(index, int(component), dims).Declare(type);
        }

        template <typename T>
        shared_ptr<T> RequireNonNull(shared_ptr<T> ptr, const char *name)
        {
            if (!ptr)
                throw Exception(std::string(name) + ": input coefficient is null");
            return ptr;
        }

        int InferWrappedOrTensorDimension(shared_ptr<CoefficientFunction> cf, int dim)
        {
            if (dim > 0)
                return dim;
            if (auto kf = dynamic_pointer_cast<KFormCoefficientFunction>(cf))
                return kf->DimensionOfSpace();
            if (auto df = dynamic_pointer_cast<DoubleFormCoefficientFunction>(cf))
                return df->DimensionOfSpace();
            if (cf->Dimensions().Size() > 0)
                return int(cf->Dimensions()[0]);
            return 0;
        }

        void RequireScalarFormDimension(int used_dim, const char *name)
        {
            if (used_dim <= 0)
                throw Exception(std::string(name) + ": dim must be provided for scalar forms");
        }

        void ValidateKFormInput(const CoefficientFunction &cf, int k)
        {
            if (cf.Dimensions().Size() != size_t(k))
                throw Exception("KFormCF: degree k requires rank-k coefficient");
        }

        void ValidateDoubleFormInput(const CoefficientFunction &cf, int p, int q)
        {
            if (cf.Dimensions().Size() != size_t(p + q))
                throw Exception("DoubleFormCF: degrees p,q require rank-(p+q) coefficient");
        }

        template <typename TWrapped>
        shared_ptr<TWrapped> ReuseIfCompatible(shared_ptr<CoefficientFunction> cf, int used_dim, const char *name)
        {
            auto existing = dynamic_pointer_cast<TWrapped>(cf);
            if (!existing)
                return nullptr;
            if (used_dim > 0 && existing->DimensionOfSpace() != used_dim)
                throw Exception(std::string(name) + ": requested dim does not match wrapped dimension");
            return existing;
        }

        shared_ptr<KFormCoefficientFunction> WrapKFormImpl(shared_ptr<CoefficientFunction> cf, int k, int dim)
        {
            int used_dim = InferWrappedOrTensorDimension(cf, dim);
            if (k == 0)
                RequireScalarFormDimension(used_dim, "KFormCF");

            ValidateKFormInput(*cf, k);

            if (k == 0)
            {
                if (auto sf = ReuseIfCompatible<ScalarFieldCoefficientFunction>(cf, used_dim, "KFormCF"))
                    return sf;
                return make_shared<ScalarFieldCoefficientFunction>(cf, used_dim);
            }
            if (k == 1)
            {
                if (auto of = ReuseIfCompatible<OneFormCoefficientFunction>(cf, used_dim, "KFormCF"))
                    return of;
                return make_shared<OneFormCoefficientFunction>(cf, used_dim);
            }
            if (k == 2)
            {
                if (auto tf = ReuseIfCompatible<TwoFormCoefficientFunction>(cf, used_dim, "KFormCF"))
                    return tf;
                return make_shared<TwoFormCoefficientFunction>(cf, used_dim);
            }
            if (k == 3)
            {
                if (auto tf = ReuseIfCompatible<ThreeFormCoefficientFunction>(cf, used_dim, "KFormCF"))
                    return tf;
                return make_shared<ThreeFormCoefficientFunction>(cf, used_dim);
            }

            if (auto kf = dynamic_pointer_cast<KFormCoefficientFunction>(cf))
            {
                if (kf->Degree() != k)
                    throw Exception("KFormCF: cannot reinterpret existing k-form with different degree");
                if (used_dim > 0 && kf->DimensionOfSpace() != used_dim)
                    throw Exception("KFormCF: requested dim does not match wrapped dimension");
                return kf;
            }

            return make_shared<KFormCoefficientFunction>(cf, uint8_t(k), uint8_t(used_dim));
        }

        shared_ptr<DoubleFormCoefficientFunction> WrapDoubleFormImpl(shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
        {
            int used_dim = InferWrappedOrTensorDimension(cf, dim);
            if (p + q == 0)
                RequireScalarFormDimension(used_dim, "DoubleFormCF");

            ValidateDoubleFormInput(*cf, p, q);

            if (auto df = dynamic_pointer_cast<DoubleFormCoefficientFunction>(cf))
            {
                if (df->LeftDegree() != p || df->RightDegree() != q)
                    throw Exception("DoubleFormCF: cannot reinterpret existing DoubleForm with different left/right degrees");
                if (used_dim > 0 && df->DimensionOfSpace() != used_dim)
                    throw Exception("DoubleFormCF: requested dim does not match wrapped double-form dimension");
                return df;
            }

            return make_shared<DoubleFormCoefficientFunction>(cf, uint8_t(p), uint8_t(q), uint8_t(used_dim));
        }

        template <typename TWrapped>
        shared_ptr<TWrapped> WrapSpecializedKForm(shared_ptr<CoefficientFunction> cf, int k, int dim, const char *name)
        {
            int used_dim = InferWrappedOrTensorDimension(cf, dim);
            auto kf = KFormCF(cf, k, used_dim);
            auto wrapped = dynamic_pointer_cast<TWrapped>(kf);
            if (!wrapped)
                throw Exception(std::string(name) + ": internal type mismatch");
            return wrapped;
        }

        int ParseDoubleFormSlot(const std::string &slot)
        {
            if (slot == "both" || slot == "all")
                return -1;
            if (slot == "left" || slot == "0")
                return 0;
            if (slot == "right" || slot == "1")
                return 1;
            throw Exception("slot must be 'left', 'right', or 'both'");
        }

        const std::vector<std::array<int, 4>> &GeneratePermutations(int rank)
        {
            if (rank < 0 || rank > MAX_PERMUTATION_RANK)
                throw Exception("GeneratePermutations: rank must be in [0, " + ToString(MAX_PERMUTATION_RANK) + "]");

            static std::array<std::vector<std::array<int, 4>>, MAX_PERMUTATION_RANK + 1> cached_perms = []()
            {
                std::array<std::vector<std::array<int, 4>>, MAX_PERMUTATION_RANK + 1> result;
                for (int r = 0; r <= MAX_PERMUTATION_RANK; ++r)
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

        enum class PermutationFamily : uint8_t
        {
            Full,
            Shuffle
        };

        struct PermutationSpec
        {
            PermutationFamily family;
            int a;
            int b;
        };

        int BlockLength(const PermutationSpec &spec)
        {
            switch (spec.family)
            {
            case PermutationFamily::Full:
                if (spec.a < 0 || spec.a > MAX_PERMUTATION_RANK)
                    throw Exception("BlockLength: full permutation rank must be in [0, " + ToString(MAX_PERMUTATION_RANK) + "]");
                if (spec.b != 0)
                    throw Exception("BlockLength: full permutation spec requires b == 0");
                return spec.a;
            case PermutationFamily::Shuffle:
                if (spec.a < 0 || spec.b < 0 || spec.a + spec.b > MAX_PERMUTATION_RANK)
                    throw Exception("BlockLength: shuffle block sizes must be non-negative and sum to <= " + ToString(MAX_PERMUTATION_RANK));
                return spec.a + spec.b;
            }
            throw Exception("BlockLength: unsupported permutation family");
        }

        bool IsShuffle(const std::array<int, 4> &perm, int left, int right)
        {
            int total = left + right;
            int prev_left = -1;
            int prev_right = left - 1;

            for (int i = 0; i < total; ++i)
            {
                int v = perm[size_t(i)];
                if (v < left)
                {
                    if (v < prev_left)
                        return false;
                    prev_left = v;
                }
                else
                {
                    if (v < prev_right)
                        return false;
                    prev_right = v;
                }
            }
            return true;
        }

        struct SignedPermutations
        {
            std::once_flag flag;
            std::vector<std::array<int, 4>> perms;
            std::vector<int> signs;
        };

        const SignedPermutations &GetSignedPermutations(const PermutationSpec &spec)
        {
            constexpr size_t family_count = 2;
            static std::array<std::array<std::array<SignedPermutations, MAX_PERMUTATION_RANK + 1>, MAX_PERMUTATION_RANK + 1>, family_count> cache;

            int block_len = BlockLength(spec);
            auto family_index = size_t(spec.family);
            auto &entry = cache[family_index][size_t(spec.a)][size_t(spec.b)];

            std::call_once(entry.flag, [&]()
                           {
            const auto &base_perms = GeneratePermutations(block_len);
            entry.signs.reserve(base_perms.size());

            switch (spec.family)
            {
            case PermutationFamily::Full:
                for (const auto &perm : base_perms)
                {
                    entry.perms.push_back(perm);
                    entry.signs.push_back(PermutationSign(perm, block_len));
                }
                break;
            case PermutationFamily::Shuffle:
                for (const auto &perm : base_perms)
                {
                    if (!IsShuffle(perm, spec.a, spec.b))
                        continue;
                    entry.perms.push_back(perm);
                    entry.signs.push_back(PermutationSign(perm, block_len));
                }
                break;
            } });

            return entry;
        }

        struct SignedPermutationOrders
        {
            std::once_flag flag;
            std::vector<std::vector<int>> orders;
            std::vector<int> signs;
        };

        const SignedPermutationOrders &GetSignedWedgeOrders(int left, int right)
        {
            static std::array<std::array<SignedPermutationOrders, MAX_PERMUTATION_RANK + 1>, MAX_PERMUTATION_RANK + 1> cache;
            auto &entry = cache[size_t(left)][size_t(right)];

            std::call_once(entry.flag, [&]()
                           {
            const auto &data = GetSignedPermutations(
                PermutationSpec{PermutationFamily::Shuffle, left, right});

            entry.orders.resize(data.perms.size());
            entry.signs = data.signs;

            int total = left + right;
            for (size_t p = 0; p < data.perms.size(); ++p)
            {
                auto &order = entry.orders[p];
                order.resize(size_t(total));
                for (int i = 0; i < total; ++i)
                    order[size_t(i)] = data.perms[p][size_t(i)];
            } });

            return entry;
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

        shared_ptr<KFormCoefficientFunction> BoundaryHodgeStarKForm(shared_ptr<KFormCoefficientFunction> a,
                                                                    const RiemannianManifold &M)
        {
            int ambient_dim = M.Dimension();
            int n = ambient_dim - 1;
            int k = a->Degree();

            auto star_vol = HodgeStar(a, M, VOL);
            auto normal = M.GetNV();
            auto contracted = M.Contraction(star_vol, normal); // reduce degree by 1
            // if (!extended && (n - k) > 0)
            // {
            //     auto projected = M.ProjectTensorToEuclideanTangent(contracted);
            //     return KFormCF(projected, n - k, ambient_dim);
            // }
            double sign = (k % 2 == 0) ? 1.0 : -1.0;
            return KFormCF(sign * contracted->GetCoefficients(), n - k, ambient_dim);
        }

        shared_ptr<DoubleFormCoefficientFunction> BoundaryHodgeStarDoubleForm(shared_ptr<DoubleFormCoefficientFunction> a,
                                                                              const RiemannianManifold &M)
        {
            int ambient_dim = M.Dimension();
            int n = ambient_dim - 1;
            int p = a->LeftDegree();
            int q = a->RightDegree();

            auto star_vol = HodgeStar(a, M, VOL);
            auto normal = M.GetNV();

            int left_deg = star_vol->LeftDegree();
            auto contracted_left = M.Contraction(star_vol, normal, 0);
            auto contracted_right = M.Contraction(contracted_left, normal, size_t(left_deg - 1));
            // if (!extended && (n - p + n - q) > 0)
            // {
            //     auto projected = M.ProjectTensorToEuclideanTangent(contracted_right);
            //     return DoubleFormCF(projected, n - p, n - q, ambient_dim);
            // }
            double sign = ((p + q) % 2 == 0) ? 1.0 : -1.0;
            return DoubleFormCF(sign * contracted_right->GetCoefficients(), n - p, n - q, ambient_dim);
        }

        shared_ptr<KFormCoefficientFunction> BBNDHodgeStarKForm(shared_ptr<KFormCoefficientFunction> a,
                                                                const RiemannianManifold &M)
        {
            int ambient_dim = M.Dimension();
            int n = ambient_dim - 2;
            int k = a->Degree();

            if (k > n)
                throw Exception("HodgeStar (BBND): form degree exceeds codimension-2 manifold dimension");

            if (a->IsZeroCF())
                return ZeroKForm(n - k, ambient_dim);

            if (n == 0)
                return KFormCF(a->GetCoefficients(), 0, ambient_dim);

            auto star_vol = HodgeStar(a, M, VOL);
            auto n1 = M.GetEdgeNormal(0);
            auto cn1 = M.GetEdgeConormal(0);
            auto c1 = M.Contraction(star_vol, n1);
            auto c2 = M.Contraction(c1, cn1);

            // Codim-2 induced orientation uses two contractions; no extra k-dependent sign.
            return KFormCF(c2->GetCoefficients(), n - k, ambient_dim);
        }

        shared_ptr<DoubleFormCoefficientFunction> BBNDHodgeStarDoubleForm(shared_ptr<DoubleFormCoefficientFunction> a,
                                                                          const RiemannianManifold &M)
        {
            int ambient_dim = M.Dimension();
            int n = ambient_dim - 2;
            int p = a->LeftDegree();
            int q = a->RightDegree();

            if (p > n || q > n)
                throw Exception("HodgeStar (double-form, BBND): form degree exceeds codimension-2 manifold dimension");

            if (a->IsZeroCF())
                return ZeroDoubleForm(n - p, n - q, ambient_dim);

            if (n == 0)
                return DoubleFormCF(a->GetCoefficients(), 0, 0, ambient_dim);

            auto n1 = M.GetEdgeNormal(0);
            auto cn1 = M.GetEdgeConormal(0);
            auto star_vol = HodgeStar(a, M, VOL);

            int left_deg = star_vol->LeftDegree();
            auto c_l1 = M.Contraction(star_vol, n1, 0);
            auto c_l2 = M.Contraction(c_l1, cn1, 0);
            auto c_r1 = M.Contraction(c_l2, n1, size_t(left_deg - 2));
            auto c_r2 = M.Contraction(c_r1, cn1, size_t(left_deg - 2));

            // Codim-2 induced orientation uses two contractions per slot; no extra degree sign.
            return DoubleFormCF(c_r2->GetCoefficients(), n - p, n - q, ambient_dim);
        }

    } // namespace

    KFormCoefficientFunction::KFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, uint8_t ak, uint8_t adim)
        : TensorFieldCoefficientFunction(ac1, std::string(size_t(ak), '1')), degree(ak), dim(adim)
    {
        if (!((adim >= 1 && adim <= MAX_SPACE_DIM) || (adim == 0 && ak == 0)))
            throw Exception("KFormCF: dim must be in {1,...," + ToString(MAX_SPACE_DIM) + "} (or 0 for scalar forms)");
        if (ak > MAX_FORM_RANK)
            throw Exception("KFormCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");

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
        if (!((adim >= 1 && adim <= MAX_SPACE_DIM) || (adim == 0 && ap == 0 && aq == 0)))
            throw Exception("DoubleFormCF: dim must be in {1,...," + ToString(MAX_SPACE_DIM) + "} (or 0 for scalar forms)");
        if (ap + aq > MAX_FORM_RANK && !ac1->IsZeroCF())
            throw Exception("DoubleFormCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");

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

    template <typename VIN, typename VOUT>
    void ApplyLinearNonZeroPattern(const std::vector<int> &valid_indices,
                                   const std::vector<int> &lin_table,
                                   size_t terms_per_index,
                                   VIN input,
                                   VOUT values)
    {
        values = AutoDiffDiff<1, NonZero>(false);
        for (size_t vi = 0; vi < valid_indices.size(); ++vi)
        {
            int idx = valid_indices[vi];
            auto accum = AutoDiffDiff<1, NonZero>(false);
            const size_t base = vi * terms_per_index;
            for (size_t p = 0; p < terms_per_index; ++p)
                accum = accum + input(lin_table[base + p]);
            values(idx) = accum;
        }
    }

    template <typename MI, typename MV>
    void ApplySignedLinearTable(size_t nir,
                                int comp_dim,
                                const std::vector<int> &valid_indices,
                                const std::vector<int> &signs,
                                const std::vector<int> &lin_table,
                                size_t terms_per_index,
                                MI input,
                                MV values)
    {
        using T = typename MV::TSCAL;
        for (size_t ip = 0; ip < nir; ++ip)
        {
            for (int idx = 0; idx < comp_dim; ++idx)
                values(idx, ip) = T(0);
            for (size_t vi = 0; vi < valid_indices.size(); ++vi)
            {
                int idx = valid_indices[vi];
                T accum = T(0);
                const size_t base = vi * terms_per_index;
                for (size_t p = 0; p < terms_per_index; ++p)
                    accum += T(signs[p]) * input(lin_table[base + p], ip);
                values(idx, ip) = accum;
            }
        }
    }

    template <typename MA, typename MB, typename MV>
    void ApplySignedProductTable(size_t nir,
                                 int comp_dim,
                                 const std::vector<int> &valid_indices,
                                 const std::vector<int> &signs,
                                 const std::vector<int> &lin_a,
                                 const std::vector<int> &lin_b,
                                 size_t terms_per_index,
                                 MA values_a,
                                 MB values_b,
                                 MV values)
    {
        using T = typename MV::TSCAL;
        for (size_t ip = 0; ip < nir; ++ip)
        {
            for (int idx = 0; idx < comp_dim; ++idx)
                values(idx, ip) = T(0);
            for (size_t vi = 0; vi < valid_indices.size(); ++vi)
            {
                T accum = T(0);
                size_t base = vi * terms_per_index;
                for (size_t t = 0; t < terms_per_index; ++t)
                    accum += T(signs[base + t]) * values_a(lin_a[base + t], ip) * values_b(lin_b[base + t], ip);
                values(valid_indices[vi], ip) = accum;
            }
        }
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
            if (rank < 0 || rank > MAX_PERMUTATION_RANK)
                throw Exception("AlternationCF: only ranks 0-" + ToString(MAX_PERMUTATION_RANK) + " supported");
            if (dim < 1 || dim > MAX_SPACE_DIM)
                throw Exception("AlternationCF: dim must be in {1,...," + ToString(MAX_SPACE_DIM) + "}");
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
            ApplyLinearNonZeroPattern(valid_indices, lin_table, perms.size(), input, values);
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            ApplyLinearNonZeroPattern(valid_indices, lin_table, perms.size(), input[0], values);
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
            ApplySignedLinearTable(mir.Size(), comp_dim, valid_indices, signs, lin_table, perms.size(), input, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            ApplySignedLinearTable(ir.Size(), this->Dimension(), valid_indices, signs, lin_table, perms.size(), input[0], values);
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

    class BlockAlternationCoefficientFunction : public T_CoefficientFunction<BlockAlternationCoefficientFunction>
    {
        shared_ptr<CoefficientFunction> c1;
        int rank_total;
        int block_start;
        int block_len;
        int dim;
        std::vector<std::array<int, 4>> perms;
        std::vector<int> signs;
        std::vector<int> valid_indices;
        std::vector<int> lin_table;

    public:
        BlockAlternationCoefficientFunction(shared_ptr<CoefficientFunction> ac1,
                                            int arank_total,
                                            int ablock_start,
                                            int ablock_len)
            : T_CoefficientFunction<BlockAlternationCoefficientFunction>(ac1->Dimension(), ac1->IsComplex()),
              c1(ac1), rank_total(arank_total), block_start(ablock_start), block_len(ablock_len)
        {
            if (rank_total < 0 || rank_total > MAX_FORM_RANK)
                throw Exception("BlockAlternationCF: only ranks 0-" + ToString(MAX_FORM_RANK) + " supported");
            if (block_len < 0 || block_len > MAX_PERMUTATION_RANK)
                throw Exception("BlockAlternationCF: block size must be in [0, " + ToString(MAX_PERMUTATION_RANK) + "]");
            if (block_start < 0 || block_start + block_len > rank_total)
                throw Exception("BlockAlternationCF: block range out of bounds");
            if (c1->Dimensions().Size() != size_t(rank_total))
                throw Exception("BlockAlternationCF: tensor rank mismatch");
            if (rank_total == 0)
                throw Exception("BlockAlternationCF: rank-zero tensors do not need block alternation");

            dim = c1->Dimensions()[0];
            if (dim < 1 || dim > MAX_SPACE_DIM)
                throw Exception("BlockAlternationCF: dim must be in {1,...," + ToString(MAX_SPACE_DIM) + "}");
            for (auto d : c1->Dimensions())
                if (d != dim)
                    throw Exception("BlockAlternationCF: tensor dimensions must equal dim");

            this->SetDimensions(c1->Dimensions());

            perms = GeneratePermutations(block_len);
            signs.resize(perms.size());
            for (size_t i = 0; i < perms.size(); ++i)
                signs[i] = PermutationSign(perms[i], block_len);

            int comp_dim = this->Dimension();
            valid_indices.reserve(comp_dim);
            lin_table.reserve(size_t(comp_dim) * perms.size());

            std::array<int, MAX_FORM_RANK> multi = {};
            std::array<int, MAX_FORM_RANK> src_multi = {};
            for (int idx = 0; idx < comp_dim; ++idx)
            {
                int rem = idx;
                bool repeated = false;
                for (int j = rank_total - 1; j >= 0; --j)
                {
                    multi[size_t(j)] = rem % dim;
                    rem /= dim;
                }

                for (int a = 0; a < block_len && !repeated; ++a)
                    for (int b = a + 1; b < block_len; ++b)
                        if (multi[size_t(block_start + a)] == multi[size_t(block_start + b)])
                        {
                            repeated = true;
                            break;
                        }

                if (repeated)
                    continue;

                valid_indices.push_back(idx);
                for (size_t p = 0; p < perms.size(); ++p)
                {
                    for (int j = 0; j < rank_total; ++j)
                        src_multi[size_t(j)] = multi[size_t(j)];
                    for (int j = 0; j < block_len; ++j)
                        src_multi[size_t(block_start + perms[p][size_t(j)])] = multi[size_t(block_start + j)];

                    int lin = 0;
                    for (int j = 0; j < rank_total; ++j)
                        lin = lin * dim + src_multi[size_t(j)];
                    lin_table.push_back(lin);
                }
            }
        }

        virtual string GetDescription() const override
        {
            return "BlockAlternationCF";
        }

        auto GetCArgs() const { return tuple{c1}; }

        void DoArchive(Archive &ar) override
        {
        }

        virtual void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
        {
            DeclareGeneratedCoefficient(code, index, Dimensions(), IsComplex());

            size_t vi = 0;
            for (int idx = 0; idx < this->Dimension(); ++idx)
            {
                if (vi >= valid_indices.size() || valid_indices[vi] != idx)
                {
                    code.body += Var(index, idx, Dimensions()).Assign(string("0.0"), false);
                    continue;
                }

                CodeExpr result;
                const size_t base = vi * perms.size();
                for (size_t p = 0; p < perms.size(); ++p)
                {
                    CodeExpr term = Var(inputs[0], lin_table[base + p], c1->Dimensions());
                    if (signs[p] == -1)
                        result -= term;
                    else
                        result += term;
                }
                code.body += Var(index, idx, Dimensions()).Assign(result.S(), false);
                ++vi;
            }
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
            ApplyLinearNonZeroPattern(valid_indices, lin_table, perms.size(), input, values);
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            ApplyLinearNonZeroPattern(valid_indices, lin_table, perms.size(), input[0], values);
        }

        shared_ptr<CoefficientFunction>
        Transform(CoefficientFunction::T_Transform &transformation) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (transformation.cache.count(thisptr))
                return transformation.cache[thisptr];
            if (transformation.replace.count(thisptr))
                return transformation.replace[thisptr];
            auto newcf = make_shared<BlockAlternationCoefficientFunction>(
                c1->Transform(transformation), rank_total, block_start, block_len);
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        using T_CoefficientFunction<BlockAlternationCoefficientFunction>::Evaluate;

        virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            throw Exception("BlockAlternationCF:: scalar evaluate called");
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir, BareSliceMatrix<T, ORD> values) const
        {
            int comp_dim = this->Dimension();

            Array<T> temp(comp_dim * mir.Size());
            FlatMatrix<T, ORD> input(comp_dim, mir.Size(), temp.Data());
            c1->Evaluate(mir, input);

            EvalFromInput(mir.Size(), input, values);
        }

        template <typename MI, typename MV>
        void EvalFromInput(size_t nir, MI input, MV values) const
        {
            ApplySignedLinearTable(nir, this->Dimension(), valid_indices, signs, lin_table, perms.size(), input, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            EvalFromInput(ir.Size(), input[0], values);
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return make_shared<BlockAlternationCoefficientFunction>(
                c1->Diff(var, dir), rank_total, block_start, block_len);
        }

        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            auto res = make_shared<BlockAlternationCoefficientFunction>(
                c1->DiffJacobi(var, cache), rank_total, block_start, block_len);
            cache[thisptr] = res;
            return res;
        }

        virtual bool IsZeroCF() const override { return c1->IsZeroCF(); }
    };

    shared_ptr<CoefficientFunction> BlockAlternationByPermutationCF(shared_ptr<CoefficientFunction> T, int rank_total, int block_start, int block_len)
    {
        if (!T)
            throw Exception("BlockAlternationByPermutationCF: input coefficient is null");
        if (rank_total < 0 || rank_total > MAX_FORM_RANK)
            throw Exception("BlockAlternationByPermutationCF: only ranks 0-" + ToString(MAX_FORM_RANK) + " supported");
        if (block_len < 0 || block_len > MAX_PERMUTATION_RANK)
            throw Exception("BlockAlternationByPermutationCF: block size must be in [0, " + ToString(MAX_PERMUTATION_RANK) + "]");
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

        return make_shared<BlockAlternationCoefficientFunction>(tf, rank_total, block_start, block_len);
    }

    int TensorComponentCount(int dim, int rank)
    {
        int count = 1;
        for (int i = 0; i < rank; ++i)
            count *= dim;
        return count;
    }

    int EncodeRowMajor(const std::array<int, MAX_FORM_RANK> &multi, const std::vector<int> &positions, int dim)
    {
        int lin = 0;
        for (int pos : positions)
            lin = lin * dim + multi[size_t(pos)];
        return lin;
    }

    class DoubleFormWedgeCoefficientFunction : public T_CoefficientFunction<DoubleFormWedgeCoefficientFunction>
    {
        shared_ptr<DoubleFormCoefficientFunction> a;
        shared_ptr<DoubleFormCoefficientFunction> b;
        int p;
        int q;
        int r;
        int s;
        int dim;
        int total;
        std::vector<int> valid_indices;
        std::vector<int> signs;
        std::vector<int> lin_a;
        std::vector<int> lin_b;
        bool structural_zero = false;
        size_t terms_per_index = 0;

    public:
        DoubleFormWedgeCoefficientFunction(shared_ptr<DoubleFormCoefficientFunction> aa,
                                           shared_ptr<DoubleFormCoefficientFunction> bb)
            : T_CoefficientFunction<DoubleFormWedgeCoefficientFunction>(
                  TensorComponentCount(aa->DimensionOfSpace(), aa->LeftDegree() + bb->LeftDegree() + aa->RightDegree() + bb->RightDegree()),
                  aa->IsComplex() || bb->IsComplex()),
              a(aa), b(bb),
              p(aa->LeftDegree()), q(aa->RightDegree()),
              r(bb->LeftDegree()), s(bb->RightDegree()),
              dim(aa->DimensionOfSpace()),
              total(p + q + r + s)
        {
            if (!a || !b)
                throw Exception("DoubleFormWedgeCF: inputs must be non-null");
            if (a->DimensionOfSpace() != b->DimensionOfSpace())
                throw Exception("DoubleFormWedgeCF: input double-forms must have the same dimension of space");
            if (total > MAX_FORM_RANK)
                throw Exception("DoubleFormWedgeCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");

            Array<int> dims(total);
            for (int i = 0; i < total; ++i)
                dims[i] = dim;
            this->SetDimensions(dims);

            const int left_len = p + r;
            const int right_len = q + s;
            const auto &left_data = GetSignedPermutations(PermutationSpec{PermutationFamily::Shuffle, p, r});
            const auto &right_data = GetSignedPermutations(PermutationSpec{PermutationFamily::Shuffle, q, s});
            terms_per_index = left_data.perms.size() * right_data.perms.size();

            std::array<int, MAX_FORM_RANK> multi = {};
            std::array<int, MAX_FORM_RANK> src_multi = {};
            std::vector<int> a_positions;
            std::vector<int> b_positions;
            a_positions.reserve(size_t(p + q));
            b_positions.reserve(size_t(r + s));
            for (int i = 0; i < p; ++i)
                a_positions.push_back(i);
            for (int i = 0; i < q; ++i)
                a_positions.push_back(left_len + i);
            for (int i = 0; i < r; ++i)
                b_positions.push_back(p + i);
            for (int i = 0; i < s; ++i)
                b_positions.push_back(left_len + q + i);

            for (int idx = 0; idx < this->Dimension(); ++idx)
            {
                int rem = idx;
                bool repeated = false;
                for (int j = total - 1; j >= 0; --j)
                {
                    multi[size_t(j)] = rem % dim;
                    rem /= dim;
                }

                for (int i = 0; i < left_len && !repeated; ++i)
                    for (int j = i + 1; j < left_len; ++j)
                        if (multi[size_t(i)] == multi[size_t(j)])
                        {
                            repeated = true;
                            break;
                        }
                for (int i = 0; i < right_len && !repeated; ++i)
                    for (int j = i + 1; j < right_len; ++j)
                        if (multi[size_t(left_len + i)] == multi[size_t(left_len + j)])
                        {
                            repeated = true;
                            break;
                        }
                if (repeated)
                    continue;

                valid_indices.push_back(idx);
                for (size_t pl = 0; pl < left_data.perms.size(); ++pl)
                {
                    for (size_t pr = 0; pr < right_data.perms.size(); ++pr)
                    {
                        std::array<int, MAX_FORM_RANK> order = {};
                        for (int i = 0; i < total; ++i)
                            order[size_t(i)] = i;
                        for (int i = 0; i < left_len; ++i)
                            order[size_t(i)] = left_data.perms[pl][size_t(i)];
                        for (int i = 0; i < right_len; ++i)
                            order[size_t(left_len + i)] = left_len + right_data.perms[pr][size_t(i)];

                        for (int i = 0; i < total; ++i)
                            src_multi[size_t(order[size_t(i)])] = multi[size_t(i)];

                        signs.push_back(left_data.signs[pl] * right_data.signs[pr]);
                        lin_a.push_back(EncodeRowMajor(src_multi, a_positions, dim));
                        lin_b.push_back(EncodeRowMajor(src_multi, b_positions, dim));
                    }
                }
            }

            structural_zero = true;
            for (size_t i = 0; i < lin_a.size(); ++i)
            {
                auto ca = MakeComponentCoefficientFunction(a->GetCoefficients(), lin_a[i]);
                auto cb = MakeComponentCoefficientFunction(b->GetCoefficients(), lin_b[i]);
                if (!ca->IsZeroCF() && !cb->IsZeroCF())
                {
                    structural_zero = false;
                    break;
                }
            }
        }

        virtual string GetDescription() const override
        {
            return "DoubleFormWedgeCF";
        }

        auto GetCArgs() const { return tuple{a, b}; }

        void DoArchive(Archive &ar) override
        {
        }

        virtual void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
        {
            DeclareGeneratedCoefficient(code, index, Dimensions(), IsComplex());

            size_t vi = 0;
            for (int idx = 0; idx < this->Dimension(); ++idx)
            {
                if (vi >= valid_indices.size() || valid_indices[vi] != idx)
                {
                    code.body += Var(index, idx, Dimensions()).Assign(string("0.0"), false);
                    continue;
                }

                CodeExpr result;
                size_t base = vi * terms_per_index;
                for (size_t t = 0; t < terms_per_index; ++t)
                {
                    CodeExpr term = Var(inputs[0], lin_a[base + t], a->Dimensions()) *
                                    Var(inputs[1], lin_b[base + t], b->Dimensions());
                    if (signs[base + t] == -1)
                        result -= term;
                    else
                        result += term;
                }
                code.body += Var(index, idx, Dimensions()).Assign(result.S(), false);
                ++vi;
            }
        }

        virtual void TraverseTree(const function<void(CoefficientFunction &)> &func) override
        {
            a->TraverseTree(func);
            b->TraverseTree(func);
            func(*this);
        }

        virtual Array<shared_ptr<CoefficientFunction>> InputCoefficientFunctions() const override
        {
            return Array<shared_ptr<CoefficientFunction>>({a, b});
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            values = AutoDiffDiff<1, NonZero>(false);
            for (int idx : valid_indices)
                values(idx) = AutoDiffDiff<1, NonZero>(true);
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            values = AutoDiffDiff<1, NonZero>(false);
            for (size_t vi = 0; vi < valid_indices.size(); ++vi)
            {
                auto accum = AutoDiffDiff<1, NonZero>(false);
                size_t base = vi * terms_per_index;
                for (size_t t = 0; t < terms_per_index; ++t)
                    accum = accum + input[0](lin_a[base + t]) * input[1](lin_b[base + t]);
                values(valid_indices[vi]) = accum;
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

            auto ta = DoubleFormCF(a->GetCoefficients()->Transform(transformation), p, q, dim);
            auto tb = DoubleFormCF(b->GetCoefficients()->Transform(transformation), r, s, dim);
            auto newcf = make_shared<DoubleFormWedgeCoefficientFunction>(ta, tb);
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        using T_CoefficientFunction<DoubleFormWedgeCoefficientFunction>::Evaluate;

        virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            throw Exception("DoubleFormWedgeCF:: scalar evaluate called");
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir, BareSliceMatrix<T, ORD> values) const
        {
            int dim_a = a->Dimension();
            int dim_b = b->Dimension();

            Array<T> temp_a(dim_a * mir.Size());
            Array<T> temp_b(dim_b * mir.Size());
            FlatMatrix<T, ORD> values_a(dim_a, mir.Size(), temp_a.Data());
            FlatMatrix<T, ORD> values_b(dim_b, mir.Size(), temp_b.Data());
            a->GetCoefficients()->Evaluate(mir, values_a);
            b->GetCoefficients()->Evaluate(mir, values_b);

            EvalFromInputs(mir.Size(), values_a, values_b, values);
        }

        template <typename MA, typename MB, typename MV>
        void EvalFromInputs(size_t nir,
                            MA values_a,
                            MB values_b,
                            MV values) const
        {
            ApplySignedProductTable(nir, this->Dimension(), valid_indices, signs, lin_a, lin_b, terms_per_index, values_a, values_b, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            EvalFromInputs(ir.Size(), input[0], input[1], values);
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            auto da = DoubleFormCF(a->GetCoefficients()->Diff(var, dir), p, q, dim);
            auto db = DoubleFormCF(b->GetCoefficients()->Diff(var, dir), r, s, dim);
            return Wedge(da, b)->GetCoefficients() + Wedge(a, db)->GetCoefficients();
        }

        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            auto da = DoubleFormCF(a->GetCoefficients()->DiffJacobi(var, cache), p, q, dim);
            auto db = DoubleFormCF(b->GetCoefficients()->DiffJacobi(var, cache), r, s, dim);
            auto res = Wedge(da, b)->GetCoefficients() + Wedge(a, db)->GetCoefficients();
            cache[thisptr] = res;
            return res;
        }

        virtual bool IsZeroCF() const override { return structural_zero || a->IsZeroCF() || b->IsZeroCF(); }
    };

    shared_ptr<KFormCoefficientFunction> KFormCF(shared_ptr<CoefficientFunction> cf, int k, int dim)
    {
        cf = RequireNonNull(std::move(cf), "KFormCF");
        if (k < 0)
            throw Exception("KFormCF: degree must be non-negative");
        if (k > MAX_FORM_RANK)
            throw Exception("KFormCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");
        return WrapKFormImpl(cf, k, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> DoubleFormCF(shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
    {
        cf = RequireNonNull(std::move(cf), "DoubleFormCF");
        if (p < 0 || q < 0)
            throw Exception("DoubleFormCF: degrees must be non-negative");
        if (p + q > MAX_FORM_RANK && !cf->IsZeroCF())
            throw Exception("DoubleFormCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");
        return WrapDoubleFormImpl(cf, p, q, dim);
    }
    shared_ptr<ScalarFieldCoefficientFunction> ScalarFieldCF(shared_ptr<CoefficientFunction> cf, int dim)
    {
        cf = RequireNonNull(std::move(cf), "ScalarFieldCF");
        return WrapSpecializedKForm<ScalarFieldCoefficientFunction>(cf, 0, dim, "ScalarFieldCF");
    }

    shared_ptr<OneFormCoefficientFunction> OneFormCF(shared_ptr<CoefficientFunction> cf)
    {
        cf = RequireNonNull(std::move(cf), "OneFormCF");
        if (cf->Dimensions().Size() != 1)
            throw Exception("OneFormCF: input coefficient must be vector valued");
        return WrapSpecializedKForm<OneFormCoefficientFunction>(cf, 1, -1, "OneFormCF");
    }

    shared_ptr<TwoFormCoefficientFunction> TwoFormCF(shared_ptr<CoefficientFunction> cf, int dim)
    {
        cf = RequireNonNull(std::move(cf), "TwoFormCF");
        return WrapSpecializedKForm<TwoFormCoefficientFunction>(cf, 2, dim, "TwoFormCF");
    }

    shared_ptr<ThreeFormCoefficientFunction> ThreeFormCF(shared_ptr<CoefficientFunction> cf, int dim)
    {
        cf = RequireNonNull(std::move(cf), "ThreeFormCF");
        return WrapSpecializedKForm<ThreeFormCoefficientFunction>(cf, 3, dim, "ThreeFormCF");
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
        const auto &shuffle_data = GetSignedWedgeOrders(k, l);
        shared_ptr<CoefficientFunction> accum;
        for (size_t p = 0; p < shuffle_data.orders.size(); ++p)
        {
            shared_ptr<CoefficientFunction> term = PermuteTensorCF(T, shuffle_data.orders[p]);
            if (shuffle_data.signs[p] == -1)
                term = (-1.0) * term;
            accum = accum ? (accum + term) : term;
        }

        return KFormCF(accum, k + l, dim);
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

        auto wedge_cf = make_shared<DoubleFormWedgeCoefficientFunction>(a, b);
        if (wedge_cf->IsZeroCF())
            return ZeroDoubleForm(p + r, q + s, dim);
        return DoubleFormCF(wedge_cf, p + r, q + s, dim);
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
        if (vb != VOL && vb != BND && vb != BBND)
            throw Exception("HodgeStar: only implemented for VOL, BND, and BBND");
        int n = (vb == VOL) ? ambient_dim : (vb == BND ? ambient_dim - 1 : ambient_dim - 2);
        int k = a->Degree();
        if (k > n)
            throw Exception("HodgeStar: form degree exceeds manifold dimension");

        if (a->IsZeroCF())
            return ZeroKForm(n - k, vb == VOL ? n : ambient_dim);

        if (vb == BND)
            return BoundaryHodgeStarKForm(a, M);
        if (vb == BBND)
            return BBNDHodgeStarKForm(a, M);

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
        if (vb != VOL && vb != BND && vb != BBND)
            throw Exception("InverseHodgeStar: only implemented for VOL, BND, and BBND");
        int n = (vb == VOL) ? M.Dimension() : (vb == BND ? M.Dimension() - 1 : M.Dimension() - 2);
        int k = a->Degree();
        int exponent = k * (n - k);
        int sign = (exponent % 2 == 0) ? 1 : -1;

        auto star = HodgeStar(a, M, vb);
        if (sign == 1)
            return star;
        return KFormCF((-1.0) * star->GetCoefficients(), n - k, vb == VOL ? n : M.Dimension());
    }

    shared_ptr<DoubleFormCoefficientFunction> HodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb, int slot)
    {
        int ambient_dim = M.Dimension();
        if (vb != VOL && vb != BND && vb != BBND)
            throw Exception("HodgeStar (double-form): only implemented for VOL, BND, and BBND");
        int n = (vb == VOL) ? ambient_dim : (vb == BND ? ambient_dim - 1 : ambient_dim - 2);
        int p = a->LeftDegree();
        int q = a->RightDegree();
        if (p > n || q > n)
            throw Exception("HodgeStar (double-form): form degree exceeds manifold dimension");

        if (a->IsZeroCF())
            return ZeroDoubleForm(n - p, n - q, vb == VOL ? n : ambient_dim);

        if (slot == 0)
        {
            if (vb == BBND)
            {
                if (n == 0)
                    return DoubleFormCF(a->GetCoefficients(), 0, q, ambient_dim);
                auto n1 = M.GetEdgeNormal(0);
                auto n2 = M.GetEdgeConormal(0);
                auto star_vol_left = BlockHodgeStar(a, 0, p, ambient_dim, M);
                auto left_tf = TensorFieldCF(star_vol_left, std::string(size_t(ambient_dim - p + q), '1'));
                auto c1 = M.Contraction(left_tf, n1, 0);
                auto c2 = M.Contraction(c1, n2, 0);
                return DoubleFormCF(c2->GetCoefficients(), n - p, q, ambient_dim);
            }
            if (vb == BND)
            {
                auto star_vol_left = BlockHodgeStar(a, 0, p, ambient_dim, M);
                auto left_tf = TensorFieldCF(star_vol_left, std::string(size_t(ambient_dim - p + q), '1'));
                auto contracted = M.Contraction(left_tf, M.GetNV(), 0);
                double sign = (p % 2 == 0) ? 1.0 : -1.0;
                return DoubleFormCF(sign * contracted->GetCoefficients(), n - p, q, ambient_dim);
            }
            auto left_star = BlockHodgeStar(a, 0, p, n, M);
            return DoubleFormCF(left_star, n - p, q, n);
        }

        if (slot == 1)
        {
            if (vb == BBND)
            {
                if (n == 0)
                    return DoubleFormCF(a->GetCoefficients(), p, 0, ambient_dim);
                auto n1 = M.GetEdgeNormal(0);
                auto n2 = M.GetEdgeConormal(0);
                auto star_vol_right = BlockHodgeStar(a, p, q, ambient_dim, M);
                auto right_tf = TensorFieldCF(star_vol_right, std::string(size_t(p + ambient_dim - q), '1'));
                auto c1 = M.Contraction(right_tf, n1, size_t(p));
                auto c2 = M.Contraction(c1, n2, size_t(p));
                return DoubleFormCF(c2->GetCoefficients(), p, n - q, ambient_dim);
            }
            if (vb == BND)
            {
                auto star_vol_right = BlockHodgeStar(a, p, q, ambient_dim, M);
                auto right_tf = TensorFieldCF(star_vol_right, std::string(size_t(p + ambient_dim - q), '1'));
                auto contracted = M.Contraction(right_tf, M.GetNV(), size_t(p));
                double sign = (q % 2 == 0) ? 1.0 : -1.0;
                return DoubleFormCF(sign * contracted->GetCoefficients(), p, n - q, ambient_dim);
            }
            auto right_star = BlockHodgeStar(a, p, q, n, M);
            return DoubleFormCF(right_star, p, n - q, n);
        }

        if (vb == BBND)
            return BBNDHodgeStarDoubleForm(a, M);

        if (vb == BND)
            return BoundaryHodgeStarDoubleForm(a, M);

        auto left_star = BlockHodgeStar(a, 0, p, n, M);
        auto left_tf = TensorFieldCF(left_star, std::string(size_t(n - p + q), '1'));
        auto right_star = BlockHodgeStar(left_tf, n - p, q, n, M);

        return DoubleFormCF(right_star, n - p, n - q, n);
    }

    shared_ptr<DoubleFormCoefficientFunction> InverseHodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb, int slot)
    {
        if (vb != VOL && vb != BND && vb != BBND)
            throw Exception("InverseHodgeStar (double-form): only implemented for VOL, BND, and BBND");
        int n = (vb == VOL) ? M.Dimension() : (vb == BND ? M.Dimension() - 1 : M.Dimension() - 2);
        int p = a->LeftDegree();
        int q = a->RightDegree();
        int exponent = 0;
        if (slot == 0)
            exponent = p * (n - p);
        else if (slot == 1)
            exponent = q * (n - q);
        else
            exponent = p * (n - p) + q * (n - q);
        int sign = (exponent % 2 == 0) ? 1 : -1;

        auto star = HodgeStar(a, M, vb, slot);
        if (sign == 1)
            return star;
        return DoubleFormCF((-1.0) * star->GetCoefficients(), star->LeftDegree(), star->RightDegree(), vb == VOL ? n : M.Dimension());
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
        .def_property_readonly("is_zero", [](shared_ptr<DoubleFormCoefficientFunction> a)
                               { return a->IsZeroCF(); })
        .def("wedge", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
             { return Wedge(a, b); }, py::arg("b"))
        .def("star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
             { return HodgeStar(a, *M, vb, ParseDoubleFormSlot(slot)); }, py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both")
        .def("inv_star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
             { return InverseHodgeStar(a, *M, vb, ParseDoubleFormSlot(slot)); }, py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both")
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
    m.def("star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
          { return HodgeStar(a, *M, vb, ParseDoubleFormSlot(slot)); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both");
    m.def("inv_star", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, const std::string &slot)
          { return InverseHodgeStar(a, *M, vb, ParseDoubleFormSlot(slot)); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("slot") = "both");
    m.def("slot_inner_product", [](shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M, VorB vb, bool forms)
          { return SlotInnerProduct(a, *M, vb, forms); }, py::arg("a"), py::arg("M"), py::arg("vb") = VOL, py::arg("forms") = true);

    m.def("delta", [](shared_ptr<KFormCoefficientFunction> a, shared_ptr<RiemannianManifold> M)
          { return M->Coderivative(a); }, py::arg("a"), py::arg("M"));
}
