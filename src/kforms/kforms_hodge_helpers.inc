// Implementation fragment included only by ../kforms.cpp.
// Permutation tables and boundary/codimension Hodge-star helpers.

namespace
{
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

    int TopFormComponentIndex(int n)
    {
        int index = 0;
        for (int i = 0; i < n; ++i)
            index = index * n + i;
        return index;
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
            return ScaleCoefficientCF(eps, raised);

        std::string pre = sig.substr(0, size_t(block_start));
        std::string block = sig.substr(size_t(block_start), size_t(block_len));
        std::string post = sig.substr(size_t(block_start + block_len));
        std::string new_block = FreshSignature(sig, n - block_len);

        std::string eps_sig = block + new_block;
        std::string out_sig = pre + new_block + post;

        std::string eins = sig + "," + eps_sig + "->" + out_sig;
        auto contracted = SymbolicEinsumCF(eins, {raised, eps});
        double scale = 1.0 / double(Factorial(block_len));
        return ScaleCoefficientCF(contracted, ConstantCF(scale));
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
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        return KFormCF(ScaleCoefficientCF(contracted, ConstantCF(sign)), n - k, ambient_dim);
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
        double sign = ((p + q) % 2 == 0) ? 1.0 : -1.0;
        return DoubleFormCF(ScaleCoefficientCF(contracted_right, ConstantCF(sign)), n - p, n - q, ambient_dim);
    }

    shared_ptr<KFormCoefficientFunction> BBNDHodgeStarKForm(shared_ptr<KFormCoefficientFunction> a,
                                                            const RiemannianManifold &M)
    {
        int ambient_dim = M.Dimension();
        int n = ambient_dim - 2;
        int k = a->Degree();

        if (k > n)
            throw Exception("HodgeStar (BBND): form degree exceeds codimension-2 manifold dimension");

        if (n == 0)
            return a;

        auto star_vol = HodgeStar(a, M, VOL);
        auto n1 = M.GetEdgeNormal(0);
        auto cn1 = M.GetEdgeConormal(0);
        auto c1 = M.Contraction(star_vol, n1);
        auto c2 = M.Contraction(c1, cn1);

        // Codim-2 induced orientation uses two contractions; no extra k-dependent sign.
        return KFormCF(c2->GetFullCoefficient(), n - k, ambient_dim);
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

        if (n == 0)
            return a;

        auto n1 = M.GetEdgeNormal(0);
        auto cn1 = M.GetEdgeConormal(0);
        auto star_vol = HodgeStar(a, M, VOL);

        int left_deg = star_vol->LeftDegree();
        auto c_l1 = M.Contraction(star_vol, n1, 0);
        auto c_l2 = M.Contraction(c_l1, cn1, 0);
        auto c_r1 = M.Contraction(c_l2, n1, size_t(left_deg - 2));
        auto c_r2 = M.Contraction(c_r1, cn1, size_t(left_deg - 2));

        // Codim-2 induced orientation uses two contractions per slot. No extra degree sign.
        return DoubleFormCF(c_r2->GetFullCoefficient(), n - p, n - q, ambient_dim);
    }

} // namespace
