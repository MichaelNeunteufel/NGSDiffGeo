// Implementation fragment included only by ../kforms.cpp.
// Dense reference nodes and shared signed-table evaluation kernels.

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

// Full and block alternation share one signed-linear code-generation rule.
static void GenerateSignedLinearCode(
    Code &code, const CoefficientFunction &output,
    const CoefficientFunction &input, int input_index, int index,
    const std::vector<int> &valid_indices, const std::vector<int> &signs,
    const std::vector<int> &lin_table)
{
    DeclareTensorFieldGeneratedCoefficient(
        code, index, output.Dimensions(), output.IsComplex());
    size_t vi = 0;
    for (size_t component = 0; component < output.Dimension(); ++component)
    {
        const int idx = int(component);
        if (vi >= valid_indices.size() || valid_indices[vi] != idx)
        {
            code.body += Var(index, idx, output.Dimensions()).Assign(string("0.0"), false);
            continue;
        }
        CodeExpr result;
        const size_t base = vi * signs.size();
        for (size_t term = 0; term < signs.size(); ++term)
        {
            CodeExpr value = Var(input_index, lin_table[base + term], input.Dimensions());
            AccumulateSignedCodeExpression(
                result, std::move(value), signs[term], term);
        }
        code.body += Var(index, idx, output.Dimensions()).Assign(result.S(), false);
        ++vi;
    }
}

/**
 * Unnormalized signed sum over all permutations of the tensor axes.
 *
 * Each node builds lookup tables used by interpreted evaluation,
 * generated code, and nonzero-pattern propagation.
 */
class AlternationCoefficientFunction : public T_CoefficientFunction<AlternationCoefficientFunction>
{
    shared_ptr<CoefficientFunction> c1;
    int rank;
    int dim;
    std::vector<int> signs;
    std::vector<int> valid_indices;
    std::vector<int> lin_table;

public:
    AlternationCoefficientFunction(shared_ptr<CoefficientFunction> ac1, int arank, int adim)
        : T_CoefficientFunction<AlternationCoefficientFunction>(
              RequireNonNull(ac1, "AlternationCF")->Dimension(),
              RequireNonNull(ac1, "AlternationCF")->IsComplex()),
          c1(ac1), rank(arank), dim(adim)
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
        this->elementwise_constant = c1->ElementwiseConstant();

        const auto &data = GetSignedPermutations(
            PermutationSpec{PermutationFamily::Full, rank, 0});
        const auto &perms = data.perms;
        signs = data.signs;

        int comp_dim = this->Dimension();
        valid_indices.reserve(comp_dim);
        lin_table.clear();
        lin_table.reserve(size_t(comp_dim) * signs.size());

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
            for (size_t p = 0; p < signs.size(); ++p)
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

    bool DefinedOn(const ElementTransformation &trafo) override
    {
        return c1->DefinedOn(trafo);
    }

    void CalcEquivalenceKey() override
    {
        equivalence_key = "AlternationCF[rank=" + ToString(rank) +
                          ",dim=" + ToString(dim) + "](" +
                          c1->EquivalenceKey() + ")";
    }

    int Rank() const { return rank; }
    int DimSpace() const { return dim; }

    auto GetCArgs() const { return tuple{c1, rank, dim}; }

    void DoArchive(Archive &ar) override
    {
    }
    virtual void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
    {
        GenerateSignedLinearCode(code, *this, *c1, inputs[0], index,
                                 valid_indices, signs, lin_table);
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
        ApplyLinearNonZeroPattern(valid_indices, lin_table, signs.size(), input, values);
    }

    virtual void NonZeroPattern(const class ProxyUserData &ud,
                                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
                                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
    {
        ApplyLinearNonZeroPattern(valid_indices, lin_table, signs.size(), input[0], values);
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

    double Evaluate(const BaseMappedIntegrationPoint &ip) const override
    {
        if (Dimensions().Size() != 0)
            throw Exception("AlternationCoefficientFunction: scalar evaluation requires scalar shape");
        return T_CoefficientFunction<AlternationCoefficientFunction>::Evaluate(ip);
    }

    template <typename MIR, typename T, ORDERING ORD>
    void T_Evaluate(const MIR &mir, BareSliceMatrix<T, ORD> values) const
    {
        int comp_dim = this->Dimension();

        Array<T> temp(comp_dim * mir.Size());
        FlatMatrix<T, ORD> input(comp_dim, mir.Size(), temp.Data());
        c1->Evaluate(mir, input);
        ApplySignedLinearTable(mir.Size(), comp_dim, valid_indices, signs, lin_table, signs.size(), input, values);
    }

    template <typename MIR, typename T, ORDERING ORD>
    void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                    BareSliceMatrix<T, ORD> values) const
    {
        ApplySignedLinearTable(ir.Size(), this->Dimension(), valid_indices, signs, lin_table, signs.size(), input[0], values);
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

        if (var->Dimensions().Size() != 0)
            return CoefficientFunction::DiffJacobi(var, cache);

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
    T = RequireNonNull(std::move(T), "AlternationCF");
    return make_shared<AlternationCoefficientFunction>(T, rank, dim);
}

/// Unnormalized alternation restricted to one contiguous block of axes.
class BlockAlternationCoefficientFunction : public T_CoefficientFunction<BlockAlternationCoefficientFunction>
{
    shared_ptr<CoefficientFunction> c1;
    int rank_total;
    int block_start;
    int block_len;
    int dim;
    std::vector<int> signs;
    std::vector<int> valid_indices;
    std::vector<int> lin_table;

public:
    BlockAlternationCoefficientFunction(shared_ptr<CoefficientFunction> ac1,
                                        int arank_total,
                                        int ablock_start,
                                        int ablock_len)
        : T_CoefficientFunction<BlockAlternationCoefficientFunction>(
              RequireNonNull(ac1, "BlockAlternationCF")->Dimension(),
              RequireNonNull(ac1, "BlockAlternationCF")->IsComplex()),
          c1(ac1), rank_total(arank_total), block_start(ablock_start), block_len(ablock_len)
    {
        if (rank_total < 0 || rank_total > MAX_FORM_RANK)
            throw Exception("BlockAlternationCF: only ranks 0-" + ToString(MAX_FORM_RANK) + " supported");
        if (block_len < 0 || block_len > MAX_PERMUTATION_RANK)
            throw Exception("BlockAlternationCF: block size must be in [0, " + ToString(MAX_PERMUTATION_RANK) + "]");
        if (block_start < 0 || block_start > rank_total || block_len > rank_total - block_start)
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
        this->elementwise_constant = c1->ElementwiseConstant();

        const auto &data = GetSignedPermutations(
            PermutationSpec{PermutationFamily::Full, block_len, 0});
        const auto &perms = data.perms;
        signs = data.signs;

        int comp_dim = this->Dimension();
        valid_indices.reserve(comp_dim);
        lin_table.reserve(size_t(comp_dim) * signs.size());

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
            for (size_t p = 0; p < signs.size(); ++p)
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

    bool DefinedOn(const ElementTransformation &trafo) override
    {
        return c1->DefinedOn(trafo);
    }

    void CalcEquivalenceKey() override
    {
        equivalence_key = "BlockAlternationCF[rank=" + ToString(rank_total) +
                          ",start=" + ToString(block_start) +
                          ",length=" + ToString(block_len) +
                          ",dim=" + ToString(dim) + "](" +
                          c1->EquivalenceKey() + ")";
    }

    auto GetCArgs() const
    {
        return tuple{c1, rank_total, block_start, block_len};
    }

    void DoArchive(Archive &ar) override
    {
    }

    virtual void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
    {
        GenerateSignedLinearCode(code, *this, *c1, inputs[0], index,
                                 valid_indices, signs, lin_table);
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
        ApplyLinearNonZeroPattern(valid_indices, lin_table, signs.size(), input, values);
    }

    virtual void NonZeroPattern(const class ProxyUserData &ud,
                                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
                                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
    {
        ApplyLinearNonZeroPattern(valid_indices, lin_table, signs.size(), input[0], values);
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

    double Evaluate(const BaseMappedIntegrationPoint &ip) const override
    {
        if (Dimensions().Size() != 0)
            throw Exception("BlockAlternationCoefficientFunction: scalar evaluation requires scalar shape");
        return T_CoefficientFunction<BlockAlternationCoefficientFunction>::Evaluate(ip);
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
        ApplySignedLinearTable(nir, this->Dimension(), valid_indices, signs, lin_table, signs.size(), input, values);
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

        if (var->Dimensions().Size() != 0)
            return CoefficientFunction::DiffJacobi(var, cache);

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
    if (block_start < 0 || block_start > rank_total || block_len > rank_total - block_start)
        throw Exception("BlockAlternationByPermutationCF: block range out of bounds");
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

    if (rank_total > 0)
        ValidateFormSpaceDimension(int(tf->Dimensions()[0]), false, "BlockAlternationByPermutationCF");
    if (block_len <= 1)
        return T;

    return make_shared<BlockAlternationCoefficientFunction>(tf, rank_total, block_start, block_len);
}

int CheckedDoubleWedgeComponentCount(
    const shared_ptr<DoubleFormCoefficientFunction> &a,
    const shared_ptr<DoubleFormCoefficientFunction> &b)
{
    RequireNonNull(a, "DoubleFormWedgeCF");
    RequireNonNull(b, "DoubleFormWedgeCF");
    if (a->DimensionOfSpace() != b->DimensionOfSpace())
        throw Exception("DoubleFormWedgeCF: input double-forms must have the same dimension of space");
    const int rank = a->LeftDegree() + a->RightDegree() +
                     b->LeftDegree() + b->RightDegree();
    if (rank > MAX_FORM_RANK)
        throw Exception("DoubleFormWedgeCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");
    int count = 1;
    for (int i = 0; i < rank; ++i)
        count *= a->DimensionOfSpace(); // bounded by 4^8
    return count;
}

int EncodeRowMajor(const std::array<int, MAX_FORM_RANK> &multi, const std::vector<int> &positions, int dim)
{
    int lin = 0;
    for (int pos : positions)
        lin = lin * dim + multi[size_t(pos)];
    return lin;
}

/**
 * Double-form wedge evaluated from precomputed left/right shuffle tables.
 *
 * Result axes are ordered as the combined left block followed by the
 * combined right block.
 */
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
              CheckedDoubleWedgeComponentCount(aa, bb),
              RequireNonNull(aa, "DoubleFormWedgeCF")->IsComplex() ||
                  RequireNonNull(bb, "DoubleFormWedgeCF")->IsComplex()),
          a(aa), b(bb),
          p(aa->LeftDegree()), q(aa->RightDegree()),
          r(bb->LeftDegree()), s(bb->RightDegree()),
          dim(aa->DimensionOfSpace()),
          total(p + q + r + s)
    {
        Array<int> dims(total);
        for (int i = 0; i < total; ++i)
            dims[i] = dim;
        this->SetDimensions(dims);
        this->elementwise_constant = a->ElementwiseConstant() && b->ElementwiseConstant();

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
            auto ca = MakeComponentCoefficientFunction(a->GetFullCoefficient(), lin_a[i]);
            auto cb = MakeComponentCoefficientFunction(b->GetFullCoefficient(), lin_b[i]);
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

    bool DefinedOn(const ElementTransformation &trafo) override
    {
        return a->DefinedOn(trafo) && b->DefinedOn(trafo);
    }

    void CalcEquivalenceKey() override
    {
        equivalence_key = "DoubleFormWedgeCF[p=" + ToString(p) +
                          ",q=" + ToString(q) + ",r=" + ToString(r) +
                          ",s=" + ToString(s) + ",dim=" + ToString(dim) +
                          "](" + a->EquivalenceKey() + "," +
                          b->EquivalenceKey() + ")";
    }

    auto GetCArgs() const { return tuple{a, b}; }

    void DoArchive(Archive &ar) override
    {
    }

    virtual void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
    {
        DeclareTensorFieldGeneratedCoefficient(code, index, Dimensions(), IsComplex());

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
                AccumulateSignedCodeExpression(
                    result, std::move(term), signs[base + t], t);
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
        Vector<AutoDiffDiff<1, NonZero>> left(a->Dimension()), right(b->Dimension());
        a->NonZeroPattern(ud, left);
        b->NonZeroPattern(ud, right);
        FlatVector<AutoDiffDiff<1, NonZero>> input_storage[] = {left, right};
        FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs(2, input_storage);
        NonZeroPattern(ud, inputs, values);
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

        auto ta = DoubleFormCF(a->Transform(transformation), p, q, dim);
        auto tb = DoubleFormCF(b->Transform(transformation), r, s, dim);
        auto newcf = make_shared<DoubleFormWedgeCoefficientFunction>(ta, tb);
        transformation.cache[thisptr] = newcf;
        return newcf;
    }

    using T_CoefficientFunction<DoubleFormWedgeCoefficientFunction>::Evaluate;

    double Evaluate(const BaseMappedIntegrationPoint &ip) const override
    {
        if (Dimensions().Size() != 0)
            throw Exception("DoubleFormWedgeCoefficientFunction: scalar evaluation requires scalar shape");
        return T_CoefficientFunction<DoubleFormWedgeCoefficientFunction>::Evaluate(ip);
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
        static_cast<const CoefficientFunction &>(*a).Evaluate(mir, values_a);
        static_cast<const CoefficientFunction &>(*b).Evaluate(mir, values_b);

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
        auto da = DoubleFormCF(a->Diff(var, dir), p, q, dim);
        auto db = DoubleFormCF(b->Diff(var, dir), r, s, dim);
        return SymbolicSumCF(Wedge(da, b), Wedge(a, db));
    }

    shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
    {
        auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        if (cache.find(thisptr) != cache.end())
            return cache[thisptr];

        if (var->Dimensions().Size() != 0)
            return CoefficientFunction::DiffJacobi(var, cache);

        if (this == var)
            return IdentityCF(this->Dimensions());

        auto da = DoubleFormCF(a->DiffJacobi(var, cache), p, q, dim);
        auto db = DoubleFormCF(b->DiffJacobi(var, cache), r, s, dim);
        auto res = SymbolicSumCF(Wedge(da, b), Wedge(a, db));
        cache[thisptr] = res;
        return res;
    }

    virtual bool IsZeroCF() const override { return structural_zero; }
};

// Dense reference nodes own their archive registrations.
static ngcore::RegisterClassForArchive<AlternationCoefficientFunction,
                                       CoefficientFunction>
    reg_alternation_cf;
static ngcore::RegisterClassForArchive<BlockAlternationCoefficientFunction,
                                       CoefficientFunction>
    reg_block_alternation_cf;
static ngcore::RegisterClassForArchive<DoubleFormWedgeCoefficientFunction,
                                       CoefficientFunction>
    reg_double_form_wedge_cf;

void ExportAlternationBinding(py::module m)
{
    py::class_<AlternationCoefficientFunction,
               CoefficientFunction,
               shared_ptr<AlternationCoefficientFunction>>(
        m, "Alternation",
        "Unnormalized signed sum over all tensor-axis permutations.")
        .def(py::init([](shared_ptr<CoefficientFunction> cf,
                         int rank, int dim)
                      {
                     auto base = AlternationCF(cf, rank, dim);
                     auto casted =
                         dynamic_pointer_cast<AlternationCoefficientFunction>(
                             base);
                     if (!casted)
                         throw Exception(
                             "Alternation pybind: returned object is not "
                             "AlternationCoefficientFunction");
                     return casted; }),
             py::arg("cf"), py::arg("rank"), py::arg("dim"))
        .def_property_readonly(
            "rank", &AlternationCoefficientFunction::Rank)
        .def_property_readonly(
            "dim", &AlternationCoefficientFunction::DimSpace)
        .def(NGSPickle<AlternationCoefficientFunction>());
}
