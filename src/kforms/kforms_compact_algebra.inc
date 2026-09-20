// Implementation fragment included only by ../kforms.cpp.
// Compact wedge lookup tables and evaluation nodes.

namespace kforms_internal
{
    shared_ptr<DoubleFormCoefficientFunction> AddDoubleForms(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right,
        bool subtract)
    {
        left = RequireCompactNonNull(std::move(left));
        right = RequireCompactNonNull(std::move(right));
        if (left->LeftDegree() != right->LeftDegree() ||
            left->RightDegree() != right->RightDegree() ||
            left->DimensionOfSpace() != right->DimensionOfSpace())
            throw Exception("AddDoubleForms: form metadata mismatch");
        auto left_view = TryGetProvenCompactInput(left);
        auto right_view = TryGetProvenCompactInput(right);
        if (!left_view || !right_view)
        {
            auto right_value =
                subtract
                    ? ScaleCoefficientCF(right, ConstantCF(-1.0))
                    : static_pointer_cast<CoefficientFunction>(right);
            return DoubleFormCF(
                SymbolicSumCF(left, right_value), left->LeftDegree(),
                left->RightDegree(), left->DimensionOfSpace());
        }

        const bool left_zero = IsConstantZero(left_view->independent);
        const bool right_zero = IsConstantZero(right_view->independent);
        shared_ptr<CoefficientFunction> independent;
        if (left_zero && right_zero)
            independent = ZeroCF(left_view->independent->Dimensions());
        else if (left_zero)
            independent = subtract ? -right_view->independent
                                   : right_view->independent;
        else if (right_zero)
            independent = left_view->independent;
        else
            independent = subtract
                              ? left_view->independent -
                                    right_view->independent
                              : left_view->independent +
                                    right_view->independent;
        const auto operation =
            subtract ? CompactDoubleFormOperation::Difference
                     : CompactDoubleFormOperation::Sum;
        auto facade =
            make_shared<CompactDoubleFormOperationCoefficientFunction>(
                int(operation),
                Array<shared_ptr<CoefficientFunction>>{left, right},
                std::move(independent), left->LeftDegree(),
                left->RightDegree(), left->DimensionOfSpace());
        return DoubleFormCF(facade, left->LeftDegree(),
                            left->RightDegree(),
                            left->DimensionOfSpace());
    }

    shared_ptr<DoubleFormCoefficientFunction> ScaleDoubleForm(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> scalar)
    {
        form = RequireCompactNonNull(std::move(form));
        scalar = RequireCompactNonNull(std::move(scalar));
        if (scalar->Dimensions().Size() != 0)
            throw Exception("ScaleDoubleForm: factor must be scalar");
        auto view = TryGetProvenCompactInput(form);
        if (!view)
            return DoubleFormCF(
                ScaleCoefficientCF(form, scalar), form->LeftDegree(),
                form->RightDegree(), form->DimensionOfSpace());
        auto native_scalar = NativeCoefficientValue(scalar);
        auto independent =
            IsConstantZero(view->independent) ||
                    IsProvablyZeroNativeValue(native_scalar)
                ? ZeroCF(view->independent->Dimensions())
                : ScaleCoefficientCF(view->independent, native_scalar);
        auto facade =
            make_shared<CompactDoubleFormOperationCoefficientFunction>(
                int(CompactDoubleFormOperation::Scale),
                Array<shared_ptr<CoefficientFunction>>{form, scalar},
                std::move(independent), form->LeftDegree(),
                form->RightDegree(), form->DimensionOfSpace());
        return DoubleFormCF(facade, form->LeftDegree(),
                            form->RightDegree(),
                            form->DimensionOfSpace());
    }

    struct CompactDenseFormEntry
    {
        int dense_index;
        int compact_index;
        int sign;
    };

    struct CompactWedgeTable
    {
        int output_size;
        int full_output_size;
        int left_size;
        int right_size;
        size_t terms_per_index;
        std::vector<int> signs;
        std::vector<int> lin_left;
        std::vector<int> lin_right;
        std::vector<CompactDenseFormEntry> dense_entries;
        std::vector<int> canonical_dense;

        size_t OwnedStorageBytesLowerBound() const
        {
            return sizeof(CompactWedgeTable) +
                   (signs.capacity() + lin_left.capacity() +
                    lin_right.capacity() + canonical_dense.capacity()) *
                       sizeof(int) +
                   dense_entries.capacity() *
                       sizeof(CompactDenseFormEntry);
        }
    };

    shared_ptr<const CompactWedgeTable>
    GetCompactWedgeTable(int dim, int p, int q, int r, int s)
    {
        if (dim < 1 || dim > MAX_SPACE_DIM || p < 0 || q < 0 ||
            r < 0 || s < 0 || p + r > dim || q + s > dim)
            throw Exception(
                "CompactWedgeTable: invalid dimension or degrees");
        constexpr size_t extent = MAX_SPACE_DIM + 1;
        constexpr size_t cache_size =
            extent * extent * extent * extent * extent;
        size_t key = size_t(dim);
        for (int degree : {p, q, r, s})
            key = key * extent + size_t(degree);
        static std::array<std::once_flag, cache_size> flags;
        static std::array<shared_ptr<const CompactWedgeTable>,
                          cache_size>
            cache;

        std::call_once(flags[key], [&]
                       {
                auto left_left_basis = GetFormBasis(dim, p);
                auto left_right_basis = GetFormBasis(dim, q);
                auto right_left_basis = GetFormBasis(dim, r);
                auto right_right_basis = GetFormBasis(dim, s);
                auto output_left_basis = GetFormBasis(dim, p + r);
                auto output_right_basis = GetFormBasis(dim, q + s);
                auto table = make_shared<CompactWedgeTable>();
                table->output_size =
                    int(output_left_basis->IndependentSize() *
                        output_right_basis->IndependentSize());
                table->full_output_size =
                    DenseFormComponentCount(dim, p + r) *
                    DenseFormComponentCount(dim, q + s);
                table->left_size =
                    int(left_left_basis->IndependentSize() *
                        left_right_basis->IndependentSize());
                table->right_size =
                    int(right_left_basis->IndependentSize() *
                        right_right_basis->IndependentSize());
                table->terms_per_index =
                    output_left_basis->Shuffles(p).size() *
                    output_right_basis->Shuffles(q).size();
                const int storage_sign = (p * r + q * s) % 2 ? -1 : 1;
                const size_t entries =
                    size_t(table->output_size) * table->terms_per_index;
                table->signs.reserve(table->terms_per_index);
                table->lin_left.reserve(entries);
                table->lin_right.reserve(entries);
                table->canonical_dense.assign(
                    size_t(table->output_size), -1);
                auto alternating_dense_size = [dim](int degree) {
                    size_t result = 1;
                    for (int slot = 0; slot < degree; ++slot)
                        result *= size_t(dim - slot);
                    return result;
                };
                table->dense_entries.reserve(
                    alternating_dense_size(p + r) *
                    alternating_dense_size(q + s));

                const auto &left_shuffles = output_left_basis->Shuffles(p);
                const auto &right_shuffles = output_right_basis->Shuffles(q);
                for (const auto &left_shuffle : left_shuffles)
                    for (const auto &right_shuffle : right_shuffles)
                        table->signs.push_back(
                            left_shuffle.sign * right_shuffle.sign *
                            storage_sign);

                for (size_t output_left = 0;
                     output_left < output_left_basis->IndependentSize();
                     ++output_left)
                    for (size_t output_right = 0;
                         output_right < output_right_basis->IndependentSize();
                         ++output_right)
                    {
                        const auto &canonical_left =
                            output_left_basis->CanonicalIndex(output_left);
                        const auto &canonical_right =
                            output_right_basis->CanonicalIndex(output_right);
                        for (const auto &left_shuffle : left_shuffles)
                            for (const auto &right_shuffle : right_shuffles)
                            {
                                std::vector<int> a_left, b_left,
                                    a_right, b_right;
                                for (int position :
                                     left_shuffle.left_positions)
                                    a_left.push_back(
                                        canonical_left[size_t(position)]);
                                for (int position :
                                     left_shuffle.right_positions)
                                    b_left.push_back(
                                        canonical_left[size_t(position)]);
                                for (int position :
                                     right_shuffle.left_positions)
                                    a_right.push_back(
                                        canonical_right[size_t(position)]);
                                for (int position :
                                     right_shuffle.right_positions)
                                    b_right.push_back(
                                        canonical_right[size_t(position)]);

                                table->lin_left.push_back(
                                    left_left_basis->CompactIndex(a_left) *
                                        int(left_right_basis
                                                ->IndependentSize()) +
                                    left_right_basis->CompactIndex(a_right));
                                table->lin_right.push_back(
                                    right_left_basis->CompactIndex(b_left) *
                                        int(right_right_basis
                                                ->IndependentSize()) +
                                    right_right_basis->CompactIndex(b_right));
                            }
                    }

                const size_t right_dense_size =
                    output_right_basis->DenseSize();
                for (int dense = 0; dense < table->full_output_size; ++dense)
                {
                    const auto left_entry = output_left_basis->DenseEntry(
                        size_t(dense) / right_dense_size);
                    const auto right_entry = output_right_basis->DenseEntry(
                        size_t(dense) % right_dense_size);
                    if (left_entry.sign == 0 || right_entry.sign == 0)
                        continue;
                    const int compact =
                        left_entry.independent_index *
                            int(output_right_basis->IndependentSize()) +
                        right_entry.independent_index;
                    const int sign = left_entry.sign * right_entry.sign;
                    table->dense_entries.push_back(
                        {dense, compact, sign});
                    if (sign > 0 &&
                        table->canonical_dense[size_t(compact)] < 0)
                        table->canonical_dense[size_t(compact)] = dense;
                }
                for (int dense : table->canonical_dense)
                    if (dense < 0)
                        throw Exception(
                            "CompactWedgeTable: incomplete basis map");

                cache[key] = std::move(table); });
        return cache[key];
    }

    template <typename T, typename LEFT, typename RIGHT>
    T EvaluateCompactWedgeEntry(
        const CompactWedgeTable &table, int output, size_t ip,
        LEFT left, RIGHT right)
    {
        T sum = T(0);
        const size_t base =
            size_t(output) * table.terms_per_index;
        for (size_t term = 0; term < table.terms_per_index; ++term)
            sum += T(table.signs[term]) *
                   left(table.lin_left[base + term], ip) *
                   right(table.lin_right[base + term], ip);
        return sum;
    }

    CodeExpr CompactWedgeCodeExpression(
        const CompactWedgeTable &table, int output,
        int left_input, int right_input,
        FlatArray<int> left_dimensions,
        FlatArray<int> right_dimensions)
    {
        CodeExpr sum;
        const size_t base =
            size_t(output) * table.terms_per_index;
        for (size_t term = 0; term < table.terms_per_index; ++term)
        {
            CodeExpr value =
                Var(left_input, table.lin_left[base + term],
                    left_dimensions) *
                Var(right_input, table.lin_right[base + term],
                    right_dimensions);
            AccumulateSignedCodeExpression(
                sum, std::move(value), table.signs[term], term);
        }
        return sum;
    }

    AutoDiffDiff<1, NonZero> CompactWedgePattern(
        const CompactWedgeTable &table, int output,
        FlatVector<AutoDiffDiff<1, NonZero>> left,
        FlatVector<AutoDiffDiff<1, NonZero>> right)
    {
        auto sum = AutoDiffDiff<1, NonZero>(false);
        const size_t base =
            size_t(output) * table.terms_per_index;
        for (size_t term = 0; term < table.terms_per_index; ++term)
            sum = sum + left(table.lin_left[base + term]) *
                            right(table.lin_right[base + term]);
        return sum;
    }

    class CompactWedgeIndependentCoefficientFunction
        : public T_CoefficientFunction<
              CompactWedgeIndependentCoefficientFunction>
    {
        using BASE = T_CoefficientFunction<
            CompactWedgeIndependentCoefficientFunction>;

        shared_ptr<CoefficientFunction> left;
        shared_ptr<CoefficientFunction> right;
        int p;
        int q;
        int r;
        int s;
        int dim;
        shared_ptr<const CompactWedgeTable> table;
        bool structural_zero;

        template <typename LEFT, typename RIGHT, typename VALUES>
        void EvaluateFromInputs(size_t points, LEFT left_values,
                                RIGHT right_values, VALUES values) const
        {
            using T = typename VALUES::TSCAL;
            for (size_t ip = 0; ip < points; ++ip)
                for (int output = 0; output < Dimension(); ++output)
                    values(output, ip) =
                        EvaluateCompactWedgeEntry<T>(
                            *table, output, ip, left_values,
                            right_values);
        }

    public:
        CompactWedgeIndependentCoefficientFunction(
            shared_ptr<CoefficientFunction> aleft,
            shared_ptr<CoefficientFunction> aright,
            int ap, int aq, int ar, int as, int adim,
            shared_ptr<const CompactWedgeTable> atable)
            : BASE(atable->output_size,
                   RequireCompactNonNull(aleft)->IsComplex() ||
                       RequireCompactNonNull(aright)->IsComplex()),
              left(std::move(aleft)), right(std::move(aright)),
              p(ap), q(aq), r(ar), s(as), dim(adim),
              table(std::move(atable)),
              structural_zero(left->IsZeroCF() || right->IsZeroCF())
        {
            const bool scalar_left =
                table->left_size == 1 && left->Dimensions().Size() == 0;
            const bool scalar_right =
                table->right_size == 1 && right->Dimensions().Size() == 0;
            if ((!scalar_left &&
                 left->Dimensions() != Array<int>{table->left_size}) ||
                (!scalar_right &&
                 right->Dimensions() != Array<int>{table->right_size}))
                throw Exception(
                    "CompactWedgeIndependentCF: input shape mismatch");

            SetDimensions(Array<int>{table->output_size});
            elementwise_constant =
                left->ElementwiseConstant() && right->ElementwiseConstant();
        }

        CompactWedgeIndependentCoefficientFunction(
            shared_ptr<CoefficientFunction> aleft,
            shared_ptr<CoefficientFunction> aright,
            int ap, int aq, int ar, int as, int adim)
            : CompactWedgeIndependentCoefficientFunction(
                  std::move(aleft), std::move(aright),
                  ap, aq, ar, as, adim,
                  GetCompactWedgeTable(adim, ap, aq, ar, as))
        {
        }

        auto GetCArgs() const
        {
            return tuple{left, right, p, q, r, s, dim};
        }

        string GetDescription() const override
        {
            return "CompactWedgeIndependentCF";
        }

        void CalcEquivalenceKey() override
        {
            equivalence_key =
                GetDescription() + "[p=" + ToString(p) +
                ",q=" + ToString(q) + ",r=" + ToString(r) +
                ",s=" + ToString(s) + ",dim=" + ToString(dim) +
                "](" + left->EquivalenceKey() + "," +
                right->EquivalenceKey() + ")";
        }

        void DoArchive(Archive &) override {}

        bool DefinedOn(const ElementTransformation &trafo) override
        {
            return left->DefinedOn(trafo) && right->DefinedOn(trafo);
        }

        bool IsZeroCF() const override
        {
            return structural_zero;
        }

        Array<shared_ptr<CoefficientFunction>>
        InputCoefficientFunctions() const override
        {
            return {left, right};
        }

        void TraverseTree(
            const function<void(CoefficientFunction &)> &func) override
        {
            left->TraverseTree(func);
            right->TraverseTree(func);
            func(*this);
        }

        using BASE::Evaluate;

        double Evaluate(
            const BaseMappedIntegrationPoint &ip) const override
        {
            if (Dimensions().Size() != 0)
                throw Exception(
                    "CompactWedgeIndependentCF: scalar evaluation requires scalar shape");
            return BASE::Evaluate(ip);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir,
                        BareSliceMatrix<T, ORD> values) const
        {
            auto &heap = TLHeap();
            HeapReset reset(heap);
            FlatMatrix<T, ORD> left_values(
                left->Dimension(), mir.Size(), heap);
            FlatMatrix<T, ORD> right_values(
                right->Dimension(), mir.Size(), heap);
            left->Evaluate(mir, left_values);
            right->Evaluate(mir, right_values);
            EvaluateFromInputs(mir.Size(), left_values,
                               right_values, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(
            const MIR &mir,
            FlatArray<BareSliceMatrix<T, ORD>> inputs,
            BareSliceMatrix<T, ORD> values) const
        {
            EvaluateFromInputs(mir.Size(), inputs[0], inputs[1], values);
        }

        void GenerateCode(Code &code, FlatArray<int> inputs,
                          int index) const override
        {
            DeclareTensorFieldGeneratedCoefficient(
                code, index, Dimensions(), IsComplex());
            for (int output = 0; output < Dimension(); ++output)
            {
                auto sum = CompactWedgeCodeExpression(
                    *table, output, inputs[0], inputs[1],
                    left->Dimensions(), right->Dimensions());
                code.body += Var(index, output, Dimensions()).Assign(sum.S(), false);
            }
        }

        void NonZeroPattern(
            const ProxyUserData &ud,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            Vector<AutoDiffDiff<1, NonZero>> left_values(
                left->Dimension());
            Vector<AutoDiffDiff<1, NonZero>> right_values(
                right->Dimension());
            left->NonZeroPattern(ud, left_values);
            right->NonZeroPattern(ud, right_values);
            FlatVector<AutoDiffDiff<1, NonZero>> input_storage[] = {
                left_values, right_values};
            NonZeroPattern(
                ud,
                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                    2, input_storage),
                values);
        }

        void NonZeroPattern(
            const ProxyUserData &,
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            values = AutoDiffDiff<1, NonZero>(false);
            for (int output = 0; output < Dimension(); ++output)
                values(output) = CompactWedgePattern(
                    *table, output, inputs[0], inputs[1]);
        }

        shared_ptr<CoefficientFunction> Transform(
            T_Transform &transformation) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(
                shared_from_this());
            if (auto found = transformation.cache.find(self);
                found != transformation.cache.end())
                return found->second;
            if (auto found = transformation.replace.find(self);
                found != transformation.replace.end())
                return transformation.cache[self] = found->second;
            return transformation.cache[self] = make_shared<
                       CompactWedgeIndependentCoefficientFunction>(
                       left->Transform(transformation),
                       right->Transform(transformation), p, q, r, s, dim);
        }

        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            auto left_term = make_shared<
                CompactWedgeIndependentCoefficientFunction>(
                left->Diff(var, dir), right, p, q, r, s, dim);
            auto right_term = make_shared<
                CompactWedgeIndependentCoefficientFunction>(
                left, right->Diff(var, dir), p, q, r, s, dim);
            return left_term + right_term;
        }

        shared_ptr<CoefficientFunction> DiffJacobi(
            const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(
                shared_from_this());
            if (auto found = cache.find(self); found != cache.end())
                return found->second;
            if (this == var)
                return cache[self] = IdentityCF(Dimensions());
            if (var->Dimensions().Size() == 0)
                return cache[self] = Diff(
                           var, make_shared<ConstantCoefficientFunction>(1.0));
            return cache[self] =
                       CoefficientFunction::DiffJacobi(var, cache);
        }
    };

    class CompactDoubleWedgeFullCoefficientFunction
        : public T_CoefficientFunction<
              CompactDoubleWedgeFullCoefficientFunction>
    {
        using BASE = T_CoefficientFunction<
            CompactDoubleWedgeFullCoefficientFunction>;

        shared_ptr<CoefficientFunction> left;
        shared_ptr<CoefficientFunction> right;
        int p;
        int q;
        int r;
        int s;
        int dim;
        shared_ptr<const CompactWedgeTable> table;
        bool structural_zero;

        template <typename LEFT, typename RIGHT, typename VALUES>
        void EvaluateFromInputs(size_t points, LEFT left_values,
                                RIGHT right_values, VALUES values) const
        {
            using T = typename VALUES::TSCAL;
            for (size_t ip = 0; ip < points; ++ip)
            {
                for (int dense = 0; dense < Dimension(); ++dense)
                    values(dense, ip) = T(0);
                for (int output = 0;
                     output < table->output_size; ++output)
                    values(
                        table->canonical_dense[size_t(output)], ip) =
                        EvaluateCompactWedgeEntry<T>(
                            *table, output, ip, left_values,
                            right_values);

                for (const auto &entry : table->dense_entries)
                    if (entry.dense_index !=
                        table->canonical_dense[size_t(entry.compact_index)])
                        values(entry.dense_index, ip) =
                            T(entry.sign) *
                            values(table->canonical_dense[size_t(entry.compact_index)],
                                   ip);
            }
        }

    public:
        CompactDoubleWedgeFullCoefficientFunction(
            shared_ptr<CoefficientFunction> aleft,
            shared_ptr<CoefficientFunction> aright,
            int ap, int aq, int ar, int as, int adim,
            shared_ptr<const CompactWedgeTable> atable)
            : BASE(atable->full_output_size,
                   RequireCompactNonNull(aleft)->IsComplex() ||
                       RequireCompactNonNull(aright)->IsComplex()),
              left(std::move(aleft)), right(std::move(aright)),
              p(ap), q(aq), r(ar), s(as), dim(adim),
              table(std::move(atable)),
              structural_zero(left->IsZeroCF() || right->IsZeroCF())
        {
            const bool scalar_left =
                table->left_size == 1 && left->Dimensions().Size() == 0;
            const bool scalar_right =
                table->right_size == 1 && right->Dimensions().Size() == 0;
            if ((!scalar_left &&
                 left->Dimensions() != Array<int>{table->left_size}) ||
                (!scalar_right &&
                 right->Dimensions() != Array<int>{table->right_size}))
                throw Exception(
                    "CompactDoubleWedgeFullCF: input shape mismatch");

            Array<int> dimensions(size_t(p + q + r + s));
            dimensions = dim;
            SetDimensions(dimensions);
            elementwise_constant =
                left->ElementwiseConstant() && right->ElementwiseConstant();
        }

        void ApplyPattern(
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const
        {
            Vector<AutoDiffDiff<1, NonZero>> compact(
                table->output_size);
            for (int output = 0; output < table->output_size; ++output)
                compact(output) = CompactWedgePattern(
                    *table, output, inputs[0], inputs[1]);
            values = AutoDiffDiff<1, NonZero>(false);
            for (const auto &entry : table->dense_entries)
                values(entry.dense_index) =
                    compact(entry.compact_index);
        }

        CompactDoubleWedgeFullCoefficientFunction(
            shared_ptr<CoefficientFunction> aleft,
            shared_ptr<CoefficientFunction> aright,
            int ap, int aq, int ar, int as, int adim)
            : CompactDoubleWedgeFullCoefficientFunction(
                  std::move(aleft), std::move(aright),
                  ap, aq, ar, as, adim,
                  GetCompactWedgeTable(adim, ap, aq, ar, as))
        {
        }

        auto GetCArgs() const
        {
            return tuple{left, right, p, q, r, s, dim};
        }

        string GetDescription() const override
        {
            return "CompactDoubleWedgeFullCF";
        }

        void CalcEquivalenceKey() override
        {
            equivalence_key =
                GetDescription() + "[p=" + ToString(p) +
                ",q=" + ToString(q) + ",r=" + ToString(r) +
                ",s=" + ToString(s) + ",dim=" + ToString(dim) +
                "](" + left->EquivalenceKey() + "," +
                right->EquivalenceKey() + ")";
        }

        void DoArchive(Archive &) override {}

        bool DefinedOn(const ElementTransformation &trafo) override
        {
            return left->DefinedOn(trafo) && right->DefinedOn(trafo);
        }

        bool IsZeroCF() const override
        {
            return structural_zero;
        }

        Array<shared_ptr<CoefficientFunction>>
        InputCoefficientFunctions() const override
        {
            return {left, right};
        }

        void TraverseTree(
            const function<void(CoefficientFunction &)> &func) override
        {
            left->TraverseTree(func);
            right->TraverseTree(func);
            func(*this);
        }

        using BASE::Evaluate;

        double Evaluate(
            const BaseMappedIntegrationPoint &ip) const override
        {
            if (Dimensions().Size() != 0)
                throw Exception(
                    "CompactDoubleWedgeFullCF: scalar evaluation requires scalar shape");
            return BASE::Evaluate(ip);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir,
                        BareSliceMatrix<T, ORD> values) const
        {
            auto &heap = TLHeap();
            HeapReset reset(heap);
            FlatMatrix<T, ORD> left_values(
                left->Dimension(), mir.Size(), heap);
            FlatMatrix<T, ORD> right_values(
                right->Dimension(), mir.Size(), heap);
            left->Evaluate(mir, left_values);
            right->Evaluate(mir, right_values);
            EvaluateFromInputs(mir.Size(), left_values,
                               right_values, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(
            const MIR &mir,
            FlatArray<BareSliceMatrix<T, ORD>> inputs,
            BareSliceMatrix<T, ORD> values) const
        {
            EvaluateFromInputs(mir.Size(), inputs[0], inputs[1], values);
        }

        void GenerateCode(Code &code, FlatArray<int> inputs,
                          int index) const override
        {
            DeclareTensorFieldGeneratedCoefficient(
                code, index, Dimensions(), IsComplex());
            for (int output = 0;
                 output < table->output_size; ++output)
            {
                auto sum = CompactWedgeCodeExpression(
                    *table, output, inputs[0], inputs[1],
                    left->Dimensions(), right->Dimensions());
                const int dense =
                    table->canonical_dense[size_t(output)];
                code.body += Var(index, dense, Dimensions()).Assign(sum.S(), false);
            }
            size_t entry_index = 0;
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                if (entry_index >= table->dense_entries.size() ||
                    table->dense_entries[entry_index].dense_index != dense)
                {
                    code.body += Var(index, dense, Dimensions()).Assign(string("0.0"), false);
                    continue;
                }
                const auto &entry =
                    table->dense_entries[entry_index++];
                const int compact = entry.compact_index;
                const int source =
                    table->canonical_dense[size_t(compact)];
                if (dense == source)
                    continue;
                CodeExpr value = Var(index, source, Dimensions());
                if (entry.sign < 0)
                    value = CodeExpr("-(" + value.S() + ")");
                code.body += Var(index, dense, Dimensions()).Assign(value.S(), false);
            }
        }

        void NonZeroPattern(
            const ProxyUserData &ud,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            Vector<AutoDiffDiff<1, NonZero>> left_values(
                left->Dimension());
            Vector<AutoDiffDiff<1, NonZero>> right_values(
                right->Dimension());
            left->NonZeroPattern(ud, left_values);
            right->NonZeroPattern(ud, right_values);
            FlatVector<AutoDiffDiff<1, NonZero>> input_storage[] = {
                left_values, right_values};
            ApplyPattern(
                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                    2, input_storage),
                values);
        }

        void NonZeroPattern(
            const ProxyUserData &,
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            ApplyPattern(inputs, values);
        }

        shared_ptr<CoefficientFunction> Transform(
            T_Transform &transformation) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(
                shared_from_this());
            if (auto found = transformation.cache.find(self);
                found != transformation.cache.end())
                return found->second;
            if (auto found = transformation.replace.find(self);
                found != transformation.replace.end())
                return transformation.cache[self] = found->second;
            return transformation.cache[self] = make_shared<
                       CompactDoubleWedgeFullCoefficientFunction>(
                       left->Transform(transformation),
                       right->Transform(transformation), p, q, r, s, dim);
        }

        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            auto left_term = make_shared<
                CompactDoubleWedgeFullCoefficientFunction>(
                left->Diff(var, dir), right, p, q, r, s, dim);
            auto right_term = make_shared<
                CompactDoubleWedgeFullCoefficientFunction>(
                left, right->Diff(var, dir), p, q, r, s, dim);
            return left_term + right_term;
        }

        shared_ptr<CoefficientFunction> DiffJacobi(
            const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(
                shared_from_this());
            if (auto found = cache.find(self); found != cache.end())
                return found->second;
            if (this == var)
                return cache[self] = IdentityCF(Dimensions());
            if (var->Dimensions().Size() == 0)
                return cache[self] = Diff(
                           var, make_shared<ConstantCoefficientFunction>(1.0));
            return cache[self] =
                       CoefficientFunction::DiffJacobi(var, cache);
        }
    };

    static ngcore::RegisterClassForArchive<
        CompactWedgeIndependentCoefficientFunction,
        CoefficientFunction>
        reg_compact_wedge_independent_cf;
    static ngcore::RegisterClassForArchive<
        CompactDoubleWedgeFullCoefficientFunction,
        CoefficientFunction>
        reg_compact_double_wedge_full_cf;

    size_t CompactWedgeTableStorageBytesLowerBound(
        int dim, int p, int q, int r, int s)
    {
        return GetCompactWedgeTable(dim, p, q, r, s)
            ->OwnedStorageBytesLowerBound();
    }

} // namespace kforms_internal
