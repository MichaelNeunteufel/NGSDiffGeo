// Implementation fragment included only by ../kforms.cpp.
// Alternating storage facades, expansion tables, and their cache ownership.

namespace kforms_internal
{
    template <typename T>
    shared_ptr<T> RequireCompactNonNull(shared_ptr<T> ptr)
    {
        if (!ptr)
            throw Exception("AlternatingFormCF: input coefficient is null");
        return ptr;
    }

    bool IsProvablyZeroNativeValue(
        const shared_ptr<CoefficientFunction> &value)
    {
        if (IsConstantZero(value))
            return true;
        auto constant = dynamic_pointer_cast<ConstantCoefficientFunction>(
            value);
        return constant && constant->EvaluateConst() == 0.0;
    }

    class AlternatingFormCoefficientFunction
        : public T_CoefficientFunction<AlternatingFormCoefficientFunction>
    {
        using BASE = T_CoefficientFunction<AlternatingFormCoefficientFunction>;
        shared_ptr<CoefficientFunction> independent;
        int degree;
        int dim;
        shared_ptr<const FormBasis> basis;

    public:
        AlternatingFormCoefficientFunction(
            shared_ptr<CoefficientFunction> aindependent, int adegree, int adim)
            : BASE(DenseFormComponentCount(adim, adegree),
                   RequireCompactNonNull(aindependent)->IsComplex()),
              independent(std::move(aindependent)), degree(adegree), dim(adim),
              basis(GetFormBasis(dim, degree))
        {
            const bool scalar_singleton =
                basis->IndependentSize() == 1 &&
                independent->Dimensions().Size() == 0;
            if (!scalar_singleton &&
                independent->Dimensions() !=
                    Array<int>({int(basis->IndependentSize())}))
                throw Exception(
                    "AlternatingFormCF: expected independent dimensions (" +
                    ToString(basis->IndependentSize()) + "), got " +
                    ToString(independent->Dimensions()));
            Array<int> full_dimensions(static_cast<size_t>(degree));
            full_dimensions = dim;
            SetDimensions(full_dimensions);
            elementwise_constant = independent->ElementwiseConstant();
        }

        auto GetCArgs() const { return tuple{independent, degree, dim}; }
        string GetDescription() const override
        {
            return "AlternatingFormCF";
        }
        void CalcEquivalenceKey() override
        {
            equivalence_key = GetDescription() + "[k=" + ToString(degree) +
                              ",dim=" + ToString(dim) + "](" +
                              independent->EquivalenceKey() + ")";
        }
        void DoArchive(Archive &) override {}
        bool DefinedOn(const ElementTransformation &trafo) override
        {
            return independent->DefinedOn(trafo);
        }
        bool IsZeroCF() const override { return independent->IsZeroCF(); }
        Array<shared_ptr<CoefficientFunction>> InputCoefficientFunctions() const override
        {
            return {independent};
        }
        void TraverseTree(const function<void(CoefficientFunction &)> &func) override
        {
            independent->TraverseTree(func);
            func(*this);
        }

        using BASE::Evaluate;
        double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            if (degree != 0)
                throw Exception("AlternatingFormCF: scalar evaluation requires degree zero");
            return BASE::Evaluate(ip);
        }
        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir, BareSliceMatrix<T, ORD> values) const
        {
            auto &heap = TLHeap();
            HeapReset reset(heap);
            FlatMatrix<T, ORD> compact(
                independent->Dimension(), mir.Size(), heap);
            independent->Evaluate(mir, compact);
            Expand(mir.Size(), compact, values);
        }
        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir,
                        FlatArray<BareSliceMatrix<T, ORD>> inputs,
                        BareSliceMatrix<T, ORD> values) const
        {
            Expand(mir.Size(), inputs[0], values);
        }
        template <typename COMPACT, typename VALUES>
        void Expand(size_t points, const COMPACT &compact, VALUES values) const
        {
            const auto &entries = basis->DenseEntries();
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                const auto entry = entries[size_t(dense)];
                if (entry.sign == 0)
                    for (size_t ip = 0; ip < points; ++ip)
                        values(dense, ip) = 0.0;
                else
                    for (size_t ip = 0; ip < points; ++ip)
                        values(dense, ip) =
                            entry.sign * compact(entry.independent_index, ip);
            }
        }
        void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
        {
            DeclareTensorFieldGeneratedCoefficient(code, index, Dimensions(), IsComplex());
            const auto &entries = basis->DenseEntries();
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                const auto entry = entries[size_t(dense)];
                if (entry.sign == 0)
                    code.body += Var(index, dense, Dimensions()).Assign(CodeExpr(std::string("0.0")), false);
                else
                {
                    CodeExpr value = Var(inputs[0], entry.independent_index,
                                         independent->Dimensions());
                    if (entry.sign < 0)
                        value = CodeExpr("-(" + value.S() + ")");
                    code.body += Var(index, dense, Dimensions()).Assign(value.S(), false);
                }
            }
        }
        void NonZeroPattern(const ProxyUserData &ud,
                            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            Vector<AutoDiffDiff<1, NonZero>> compact(independent->Dimension());
            independent->NonZeroPattern(ud, compact);
            ApplyPattern(compact, values);
        }
        void NonZeroPattern(
            const ProxyUserData &,
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            ApplyPattern(inputs[0], values);
        }
        void ApplyPattern(FlatVector<AutoDiffDiff<1, NonZero>> compact,
                          FlatVector<AutoDiffDiff<1, NonZero>> values) const
        {
            values = AutoDiffDiff<1, NonZero>(false);
            const auto &entries = basis->DenseEntries();
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                const auto entry = entries[size_t(dense)];
                if (entry.sign != 0)
                    values(dense) = compact(entry.independent_index);
            }
        }
        shared_ptr<CoefficientFunction> Transform(T_Transform &transformation) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(shared_from_this());
            if (auto found = transformation.cache.find(self);
                found != transformation.cache.end())
                return found->second;
            if (auto found = transformation.replace.find(self);
                found != transformation.replace.end())
                return transformation.cache[self] = found->second;
            return transformation.cache[self] =
                       make_shared<AlternatingFormCoefficientFunction>(
                           independent->Transform(transformation), degree, dim);
        }
        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return make_shared<AlternatingFormCoefficientFunction>(
                independent->Diff(var, dir), degree, dim);
        }
        shared_ptr<CoefficientFunction> DiffJacobi(
            const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(shared_from_this());
            if (auto found = cache.find(self); found != cache.end())
                return found->second;
            if (this == var)
                return cache[self] = IdentityCF(Dimensions());
            if (var->Dimensions().Size() == 0)
                return cache[self] =
                           make_shared<AlternatingFormCoefficientFunction>(
                               independent->Diff(
                                   var,
                                   make_shared<ConstantCoefficientFunction>(1.0)),
                               degree, dim);
            return cache[self] = CoefficientFunction::DiffJacobi(var, cache);
        }

        shared_ptr<CoefficientFunction> IndependentValue() const
        {
            return independent;
        }
        int Degree() const { return degree; }
        int DimensionOfSpace() const { return dim; }
    };

    struct CompactDoubleFormExpansionTable
    {
        int independent_size;
        std::vector<int> independent_indices;
        std::vector<int> signs;

        size_t OwnedStorageBytesLowerBound() const
        {
            return sizeof(CompactDoubleFormExpansionTable) +
                   (independent_indices.capacity() +
                    signs.capacity()) *
                       sizeof(int);
        }
    };

    shared_ptr<const CompactDoubleFormExpansionTable>
    GetCompactDoubleFormExpansionTable(
        int dim, int left_degree, int right_degree)
    {
        if (dim < 1 || dim > MAX_SPACE_DIM || left_degree < 0 ||
            right_degree < 0 || left_degree > dim ||
            right_degree > dim)
            throw Exception(
                "CompactDoubleFormExpansionTable: invalid dimension or degrees");
        constexpr size_t extent = MAX_SPACE_DIM + 1;
        static std::array<
            std::array<std::array<std::once_flag, extent>, extent>,
            extent>
            flags;
        static std::array<
            std::array<
                std::array<
                    shared_ptr<const CompactDoubleFormExpansionTable>,
                    extent>,
                extent>,
            extent>
            cache;
        std::call_once(
            flags[size_t(dim)][size_t(left_degree)]
                 [size_t(right_degree)],
            [&]
            {
                auto left_basis = GetFormBasis(dim, left_degree);
                auto right_basis = GetFormBasis(dim, right_degree);
                auto table =
                    make_shared<CompactDoubleFormExpansionTable>();
                table->independent_size =
                    int(left_basis->IndependentSize() *
                        right_basis->IndependentSize());
                const size_t dense_size =
                    left_basis->DenseSize() * right_basis->DenseSize();
                table->independent_indices.assign(dense_size, -1);
                table->signs.assign(dense_size, 0);
                for (size_t dense = 0; dense < dense_size; ++dense)
                {
                    const auto left = left_basis->DenseEntry(
                        dense / right_basis->DenseSize());
                    const auto right = right_basis->DenseEntry(
                        dense % right_basis->DenseSize());
                    if (left.sign == 0 || right.sign == 0)
                        continue;
                    table->independent_indices[dense] =
                        left.independent_index *
                            int(right_basis->IndependentSize()) +
                        right.independent_index;
                    table->signs[dense] = left.sign * right.sign;
                }
                cache[size_t(dim)][size_t(left_degree)]
                     [size_t(right_degree)] = std::move(table);
            });
        return cache[size_t(dim)][size_t(left_degree)]
                    [size_t(right_degree)];
    }

    class AlternatingDoubleFormCoefficientFunction
        : public T_CoefficientFunction<AlternatingDoubleFormCoefficientFunction>
    {
        using BASE =
            T_CoefficientFunction<AlternatingDoubleFormCoefficientFunction>;
        shared_ptr<CoefficientFunction> independent;
        int left_degree;
        int right_degree;
        int dim;
        shared_ptr<const CompactDoubleFormExpansionTable> expansion;

    public:
        AlternatingDoubleFormCoefficientFunction(
            shared_ptr<CoefficientFunction> aindependent,
            int aleft_degree, int aright_degree, int adim)
            : BASE(DenseFormComponentCount(adim, aleft_degree) *
                       DenseFormComponentCount(adim, aright_degree),
                   RequireCompactNonNull(aindependent)->IsComplex()),
              independent(std::move(aindependent)),
              left_degree(aleft_degree), right_degree(aright_degree),
              dim(adim), expansion(GetCompactDoubleFormExpansionTable(
                             dim, left_degree, right_degree))
        {
            const size_t compact_size = size_t(expansion->independent_size);
            const bool scalar_singleton =
                compact_size == 1 && independent->Dimensions().Size() == 0;
            if (!scalar_singleton &&
                independent->Dimensions() !=
                    Array<int>({int(compact_size)}))
                throw Exception(
                    "AlternatingDoubleFormCF: expected independent dimensions (" +
                    ToString(compact_size) + "), got " +
                    ToString(independent->Dimensions()));
            Array<int> full_dimensions(
                size_t(left_degree + right_degree));
            full_dimensions = dim;
            SetDimensions(full_dimensions);
            elementwise_constant = independent->ElementwiseConstant();
        }

        auto GetCArgs() const
        {
            return tuple{independent, left_degree, right_degree, dim};
        }
        string GetDescription() const override
        {
            return "AlternatingDoubleFormCF";
        }
        void CalcEquivalenceKey() override
        {
            equivalence_key =
                GetDescription() + "[p=" + ToString(left_degree) +
                ",q=" + ToString(right_degree) +
                ",dim=" + ToString(dim) + "](" +
                independent->EquivalenceKey() + ")";
        }
        void DoArchive(Archive &) override {}
        bool DefinedOn(const ElementTransformation &trafo) override
        {
            return independent->DefinedOn(trafo);
        }
        bool IsZeroCF() const override { return independent->IsZeroCF(); }
        Array<shared_ptr<CoefficientFunction>>
        InputCoefficientFunctions() const override
        {
            return {independent};
        }
        void TraverseTree(
            const function<void(CoefficientFunction &)> &func) override
        {
            independent->TraverseTree(func);
            func(*this);
        }

        using BASE::Evaluate;
        double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            if (left_degree + right_degree != 0)
                throw Exception(
                    "AlternatingDoubleFormCF: scalar evaluation requires rank zero");
            return BASE::Evaluate(ip);
        }

        template <typename COMPACT, typename VALUES>
        void Expand(size_t points, const COMPACT &compact,
                    VALUES values) const
        {
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                const int compact_index =
                    expansion->independent_indices[size_t(dense)];
                if (compact_index < 0)
                    for (size_t ip = 0; ip < points; ++ip)
                        values(dense, ip) = 0.0;
                else
                {
                    const int sign = expansion->signs[size_t(dense)];
                    for (size_t ip = 0; ip < points; ++ip)
                        values(dense, ip) =
                            sign * compact(compact_index, ip);
                }
            }
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir,
                        BareSliceMatrix<T, ORD> values) const
        {
            auto &heap = TLHeap();
            HeapReset reset(heap);
            FlatMatrix<T, ORD> compact(
                independent->Dimension(), mir.Size(), heap);
            independent->Evaluate(mir, compact);
            Expand(mir.Size(), compact, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(
            const MIR &mir,
            FlatArray<BareSliceMatrix<T, ORD>> inputs,
            BareSliceMatrix<T, ORD> values) const
        {
            Expand(mir.Size(), inputs[0], values);
        }

        void GenerateCode(Code &code, FlatArray<int> inputs,
                          int index) const override
        {
            DeclareTensorFieldGeneratedCoefficient(
                code, index, Dimensions(), IsComplex());
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                const int compact_index =
                    expansion->independent_indices[size_t(dense)];
                if (compact_index < 0)
                {
                    code.body += Var(index, dense, Dimensions()).Assign(CodeExpr(std::string("0.0")), false);
                    continue;
                }
                CodeExpr value = Var(
                    inputs[0], compact_index,
                    independent->Dimensions());
                if (expansion->signs[size_t(dense)] < 0)
                    value = CodeExpr("-(" + value.S() + ")");
                code.body += Var(index, dense, Dimensions()).Assign(value.S(), false);
            }
        }

        void ApplyPattern(
            FlatVector<AutoDiffDiff<1, NonZero>> compact,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const
        {
            values = AutoDiffDiff<1, NonZero>(false);
            for (int dense = 0; dense < Dimension(); ++dense)
            {
                const int compact_index =
                    expansion->independent_indices[size_t(dense)];
                if (compact_index >= 0)
                    values(dense) = compact(compact_index);
            }
        }

        void NonZeroPattern(
            const ProxyUserData &ud,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            Vector<AutoDiffDiff<1, NonZero>> compact(
                independent->Dimension());
            independent->NonZeroPattern(ud, compact);
            ApplyPattern(compact, values);
        }

        void NonZeroPattern(
            const ProxyUserData &,
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            ApplyPattern(inputs[0], values);
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
            return transformation.cache[self] =
                       make_shared<AlternatingDoubleFormCoefficientFunction>(
                           independent->Transform(transformation),
                           left_degree, right_degree, dim);
        }

        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return make_shared<AlternatingDoubleFormCoefficientFunction>(
                independent->Diff(var, dir), left_degree,
                right_degree, dim);
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
                return cache[self] =
                           make_shared<AlternatingDoubleFormCoefficientFunction>(
                               independent->Diff(
                                   var,
                                   make_shared<ConstantCoefficientFunction>(
                                       1.0)),
                               left_degree, right_degree, dim);
            return cache[self] = CoefficientFunction::DiffJacobi(var,
                                                                 cache);
        }

        shared_ptr<CoefficientFunction> IndependentValue() const
        {
            return independent;
        }
        int LeftDegree() const { return left_degree; }
        int RightDegree() const { return right_degree; }
        int DimensionOfSpace() const { return dim; }
    };

    shared_ptr<AlternatingFormCoefficientFunction>
    GetAlternatingForm(const shared_ptr<CoefficientFunction> &value)
    {
        auto full = value;
        while (auto wrapper =
                   dynamic_pointer_cast<TensorFieldCoefficientFunction>(full))
            full = wrapper->GetFullCoefficient();
        return dynamic_pointer_cast<AlternatingFormCoefficientFunction>(full);
    }

    shared_ptr<AlternatingDoubleFormCoefficientFunction>
    GetAlternatingDoubleForm(
        const shared_ptr<CoefficientFunction> &value)
    {
        auto full = value;
        while (auto wrapper =
                   dynamic_pointer_cast<TensorFieldCoefficientFunction>(full))
            full = wrapper->GetFullCoefficient();
        return dynamic_pointer_cast<
            AlternatingDoubleFormCoefficientFunction>(full);
    }

    static ngcore::RegisterClassForArchive<
        AlternatingFormCoefficientFunction, CoefficientFunction>
        reg_alternating_form_cf;
    static ngcore::RegisterClassForArchive<
        AlternatingDoubleFormCoefficientFunction, CoefficientFunction>
        reg_alternating_double_form_cf;

    size_t CompactDoubleFormExpansionTableStorageBytesLowerBound(
        int dim, int left_degree, int right_degree)
    {
        return GetCompactDoubleFormExpansionTable(
                   dim, left_degree, right_degree)
            ->OwnedStorageBytesLowerBound();
    }

} // namespace kforms_internal
