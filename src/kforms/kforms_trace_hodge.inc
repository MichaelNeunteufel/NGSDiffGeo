// Implementation fragment included only by ../kforms.cpp.
// Compact trace and metric Hodge-star implementations.

namespace kforms_internal
{
    namespace
    {
        struct CompactDoubleTraceTerm
        {
            int input_index;
            int metric_index;
            int sign;
        };

        class CompactDoubleTraceTable
        {
        public:
            std::vector<std::vector<CompactDoubleTraceTerm>> outputs;

            size_t OwnedStorageBytesLowerBound() const
            {
                size_t bytes =
                    sizeof(CompactDoubleTraceTable) +
                    outputs.capacity() *
                        sizeof(std::vector<CompactDoubleTraceTerm>);
                for (const auto &terms : outputs)
                    bytes += terms.capacity() *
                             sizeof(CompactDoubleTraceTerm);
                return bytes;
            }

            CompactDoubleTraceTable(int dim, int left_degree,
                                    int right_degree)
            {
                if (left_degree < 1 || right_degree < 1)
                    throw Exception(
                        "CompactDoubleTraceTable: both degrees must be positive");
                auto input_left = GetFormBasis(dim, left_degree);
                auto input_right = GetFormBasis(dim, right_degree);
                auto output_left = GetFormBasis(dim, left_degree - 1);
                auto output_right = GetFormBasis(dim, right_degree - 1);
                outputs.resize(output_left->IndependentSize() *
                               output_right->IndependentSize());

                auto prepend_dense_index = [dim](
                                               int first, const std::vector<int> &tail)
                {
                    int dense = first;
                    int stride = dim;
                    for (int axis : tail)
                    {
                        dense += axis * stride;
                        stride *= dim;
                    }
                    return dense;
                };

                for (size_t left_output = 0;
                     left_output < output_left->IndependentSize();
                     ++left_output)
                    for (size_t right_output = 0;
                         right_output < output_right->IndependentSize();
                         ++right_output)
                    {
                        auto &terms = outputs[left_output *
                                                  output_right->IndependentSize() +
                                              right_output];
                        for (int left_axis = 0; left_axis < dim;
                             ++left_axis)
                        {
                            auto left_entry = input_left->DenseEntry(
                                size_t(prepend_dense_index(
                                    left_axis,
                                    output_left->CanonicalIndex(
                                        left_output))));
                            if (left_entry.independent_index < 0)
                                continue;
                            for (int right_axis = 0; right_axis < dim;
                                 ++right_axis)
                            {
                                auto right_entry = input_right->DenseEntry(
                                    size_t(prepend_dense_index(
                                        right_axis,
                                        output_right->CanonicalIndex(
                                            right_output))));
                                if (right_entry.independent_index < 0)
                                    continue;
                                terms.push_back({left_entry.independent_index *
                                                         int(input_right
                                                                 ->IndependentSize()) +
                                                     right_entry.independent_index,
                                                 left_axis * dim + right_axis,
                                                 left_entry.sign * right_entry.sign});
                            }
                        }
                    }
            }
        };

        shared_ptr<const CompactDoubleTraceTable>
        GetCompactDoubleTraceTable(int dim, int left_degree,
                                   int right_degree)
        {
            constexpr size_t size = MAX_SPACE_DIM + 1;
            static std::array<
                std::array<std::array<std::once_flag, size>, size>,
                size>
                flags;
            static std::array<
                std::array<
                    std::array<
                        shared_ptr<const CompactDoubleTraceTable>, size>,
                    size>,
                size>
                tables;
            if (dim < 1 || dim > MAX_SPACE_DIM || left_degree < 1 ||
                left_degree > dim || right_degree < 1 ||
                right_degree > dim)
                throw Exception(
                    "CompactDoubleTraceTable: invalid dimension or degree");
            std::call_once(
                flags[size_t(dim)][size_t(left_degree)]
                     [size_t(right_degree)],
                [&]
                {
                    tables[size_t(dim)][size_t(left_degree)]
                          [size_t(right_degree)] =
                              make_shared<const CompactDoubleTraceTable>(
                                  dim, left_degree, right_degree);
                });
            return tables[size_t(dim)][size_t(left_degree)]
                         [size_t(right_degree)];
        }

        class CompactDoubleTraceIndependentCoefficientFunction
            : public T_CoefficientFunction<
                  CompactDoubleTraceIndependentCoefficientFunction>
        {
            using BASE = T_CoefficientFunction<
                CompactDoubleTraceIndependentCoefficientFunction>;
            shared_ptr<CoefficientFunction> input;
            shared_ptr<CoefficientFunction> inverse_metric;
            int left_degree;
            int right_degree;
            int dim;
            shared_ptr<const CompactDoubleTraceTable> table;

            template <typename INPUT, typename METRIC, typename VALUES>
            void EvaluateFromInputs(size_t points, INPUT input_values,
                                    METRIC metric_values,
                                    VALUES values) const
            {
                using T = typename VALUES::TSCAL;
                for (size_t output = 0; output < table->outputs.size();
                     ++output)
                    for (size_t ip = 0; ip < points; ++ip)
                    {
                        T sum = T(0);
                        for (const auto &term : table->outputs[output])
                            sum += T(term.sign) *
                                   input_values(term.input_index, ip) *
                                   metric_values(term.metric_index, ip);
                        values(int(output), ip) = sum;
                    }
            }

        public:
            CompactDoubleTraceIndependentCoefficientFunction(
                shared_ptr<CoefficientFunction> ainput,
                shared_ptr<CoefficientFunction> ainverse_metric,
                int aleft_degree, int aright_degree, int adim)
                : BASE(
                      int(GetFormBasis(adim, aleft_degree - 1)
                              ->IndependentSize() *
                          GetFormBasis(adim, aright_degree - 1)
                              ->IndependentSize()),
                      RequireCompactNonNull(ainput)->IsComplex() ||
                          RequireCompactNonNull(ainverse_metric)
                              ->IsComplex()),
                  input(std::move(ainput)),
                  inverse_metric(std::move(ainverse_metric)),
                  left_degree(aleft_degree),
                  right_degree(aright_degree), dim(adim),
                  table(GetCompactDoubleTraceTable(
                      dim, left_degree, right_degree))
            {
                const int expected_input =
                    int(GetFormBasis(dim, left_degree)
                            ->IndependentSize() *
                        GetFormBasis(dim, right_degree)
                            ->IndependentSize());
                if (input->Dimension() != expected_input)
                    throw Exception(
                        "CompactDoubleTraceIndependentCF: input shape mismatch");
                if (inverse_metric->Dimensions() !=
                    Array<int>{dim, dim})
                    throw Exception(
                        "CompactDoubleTraceIndependentCF: metric shape mismatch");
                SetDimensions(Array<int>{int(Dimension())});
                elementwise_constant =
                    input->ElementwiseConstant() &&
                    inverse_metric->ElementwiseConstant();
            }

            auto GetCArgs() const
            {
                return tuple{input, inverse_metric, left_degree,
                             right_degree, dim};
            }
            string GetDescription() const override
            {
                return "CompactDoubleTraceIndependentCF";
            }
            void CalcEquivalenceKey() override
            {
                equivalence_key =
                    GetDescription() + "[p=" + ToString(left_degree) +
                    ",q=" + ToString(right_degree) +
                    ",dim=" + ToString(dim) + "](" +
                    input->EquivalenceKey() + "," +
                    inverse_metric->EquivalenceKey() + ")";
            }
            void DoArchive(Archive &) override {}
            bool DefinedOn(const ElementTransformation &trafo) override
            {
                return input->DefinedOn(trafo) &&
                       inverse_metric->DefinedOn(trafo);
            }
            bool IsZeroCF() const override
            {
                return input->IsZeroCF() || inverse_metric->IsZeroCF();
            }
            Array<shared_ptr<CoefficientFunction>>
            InputCoefficientFunctions() const override
            {
                return {input, inverse_metric};
            }
            void TraverseTree(
                const function<void(CoefficientFunction &)> &func) override
            {
                input->TraverseTree(func);
                inverse_metric->TraverseTree(func);
                func(*this);
            }
            using BASE::Evaluate;
            template <typename MIR, typename T, ORDERING ORD>
            void T_Evaluate(const MIR &mir,
                            BareSliceMatrix<T, ORD> values) const
            {
                auto &heap = TLHeap();
                HeapReset reset(heap);
                FlatMatrix<T, ORD> input_values(
                    input->Dimension(), mir.Size(), heap);
                FlatMatrix<T, ORD> metric_values(
                    inverse_metric->Dimension(), mir.Size(), heap);
                input->Evaluate(mir, input_values);
                inverse_metric->Evaluate(mir, metric_values);
                EvaluateFromInputs(mir.Size(), input_values,
                                   metric_values, values);
            }
            template <typename MIR, typename T, ORDERING ORD>
            void T_Evaluate(
                const MIR &mir,
                FlatArray<BareSliceMatrix<T, ORD>> inputs,
                BareSliceMatrix<T, ORD> values) const
            {
                EvaluateFromInputs(mir.Size(), inputs[0], inputs[1],
                                   values);
            }
            void GenerateCode(Code &code, FlatArray<int> inputs,
                              int index) const override
            {
                DeclareTensorFieldGeneratedCoefficient(
                    code, index, Dimensions(), IsComplex());
                for (size_t output = 0; output < table->outputs.size();
                     ++output)
                {
                    CodeExpr sum;
                    size_t term_index = 0;
                    for (const auto &term : table->outputs[output])
                    {
                        auto product =
                            Var(inputs[0], term.input_index,
                                input->Dimensions()) *
                            Var(inputs[1], term.metric_index,
                                inverse_metric->Dimensions());
                        AccumulateSignedCodeExpression(
                            sum, std::move(product), term.sign,
                            term_index++);
                    }
                    code.body +=
                        Var(index, int(output), Dimensions()).Assign(sum.S(), false);
                }
            }
            void NonZeroPattern(
                const ProxyUserData &ud,
                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
            {
                Vector<AutoDiffDiff<1, NonZero>> input_pattern(
                    input->Dimension());
                Vector<AutoDiffDiff<1, NonZero>> metric_pattern(
                    inverse_metric->Dimension());
                input->NonZeroPattern(ud, input_pattern);
                inverse_metric->NonZeroPattern(ud, metric_pattern);
                FlatVector<AutoDiffDiff<1, NonZero>> inputs[] = {
                    input_pattern, metric_pattern};
                NonZeroPattern(
                    ud,
                    FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                        2, inputs),
                    values);
            }
            void NonZeroPattern(
                const ProxyUserData &,
                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
            {
                values = AutoDiffDiff<1, NonZero>(false);
                for (size_t output = 0; output < table->outputs.size();
                     ++output)
                    for (const auto &term : table->outputs[output])
                        values(int(output)) =
                            values(int(output)) +
                            inputs[0](term.input_index) *
                                inputs[1](term.metric_index);
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
                           CompactDoubleTraceIndependentCoefficientFunction>(
                           input->Transform(transformation),
                           inverse_metric->Transform(transformation),
                           left_degree, right_degree, dim);
            }
            shared_ptr<CoefficientFunction> Diff(
                const CoefficientFunction *var,
                shared_ptr<CoefficientFunction> dir) const override
            {
                if (this == var)
                    return dir;
                auto input_derivative = input->Diff(var, dir);
                auto metric_derivative = inverse_metric->Diff(var, dir);
                shared_ptr<CoefficientFunction> result;
                if (!IsConstantZero(input_derivative))
                    result = make_shared<
                        CompactDoubleTraceIndependentCoefficientFunction>(
                        input_derivative, inverse_metric, left_degree,
                        right_degree, dim);
                if (!IsConstantZero(metric_derivative))
                {
                    auto metric_term = make_shared<
                        CompactDoubleTraceIndependentCoefficientFunction>(
                        input, metric_derivative, left_degree,
                        right_degree, dim);
                    result = result
                                 ? SymbolicSumCF(result, metric_term)
                                 : metric_term;
                }
                return result ? result : ZeroCF(Dimensions());
            }
            shared_ptr<CoefficientFunction> DiffJacobi(
                const CoefficientFunction *var,
                T_DJC &cache) const override
            {
                auto self = const_pointer_cast<CoefficientFunction>(
                    shared_from_this());
                if (auto found = cache.find(self); found != cache.end())
                    return found->second;
                if (this == var)
                    return cache[self] = IdentityCF(Dimensions());
                if (var->Dimensions().Size() == 0)
                    return cache[self] = Diff(
                               var, ConstantCF(1.0));
                return cache[self] =
                           CoefficientFunction::DiffJacobi(var, cache);
            }
        };
    }

    shared_ptr<DoubleFormCoefficientFunction> TraceDoubleFormWithMetric(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric)
    {
        form = RequireCompactNonNull(std::move(form));
        inverse_metric = RequireCompactNonNull(std::move(inverse_metric));
        const int dim = form->DimensionOfSpace();
        const int left_degree = form->LeftDegree();
        const int right_degree = form->RightDegree();
        if (left_degree < 1 || right_degree < 1)
            throw Exception(
                "TraceDoubleFormWithMetric: both degrees must be positive");
        if (inverse_metric->Dimensions() != Array<int>{dim, dim})
            throw Exception(
                "TraceDoubleFormWithMetric: metric shape mismatch");

        if (auto view = TryGetProvenCompactInput(form))
        {
            auto independent = make_shared<
                CompactDoubleTraceIndependentCoefficientFunction>(
                view->independent, inverse_metric, left_degree,
                right_degree, dim);
            auto facade = make_shared<
                CompactDoubleFormOperationCoefficientFunction>(
                int(CompactDoubleFormOperation::Trace),
                Array<shared_ptr<CoefficientFunction>>{form,
                                                       inverse_metric},
                independent, left_degree - 1, right_degree - 1, dim);
            return DoubleFormCF(facade, left_degree - 1,
                                right_degree - 1, dim);
        }

        std::string signature = form->GetSignature();
        const char contracted_left = signature[0];
        const char contracted_right = signature[size_t(left_degree)];
        std::string output_signature;
        for (size_t axis = 0; axis < signature.size(); ++axis)
            if (axis != 0 && axis != size_t(left_degree))
                output_signature.push_back(signature[axis]);
        std::string einsum = signature + ",";
        einsum.push_back(contracted_left);
        einsum.push_back(contracted_right);
        einsum += "->" + output_signature;
        auto result = SymbolicEinsumCF(
            einsum, {form, inverse_metric});
        return DoubleFormCF(result, left_degree - 1,
                            right_degree - 1, dim);
    }

    namespace
    {
        struct CompactHodgeMapTable
        {
            int size = 0;
            std::vector<int> rows;
            std::vector<int> signs;

            size_t OwnedStorageBytesLowerBound() const
            {
                return sizeof(CompactHodgeMapTable) +
                       (rows.capacity() + signs.capacity()) *
                           sizeof(int);
            }
        };

        shared_ptr<const CompactHodgeMapTable>
        GetCompactHodgeMapTable(int dim, int degree)
        {
            auto input_basis = GetFormBasis(dim, degree);
            constexpr size_t extent = MAX_SPACE_DIM + 1;
            static std::array<std::array<std::once_flag, extent>, extent>
                flags;
            static std::array<
                std::array<shared_ptr<const CompactHodgeMapTable>,
                           extent>,
                extent>
                cache;
            std::call_once(flags[size_t(dim)][size_t(degree)], [&]
                           {
                    auto table = make_shared<CompactHodgeMapTable>();
                    table->size = int(input_basis->IndependentSize());
                    auto output_basis =
                        GetFormBasis(dim, dim - degree);
                    table->rows.reserve(output_basis->IndependentSize());
                    table->signs.reserve(output_basis->IndependentSize());
                    for (size_t output = 0;
                         output < output_basis->IndependentSize(); ++output)
                    {
                        const auto &output_index =
                            output_basis->CanonicalIndex(output);
                        std::vector<bool> selected(size_t(dim), false);
                        for (int axis : output_index)
                            selected[size_t(axis)] = true;
                        std::vector<int> complement;
                        complement.reserve(size_t(degree));
                        for (int axis = 0; axis < dim; ++axis)
                            if (!selected[size_t(axis)])
                                complement.push_back(axis);
                        // The compact output block precedes the complementary
                        // raised input block: epsilon(J, I). FormBasis stores
                        // both blocks in reverse stride order relative to the
                        // row-major Levi-Civita tensor.
                        std::vector<int> permutation(
                            output_index.begin(), output_index.end());
                        permutation.insert(permutation.end(),
                                           complement.begin(),
                                           complement.end());
                        table->rows.push_back(
                            input_basis->CompactIndex(complement));
                        const int output_degree = dim - degree;
                        const int input_storage_sign =
                            (degree * (degree - 1) / 2) % 2 ? -1 : 1;
                        const int output_storage_sign =
                            (output_degree * (output_degree - 1) / 2) % 2
                                ? -1
                                : 1;
                        table->signs.push_back(
                            input_storage_sign * output_storage_sign *
                            PermutationSign(permutation));
                    }
                    cache[size_t(dim)][size_t(degree)] =
                        std::move(table); });
            return cache[size_t(dim)][size_t(degree)];
        }

        class CompactHodgeMapCoefficientFunction
            : public T_CoefficientFunction<
                  CompactHodgeMapCoefficientFunction>
        {
            using BASE = T_CoefficientFunction<
                CompactHodgeMapCoefficientFunction>;
            shared_ptr<CoefficientFunction> induced_metric;
            shared_ptr<CoefficientFunction> volume;
            int degree;
            int dim;
            shared_ptr<const CompactHodgeMapTable> table;

            template <typename INDUCED, typename VOLUME,
                      typename VALUES>
            void EvaluateFromInputs(size_t points, INDUCED induced,
                                    VOLUME volume_values,
                                    VALUES values) const
            {
                for (int output_row = 0; output_row < table->size;
                     ++output_row)
                    for (int input_column = 0;
                         input_column < table->size;
                         ++input_column)
                        for (size_t ip = 0; ip < points; ++ip)
                            values(output_row * table->size +
                                       input_column,
                                   ip) =
                                table->signs[size_t(output_row)] *
                                volume_values(0, ip) *
                                induced(table->rows[size_t(output_row)] *
                                                table->size +
                                            input_column,
                                        ip);
            }

        public:
            CompactHodgeMapCoefficientFunction(
                shared_ptr<CoefficientFunction> ainduced_metric,
                shared_ptr<CoefficientFunction> avolume, int adegree,
                int adim)
                : BASE(
                      int(GetFormBasis(adim, adegree)
                              ->IndependentSize() *
                          GetFormBasis(adim, adegree)
                              ->IndependentSize()),
                      RequireCompactNonNull(ainduced_metric)
                              ->IsComplex() ||
                          RequireCompactNonNull(avolume)->IsComplex()),
                  induced_metric(std::move(ainduced_metric)),
                  volume(std::move(avolume)), degree(adegree), dim(adim),
                  table(GetCompactHodgeMapTable(dim, degree))
            {
                if (induced_metric->Dimensions() !=
                    Array<int>{table->size, table->size})
                    throw Exception(
                        "CompactHodgeMapCF: induced metric shape mismatch");
                if (volume->Dimensions().Size() != 0)
                    throw Exception(
                        "CompactHodgeMapCF: volume must be scalar");
                SetDimensions(Array<int>{table->size, table->size});
                elementwise_constant =
                    induced_metric->ElementwiseConstant() &&
                    volume->ElementwiseConstant();
            }

            auto GetCArgs() const
            {
                return tuple{induced_metric, volume, degree, dim};
            }
            string GetDescription() const override
            {
                return "CompactHodgeMapCF";
            }
            void CalcEquivalenceKey() override
            {
                equivalence_key =
                    GetDescription() + "[k=" + ToString(degree) +
                    ",dim=" + ToString(dim) + "](" +
                    induced_metric->EquivalenceKey() + "," +
                    volume->EquivalenceKey() + ")";
            }
            void DoArchive(Archive &) override {}
            bool DefinedOn(const ElementTransformation &trafo) override
            {
                return induced_metric->DefinedOn(trafo) &&
                       volume->DefinedOn(trafo);
            }
            bool IsZeroCF() const override
            {
                return induced_metric->IsZeroCF() || volume->IsZeroCF();
            }
            Array<shared_ptr<CoefficientFunction>>
            InputCoefficientFunctions() const override
            {
                return {induced_metric, volume};
            }
            void TraverseTree(
                const function<void(CoefficientFunction &)> &func) override
            {
                induced_metric->TraverseTree(func);
                volume->TraverseTree(func);
                func(*this);
            }
            using BASE::Evaluate;
            template <typename MIR, typename T, ORDERING ORD>
            void T_Evaluate(const MIR &mir,
                            BareSliceMatrix<T, ORD> values) const
            {
                auto &heap = TLHeap();
                HeapReset reset(heap);
                FlatMatrix<T, ORD> induced_values(
                    induced_metric->Dimension(), mir.Size(), heap);
                FlatMatrix<T, ORD> volume_values(
                    volume->Dimension(), mir.Size(), heap);
                induced_metric->Evaluate(mir, induced_values);
                volume->Evaluate(mir, volume_values);
                EvaluateFromInputs(mir.Size(), induced_values,
                                   volume_values, values);
            }
            template <typename MIR, typename T, ORDERING ORD>
            void T_Evaluate(
                const MIR &mir,
                FlatArray<BareSliceMatrix<T, ORD>> inputs,
                BareSliceMatrix<T, ORD> values) const
            {
                EvaluateFromInputs(mir.Size(), inputs[0], inputs[1],
                                   values);
            }
            void GenerateCode(Code &code, FlatArray<int> inputs,
                              int index) const override
            {
                DeclareTensorFieldGeneratedCoefficient(
                    code, index, Dimensions(), IsComplex());
                for (int output_row = 0; output_row < table->size;
                     ++output_row)
                    for (int input_column = 0;
                         input_column < table->size;
                         ++input_column)
                    {
                        CodeExpr value =
                            Var(inputs[1], 0, volume->Dimensions()) *
                            Var(inputs[0],
                                table->rows[size_t(output_row)] *
                                        table->size +
                                    input_column,
                                induced_metric->Dimensions());
                        if (table->signs[size_t(output_row)] < 0)
                            value = CodeExpr("-(" + value.S() + ")");
                        code.body +=
                            Var(index,
                                output_row * table->size +
                                    input_column,
                                Dimensions())
                                .Assign(value.S(), false);
                    }
            }
            void NonZeroPattern(
                const ProxyUserData &ud,
                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
            {
                Vector<AutoDiffDiff<1, NonZero>> induced_pattern(
                    induced_metric->Dimension());
                Vector<AutoDiffDiff<1, NonZero>> volume_pattern(
                    volume->Dimension());
                induced_metric->NonZeroPattern(ud, induced_pattern);
                volume->NonZeroPattern(ud, volume_pattern);
                FlatVector<AutoDiffDiff<1, NonZero>> inputs[] = {
                    induced_pattern, volume_pattern};
                NonZeroPattern(
                    ud,
                    FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                        2, inputs),
                    values);
            }
            void NonZeroPattern(
                const ProxyUserData &,
                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
            {
                for (int output_row = 0; output_row < table->size;
                     ++output_row)
                    for (int input_column = 0;
                         input_column < table->size;
                         ++input_column)
                        values(output_row * table->size +
                               input_column) =
                            inputs[1](0) *
                            inputs[0](
                                table->rows[size_t(output_row)] *
                                    table->size +
                                input_column);
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
                           CompactHodgeMapCoefficientFunction>(
                           induced_metric->Transform(transformation),
                           volume->Transform(transformation), degree,
                           dim);
            }
            shared_ptr<CoefficientFunction> Diff(
                const CoefficientFunction *var,
                shared_ptr<CoefficientFunction> dir) const override
            {
                if (this == var)
                    return dir;
                auto induced_derivative =
                    induced_metric->Diff(var, dir);
                auto volume_derivative = volume->Diff(var, dir);
                shared_ptr<CoefficientFunction> result;
                if (!IsConstantZero(induced_derivative))
                    result = make_shared<
                        CompactHodgeMapCoefficientFunction>(
                        induced_derivative, volume, degree, dim);
                if (!IsConstantZero(volume_derivative))
                {
                    auto volume_term = make_shared<
                        CompactHodgeMapCoefficientFunction>(
                        induced_metric, volume_derivative, degree, dim);
                    result = result
                                 ? SymbolicSumCF(result, volume_term)
                                 : volume_term;
                }
                return result ? result : ZeroCF(Dimensions());
            }
        };

        shared_ptr<CoefficientFunction> CompactHodgeMap(
            const shared_ptr<CoefficientFunction> &inverse_metric,
            const shared_ptr<CoefficientFunction> &volume,
            const shared_ptr<const FormBasis> &input_basis)
        {
            auto induced = InducedMetric(inverse_metric, input_basis);
            return make_shared<CompactHodgeMapCoefficientFunction>(
                std::move(induced), volume, input_basis->Degree(),
                input_basis->DimensionOfSpace());
        }

        shared_ptr<CoefficientFunction> DenseBlockHodgeWithMetric(
            shared_ptr<CoefficientFunction> value, int rank,
            int block_start, int block_degree, int dim,
            const shared_ptr<CoefficientFunction> &inverse_metric,
            const shared_ptr<CoefficientFunction> &volume)
        {
            if (block_start < 0 || block_degree < 0 ||
                block_start + block_degree > rank ||
                block_degree > dim)
                throw Exception(
                    "DenseBlockHodgeWithMetric: invalid block");
            std::string signature = SIGNATURE.substr(0, size_t(rank));
            for (int axis = block_start;
                 axis < block_start + block_degree; ++axis)
            {
                const char output_axis = signature[size_t(axis)];
                const char contracted_axis =
                    FreshSignature(signature, 1)[0];
                std::string input_signature = signature;
                input_signature[size_t(axis)] = contracted_axis;
                std::string metric_signature;
                metric_signature.push_back(output_axis);
                metric_signature.push_back(contracted_axis);
                value = SymbolicEinsumCF(
                    input_signature + "," + metric_signature + "->" +
                        signature,
                    {value, inverse_metric});
            }

            auto epsilon = ScaleCoefficientCF(
                LeviCivitaCF(dim), volume);
            if (signature.empty())
                return ScaleCoefficientCF(epsilon, value);

            const std::string prefix =
                signature.substr(0, size_t(block_start));
            const std::string block = signature.substr(
                size_t(block_start), size_t(block_degree));
            const std::string suffix = signature.substr(
                size_t(block_start + block_degree));
            const std::string new_block = FreshSignature(
                signature, dim - block_degree);
            auto result = SymbolicEinsumCF(
                signature + "," + block + new_block + "->" +
                    prefix + new_block + suffix,
                {value, epsilon});
            if (block_degree > 1)
                result = ScaleCoefficientCF(
                    result,
                    ConstantCF(1.0 / double(Factorial(block_degree))));
            return result;
        }
    }

    shared_ptr<DoubleFormCoefficientFunction> TryHodgeCompactDoubleForm(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric,
        shared_ptr<CoefficientFunction> volume, int slot)
    {
        form = RequireCompactNonNull(std::move(form));
        inverse_metric = RequireCompactNonNull(std::move(inverse_metric));
        volume = RequireCompactNonNull(std::move(volume));
        if (slot != -1 && slot != 0 && slot != 1)
            throw Exception(
                "TryHodgeCompactDoubleForm: invalid slot");
        const int dim = form->DimensionOfSpace();
        if (inverse_metric->Dimensions() != Array<int>{dim, dim} ||
            volume->Dimensions().Size() != 0)
            throw Exception(
                "TryHodgeCompactDoubleForm: metric shape mismatch");
        auto view = TryGetProvenCompactInput(form);
        if (!view)
            return nullptr;

        auto left_basis = GetFormBasis(dim, view->left_degree);
        auto right_basis = GetFormBasis(dim, view->right_degree);
        const int left_size = int(left_basis->IndependentSize());
        const int right_size = int(right_basis->IndependentSize());
        auto input = AsCompactVector(
                         view->independent, left_size * right_size)
                         ->Reshape(
                             Array<int>{left_size, right_size});

        const int output_left_degree =
            slot == 1 ? view->left_degree
                      : dim - view->left_degree;
        const int output_right_degree =
            slot == 0 ? view->right_degree
                      : dim - view->right_degree;
        shared_ptr<CoefficientFunction> independent;
        CompactDoubleFormOperation operation;
        if (slot == 0)
        {
            auto left_map = CompactHodgeMap(
                inverse_metric, volume, left_basis);
            independent = SymbolicEinsumCF(
                              "ij,ai->aj", {input, left_map})
                              ->Reshape(Array<int>{
                                  int(GetFormBasis(
                                          dim, output_left_degree)
                                          ->IndependentSize() *
                                      right_basis->IndependentSize())});
            operation = CompactDoubleFormOperation::HodgeLeft;
        }
        else if (slot == 1)
        {
            auto right_map = CompactHodgeMap(
                inverse_metric, volume, right_basis);
            independent = SymbolicEinsumCF(
                              "ij,bj->ib", {input, right_map})
                              ->Reshape(Array<int>{
                                  int(left_basis->IndependentSize() *
                                      GetFormBasis(
                                          dim, output_right_degree)
                                          ->IndependentSize())});
            operation = CompactDoubleFormOperation::HodgeRight;
        }
        else
        {
            auto left_map = CompactHodgeMap(
                inverse_metric, volume, left_basis);
            auto right_map = CompactHodgeMap(
                inverse_metric, volume, right_basis);
            independent = SymbolicEinsumCF(
                              "ij,ai,bj->ab",
                              {input, left_map, right_map})
                              ->Reshape(Array<int>{
                                  int(GetFormBasis(
                                          dim, output_left_degree)
                                          ->IndependentSize() *
                                      GetFormBasis(
                                          dim, output_right_degree)
                                          ->IndependentSize())});
            operation = CompactDoubleFormOperation::HodgeBoth;
        }

        auto facade =
            make_shared<CompactDoubleFormOperationCoefficientFunction>(
                int(operation),
                Array<shared_ptr<CoefficientFunction>>{
                    form, inverse_metric, volume},
                std::move(independent), output_left_degree,
                output_right_degree, dim);
        return DoubleFormCF(facade, output_left_degree,
                            output_right_degree, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction>
    HodgeDenseDoubleFormWithMetric(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric,
        shared_ptr<CoefficientFunction> volume, int slot)
    {
        form = RequireCompactNonNull(std::move(form));
        inverse_metric = RequireCompactNonNull(std::move(inverse_metric));
        volume = RequireCompactNonNull(std::move(volume));
        if (slot != -1 && slot != 0 && slot != 1)
            throw Exception(
                "HodgeDenseDoubleFormWithMetric: invalid slot");
        const int dim = form->DimensionOfSpace();
        const int left_degree = form->LeftDegree();
        const int right_degree = form->RightDegree();
        if (inverse_metric->Dimensions() != Array<int>{dim, dim} ||
            volume->Dimensions().Size() != 0)
            throw Exception(
                "HodgeDenseDoubleFormWithMetric: metric shape mismatch");

        shared_ptr<CoefficientFunction> result = form;
        int output_left_degree = left_degree;
        int output_right_degree = right_degree;
        if (slot != 1)
        {
            result = DenseBlockHodgeWithMetric(
                result, left_degree + right_degree, 0, left_degree,
                dim, inverse_metric, volume);
            output_left_degree = dim - left_degree;
        }
        if (slot != 0)
        {
            result = DenseBlockHodgeWithMetric(
                result, output_left_degree + right_degree,
                output_left_degree, right_degree, dim, inverse_metric,
                volume);
            output_right_degree = dim - right_degree;
        }
        return DoubleFormCF(result, output_left_degree,
                            output_right_degree, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> HodgeDoubleFormWithMetric(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric,
        shared_ptr<CoefficientFunction> volume, int slot)
    {
        if (auto compact = TryHodgeCompactDoubleForm(
                form, inverse_metric, volume, slot))
            return compact;
        return HodgeDenseDoubleFormWithMetric(
            std::move(form), std::move(inverse_metric),
            std::move(volume), slot);
    }

    static ngcore::RegisterClassForArchive<
        CompactDoubleTraceIndependentCoefficientFunction,
        CoefficientFunction>
        reg_compact_double_trace_independent_cf;
    static ngcore::RegisterClassForArchive<
        CompactHodgeMapCoefficientFunction, CoefficientFunction>
        reg_compact_hodge_map_cf;

    size_t CompactHodgeMapTableStorageBytesLowerBound(
        int dim, int degree)
    {
        return GetCompactHodgeMapTable(dim, degree)
            ->OwnedStorageBytesLowerBound();
    }

    size_t CompactDoubleTraceTableStorageBytesLowerBound(
        int dim, int left_degree, int right_degree)
    {
        return GetCompactDoubleTraceTable(
                   dim, left_degree, right_degree)
            ->OwnedStorageBytesLowerBound();
    }

} // namespace kforms_internal
