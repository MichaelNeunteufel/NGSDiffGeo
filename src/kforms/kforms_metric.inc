// Implementation fragment included only by ../kforms.cpp.
// Induced form metrics, compact metric products, and compact-input recognition.

namespace kforms_internal
{
    namespace
    {
        shared_ptr<CoefficientFunction> AsCompactVector(
            shared_ptr<CoefficientFunction> value, int size)
        {
            if (value->Dimensions().Size() == 0)
                return value->Reshape(Array<int>{size});
            if (value->Dimensions() != Array<int>{size})
                throw Exception(
                    "CompactFormInnerProduct: independent shape mismatch");
            return value;
        }

        int PermutationSign(const std::vector<int> &permutation)
        {
            int inversions = 0;
            for (size_t i = 0; i < permutation.size(); ++i)
                for (size_t j = i + 1; j < permutation.size(); ++j)
                    inversions += permutation[i] > permutation[j];
            return inversions % 2 ? -1 : 1;
        }

        struct InducedFormMetricTable
        {
            shared_ptr<const FormBasis> basis;
            size_t terms_per_minor = 0;
            std::vector<int> signs;
            std::vector<int> metric_indices;

            size_t OwnedStorageBytesLowerBound() const
            {
                return sizeof(InducedFormMetricTable) +
                       (signs.capacity() + metric_indices.capacity()) *
                           sizeof(int);
            }
        };

        shared_ptr<const InducedFormMetricTable>
        GetInducedFormMetricTable(int dim, int degree)
        {
            if (degree < 2)
                throw Exception(
                    "InducedFormMetricCF: degree must be at least two");
            auto basis = GetFormBasis(dim, degree);
            constexpr size_t extent = MAX_SPACE_DIM + 1;
            static std::array<std::array<std::once_flag, extent>, extent>
                flags;
            static std::array<
                std::array<shared_ptr<const InducedFormMetricTable>,
                           extent>,
                extent>
                cache;
            std::call_once(flags[size_t(dim)][size_t(degree)], [&]
                           {
                    auto table = make_shared<InducedFormMetricTable>();
                    table->basis = std::move(basis);

                    std::vector<int> permutation(
                        static_cast<size_t>(degree));
                    std::iota(permutation.begin(), permutation.end(), 0);
                    do
                    {
                        table->signs.push_back(
                            PermutationSign(permutation));
                    } while (std::next_permutation(permutation.begin(),
                                                   permutation.end()));
                    table->terms_per_minor = table->signs.size();

                    const int size =
                        int(table->basis->IndependentSize());
                    table->metric_indices.reserve(
                        size_t(size) * size_t(size) *
                        table->terms_per_minor * size_t(degree));
                    for (int row = 0; row < size; ++row)
                        for (int column = 0; column < size; ++column)
                        {
                            std::iota(permutation.begin(),
                                      permutation.end(), 0);
                            do
                            {
                                for (int factor = 0; factor < degree;
                                     ++factor)
                                    table->metric_indices.push_back(
                                        table->basis->CanonicalIndex(
                                            size_t(row))[size_t(factor)] *
                                            dim +
                                        table->basis->CanonicalIndex(
                                            size_t(column))[size_t(
                                            permutation[size_t(factor)])]);
                            } while (std::next_permutation(
                                permutation.begin(), permutation.end()));
                        }
                    cache[size_t(dim)][size_t(degree)] =
                        std::move(table); });
            return cache[size_t(dim)][size_t(degree)];
        }

        class InducedFormMetricCoefficientFunction
            : public T_CoefficientFunction<
                  InducedFormMetricCoefficientFunction>
        {
            using BASE = T_CoefficientFunction<
                InducedFormMetricCoefficientFunction>;

            shared_ptr<CoefficientFunction> metric;
            shared_ptr<CoefficientFunction> metric_derivative;
            int degree;
            int dim;
            shared_ptr<const InducedFormMetricTable> table;

            template <typename INPUT, typename VALUES>
            void EvaluateFromInput(size_t points, INPUT input,
                                   VALUES values) const
            {
                using T = typename VALUES::TSCAL;
                for (size_t ip = 0; ip < points; ++ip)
                    for (int output = 0; output < Dimension(); ++output)
                    {
                        T sum = T(0);
                        const size_t base = size_t(output) *
                                            table->terms_per_minor *
                                            size_t(degree);
                        for (size_t term = 0;
                             term < table->terms_per_minor;
                             ++term)
                        {
                            T product = T(1);
                            for (int factor = 0; factor < degree;
                                 ++factor)
                                product *= input(
                                    table->metric_indices[base + term * size_t(degree) +
                                                          size_t(factor)],
                                    ip);
                            sum += T(table->signs[term]) * product;
                        }
                        values(output, ip) = sum;
                    }
            }

            template <typename INPUT, typename DERIVATIVE,
                      typename VALUES>
            void EvaluateDerivativeFromInputs(
                size_t points, INPUT input, DERIVATIVE derivative,
                VALUES values) const
            {
                using T = typename VALUES::TSCAL;
                for (size_t ip = 0; ip < points; ++ip)
                    for (int output = 0; output < Dimension(); ++output)
                    {
                        T sum = T(0);
                        const size_t base = size_t(output) *
                                            table->terms_per_minor *
                                            size_t(degree);
                        for (size_t term = 0;
                             term < table->terms_per_minor;
                             ++term)
                        {
                            T term_derivative = T(0);
                            for (int differentiated_factor = 0;
                                 differentiated_factor < degree;
                                 ++differentiated_factor)
                            {
                                T product = derivative(
                                    table->metric_indices[base + term * size_t(degree) +
                                                          size_t(differentiated_factor)],
                                    ip);
                                for (int factor = 0; factor < degree;
                                     ++factor)
                                    if (factor != differentiated_factor)
                                        product *= input(
                                            table->metric_indices[base + term * size_t(degree) +
                                                                  size_t(factor)],
                                            ip);
                                term_derivative += product;
                            }
                            sum += T(table->signs[term]) *
                                   term_derivative;
                        }
                        values(output, ip) = sum;
                    }
            }

        public:
            InducedFormMetricCoefficientFunction(
                shared_ptr<CoefficientFunction> ametric,
                shared_ptr<CoefficientFunction> ametric_derivative,
                int adegree, int adim)
                : BASE(
                      int(GetFormBasis(adim, adegree)->IndependentSize() *
                          GetFormBasis(adim, adegree)->IndependentSize()),
                      RequireCompactNonNull(ametric)->IsComplex() ||
                          (ametric_derivative &&
                           ametric_derivative->IsComplex())),
                  metric(std::move(ametric)),
                  metric_derivative(std::move(ametric_derivative)),
                  degree(adegree), dim(adim),
                  table(GetInducedFormMetricTable(dim, degree))
            {
                if (metric->Dimensions() != Array<int>{dim, dim})
                    throw Exception(
                        "InducedFormMetricCF: metric shape mismatch");
                if (metric_derivative &&
                    metric_derivative->Dimensions() !=
                        Array<int>{dim, dim})
                    throw Exception(
                        "InducedFormMetricCF: metric derivative shape mismatch");

                const int size = int(table->basis->IndependentSize());
                SetDimensions(Array<int>{size, size});
                elementwise_constant =
                    metric->ElementwiseConstant() &&
                    (!metric_derivative ||
                     metric_derivative->ElementwiseConstant());
            }

            auto GetCArgs() const
            {
                return tuple{metric, metric_derivative, degree, dim};
            }

            string GetDescription() const override
            {
                return metric_derivative
                           ? "DirectionalInducedFormMetricCF"
                           : "InducedFormMetricCF";
            }

            void CalcEquivalenceKey() override
            {
                equivalence_key =
                    GetDescription() + "[k=" + ToString(degree) +
                    ",dim=" + ToString(dim) + "](" +
                    metric->EquivalenceKey() +
                    (metric_derivative
                         ? "," + metric_derivative->EquivalenceKey()
                         : "") +
                    ")";
            }

            void DoArchive(Archive &) override {}

            bool DefinedOn(
                const ElementTransformation &trafo) override
            {
                return metric->DefinedOn(trafo) &&
                       (!metric_derivative ||
                        metric_derivative->DefinedOn(trafo));
            }

            bool IsZeroCF() const override
            {
                return metric_derivative
                           ? (metric_derivative->IsZeroCF() ||
                              metric->IsZeroCF())
                           : metric->IsZeroCF();
            }

            Array<shared_ptr<CoefficientFunction>>
            InputCoefficientFunctions() const override
            {
                return metric_derivative
                           ? Array<shared_ptr<CoefficientFunction>>{
                                 metric, metric_derivative}
                           : Array<shared_ptr<CoefficientFunction>>{metric};
            }

            void TraverseTree(
                const function<void(CoefficientFunction &)> &func) override
            {
                metric->TraverseTree(func);
                if (metric_derivative)
                    metric_derivative->TraverseTree(func);
                func(*this);
            }

            using BASE::Evaluate;

            double Evaluate(
                const BaseMappedIntegrationPoint &ip) const override
            {
                if (Dimensions().Size() != 0)
                    throw Exception(
                        "InducedFormMetricCF: scalar evaluation requires scalar shape");
                return BASE::Evaluate(ip);
            }

            template <typename MIR, typename T, ORDERING ORD>
            void T_Evaluate(const MIR &mir,
                            BareSliceMatrix<T, ORD> values) const
            {
                auto &heap = TLHeap();
                HeapReset reset(heap);
                FlatMatrix<T, ORD> input(
                    metric->Dimension(), mir.Size(), heap);
                metric->Evaluate(mir, input);
                if (!metric_derivative)
                    EvaluateFromInput(mir.Size(), input, values);
                else
                {
                    FlatMatrix<T, ORD> derivative(
                        metric_derivative->Dimension(), mir.Size(), heap);
                    metric_derivative->Evaluate(mir, derivative);
                    EvaluateDerivativeFromInputs(
                        mir.Size(), input, derivative, values);
                }
            }

            template <typename MIR, typename T, ORDERING ORD>
            void T_Evaluate(
                const MIR &mir,
                FlatArray<BareSliceMatrix<T, ORD>> inputs,
                BareSliceMatrix<T, ORD> values) const
            {
                if (!metric_derivative)
                    EvaluateFromInput(mir.Size(), inputs[0], values);
                else
                    EvaluateDerivativeFromInputs(
                        mir.Size(), inputs[0], inputs[1], values);
            }

            void GenerateCode(Code &code, FlatArray<int> inputs,
                              int index) const override
            {
                DeclareTensorFieldGeneratedCoefficient(
                    code, index, Dimensions(), IsComplex());
                for (int output = 0; output < Dimension(); ++output)
                {
                    CodeExpr sum;
                    const size_t base = size_t(output) *
                                        table->terms_per_minor *
                                        size_t(degree);
                    for (size_t term = 0;
                         term < table->terms_per_minor;
                         ++term)
                    {
                        CodeExpr term_value;
                        if (!metric_derivative)
                        {
                            term_value = Var(
                                inputs[0],
                                table->metric_indices[base + term * size_t(degree)],
                                metric->Dimensions());
                            for (int factor = 1; factor < degree;
                                 ++factor)
                                term_value *= Var(
                                    inputs[0],
                                    table->metric_indices[base + term * size_t(degree) +
                                                          size_t(factor)],
                                    metric->Dimensions());
                        }
                        else
                            for (int differentiated_factor = 0;
                                 differentiated_factor < degree;
                                 ++differentiated_factor)
                            {
                                CodeExpr product = Var(
                                    inputs[1],
                                    table->metric_indices[base + term * size_t(degree) +
                                                          size_t(differentiated_factor)],
                                    metric_derivative->Dimensions());
                                for (int factor = 0; factor < degree;
                                     ++factor)
                                    if (factor != differentiated_factor)
                                        product *= Var(
                                            inputs[0],
                                            table->metric_indices[base + term * size_t(degree) +
                                                                  size_t(factor)],
                                            metric->Dimensions());
                                term_value += product;
                            }
                        AccumulateSignedCodeExpression(
                            sum, std::move(term_value),
                            table->signs[term], term);
                    }
                    code.body += Var(index, output, Dimensions()).Assign(sum.S(), false);
                }
            }

            void NonZeroPattern(
                const ProxyUserData &ud,
                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
            {
                Vector<AutoDiffDiff<1, NonZero>> input(
                    metric->Dimension());
                metric->NonZeroPattern(ud, input);
                if (!metric_derivative)
                {
                    FlatVector<AutoDiffDiff<1, NonZero>> inputs[] = {
                        input};
                    NonZeroPattern(
                        ud,
                        FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                            1, inputs),
                        values);
                }
                else
                {
                    Vector<AutoDiffDiff<1, NonZero>> derivative(
                        metric_derivative->Dimension());
                    metric_derivative->NonZeroPattern(ud, derivative);
                    FlatVector<AutoDiffDiff<1, NonZero>> inputs[] = {
                        input, derivative};
                    NonZeroPattern(
                        ud,
                        FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                            2, inputs),
                        values);
                }
            }

            void NonZeroPattern(
                const ProxyUserData &,
                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
                FlatVector<AutoDiffDiff<1, NonZero>> values) const override
            {
                values = AutoDiffDiff<1, NonZero>(false);
                for (int output = 0; output < Dimension(); ++output)
                {
                    const size_t base = size_t(output) *
                                        table->terms_per_minor *
                                        size_t(degree);
                    for (size_t term = 0;
                         term < table->terms_per_minor;
                         ++term)
                    {
                        AutoDiffDiff<1, NonZero> term_value(false);
                        if (!metric_derivative)
                        {
                            term_value = inputs[0](
                                table->metric_indices[base + term * size_t(degree)]);
                            for (int factor = 1; factor < degree;
                                 ++factor)
                                term_value = term_value * inputs[0](
                                                              table->metric_indices[base + term * size_t(degree) +
                                                                                    size_t(factor)]);
                        }
                        else
                            for (int differentiated_factor = 0;
                                 differentiated_factor < degree;
                                 ++differentiated_factor)
                            {
                                auto product = inputs[1](
                                    table->metric_indices[base + term * size_t(degree) +
                                                          size_t(differentiated_factor)]);
                                for (int factor = 0; factor < degree;
                                     ++factor)
                                    if (factor != differentiated_factor)
                                        product = product * inputs[0](
                                                                table->metric_indices[base + term * size_t(degree) +
                                                                                      size_t(factor)]);
                                term_value = term_value + product;
                            }
                        values(output) = values(output) + term_value;
                    }
                }
            }

            shared_ptr<CoefficientFunction> Diff(
                const CoefficientFunction *var,
                shared_ptr<CoefficientFunction> dir) const override
            {
                if (this == var)
                    return dir;

                bool metric_contains_var = false;
                TraverseSemanticDAG(
                    metric, [&](CoefficientFunction &node)
                    { metric_contains_var |= &node == var; });
                bool derivative_contains_var = false;
                if (metric_derivative)
                    TraverseSemanticDAG(
                        metric_derivative,
                        [&](CoefficientFunction &node)
                        {
                            derivative_contains_var |= &node == var;
                        });
                if (!metric_contains_var && !derivative_contains_var)
                    return ZeroCF(Dimensions());

                if (!metric_derivative)
                {
                    auto directional_metric =
                        metric->Diff(var, std::move(dir));
                    if (IsConstantZero(directional_metric))
                        return ZeroCF(Dimensions());
                    return make_shared<
                        InducedFormMetricCoefficientFunction>(
                        metric, std::move(directional_metric), degree,
                        dim);
                }

                // Higher derivatives are uncommon and need both the
                // derivative of the metric and of the current direction.
                // Preserve their generic semantics without complicating
                // the fast first-directional evaluator.
                Array<shared_ptr<CoefficientFunction>> components;
                for (int output = 0; output < Dimension(); ++output)
                {
                    shared_ptr<CoefficientFunction> sum;
                    const size_t base = size_t(output) *
                                        table->terms_per_minor *
                                        size_t(degree);
                    for (size_t term = 0;
                         term < table->terms_per_minor;
                         ++term)
                        for (int differentiated_factor = 0;
                             differentiated_factor < degree;
                             ++differentiated_factor)
                        {
                            shared_ptr<CoefficientFunction> product =
                                MakeComponentCoefficientFunction(
                                    metric_derivative,
                                    table->metric_indices[base + term * size_t(degree) +
                                                          size_t(differentiated_factor)]);
                            for (int factor = 0; factor < degree; ++factor)
                            {
                                if (factor == differentiated_factor)
                                    continue;
                                product = product *
                                          MakeComponentCoefficientFunction(
                                              metric,
                                              table->metric_indices[base + term * size_t(degree) +
                                                                    size_t(factor)]);
                            }
                            if (table->signs[term] < 0)
                                product = ScaleCoefficientCF(
                                    product, ConstantCF(-1.0));
                            sum = sum ? SymbolicSumCF(sum, product)
                                      : product;
                        }
                    components.Append(
                        sum ? sum : ConstantCF(0.0));
                }
                const int size = int(table->basis->IndependentSize());
                auto expression = MakeVectorialCoefficientFunction(
                                      std::move(components))
                                      ->Reshape(
                                          Array<int>{size, size});
                return expression->Diff(var, std::move(dir));
            }
        };

        shared_ptr<CoefficientFunction> InducedMetric(
            const shared_ptr<CoefficientFunction> &metric,
            const shared_ptr<const FormBasis> &basis)
        {
            if (basis->Degree() == 1)
                return metric;
            if (basis->Degree() == 0)
                return ConstantCF(1.0)->Reshape(Array<int>{1, 1});
            return make_shared<InducedFormMetricCoefficientFunction>(
                metric, nullptr, basis->Degree(),
                basis->DimensionOfSpace());
        }

        shared_ptr<CoefficientFunction> DenseNormalizedFormInnerProduct(
            const shared_ptr<CoefficientFunction> &left,
            const shared_ptr<CoefficientFunction> &right,
            const shared_ptr<CoefficientFunction> &inverse_metric,
            int rank, double normalization)
        {
            const std::string left_signature = SIGNATURE.substr(0, rank);
            const std::string right_signature = SIGNATURE.substr(rank, rank);
            std::string signature = left_signature + "," + right_signature;
            Array<shared_ptr<CoefficientFunction>> inputs{left, right};
            for (int axis = 0; axis < rank; ++axis)
            {
                signature += ",";
                signature.push_back(left_signature[size_t(axis)]);
                signature.push_back(right_signature[size_t(axis)]);
                inputs.Append(inverse_metric);
            }
            signature += "->";
            auto result = SymbolicEinsumCF(signature, inputs);
            return normalization == 1.0
                       ? result
                       : ScaleCoefficientCF(
                             result, ConstantCF(normalization));
        }

        shared_ptr<CoefficientFunction> MakeCompactKFormIPEvaluator(
            const CompactKFormView &left,
            const CompactKFormView &right,
            const shared_ptr<CoefficientFunction> &inverse_metric)
        {
            auto basis = GetFormBasis(left.dim, left.degree);
            const int size = int(basis->IndependentSize());
            auto left_value = AsCompactVector(left.independent, size);
            auto right_value = AsCompactVector(right.independent, size);
            auto induced = InducedMetric(inverse_metric, basis);
            return SymbolicEinsumCF(
                "i,j,ij->", {left_value, right_value, induced});
        }

        shared_ptr<CoefficientFunction> MakeCompactDoubleFormIPEvaluator(
            const CompactDoubleFormView &left,
            const CompactDoubleFormView &right,
            const shared_ptr<CoefficientFunction> &inverse_metric)
        {
            auto left_basis = GetFormBasis(left.dim, left.left_degree);
            auto right_basis = GetFormBasis(left.dim, left.right_degree);
            const int left_size = int(left_basis->IndependentSize());
            const int right_size = int(right_basis->IndependentSize());
            auto left_value = AsCompactVector(
                                  left.independent, left_size * right_size)
                                  ->Reshape(
                                      Array<int>{left_size, right_size});
            auto right_value = AsCompactVector(
                                   right.independent, left_size * right_size)
                                   ->Reshape(
                                       Array<int>{left_size, right_size});
            auto left_metric = InducedMetric(inverse_metric, left_basis);
            auto right_metric = left_basis == right_basis
                                    ? left_metric
                                    : InducedMetric(inverse_metric,
                                                    right_basis);
            return SymbolicEinsumCF(
                "ij,kl,ik,jl->",
                {left_value, right_value, left_metric, right_metric});
        }

        class CompactKFormInnerProductCoefficientFunction
            : public SymbolicExpressionCoefficientFunction
        {
            int degree;

            shared_ptr<CoefficientFunction> Rebuild(
                const Array<shared_ptr<CoefficientFunction>> &inputs) const override
            {
                auto left = dynamic_pointer_cast<KFormCoefficientFunction>(
                    inputs[0]);
                auto right = dynamic_pointer_cast<KFormCoefficientFunction>(
                    inputs[1]);
                if (left && right)
                    if (auto compact = CompactKFormInnerProduct(
                            left, right, inputs[2]))
                        return compact;
                return DenseNormalizedFormInnerProduct(
                    inputs[0], inputs[1], inputs[2], degree,
                    1.0 / double(Factorial(degree)));
            }

        public:
            CompactKFormInnerProductCoefficientFunction(
                shared_ptr<KFormCoefficientFunction> left,
                shared_ptr<KFormCoefficientFunction> right,
                shared_ptr<CoefficientFunction> inverse_metric)
                : SymbolicExpressionCoefficientFunction(
                      {left, right, inverse_metric},
                      MakeCompactKFormIPEvaluator(
                          *TryGetCompactView(*left),
                          *TryGetCompactView(*right), inverse_metric)),
                  degree(left->Degree())
            {
            }

            auto GetCArgs() const
            {
                return tuple{
                    dynamic_pointer_cast<KFormCoefficientFunction>(
                        operands[0]),
                    dynamic_pointer_cast<KFormCoefficientFunction>(
                        operands[1]),
                    operands[2]};
            }
            string GetDescription() const override
            {
                return "CompactKFormInnerProductCF";
            }
            shared_ptr<CoefficientFunction> Diff(
                const CoefficientFunction *var,
                shared_ptr<CoefficientFunction> dir) const override
            {
                if (this == var)
                    return dir;
                return DenseNormalizedFormInnerProduct(
                           operands[0], operands[1], operands[2], degree,
                           1.0 / double(Factorial(degree)))
                    ->Diff(var, dir);
            }
        };

        class CompactDoubleFormInnerProductCoefficientFunction
            : public SymbolicExpressionCoefficientFunction
        {
            int left_degree;
            int right_degree;

            shared_ptr<CoefficientFunction> Rebuild(
                const Array<shared_ptr<CoefficientFunction>> &inputs) const override
            {
                auto left =
                    dynamic_pointer_cast<DoubleFormCoefficientFunction>(
                        inputs[0]);
                auto right =
                    dynamic_pointer_cast<DoubleFormCoefficientFunction>(
                        inputs[1]);
                if (left && right)
                    if (auto compact = CompactDoubleFormInnerProduct(
                            left, right, inputs[2]))
                        return compact;
                return DenseNormalizedFormInnerProduct(
                    inputs[0], inputs[1], inputs[2],
                    left_degree + right_degree,
                    1.0 / double(Factorial(left_degree) *
                                 Factorial(right_degree)));
            }

        public:
            CompactDoubleFormInnerProductCoefficientFunction(
                shared_ptr<DoubleFormCoefficientFunction> left,
                shared_ptr<DoubleFormCoefficientFunction> right,
                shared_ptr<CoefficientFunction> inverse_metric)
                : SymbolicExpressionCoefficientFunction(
                      {left, right, inverse_metric},
                      MakeCompactDoubleFormIPEvaluator(
                          *TryGetCompactView(*left),
                          *TryGetCompactView(*right), inverse_metric)),
                  left_degree(left->LeftDegree()),
                  right_degree(left->RightDegree())
            {
            }

            auto GetCArgs() const
            {
                return tuple{
                    dynamic_pointer_cast<DoubleFormCoefficientFunction>(
                        operands[0]),
                    dynamic_pointer_cast<DoubleFormCoefficientFunction>(
                        operands[1]),
                    operands[2]};
            }
            string GetDescription() const override
            {
                return "CompactDoubleFormInnerProductCF";
            }
            shared_ptr<CoefficientFunction> Diff(
                const CoefficientFunction *var,
                shared_ptr<CoefficientFunction> dir) const override
            {
                if (this == var)
                    return dir;
                // Scalar variables that occur unchanged in the compact
                // independent values or induced metrics can be
                // differentiated on the evaluator directly, without
                // expanding either alternating index block. A scalar
                // semantic wrapper need not itself occur in the native
                // evaluator, however, so retain the dense fallback unless
                // the exact differentiation node is present.
                bool evaluator_contains_var = false;
                if (var->Dimensions().Size() == 0)
                    TraverseSemanticDAG(
                        evaluator, [&](CoefficientFunction &node)
                        { evaluator_contains_var |= &node == var; });
                if (evaluator_contains_var)
                    return evaluator->Diff(var, std::move(dir));
                return DenseNormalizedFormInnerProduct(
                           operands[0], operands[1], operands[2],
                           left_degree + right_degree,
                           1.0 / double(Factorial(left_degree) *
                                        Factorial(right_degree)))
                    ->Diff(var, dir);
            }
        };
    }

    shared_ptr<CoefficientFunction> CompactKFormInnerProduct(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right,
        shared_ptr<CoefficientFunction> inverse_metric)
    {
        if (!left || !right || !inverse_metric)
            throw Exception("CompactKFormInnerProduct: null input");
        if (left->Degree() != right->Degree() ||
            left->DimensionOfSpace() != right->DimensionOfSpace())
            throw Exception("CompactKFormInnerProduct: metadata mismatch");
        if (!TryGetCompactView(*left) || !TryGetCompactView(*right))
            return nullptr;
        return make_shared<CompactKFormInnerProductCoefficientFunction>(
            std::move(left), std::move(right),
            std::move(inverse_metric));
    }

    shared_ptr<CoefficientFunction> CompactDoubleFormInnerProduct(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right,
        shared_ptr<CoefficientFunction> inverse_metric)
    {
        if (!left || !right || !inverse_metric)
            throw Exception("CompactDoubleFormInnerProduct: null input");
        if (left->LeftDegree() != right->LeftDegree() ||
            left->RightDegree() != right->RightDegree() ||
            left->DimensionOfSpace() != right->DimensionOfSpace())
            throw Exception(
                "CompactDoubleFormInnerProduct: metadata mismatch");
        if (!TryGetCompactView(*left) || !TryGetCompactView(*right))
            return nullptr;
        return make_shared<
            CompactDoubleFormInnerProductCoefficientFunction>(
            std::move(left), std::move(right),
            std::move(inverse_metric));
    }

    std::optional<CompactKFormView> TryGetProvenCompactInput(
        const shared_ptr<KFormCoefficientFunction> &form)
    {
        if (auto compact = TryGetCompactView(*form))
            return compact;
        // Scalars and one-forms are alternating for every full component
        // value. Their existing public constructors remain dense. An
        // operation may nevertheless consume their full value directly as
        // independent data without dropping or projecting any component.
        if (form->DimensionOfSpace() > 0 && form->Degree() <= 1)
            return CompactKFormView{NativeCoefficientValue(form),
                                    form->Degree(),
                                    form->DimensionOfSpace()};
        return std::nullopt;
    }

    std::optional<CompactDoubleFormView> TryGetProvenCompactInput(
        const shared_ptr<DoubleFormCoefficientFunction> &form)
    {
        if (auto compact = TryGetCompactView(*form))
            return compact;
        // Each block of degree zero or one is alternating for every full
        // value, so consuming it as independent data is lossless. This
        // includes arbitrary public (1,1) matrices and establishes the
        // compact seed for ordinary double-form wedge chains.
        if (form->DimensionOfSpace() > 0 &&
            form->LeftDegree() <= 1 && form->RightDegree() <= 1)
        {
            auto independent = NativeCoefficientValue(form);
            if (form->LeftDegree() + form->RightDegree() > 1)
                independent = independent->Reshape(
                    Array<int>{int(form->Dimension())});
            return CompactDoubleFormView{
                std::move(independent), form->LeftDegree(),
                form->RightDegree(), form->DimensionOfSpace()};
        }
        return std::nullopt;
    }

    static ngcore::RegisterClassForArchive<
        InducedFormMetricCoefficientFunction, CoefficientFunction>
        reg_induced_form_metric_cf;
    static ngcore::RegisterClassForArchive<
        CompactKFormInnerProductCoefficientFunction,
        CoefficientFunction>
        reg_compact_kform_inner_product_cf;
    static ngcore::RegisterClassForArchive<
        CompactDoubleFormInnerProductCoefficientFunction,
        CoefficientFunction>
        reg_compact_double_form_inner_product_cf;

    size_t InducedFormMetricTableStorageBytesLowerBound(
        int dim, int degree)
    {
        return GetInducedFormMetricTable(dim, degree)
            ->OwnedStorageBytesLowerBound();
    }

} // namespace kforms_internal
