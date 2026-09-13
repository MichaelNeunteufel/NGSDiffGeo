// Implementation fragment included only by ../kforms.cpp.
// Compact exterior-derivative lookup table and evaluation node.

namespace kforms_internal
{
    struct CompactExteriorDerivativeTable
    {
        int input_size;
        int output_size;
        size_t terms_per_index;
        std::vector<int> signs;
        std::vector<int> gradient_indices;

        size_t OwnedStorageBytesLowerBound() const
        {
            return sizeof(CompactExteriorDerivativeTable) +
                   (signs.capacity() + gradient_indices.capacity()) *
                       sizeof(int);
        }
    };

    shared_ptr<const CompactExteriorDerivativeTable>
    GetCompactExteriorDerivativeTable(int dim, int degree)
    {
        if (dim < 1 || dim > MAX_SPACE_DIM || degree < 0 ||
            degree >= dim)
            throw Exception(
                "CompactExteriorDerivativeTable: invalid dimension or degree");
        constexpr size_t extent = MAX_SPACE_DIM + 1;
        static std::array<std::array<std::once_flag, extent>, extent>
            flags;
        static std::array<
            std::array<shared_ptr<const CompactExteriorDerivativeTable>,
                       extent>,
            extent>
            cache;
        std::call_once(flags[size_t(dim)][size_t(degree)], [&]
                       {
                auto input_basis = GetFormBasis(dim, degree);
                auto output_basis = GetFormBasis(dim, degree + 1);
                auto table = make_shared<CompactExteriorDerivativeTable>();
                table->input_size = int(input_basis->IndependentSize());
                table->output_size = int(output_basis->IndependentSize());
                table->terms_per_index = size_t(degree + 1);
                table->signs.reserve(table->terms_per_index);
                for (int omitted = 0; omitted <= degree; ++omitted)
                    table->signs.push_back(
                        (degree + omitted) % 2 ? -1 : 1);
                table->gradient_indices.reserve(
                    size_t(table->output_size) * table->terms_per_index);

                for (size_t output = 0;
                     output < output_basis->IndependentSize(); ++output)
                {
                    const auto &canonical =
                        output_basis->CanonicalIndex(output);
                    for (int omitted = 0; omitted <= degree; ++omitted)
                    {
                        std::vector<int> input_index;
                        input_index.reserve(size_t(degree));
                        for (int position = 0; position <= degree;
                             ++position)
                            if (position != omitted)
                                input_index.push_back(
                                    canonical[size_t(position)]);
                        table->gradient_indices.push_back(
                            canonical[size_t(omitted)] * table->input_size +
                            input_basis->CompactIndex(input_index));
                    }
                }
                cache[size_t(dim)][size_t(degree)] = std::move(table); });
        return cache[size_t(dim)][size_t(degree)];
    }

    // Preserve the original definition order while keeping cross-family
    // reporting out of the exterior-derivative implementation.
#include "kforms_cache_diagnostics.inc"

    class CompactExteriorDerivativeIndependentCoefficientFunction
        : public T_CoefficientFunction<
              CompactExteriorDerivativeIndependentCoefficientFunction>
    {
        using BASE = T_CoefficientFunction<
            CompactExteriorDerivativeIndependentCoefficientFunction>;

        shared_ptr<CoefficientFunction> gradient;
        int degree;
        int dim;
        shared_ptr<const CompactExteriorDerivativeTable> table;
        bool structural_zero;

        template <typename INPUT, typename VALUES>
        void EvaluateFromInput(size_t points, INPUT input,
                               VALUES values) const
        {
            using T = typename VALUES::TSCAL;
            for (size_t ip = 0; ip < points; ++ip)
                for (int output = 0; output < Dimension(); ++output)
                {
                    T sum = T(0);
                    const size_t base =
                        size_t(output) * table->terms_per_index;
                    for (size_t term = 0;
                         term < table->terms_per_index; ++term)
                        sum += T(table->signs[term]) *
                               input(table->gradient_indices[base + term],
                                     ip);
                    values(output, ip) = sum;
                }
        }

    public:
        CompactExteriorDerivativeIndependentCoefficientFunction(
            shared_ptr<CoefficientFunction> agradient,
            int adegree, int adim,
            shared_ptr<const CompactExteriorDerivativeTable> atable)
            : BASE(atable->output_size,
                   RequireCompactNonNull(agradient)->IsComplex()),
              gradient(std::move(agradient)), degree(adegree), dim(adim),
              table(std::move(atable)),
              structural_zero(gradient->IsZeroCF())
        {
            if (gradient->Dimension() != dim * table->input_size)
                throw Exception(
                    "CompactExteriorDerivativeIndependentCF: gradient shape mismatch");
            SetDimensions(Array<int>{table->output_size});
            elementwise_constant = gradient->ElementwiseConstant();
        }

        CompactExteriorDerivativeIndependentCoefficientFunction(
            shared_ptr<CoefficientFunction> agradient,
            int adegree, int adim)
            : CompactExteriorDerivativeIndependentCoefficientFunction(
                  std::move(agradient), adegree, adim,
                  GetCompactExteriorDerivativeTable(adim, adegree))
        {
        }

        auto GetCArgs() const { return tuple{gradient, degree, dim}; }

        string GetDescription() const override
        {
            return "CompactExteriorDerivativeIndependentCF";
        }

        void CalcEquivalenceKey() override
        {
            equivalence_key =
                GetDescription() + "[k=" + ToString(degree) +
                ",dim=" + ToString(dim) + "](" +
                gradient->EquivalenceKey() + ")";
        }

        void DoArchive(Archive &) override {}

        bool DefinedOn(const ElementTransformation &trafo) override
        {
            return gradient->DefinedOn(trafo);
        }

        bool IsZeroCF() const override { return structural_zero; }

        Array<shared_ptr<CoefficientFunction>>
        InputCoefficientFunctions() const override
        {
            return {gradient};
        }

        void TraverseTree(
            const function<void(CoefficientFunction &)> &func) override
        {
            gradient->TraverseTree(func);
            func(*this);
        }

        using BASE::Evaluate;

        double Evaluate(
            const BaseMappedIntegrationPoint &ip) const override
        {
            if (Dimensions().Size() != 0)
                throw Exception(
                    "CompactExteriorDerivativeIndependentCF: scalar evaluation requires scalar shape");
            return BASE::Evaluate(ip);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &mir,
                        BareSliceMatrix<T, ORD> values) const
        {
            auto &heap = TLHeap();
            HeapReset reset(heap);
            FlatMatrix<T, ORD> input(
                gradient->Dimension(), mir.Size(), heap);
            gradient->Evaluate(mir, input);
            EvaluateFromInput(mir.Size(), input, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(
            const MIR &mir,
            FlatArray<BareSliceMatrix<T, ORD>> inputs,
            BareSliceMatrix<T, ORD> values) const
        {
            EvaluateFromInput(mir.Size(), inputs[0], values);
        }

        void GenerateCode(Code &code, FlatArray<int> inputs,
                          int index) const override
        {
            DeclareTensorFieldGeneratedCoefficient(
                code, index, Dimensions(), IsComplex());
            for (int output = 0; output < Dimension(); ++output)
            {
                CodeExpr sum;
                const size_t base =
                    size_t(output) * table->terms_per_index;
                for (size_t term = 0;
                     term < table->terms_per_index; ++term)
                {
                    CodeExpr value = Var(
                        inputs[0],
                        table->gradient_indices[base + term],
                        gradient->Dimensions());
                    AccumulateSignedCodeExpression(
                        sum, std::move(value), table->signs[term], term);
                }
                code.body += Var(index, output, Dimensions()).Assign(sum.S(), false);
            }
        }

        void NonZeroPattern(
            const ProxyUserData &ud,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            Vector<AutoDiffDiff<1, NonZero>> input(
                gradient->Dimension());
            gradient->NonZeroPattern(ud, input);
            FlatVector<AutoDiffDiff<1, NonZero>> input_storage[] = {
                input};
            NonZeroPattern(
                ud,
                FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>>(
                    1, input_storage),
                values);
        }

        void NonZeroPattern(
            const ProxyUserData &,
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            values = AutoDiffDiff<1, NonZero>(false);
            for (int output = 0; output < Dimension(); ++output)
            {
                auto sum = AutoDiffDiff<1, NonZero>(false);
                const size_t base =
                    size_t(output) * table->terms_per_index;
                for (size_t term = 0;
                     term < table->terms_per_index; ++term)
                    sum = sum + inputs[0](
                                    table->gradient_indices[base + term]);
                values(output) = sum;
            }
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
                       CompactExteriorDerivativeIndependentCoefficientFunction>(
                       gradient->Transform(transformation), degree, dim);
        }

        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return make_shared<
                CompactExteriorDerivativeIndependentCoefficientFunction>(
                gradient->Diff(var, dir), degree, dim);
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
        CompactExteriorDerivativeIndependentCoefficientFunction,
        CoefficientFunction>
        reg_compact_exterior_derivative_independent_cf;

} // namespace kforms_internal
