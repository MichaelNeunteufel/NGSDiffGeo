#ifndef NGSDIFFGEO_TENSOR_FIELDS_HPP
#define NGSDIFFGEO_TENSOR_FIELDS_HPP

#include <coefficient.hpp>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

/**
 * @file tensor_fields.hpp
 * @brief Metadata-preserving coefficient-function wrappers for tensor fields.
 */

namespace ngfem
{

    class TensorFieldCoefficientFunction;
    class OneFormCoefficientFunction;
    class VectorFieldCoefficientFunction;
    class ScalarFieldCoefficientFunction;

    inline constexpr int MAX_SPACE_DIM = 4;
    inline constexpr int MAX_PERMUTATION_RANK = 4;
    inline constexpr int MAX_FORM_RANK = 8;
    inline constexpr size_t MAX_SIGNATURE_LABELS = 52;
    inline constexpr std::string_view SIGNATURE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static_assert(SIGNATURE_CHARS.size() == MAX_SIGNATURE_LABELS,
                  "SIGNATURE size must match MAX_SIGNATURE_LABELS");
    inline const std::string SIGNATURE(SIGNATURE_CHARS);

    inline std::string EraseLabel(std::string s, size_t i)
    {
        if (i >= s.size())
            throw ngstd::Exception("EraseLabel: out of range");
        s.erase(i, 1);
        return s;
    }

    inline std::string Erase2Labels(std::string s, size_t i, size_t j)
    {
        if (i == j)
            throw ngstd::Exception("Erase2Labels: indices must differ");
        if (i > j)
            std::swap(i, j);
        if (j >= s.size())
            throw ngstd::Exception("Erase2Labels: out of range");
        s.erase(j, 1);
        s.erase(i, 1);
        return s;
    }

    inline std::string TensorFieldGeneratedCoefficientType(
        const Code &code, bool is_complex)
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

    inline void DeclareTensorFieldGeneratedCoefficient(
        Code &code, int index, FlatArray<int> dims, bool is_complex)
    {
        // Code::Declare is not exported by the NGSolve DLL on Windows.
        // Generate the equivalent declaration locally so addon wheels link.
        const std::string type =
            TensorFieldGeneratedCoefficientType(code, is_complex);

        if (dims.Size() == 0)
        {
            code.body += Var(index).Declare(type);
            return;
        }

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

    /**
     * Variance metadata for the ordered component axes of a tensor.
     *
     * Bit i is one for a covariant axis and zero for a contravariant axis.
     * Instances are validated on construction: the rank fits the einsum label
     * alphabet and the mask contains no bits beyond that rank.
     */
    class TensorMeta
    {
        // rank of tensor
        // E.g. 0 for scalar, 1 for vector/1-form, 2 for matrix, etc.
        uint8_t rank = 0;
        uint64_t covmask = 0;

        static uint8_t CheckedRank(size_t arank)
        {
            if (arank > MAX_SIGNATURE_LABELS)
                throw ngstd::Exception("TensorMeta: rank overflow (>" + ToString(MAX_SIGNATURE_LABELS) + ")");
            return uint8_t(arank);
        }

        static uint64_t ValidBitsMask(size_t arank)
        {
            return arank == 0 ? 0 : (uint64_t(1) << arank) - 1;
        }

        TensorMeta(size_t arank, uint64_t acovmask)
            : rank(CheckedRank(arank)), covmask(acovmask)
        {
            const uint64_t valid_bits = ValidBitsMask(arank);
            if ((covmask & ~valid_bits) != 0)
                throw ngstd::Exception("TensorMeta: covariance mask contains bits outside the tensor rank");
        }

    public:
        TensorMeta() = default;

        static TensorMeta FromCovString(std::string_view cov)
        {
            if (cov.size() > SIGNATURE.size())
                throw ngstd::Exception("TensorMeta: rank overflow (>" + ToString(MAX_SIGNATURE_LABELS) + ")");

            uint64_t mask = 0;
            for (size_t i = 0; i < cov.size(); ++i)
            {
                char c = cov[i];
                if (c == '1')
                    mask |= (uint64_t(1) << i);
                else if (c != '0')
                    throw ngstd::Exception("TensorMeta: covariant_indices must be only '0'/'1'");
            }
            return TensorMeta(cov.size(), mask);
        }

        static TensorMeta FromRaw(size_t rank, uint64_t covmask)
        {
            return TensorMeta(rank, covmask);
        }

        size_t Rank() const { return rank; }
        uint64_t CovarianceMask() const { return covmask; }

        bool Covariant(size_t i) const
        {
            if (i >= rank)
                throw ngstd::Exception("TensorMeta: index out of range");
            return (covmask >> i) & 1;
        }

        std::string CovString() const
        {
            std::string s(rank, '0');
            for (size_t i = 0; i < rank; ++i)
                if (Covariant(i))
                    s[i] = '1';
            return s;
        }

        char Label(size_t i) const
        {
            if (i >= rank)
                throw ngstd::Exception("TensorMeta: label index out of range");
            return SIGNATURE[i];
        }

        char FreshLabel(size_t offset = 0) const
        {
            if (offset >= SIGNATURE.size() - size_t(rank))
                throw Exception("TensorMeta: signature overflow (>" + ToString(MAX_SIGNATURE_LABELS) + ")");
            return SIGNATURE[size_t(rank) + offset];
        }

        std::string Sig() const { return SIGNATURE.substr(0, rank); }

        TensorMeta WithCovariant(size_t i, bool cov) const
        {
            if (i >= rank)
                throw ngstd::Exception("TensorMeta: index out of range");
            uint64_t bit = (uint64_t(1) << i);
            uint64_t mask = cov ? (covmask | bit) : (covmask & ~bit);
            return TensorMeta(rank, mask);
        }

        TensorMeta Appended(bool cov) const
        {
            if (rank >= SIGNATURE.size())
                throw ngstd::Exception("TensorMeta: rank overflow (>" + ToString(MAX_SIGNATURE_LABELS) + ")");
            uint64_t mask = covmask;
            if (cov)
                mask |= (uint64_t(1) << rank);
            return TensorMeta(size_t(rank) + 1, mask);
        }

        TensorMeta Prepended(bool cov) const
        {
            if (rank >= SIGNATURE.size())
                throw Exception("TensorMeta: rank overflow (>" + ToString(MAX_SIGNATURE_LABELS) + ")");
            return TensorMeta(size_t(rank) + 1,
                              (cov ? 1ull : 0ull) | (covmask << 1));
        }

        TensorMeta Erased(size_t i) const
        {
            if (i >= rank)
                throw ngstd::Exception("TensorMeta: erase index out of range");
            uint64_t low = covmask & ((uint64_t(1) << i) - 1);
            uint64_t high = covmask >> (i + 1);
            return TensorMeta(size_t(rank) - 1, low | (high << i));
        }

        TensorMeta Erased2(size_t i, size_t j) const
        {
            if (i == j)
                throw Exception("TensorMeta: erase2 needs distinct indices");
            if (i > j)
                std::swap(i, j);
            return Erased(j).Erased(i);
        }

        TensorMeta Concatenated(const TensorMeta &b) const
        {
            if (size_t(rank) + size_t(b.rank) > SIGNATURE.size())
                throw Exception("TensorMeta: concat overflow (>" + ToString(MAX_SIGNATURE_LABELS) + ")");
            return TensorMeta(size_t(rank) + size_t(b.rank),
                              covmask | (b.covmask << rank));
        }

        bool operator==(const TensorMeta &other) const
        {
            return rank == other.rank && covmask == other.covmask;
        }
    };

    /**
     * Canonically wrap a value graph with tensor variance metadata.
     *
     * Compatible wrappers are reused. Incompatible metadata-only wrappers are
     * stripped so retyping does not grow the expression graph.
     */
    shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                             const string &covariant_indices);

    shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                             const TensorMeta &meta);

    /// Canonical contravariant rank-one wrapper.
    shared_ptr<VectorFieldCoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf);
    /// Native einsum evaluation with reconstructible symbolic operands.
    shared_ptr<CoefficientFunction> SymbolicEinsumCF(
        const std::string &signature,
        const Array<shared_ptr<CoefficientFunction>> &inputs);

    /// Sum and scalar product retaining zero-valued symbolic operands.
    shared_ptr<CoefficientFunction> SymbolicSumCF(
        shared_ptr<CoefficientFunction> a, shared_ptr<CoefficientFunction> b);
    shared_ptr<CoefficientFunction> ScaleCoefficientCF(
        shared_ptr<CoefficientFunction> value, shared_ptr<CoefficientFunction> scalar);

    /// Native low-rank evaluators with reconstructible semantic operands.
    shared_ptr<CoefficientFunction> SymbolicMatrixProductCF(
        shared_ptr<CoefficientFunction> a, shared_ptr<CoefficientFunction> b);
    shared_ptr<CoefficientFunction> SymbolicInnerProductCF(
        shared_ptr<CoefficientFunction> a, shared_ptr<CoefficientFunction> b);
    shared_ptr<CoefficientFunction> SymbolicMetricInnerProductCF(
        shared_ptr<CoefficientFunction> a,
        shared_ptr<CoefficientFunction> b,
        const Array<shared_ptr<CoefficientFunction>> &metrics,
        const Array<int> &metric_axes);
    shared_ptr<CoefficientFunction> SymbolicTraceCF(
        shared_ptr<CoefficientFunction> value);
    shared_ptr<CoefficientFunction> SymbolicTransposeCF(
        shared_ptr<CoefficientFunction> value);

    shared_ptr<TensorFieldCoefficientFunction> PermuteTensorCF(shared_ptr<TensorFieldCoefficientFunction> tf,
                                                               const std::vector<int> &order);
    shared_ptr<TensorFieldCoefficientFunction> ApplyProjectorToIndex(shared_ptr<TensorFieldCoefficientFunction> tf,
                                                                     shared_ptr<CoefficientFunction> proj,
                                                                     size_t index);
    inline int Factorial(int n)
    {
        static constexpr std::array<int, 13> table = {
            1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
            3628800, 39916800, 479001600};
        if (n < 0)
            throw Exception("Factorial: n must be non-negative");
        if (n >= int(table.size()))
            throw Exception("Factorial: n must not exceed 12 for an int result");
        return table[size_t(n)];
    }

    /**
     * Metadata-only wrapper around an NGSolve coefficient-function value graph.
     *
     * Evaluation, compilation, domain information, and zero structure are
     * delegated to the wrapped coefficient. The wrapper adds the ordered slot
     * variance needed by geometric operations. Its rank must equal the number
     * of component axes, and all component axes must have the same dimension.
     *
     * Derived semantic types override Rewrap so transformations and directional
     * derivatives retain their type without duplicating those algorithms.
     */
    class TensorFieldCoefficientFunction : public T_CoefficientFunction<TensorFieldCoefficientFunction>
    {
        using BASE = T_CoefficientFunction<TensorFieldCoefficientFunction>;

        shared_ptr<CoefficientFunction> c1;
        TensorMeta meta;

        static const CoefficientFunction &CheckedCoefficient(
            const shared_ptr<CoefficientFunction> &cf)
        {
            if (!cf)
                throw ngstd::Exception("TensorFieldCoefficientFunction: input coefficient is null");
            return *cf;
        }

        void Initialize()
        {
            this->SetDimensions(c1->Dimensions());
            this->elementwise_constant = c1->ElementwiseConstant();

            if (Dimensions().Size() != meta.Rank())
                throw ngstd::Exception(
                    "TensorField: covariant_indices length must equal tensor rank. Received length " +
                    ToString(meta.Rank()) + ", but dimensions " + ToString(Dimensions()));

            if (c1->Dimensions().Size() > 0)
            {
                auto dim = c1->Dimensions()[0];
                for (auto cf_dim : c1->Dimensions())
                    if (cf_dim != dim)
                        throw Exception("TensorFieldCF: all dimensions must be the same");
            }
        }

    protected:
        /**
         * Reconstruct the semantic wrapper after an operation on the value graph.
         *
         * Implementations must preserve component axes. Factories should reuse
         * compatible wrappers and avoid nesting metadata-only wrappers.
         */
        virtual shared_ptr<TensorFieldCoefficientFunction>
        Rewrap(shared_ptr<CoefficientFunction> cf) const
        {
            return TensorFieldCF(std::move(cf), meta);
        }

    public:
        /// Construct from a slot string: '1' is covariant, '0' contravariant.
        TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, std::string_view acov)
            : BASE(CheckedCoefficient(ac1).Dimension(), CheckedCoefficient(ac1).IsComplex()),
              c1(std::move(ac1)), meta(TensorMeta::FromCovString(acov))
        {
            Initialize();
        }

        TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, const TensorMeta &ameta)
            : BASE(CheckedCoefficient(ac1).Dimension(), CheckedCoefficient(ac1).IsComplex()),
              c1(std::move(ac1)), meta(ameta)
        {
            Initialize();
        }

        virtual string GetDescription() const override
        {
            return "TensorFieldCF";
        }

        const TensorMeta &Meta() const { return meta; }
        std::string GetCovariantIndices() const { return meta.CovString(); }

        /**
         * Full-shaped semantic value graph used by generic tensor operations.
         *
         * The returned node may later be backed by compact form storage, but its
         * dimensions and values always match this wrapper.
         */
        const shared_ptr<CoefficientFunction> &GetFullCoefficient() const { return c1; }

        // Compatibility name. New representation-sensitive code should use the
        // explicit full-value accessor above.
        const shared_ptr<CoefficientFunction> &GetCoefficients() const
        {
            return GetFullCoefficient();
        }

        string GetSignature() const
        {
            return meta.Sig();
        }

        // Constructor state used by NGSolve's polymorphic archive.
        auto GetCArgs() const { return tuple{GetFullCoefficient(), meta.CovString()}; }

        void DoArchive(Archive &ar) override
        {
            BASE::DoArchive(ar);
        }

        virtual bool DefinedOn(const ElementTransformation &trafo) override
        {
            return c1->DefinedOn(trafo);
        }

        void CalcEquivalenceKey() override
        {
            this->equivalence_key =
                GetDescription() + "[" + meta.CovString() + "](" + c1->EquivalenceKey() + ")";
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

        virtual void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override
        {
            DeclareTensorFieldGeneratedCoefficient(
                code, index, this->Dimensions(), this->IsComplex());
            if (this->Dimensions().Size() == 0)
                code.body += Var(index).Assign(Var(inputs[0]), false);
            else if (code_uses_tensors)
            {
                code.body += "for (size_t i = 0; i < " +
                             ToString(this->Dimension()) + "; i++)\n";
                code.body += "var_" + ToString(index) + "[i] = var_" +
                             ToString(inputs[0]) + "[i];\n";
            }
            else
                for (size_t i = 0; i < this->Dimension(); ++i)
                    code.body += Var(index, i, this->Dimensions())
                                     .Assign(Var(inputs[0], i, c1->Dimensions()), false);
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            c1->NonZeroPattern(ud, values);
        }

        virtual void NonZeroPattern(const class ProxyUserData &ud,
                                    FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
                                    FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            values = input[0];
        }

        shared_ptr<CoefficientFunction>
        Transform(CoefficientFunction::T_Transform &transformation) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (transformation.cache.count(thisptr))
                return transformation.cache[thisptr];
            if (transformation.replace.count(thisptr))
                return transformation.replace[thisptr];
            auto newcf = Rewrap(c1->Transform(transformation));
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            return c1->Evaluate(ip);
        }

        virtual Complex EvaluateComplex(const BaseMappedIntegrationPoint &ip) const override
        {
            return c1->EvaluateComplex(ip);
        }

        virtual double EvaluateConst() const override
        {
            return c1->EvaluateConst();
        }

        using BASE::Evaluate;

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, BareSliceMatrix<T, ORD> values) const
        {
            c1->Evaluate(ir, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            auto input_values = input[0];
            for (size_t ip = 0; ip < ir.Size(); ++ip)
                for (size_t i = 0; i < this->Dimension(); ++i)
                    values(i, ip) = input_values(i, ip);
        }

        void EvaluateDeriv(const BaseMappedIntegrationRule &ir,
                           FlatMatrix<Complex> values,
                           FlatMatrix<Complex> deriv) const override
        {
            c1->EvaluateDeriv(ir, values, deriv);
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return Rewrap(c1->Diff(var, dir));
        }
        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            auto jacobi = c1->DiffJacobi(var, cache);
            // Scalar differentiation adds no component axis, so the original
            // slot metadata remains valid. A tensor-valued variable appends
            // axes whose variance is not defined by this wrapper.
            auto res = var->Dimensions().Size() == 0
                           ? shared_ptr<CoefficientFunction>(Rewrap(jacobi))
                           : jacobi;
            cache[thisptr] = res;
            return res;
        }

        virtual bool IsZeroCF() const override { return c1->IsZeroCF(); }
    };

    /// Contravariant rank-one specialization.
    class VectorFieldCoefficientFunction : public TensorFieldCoefficientFunction
    {
    protected:
        shared_ptr<TensorFieldCoefficientFunction>
        Rewrap(shared_ptr<CoefficientFunction> cf) const override
        {
            return VectorFieldCF(std::move(cf));
        }

    public:
        VectorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
            : TensorFieldCoefficientFunction(ac1, "0")
        {
        }

        virtual string GetDescription() const override
        {
            return "VectorFieldCF";
        }

        auto GetCArgs() const { return tuple{GetFullCoefficient()}; }
    };

    /// Tensor product with left slots followed by right slots.
    shared_ptr<TensorFieldCoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2);

    bool IsVectorField(const TensorFieldCoefficientFunction &t);
    bool IsOneForm(const TensorFieldCoefficientFunction &t);

}

#include <python_ngstd.hpp>
void ExportTensorFields(py::module m);

#endif // NGSDIFFGEO_TENSOR_FIELDS_HPP
