#ifndef TENSOR_FIELDS
#define TENSOR_FIELDS

// #include <fem.hpp>
#include <coefficient.hpp>
#include <cstdint>
#include <string_view>

/**
 * @file tensor_fields.hpp
 * @brief This file contains the definition and implementation of scalar, vector, 1-form, and tensor field coefficient functions.
 */

namespace ngfem
{

    class TensorFieldCoefficientFunction;
    class OneFormCoefficientFunction;
    class VectorFieldCoefficientFunction;
    class ScalarFieldCoefficientFunction;
    // /**
    //  * @var const string SIGNATURE
    //  * @brief A constant string containing all possible letters for building a signature for tensor field coefficient functions.
    //  */
    inline const string SIGNATURE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

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

    struct TensorMeta
    {
        uint8_t rank = 0;     // up to 52
        uint64_t covmask = 0; // bit i = 1 means covariant

        static TensorMeta FromCovString(std::string_view cov)
        {
            if (cov.size() > SIGNATURE.size())
                throw ngstd::Exception("TensorMeta: rank overflow (>52)");

            TensorMeta m;
            m.rank = uint8_t(cov.size());
            for (size_t i = 0; i < cov.size(); ++i)
            {
                char c = cov[i];
                if (c == '1')
                    m.covmask |= (uint64_t(1) << i);
                else if (c != '0')
                    throw ngstd::Exception("TensorMeta: covariant_indices must be only '0'/'1'");
            }
            return m;
        }

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
            size_t id = size_t(rank) + offset;
            if (id >= SIGNATURE.size())
                throw Exception("TensorMeta: signature overflow (>52)");
            return SIGNATURE[id];
        }

        std::string Sig() const { return SIGNATURE.substr(0, rank); }

        // Set one slot (raise/lower)
        TensorMeta WithCovariant(size_t i, bool cov) const
        {
            TensorMeta m = *this;
            uint64_t bit = (uint64_t(1) << i);
            if (cov)
                m.covmask |= bit;
            else
                m.covmask &= ~bit;
            return m;
        }

        // Append a new index at the end
        TensorMeta Appended(bool cov) const
        {
            if (rank >= SIGNATURE.size())
                throw ngstd::Exception("TensorMeta: rank overflow (>52)");
            TensorMeta m = *this;
            if (cov)
                m.covmask |= (uint64_t(1) << rank);
            m.rank++;
            return m;
        }

        TensorMeta Prepended(bool cov) const
        {
            if (rank >= SIGNATURE.size())
                throw Exception("TensorMeta: rank overflow (>52)");
            TensorMeta m;
            m.rank = uint8_t(rank + 1);
            m.covmask = (cov ? 1ull : 0ull) | (covmask << 1);
            return m;
        }

        // Remove index slot i (used in trace/contraction)
        TensorMeta Erased(size_t i) const
        {
            if (i >= rank)
                throw ngstd::Exception("TensorMeta: erase index out of range");
            TensorMeta m;
            m.rank = uint8_t(rank - 1);

            uint64_t low = covmask & ((uint64_t(1) << i) - 1);
            uint64_t high = covmask >> (i + 1);
            m.covmask = low | (high << i);
            return m;
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
                throw Exception("TensorMeta: concat overflow (>52)");
            TensorMeta m;
            m.rank = uint8_t(rank + b.rank);
            m.covmask = covmask | (b.covmask << rank);
            return m;
        }

        bool operator==(const TensorMeta &other) const
        {
            return rank == other.rank && covmask == other.covmask;
        }
    };

    /**
     * @fn shared_ptr<CoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf, const string &covariant_indices)
     * @brief Creates a tensor field coefficient function.
     * @param cf The input coefficient function.
     * @param covariant_indices string of the from "0110" where "1" indicates a covariant index and "0" a contravariant one.
     * @return A shared pointer to the created tensor field coefficient function.
     */
    shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                             const string &covariant_indices);

    shared_ptr<TensorFieldCoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                             const TensorMeta &meta);

    /**
     * @fn shared_ptr<CoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf)
     * @brief Creates a vector field coefficient function.
     * @param cf The input coefficient function.
     * @return A shared pointer to the created vector field coefficient function.
     */
    shared_ptr<VectorFieldCoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf);

    /**
     * @class TensorFieldCoefficientFunction
     * @brief A class representing a tensor field coefficient function.
     */
    class TensorFieldCoefficientFunction : public T_CoefficientFunction<TensorFieldCoefficientFunction>
    {
        shared_ptr<CoefficientFunction> c1;
        TensorMeta meta;
        const std::string cov;

    public:
        /**
         * @fn TensorFieldCoefficientFunction::TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, const string &acovariant_indices)
         * @brief Constructor for TensorFieldCoefficientFunction.
         * @param ac1 The input coefficient function.
         * @param acovariant_indices string of the from "0110" where "1" indicates a covariant index and "0" a contravariant one.
         * @throws Exception if the number of indices does not match the number of dimensions of the input coefficient function or if not all dimensions are the same.
         */
        TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, std::string_view acov)
            : T_CoefficientFunction<TensorFieldCoefficientFunction>(ac1->Dimension(), ac1->IsComplex()), c1(ac1), meta(TensorMeta::FromCovString(acov)), cov(acov)
        {
            this->SetDimensions(ac1->Dimensions());
            if (Dimensions().Size() != meta.rank)
                throw ngstd::Exception("TensorField: covariant_indices length must equal tensor rank. Received length " + ToString(meta.rank) + ", but dimensions " + ToString(Dimensions()));

            if (ac1->Dimensions().Size() > 0)
            {
                auto dim = ac1->Dimensions()[0];
                for (auto cf_dim : ac1->Dimensions())
                    if (cf_dim != dim)
                        throw Exception("TensorFieldCF: all dimensions must be the same");
            }
        }

        TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, const TensorMeta &ameta)
            : T_CoefficientFunction<TensorFieldCoefficientFunction>(ac1->Dimension(), ac1->IsComplex()), c1(ac1), meta(ameta), cov(ameta.CovString())
        {
            this->SetDimensions(ac1->Dimensions());
            if (Dimensions().Size() != meta.rank)
                throw ngstd::Exception("TensorField: covariant_indices length must equal tensor rank. Received length " + ToString(meta.rank) + ", but dimensions " + ToString(Dimensions()));

            if (ac1->Dimensions().Size() > 0)
            {
                auto dim = ac1->Dimensions()[0];
                for (auto cf_dim : ac1->Dimensions())
                    if (cf_dim != dim)
                        throw Exception("TensorFieldCF: all dimensions must be the same");
            }
        }

        virtual string GetDescription() const override
        {
            return "TensorFieldCF";
        }

        const TensorMeta &Meta() const { return meta; }
        const std::string &GetCovariantIndices() const { return cov; }

        const shared_ptr<CoefficientFunction> &GetCoefficients() const { return c1; }

        string GetSignature() const
        {
            return SIGNATURE.substr(0, meta.rank);
        }

        auto GetCArgs() const { return tuple{c1}; }

        void DoArchive(Archive &ar) override
        {
            /*
            BASE::DoArchive(ar);
            ar.Shallow(c1);
            */
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
            return c1->NonZeroPattern(ud, values);
        }

        shared_ptr<CoefficientFunction>
        Transform(CoefficientFunction::T_Transform &transformation) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (transformation.cache.count(thisptr))
                return transformation.cache[thisptr];
            if (transformation.replace.count(thisptr))
                return transformation.replace[thisptr];
            auto newcf = make_shared<TensorFieldCoefficientFunction>(c1->Transform(transformation), meta);
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            return c1->Evaluate(ip);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, BareSliceMatrix<T, ORD> values) const
        {
            c1->Evaluate(ir, values);
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            c1->Evaluate(ir, input, values);
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return TensorFieldCF(c1->Diff(var, dir), cov);
        }
        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            auto res = TensorFieldCF(c1->DiffJacobi(var, cache), meta);
            cache[thisptr] = res;
            return res;
        }

        virtual bool IsZeroCF() const override { return c1->IsZeroCF(); }
    };

    /**
     * @class VectorFieldCoefficientFunction
     * @brief A class representing a vector field coefficient function.
     * @details This class is derived from TensorFieldCoefficientFunction and represents a vector field.
     */
    class VectorFieldCoefficientFunction : public TensorFieldCoefficientFunction
    {
    public:
        /**
         * @fn VectorFieldCoefficientFunction::VectorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
         * @brief Constructor for VectorFieldCoefficientFunction.
         * @param ac1 The input coefficient function.
         */
        VectorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
            : TensorFieldCoefficientFunction(ac1, "0")
        {
        }

        virtual string GetDescription() const override
        {
            return "VectorFieldCF";
        }
    };

    /**
     * @fn shared_ptr<CoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2)
     * @brief Computes the tensor product of two tensor field coefficient functions.
     * @param c1 The first tensor field coefficient function.
     * @param c2 The second tensor field coefficient function.
     * @return A shared pointer to the resulting tensor field coefficient function.
     */
    shared_ptr<TensorFieldCoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2);

    bool IsVectorField(const TensorFieldCoefficientFunction &t);
    bool IsOneForm(const TensorFieldCoefficientFunction &t);

}

#include <python_ngstd.hpp>
void ExportTensorFields(py::module m);

#endif // TENSOR_FIELDS
