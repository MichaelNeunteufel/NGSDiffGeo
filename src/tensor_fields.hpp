#ifndef TENSOR_FIELDS
#define TENSOR_FIELDS

#include <fem.hpp>

/**
 * @file tensor_fields.hpp
 * @brief This file contains the definition and implementation of scalar, vector, 1-form, and tensor field coefficient functions.
 */

namespace ngfem
{

    /**
     * @var const string SIGNATURE
     * @brief A constant string containing all possible letters for building a signature for tensor field coefficient functions.
     */
    const string SIGNATURE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    /**
     * @fn shared_ptr<CoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf, const string &covariant_indices)
     * @brief Creates a tensor field coefficient function.
     * @param cf The input coefficient function.
     * @param covariant_indices string of the from "0110" where "1" indicates a covariant index and "0" a contravariant one.
     * @return A shared pointer to the created tensor field coefficient function.
     */
    shared_ptr<CoefficientFunction> TensorFieldCF(const shared_ptr<CoefficientFunction> &cf,
                                                  const string &covariant_indices);

    /**
     * @fn shared_ptr<CoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf)
     * @brief Creates a vector field coefficient function.
     * @param cf The input coefficient function.
     * @return A shared pointer to the created vector field coefficient function.
     */
    shared_ptr<CoefficientFunction> VectorFieldCF(const shared_ptr<CoefficientFunction> &cf);

    /**
     * @fn shared_ptr<CoefficientFunction> OneFormCF(const shared_ptr<CoefficientFunction> &cf)
     * @brief Creates a one-form coefficient function.
     * @param cf The input coefficient function.
     * @return A shared pointer to the created one-form coefficient function.
     */
    shared_ptr<CoefficientFunction> OneFormCF(const shared_ptr<CoefficientFunction> &cf);

    /**
     * @fn shared_ptr<CoefficientFunction> ScalarFieldCF(const shared_ptr<CoefficientFunction> &cf)
     * @brief Creates a scalar field coefficient function.
     * @param cf The input coefficient function.
     * @return A shared pointer to the created scalar field coefficient function.
     */
    shared_ptr<CoefficientFunction> ScalarFieldCF(const shared_ptr<CoefficientFunction> &cf);

    /**
     * @class TensorFieldCoefficientFunction
     * @brief A class representing a tensor field coefficient function.
     */
    class TensorFieldCoefficientFunction : public T_CoefficientFunction<TensorFieldCoefficientFunction>
    {
        string covariant_indices;
        shared_ptr<CoefficientFunction> c1;

    public:
        /**
         * @fn TensorFieldCoefficientFunction::TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, const string &acovariant_indices)
         * @brief Constructor for TensorFieldCoefficientFunction.
         * @param ac1 The input coefficient function.
         * @param acovariant_indices string of the from "0110" where "1" indicates a covariant index and "0" a contravariant one.
         * @throws Exception if the number of indices does not match the number of dimensions of the input coefficient function or if not all dimensions are the same.
         */
        TensorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1, const string &acovariant_indices)
            : T_CoefficientFunction<TensorFieldCoefficientFunction>(ac1->Dimension(), ac1->IsComplex()), covariant_indices(acovariant_indices), c1(ac1)
        {
            if (ac1->Dimensions().Size() != acovariant_indices.size())
                throw Exception("TensorFieldCF: number of covariant indices must match the number of dimensions of the input coefficient function");

            this->SetDimensions(ac1->Dimensions());
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

        const string &GetCovariantIndices() const { return covariant_indices; }

        shared_ptr<CoefficientFunction> GetCoefficients() const { return c1; }

        string GetSignature() const
        {
            return SIGNATURE.substr(0, covariant_indices.size());
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

        // shared_ptr<CoefficientFunction>
        // Transform(CoefficientFunction::T_Transform &transformation) const override
        // {
        //     auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        //     if (transformation.cache.count(thisptr))
        //         return transformation.cache[thisptr];
        //     if (transformation.replace.count(thisptr))
        //         return transformation.replace[thisptr];
        //     auto newcf = make_shared<TensorFieldCoefficientFunction>(c1->Transform(transformation));
        //     transformation.cache[thisptr] = newcf;
        //     return newcf;
        // }

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
            return TensorFieldCF(c1->Diff(var, dir), covariant_indices);
        }
        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            auto res = c1->DiffJacobi(var, cache);
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
     * @class OneFormCoefficientFunction
     * @brief A class representing a one-form coefficient function.
     * @details This class is derived from TensorFieldCoefficientFunction and represents a one-form.
     */
    class OneFormCoefficientFunction : public TensorFieldCoefficientFunction
    {
    public:
        /**
         * @fn OneFormCoefficientFunction::OneFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
         * @brief Constructor for OneFormCoefficientFunction.
         * @param ac1 The input coefficient function.
         */
        OneFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
            : TensorFieldCoefficientFunction(ac1, "1")
        {
        }

        virtual string GetDescription() const override
        {
            return "OneFormCF";
        }
    };

    /**
     * @class ScalarFieldCoefficientFunction
     * @brief A class representing a scalar field coefficient function.
     * @details This class is derived from TensorFieldCoefficientFunction and represents a scalar field.
     */
    class ScalarFieldCoefficientFunction : public TensorFieldCoefficientFunction
    {
    public:
        /**
         * @fn ScalarFieldCoefficientFunction::ScalarFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
         * @brief Constructor for ScalarFieldCoefficientFunction.
         * @param ac1 The input coefficient function.
         */
        ScalarFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
            : TensorFieldCoefficientFunction(ac1, "")
        {
        }

        virtual string GetDescription() const override
        {
            return "ScalarFieldCF";
        }
    };

    /**
     * @fn shared_ptr<CoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2)
     * @brief Computes the tensor product of two tensor field coefficient functions.
     * @param c1 The first tensor field coefficient function.
     * @param c2 The second tensor field coefficient function.
     * @return A shared pointer to the resulting tensor field coefficient function.
     */
    shared_ptr<CoefficientFunction> TensorProduct(shared_ptr<TensorFieldCoefficientFunction> c1, shared_ptr<TensorFieldCoefficientFunction> c2);

}

#include <python_ngstd.hpp>
void ExportTensorFields(py::module m);

#endif // TENSOR_FIELDS
