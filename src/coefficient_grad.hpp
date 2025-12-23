#ifndef COEFFICIENT_GRAD
#define COEFFICIENT_GRAD

// #include <fem.hpp>
#include <coefficient.hpp>
// #include <fespace.hpp>
#include <symbolicintegrator.hpp>
#include <diffop.hpp>

using namespace std;

namespace ngfem
{

    shared_ptr<CoefficientFunction> GradCF(const shared_ptr<CoefficientFunction> &cf, size_t dim);

    template <int D>
    class GradCoefficientFunction : public T_CoefficientFunction<GradCoefficientFunction<D>>
    {
        shared_ptr<CoefficientFunction> c1;

    public:
        static constexpr double eps() { return 1e-4; }
        /**
         * @brief Constructs a GradCoefficientFunction object.
         *
         * @param ac1 A shared pointer to a CoefficientFunction object.
         *
         * @throws Exception if any dimension of the input CoefficientFunction is not equal to D.
         */
        GradCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
            : T_CoefficientFunction<GradCoefficientFunction<D>>(ac1->Dimension() * D, ac1->IsComplex()), c1(ac1)
        {
            for (auto cf_dim : ac1->Dimensions())
                if (cf_dim != D)
                    throw Exception("GradCF: all dimensions must be the same and equal to D");

            Array<int> tensor_dims(c1->Dimensions().Size() + 1);
            for (size_t i = 0; i < tensor_dims.Size(); i++)
                tensor_dims[i] = D;

            this->SetDimensions(tensor_dims);
        }

        virtual string GetDescription() const override
        {
            return "GradCF";
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
            Vector<AutoDiffDiff<1, NonZero>> v1(c1->Dimension());
            c1->NonZeroPattern(ud, v1);

            values = AutoDiffDiff<1, NonZero>(true);
            for (size_t d = 0; d < D; d++)
                for (size_t i = 0; i < c1->Dimension(); i++)
                    values[d * c1->Dimension() + i] = v1[i];
        }

        shared_ptr<CoefficientFunction>
        Transform(CoefficientFunction::T_Transform &transformation) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (transformation.cache.count(thisptr))
                return transformation.cache[thisptr];
            if (transformation.replace.count(thisptr))
                return transformation.replace[thisptr];
            auto newcf = make_shared<GradCoefficientFunction<D>>(c1->Transform(transformation));
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        using T_CoefficientFunction<GradCoefficientFunction>::Evaluate;

        virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const override
        {
            throw Exception("GradCF:: scalar evaluate called");
        }

        template <typename T, ORDERING ORD, typename std::enable_if<std::is_convertible<T, double>::value, int>::type = 0>
        void T_Evaluate_impl(const BaseMappedIntegrationRule &bmir, BareSliceMatrix<T, ORD> values) const
        {
            if (bmir.DimSpace() != D || bmir.DimElement() != D)
            {
                throw Exception(ToString("GradCF :: T_Evaluate_impl: D = ") + ToString(D) + ", bmir.DimSpace() = " + ToString(bmir.DimSpace()) + ", bmir.DimElement() = " + ToString(bmir.DimElement()));
            }
            LocalHeapMem<10000> lh("GradCF-lh");

            auto &mir = static_cast<const MappedIntegrationRule<D, D> &>(bmir);
            auto &ir = mir.IR();
            int hd = c1->Dimension();
            STACK_ARRAY(T, hmem, hd * 4);
            STACK_ARRAY(T, hmem2, hd * D);
            STACK_ARRAY(T, hmem3, hd * D);
            FlatMatrix<T, ORD> values_c1(hd, 4, &hmem[0]);
            FlatMatrix<T, ORD> dshape_ref(hd, D, &hmem2[0]);
            FlatMatrix<T, ORD> dshape(hd, D, &hmem3[0]);

            for (size_t i = 0; i < mir.Size(); i++)
            {
                const IntegrationPoint &ip = ir[i];
                const ElementTransformation &eltrans = mir[i].GetTransformation();

                for (int j = 0; j < D; j++) // d / d t_j
                {
                    HeapReset hr(lh);
                    IntegrationPoint ipts[4];
                    ipts[0] = ip;
                    ipts[0](j) -= eps();
                    ipts[1] = ip;
                    ipts[1](j) += eps();
                    ipts[2] = ip;
                    ipts[2](j) -= 2 * eps();
                    ipts[3] = ip;
                    ipts[3](j) += 2 * eps();

                    IntegrationRule ir_j(4, ipts);
                    MappedIntegrationRule<D, D, T> mir_j(ir_j, eltrans, lh);

                    c1->Evaluate(mir_j, values_c1);

                    dshape_ref.Col(j) = (1.0 / (12.0 * eps())) * (8.0 * values_c1.Col(1) - 8.0 * values_c1.Col(0) - values_c1.Col(3) + values_c1.Col(2));
                }

                dshape = dshape_ref * mir[i].GetJacobianInverse();
                values.Col(i) = dshape.AsVector();
            }
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &bmir, BareSliceMatrix<T, ORD> values) const
        {
            if constexpr (is_same<MIR, SIMD_BaseMappedIntegrationRule>::value)
                throw ExceptionNOSIMD("no simd in GradCF");
            if constexpr (std::is_same<T, double>::value)
                T_Evaluate_impl(bmir, values);
            else
                throw Exception("GradCF::T_Evaluate only for double!");
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> input,
                        BareSliceMatrix<T, ORD> values) const
        {
            this->T_Evaluate(ir, values);
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return GradCF(c1->Diff(var, dir), D);
        }

        // shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        // {
        //     auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
        //     if (cache.find(thisptr) != cache.end())
        //         return cache[thisptr];

        //     if (this == var)
        //         return IdentityCF(this->Dimensions());

        //     auto res = c1->DiffJacobi(var, cache);
        //     cache[thisptr] = res;
        //     return res;
        // }

        // virtual bool IsZeroCF() const override { return c1->IsZeroCF(); }
    };

    template <int D>
    class GradDiffOp : public DifferentialOperator
    {
        shared_ptr<CoefficientFunction> func;
        ProxyFunction *proxy;
        bool testfunction;

    public:
        static constexpr double eps() { return 1e-4; }
        GradDiffOp(shared_ptr<CoefficientFunction> afunc, bool atestfunction);

        void
        CalcMatrix(const FiniteElement &inner_fel,
                   const BaseMappedIntegrationRule &bmir,
                   BareSliceMatrix<double, ColMajor> mat,
                   LocalHeap &lh) const override;

        bool IsNonlinear() const override
        {
            return false;
        }
    };

    class GradProxy : public ProxyFunction
    {
    protected:
        shared_ptr<CoefficientFunction> func;
        bool testfunction;
        int dim;

    public:
        GradProxy(shared_ptr<CoefficientFunction> afunc, bool atestfunction, int adim, shared_ptr<DifferentialOperator> adiffop);

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var, shared_ptr<CoefficientFunction> dir) const override;
    };
}

#include <python_ngstd.hpp>
void ExportGradCF(py::module m);

#endif // COEFFICIENT_GRAD
