#ifndef TENSOR_FIELDS
#define TENSOR_FIELDS

#include <fem.hpp>
#include <coefficient.hpp>

namespace ngfem
{

    class VectorFieldCoefficientFunction : public T_CoefficientFunction<VectorFieldCoefficientFunction>
    {
        shared_ptr<CoefficientFunction> c1;

    public:
        VectorFieldCoefficientFunction(shared_ptr<CoefficientFunction> ac1)
            : T_CoefficientFunction<VectorFieldCoefficientFunction>(1, ac1->IsComplex()), c1(ac1)
        {
        }
        virtual ~VectorFieldCoefficientFunction();

        virtual string GetDescription() const override
        {
            return "VectorField";
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
            auto newcf = make_shared<VectorFieldCoefficientFunction>(c1->Transform(transformation));
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return VectorFieldCF(c1->Diff(var, dir));
        }
        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto thisptr = const_pointer_cast<CoefficientFunction>(this->shared_from_this());
            if (cache.find(thisptr) != cache.end())
                return cache[thisptr];

            if (this == var)
                return IdentityCF(this->Dimensions());

            cache[thisptr] = c1->DiffJacobi(var, cache);
            return res;
        }
    };
}

#include <python_ngstd.hpp>
void ExportTensorFields(py::module m);

#endif // TENSOR_FIELDS
