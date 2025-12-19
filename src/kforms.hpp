#ifndef KFORMS_HPP
#define KFORMS_HPP

#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"

namespace ngfem
{
    class RiemannianManifold;

    class KFormCoefficientFunction : public TensorFieldCoefficientFunction
    {
        uint8_t degree;
        uint8_t dim;

    public:
        KFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, uint8_t ak, uint8_t adim);

        uint8_t Degree() const { return degree; }
        uint8_t DimensionOfSpace() const { return dim; }

        virtual string GetDescription() const override { return "KFormCF"; }

        shared_ptr<CoefficientFunction>
        Transform(CoefficientFunction::T_Transform &transformation) const override;

        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override;
        shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var, T_DJC &cache) const override;
    };

    class ScalarFieldCoefficientFunction : public KFormCoefficientFunction
    {
    public:
        ScalarFieldCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim);

        virtual string GetDescription() const override
        {
            return "ScalarFieldCF";
        }
    };

    class OneFormCoefficientFunction : public KFormCoefficientFunction
    {
    public:
        OneFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim);

        virtual string GetDescription() const override
        {
            return "OneFormCF";
        }
    };

    class TwoFormCoefficientFunction : public KFormCoefficientFunction
    {
    public:
        TwoFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim);

        virtual string GetDescription() const override
        {
            return "TwoFormCF";
        }
    };

    class ThreeFormCoefficientFunction : public KFormCoefficientFunction
    {
    public:
        ThreeFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim);

        virtual string GetDescription() const override
        {
            return "ThreeFormCF";
        }
    };

    shared_ptr<KFormCoefficientFunction> KFormCF(shared_ptr<CoefficientFunction> cf, int k, int dim);

    shared_ptr<ScalarFieldCoefficientFunction> ScalarFieldCF(shared_ptr<CoefficientFunction> cf, int dim = -1);

    shared_ptr<OneFormCoefficientFunction> OneFormCF(shared_ptr<CoefficientFunction> cf);

    shared_ptr<TwoFormCoefficientFunction> TwoFormCF(shared_ptr<CoefficientFunction> cf, int dim = -1);

    shared_ptr<ThreeFormCoefficientFunction> ThreeFormCF(shared_ptr<CoefficientFunction> cf, int dim = -1);

    shared_ptr<KFormCoefficientFunction> HodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb = VOL);

    shared_ptr<CoefficientFunction> AlternationCF(shared_ptr<CoefficientFunction> T, int rank, int dim);

    shared_ptr<KFormCoefficientFunction> Wedge(shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b);

    shared_ptr<KFormCoefficientFunction> ExteriorDerivative(shared_ptr<KFormCoefficientFunction> a);

    shared_ptr<KFormCoefficientFunction> ZeroKForm(int k, int dim);

}

#include <python_ngstd.hpp>
void ExportKForms(py::module m);

#endif
