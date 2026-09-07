#ifndef KFORMS_HPP
#define KFORMS_HPP

#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"

namespace ngfem
{
    class RiemannianManifold;

    /**
     * Fully covariant rank-k tensor with an ambient-space dimension.
     *
     * This class records form semantics; it does not antisymmetrize arbitrary
     * input components. Apply AlternationCF explicitly when input alternation is required.
     */
    class KFormCoefficientFunction : public TensorFieldCoefficientFunction
    {
        uint8_t degree;
        uint8_t dim;

    protected:
        /// Preserve degree and ambient dimension after value-graph operations.
        shared_ptr<TensorFieldCoefficientFunction>
        Rewrap(shared_ptr<CoefficientFunction> cf) const override;

    public:
        KFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, int ak, int adim);

        uint8_t Degree() const { return degree; }
        uint8_t DimensionOfSpace() const { return dim; }

        virtual string GetDescription() const override { return "KFormCF"; }

        void CalcEquivalenceKey() override
        {
            equivalence_key =
                GetDescription() + "[k=" + ToString(int(degree)) +
                ",dim=" + ToString(int(dim)) + "](" +
                GetFullCoefficient()->EquivalenceKey() + ")";
        }

        auto GetCArgs() const
        {
            return tuple{GetFullCoefficient(), degree, dim};
        }
    };

    /**
     * Tensor interpreted as two form blocks of degrees p and q.
     *
     * Component axes are ordered as the p left slots followed by the q right
     * slots. Construction validates shape and metadata but does not apply
     * alternation to the input.
     */
    class DoubleFormCoefficientFunction : public TensorFieldCoefficientFunction
    {
        uint8_t degree_left;
        uint8_t degree_right;
        uint8_t dim;

    protected:
        /// Preserve both block degrees and ambient dimension after rewrapping.
        shared_ptr<TensorFieldCoefficientFunction>
        Rewrap(shared_ptr<CoefficientFunction> cf) const override;

    public:
        DoubleFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, int ap, int aq, int adim);

        uint8_t LeftDegree() const { return degree_left; }
        uint8_t RightDegree() const { return degree_right; }
        uint8_t DimensionOfSpace() const { return dim; }

        virtual string GetDescription() const override { return "DoubleFormCF"; }

        void CalcEquivalenceKey() override
        {
            equivalence_key =
                GetDescription() + "[p=" + ToString(int(degree_left)) +
                ",q=" + ToString(int(degree_right)) +
                ",dim=" + ToString(int(dim)) + "](" +
                GetFullCoefficient()->EquivalenceKey() + ")";
        }

        auto GetCArgs() const
        {
            return tuple{GetFullCoefficient(), degree_left, degree_right, dim};
        }
    };

    class ScalarFieldCoefficientFunction : public KFormCoefficientFunction
    {
    public:
        ScalarFieldCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim);

        virtual string GetDescription() const override
        {
            return "ScalarFieldCF";
        }

        auto GetCArgs() const
        {
            return tuple{GetFullCoefficient(), int(DimensionOfSpace())};
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

        auto GetCArgs() const
        {
            return tuple{GetFullCoefficient(), int(DimensionOfSpace())};
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

        auto GetCArgs() const
        {
            return tuple{GetFullCoefficient(), int(DimensionOfSpace())};
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

        auto GetCArgs() const
        {
            return tuple{GetFullCoefficient(), int(DimensionOfSpace())};
        }
    };

    shared_ptr<KFormCoefficientFunction> KFormCF(shared_ptr<CoefficientFunction> cf, int k, int dim);

    shared_ptr<DoubleFormCoefficientFunction> DoubleFormCF(shared_ptr<CoefficientFunction> cf, int p, int q, int dim);

    shared_ptr<ScalarFieldCoefficientFunction> ScalarFieldCF(shared_ptr<CoefficientFunction> cf, int dim);

    shared_ptr<OneFormCoefficientFunction> OneFormCF(shared_ptr<CoefficientFunction> cf);

    shared_ptr<TwoFormCoefficientFunction> TwoFormCF(shared_ptr<CoefficientFunction> cf, int dim = -1);

    shared_ptr<ThreeFormCoefficientFunction> ThreeFormCF(shared_ptr<CoefficientFunction> cf, int dim = -1);

    shared_ptr<KFormCoefficientFunction> HodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb = VOL);
    shared_ptr<KFormCoefficientFunction> InverseHodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb = VOL);
    shared_ptr<DoubleFormCoefficientFunction> HodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb = VOL, int slot = -1);
    shared_ptr<DoubleFormCoefficientFunction> InverseHodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb = VOL, int slot = -1);
    shared_ptr<ScalarFieldCoefficientFunction> SlotInnerProduct(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb = VOL, bool forms = true);
    shared_ptr<DoubleFormCoefficientFunction> SwapDoubleFormSlots(shared_ptr<DoubleFormCoefficientFunction> a);
    shared_ptr<CoefficientFunction> BlockAlternationByPermutationCF(shared_ptr<CoefficientFunction> T, int rank_total, int block_start, int block_len);

    /// Unnormalized alternation; rank-zero and rank-one inputs retain their shapes.
    shared_ptr<CoefficientFunction> AlternationCF(shared_ptr<CoefficientFunction> T, int rank, int dim);

    shared_ptr<KFormCoefficientFunction> Wedge(shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b);
    shared_ptr<DoubleFormCoefficientFunction> Wedge(shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b);

    /// Exterior derivative with derivative-first GradCF; nontrivial derivatives require dim <= 3.
    shared_ptr<KFormCoefficientFunction> ExteriorDerivative(shared_ptr<KFormCoefficientFunction> a);

    shared_ptr<KFormCoefficientFunction> ZeroKForm(int k, int dim);
    shared_ptr<DoubleFormCoefficientFunction> ZeroDoubleForm(int p, int q, int dim);

}

#include <python_ngstd.hpp>
void ExportKForms(py::module m);

#endif
