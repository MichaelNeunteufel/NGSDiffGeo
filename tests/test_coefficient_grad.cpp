#include "coefficient_grad.hpp"

#include <elementtransformation.hpp>

#include <functional>
#include <memory>
#include <stdexcept>

namespace
{
    class DummyElementTransformation : public ngfem::ElementTransformation
    {
    public:
        DummyElementTransformation()
            : ngfem::ElementTransformation(
                  ngfem::ET_POINT, ngfem::VOL, 0, 0)
        {
        }

        void CalcJacobian(
            const ngfem::IntegrationPoint &,
            ngbla::FlatMatrix<double>) const override
        {
        }

        void CalcPoint(
            const ngfem::IntegrationPoint &,
            ngbla::FlatVector<double>) const override
        {
        }

        void CalcPointJacobian(
            const ngfem::IntegrationPoint &,
            ngbla::FlatVector<double>,
            ngbla::FlatMatrix<double>) const override
        {
        }

        void CalcMultiPointJacobian(
            const ngfem::IntegrationRule &,
            ngfem::BaseMappedIntegrationRule &) const override
        {
        }

        int SpaceDim() const override { return 0; }
        ngfem::VorB VB() const override { return ngfem::VOL; }

        ngfem::BaseMappedIntegrationPoint &operator()(
            const ngfem::IntegrationPoint &,
            ngcore::Allocator &) const override
        {
            throw ngstd::Exception("dummy transformation cannot map points");
        }

        ngfem::BaseMappedIntegrationRule &operator()(
            const ngfem::IntegrationRule &,
            ngcore::Allocator &) const override
        {
            throw ngstd::Exception("dummy transformation cannot map rules");
        }
    };

    class RestrictedConstantCoefficientFunction
        : public ngfem::CoefficientFunction
    {
    public:
        RestrictedConstantCoefficientFunction()
            : ngfem::CoefficientFunction(1)
        {
            elementwise_constant = true;
        }

        double Evaluate(
            const ngfem::BaseMappedIntegrationPoint &) const override
        {
            return 1.0;
        }

        bool DefinedOn(
            const ngfem::ElementTransformation &) override
        {
            return false;
        }

        void CalcEquivalenceKey() override
        {
            equivalence_key = "restricted-constant";
        }
    };

    void Require(bool condition)
    {
        if (!condition)
            throw std::runtime_error(
                "coefficient-gradient unit-test assertion failed");
    }

    void ExpectException(const std::function<void()> &function)
    {
        bool threw = false;
        try
        {
            function();
        }
        catch (const ngstd::Exception &)
        {
            threw = true;
        }
        Require(threw);
    }

    void TestSIMDVolumeGradient()
    {
        ngcore::LocalHeapMem<100000> heap("gradcf-simd-test");
        ngfem::FE_ElementTransformation<2, 2> transformation(
            ngfem::ET_TRIG);
        ngfem::SIMD_IntegrationRule rule(ngfem::ET_TRIG, 4);
        ngfem::SIMD_MappedIntegrationRule<2, 2> mapped_rule(
            rule,
            transformation,
            heap);

        const auto coordinate =
            ngfem::MakeCoordinateCoefficientFunction(0);
        const auto gradient = ngfem::GradCF(coordinate, 2);
        ngbla::FlatMatrix<ngcore::SIMD<double>> values(
            2,
            mapped_rule.Size(),
            heap);
        gradient->Evaluate(mapped_rule, values);

        for (size_t i = 0; i < mapped_rule.Size(); ++i)
        {
            const auto error_x =
                values(0, i) - ngcore::SIMD<double>(1.0);
            const auto error_y = values(1, i);
            Require(ngcore::HSum(error_x * error_x) < 1e-20);
            Require(ngcore::HSum(error_y * error_y) < 1e-20);
        }

        const auto scalar_only =
            std::make_shared<RestrictedConstantCoefficientFunction>();
        const auto fallback_gradient = ngfem::GradCF(scalar_only, 2);
        ExpectException([&]() {
            fallback_gradient->Evaluate(mapped_rule, values);
        });
    }
}

int main()
{
    TestSIMDVolumeGradient();

    DummyElementTransformation transformation;
    const auto restricted =
        std::make_shared<RestrictedConstantCoefficientFunction>();
    const auto first = ngfem::GradCF(restricted, 1);
    const auto second = ngfem::GradCF(restricted, 1);

    Require(!first->DefinedOn(transformation));
    Require(first->ElementwiseConstant());
    Require(first->EquivalenceKey() == second->EquivalenceKey());

    ExpectException([]() {
        ngfem::GradCF(nullptr, 2);
    });
    ngcore::Array<int> dimensions = {3};
    const auto vector_zero = ngfem::GradCF(ngfem::ZeroCF(dimensions), 2);
    Require(vector_zero->Dimensions().Size() == 2);
    Require(vector_zero->Dimensions()[0] == 2);
    Require(vector_zero->Dimensions()[1] == 3);
    ExpectException([]() {
        std::make_shared<ngfem::GradCoefficientFunction<2>>(
            nullptr, false);
    });
}
