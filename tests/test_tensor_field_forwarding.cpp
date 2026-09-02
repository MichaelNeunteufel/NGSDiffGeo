#include "tensor_fields.hpp"

#include <elementtransformation.hpp>
#include <symbolicintegrator.hpp>

#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
  class DummyElementTransformation : public ngfem::ElementTransformation
  {
  public:
    DummyElementTransformation()
        : ngfem::ElementTransformation(ngfem::ET_POINT, ngfem::VOL, 0, 0)
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

  class ThrowingPatternCoefficientFunction
      : public ngfem::CoefficientFunction
  {
  public:
    ThrowingPatternCoefficientFunction()
        : ngfem::CoefficientFunction(1)
    {
    }

    double Evaluate(const ngfem::BaseMappedIntegrationPoint &) const override
    {
      return 1.0;
    }

    bool DefinedOn(const ngfem::ElementTransformation &) override
    {
      return false;
    }

    void NonZeroPattern(
        const ngfem::ProxyUserData &,
        ngbla::FlatVector<ngcore::AutoDiffDiff<1, ngfem::NonZero>>) const override
    {
      throw ngstd::Exception("nonzero-pattern probe");
    }
  };

  class DerivativeProbeCoefficientFunction
      : public ngfem::CoefficientFunction
  {
  public:
    mutable bool derivative_called = false;

    DerivativeProbeCoefficientFunction()
        : ngfem::CoefficientFunction(1)
    {
    }

    double Evaluate(const ngfem::BaseMappedIntegrationPoint &) const override
    {
      return 0.0;
    }

    void EvaluateDeriv(
        const ngfem::BaseMappedIntegrationRule &,
        ngbla::FlatMatrix<ngcore::Complex> values,
        ngbla::FlatMatrix<ngcore::Complex> deriv) const override
    {
      derivative_called = true;
      values = ngcore::Complex(2.0);
      deriv = ngcore::Complex(7.0);
    }
  };

  class ShapedProbeCoefficientFunction
      : public ngfem::CoefficientFunction
  {
    bool is_zero;

  public:
    ShapedProbeCoefficientFunction(std::initializer_list<int> dimensions,
                                   bool ais_zero = false)
        : ngfem::CoefficientFunction(1), is_zero(ais_zero)
    {
      ngstd::Array<int> shape(dimensions.size());
      size_t i = 0;
      for (int dimension : dimensions)
        shape[i++] = dimension;
      SetDimensions(shape);
    }

    double Evaluate(const ngfem::BaseMappedIntegrationPoint &) const override
    {
      return 0.0;
    }

    bool IsZeroCF() const override { return is_zero; }
  };

  void Require(bool condition)
  {
    if (!condition)
      throw std::runtime_error("tensor-field forwarding unit-test assertion failed");
  }

  void ExpectException(const std::function<void()> &func)
  {
    bool threw = false;
    try
    {
      func();
    }
    catch (const ngstd::Exception &)
    {
      threw = true;
    }
    Require(threw);
  }
}

int main()
{
  const auto constant =
      std::make_shared<ngfem::ConstantCoefficientFunction>(3.5);
  const auto scalar_tensor = ngfem::TensorFieldCF(constant, "");
  Require(scalar_tensor->ElementwiseConstant());
  Require(scalar_tensor->EvaluateConst() == 3.5);
  Require(scalar_tensor->GetCovariantIndices().empty());
  Require(scalar_tensor->GetFullCoefficient() == constant);
  Require(scalar_tensor->GetCoefficients() ==
          scalar_tensor->GetFullCoefficient());

  const auto inputs = scalar_tensor->InputCoefficientFunctions();
  Require(inputs.Size() == 1);
  Require(inputs[0] == constant);

  std::vector<ngfem::CoefficientFunction *> visited;
  scalar_tensor->TraverseTree(
      [&](ngfem::CoefficientFunction &node) { visited.push_back(&node); });
  Require(!visited.empty());
  Require(visited.back() == scalar_tensor.get());

  const auto same_scalar_tensor = ngfem::TensorFieldCF(scalar_tensor, "");
  Require(same_scalar_tensor == scalar_tensor);

  ExpectException([&]() { ngfem::TensorFieldCF(constant, "0"); });
  ExpectException([]() {
    ngfem::TensorFieldCF(
        std::shared_ptr<ngfem::CoefficientFunction>(), "");
  });

  const auto vector_values =
      std::make_shared<ShapedProbeCoefficientFunction>(
          std::initializer_list<int>{2});
  const auto vector = ngfem::VectorFieldCF(vector_values);
  Require(ngfem::IsVectorField(*vector));
  Require(!ngfem::IsOneForm(*vector));
  Require(ngfem::VectorFieldCF(vector) == vector);

  const auto one_form = ngfem::TensorFieldCF(vector, "1");
  Require(ngfem::IsOneForm(*one_form));
  Require(!ngfem::IsVectorField(*one_form));
  Require(one_form->GetFullCoefficient() == vector_values);

  const auto nonsquare_values =
      std::make_shared<ShapedProbeCoefficientFunction>(
          std::initializer_list<int>{2, 3});
  ExpectException([&]() { ngfem::TensorFieldCF(nonsquare_values, "00"); });

  const auto zero_values =
      std::make_shared<ShapedProbeCoefficientFunction>(
          std::initializer_list<int>{2}, true);
  Require(ngfem::TensorFieldCF(zero_values, "0")->IsZeroCF());

  const auto throwing_tensor = ngfem::TensorFieldCF(
      std::make_shared<ThrowingPatternCoefficientFunction>(), "");
  DummyElementTransformation transformation;
  Require(!throwing_tensor->DefinedOn(transformation));

  ngfem::ProxyUserData user_data;
  ngcore::AutoDiffDiff<1, ngfem::NonZero> nonzero;
  ngbla::FlatVector<ngcore::AutoDiffDiff<1, ngfem::NonZero>> values(1, &nonzero);
  ExpectException([&]() {
    throwing_tensor->NonZeroPattern(user_data, values);
  });

  using Pattern = ngcore::AutoDiffDiff<1, ngfem::NonZero>;
  Pattern input_pattern(ngfem::NonZero(true), 0);
  Pattern output_pattern(ngfem::NonZero(false));
  ngbla::FlatVector<Pattern> input_values(1, &input_pattern);
  ngbla::FlatVector<Pattern> output_values(1, &output_pattern);
  ngbla::FlatVector<Pattern> input_storage[] = {input_values};
  ngstd::FlatArray<ngbla::FlatVector<Pattern>> input(1, input_storage);
  scalar_tensor->NonZeroPattern(user_data, input, output_values);
  Require(bool(output_pattern.Value()));
  Require(bool(output_pattern.DValue(0)));

  const auto derivative_probe =
      std::make_shared<DerivativeProbeCoefficientFunction>();
  const auto derivative_tensor =
      ngfem::TensorFieldCF(derivative_probe, "");
  ngcore::LocalHeap heap(1024, "tensor-field forwarding test");
  ngfem::IntegrationRule integration_rule(ngfem::ET_POINT, 0);
  ngfem::MappedIntegrationRule<0, 0> mapped_rule(
      integration_rule, transformation, 0, heap);
  ngbla::Matrix<ngcore::Complex> derivative_values(1, mapped_rule.Size());
  ngbla::Matrix<ngcore::Complex> derivatives(1, mapped_rule.Size());
  derivative_tensor->EvaluateDeriv(
      mapped_rule, derivative_values, derivatives);
  Require(derivative_probe->derivative_called);
  Require(derivative_values(0, 0) == ngcore::Complex(2.0));
  Require(derivatives(0, 0) == ngcore::Complex(7.0));
}
