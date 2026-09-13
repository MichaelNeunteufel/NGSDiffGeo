// Implementation fragment included only by ../kforms.cpp.
// Compact and dense exterior-derivative factories.

namespace kforms_internal
{
    // Exterior-derivative factories and the exact dense fallback.
    shared_ptr<CoefficientFunction> CompactExteriorIndependentValue(
        const CompactKFormView &input)
    {
        auto gradient = GradCF(input.independent, input.dim);
        auto table = GetCompactExteriorDerivativeTable(
            input.dim, input.degree);
        return make_shared<
            CompactExteriorDerivativeIndependentCoefficientFunction>(
            std::move(gradient), input.degree, input.dim,
            std::move(table));
    }

    shared_ptr<KFormCoefficientFunction>
    ExteriorDerivativeCompactKFormWithView(
        shared_ptr<KFormCoefficientFunction> form,
        CompactKFormView input)
    {
        const int dim = input.dim;
        const int output_degree = input.degree + 1;
        auto independent = IsConstantZero(input.independent)
                               ? ZeroCF(Array<int>{int(
                                     GetFormBasis(dim, output_degree)
                                         ->IndependentSize())})
                               : CompactExteriorIndependentValue(input);
        auto facade = make_shared<CompactKFormOperationCoefficientFunction>(
            int(CompactKFormOperation::ExteriorDerivative),
            Array<shared_ptr<CoefficientFunction>>{form},
            std::move(independent), output_degree, dim);
        return KFormCF(facade, output_degree, dim);
    }

    shared_ptr<KFormCoefficientFunction> ExteriorDerivativeDenseKForm(
        shared_ptr<KFormCoefficientFunction> form)
    {
        form = RequireCompactNonNull(std::move(form));
        const int dim = form->DimensionOfSpace();
        const int degree = form->Degree();
        if (degree + 1 > dim)
            return ZeroKForm(degree + 1, dim);
        auto gradient = GradCF(form, dim);
        auto alternating = AlternationCF(gradient, degree + 1, dim);
        auto scaled = ScaleCoefficientCF(
            alternating, ConstantCF(1.0 / double(Factorial(degree))));
        return KFormCF(scaled, degree + 1, dim);
    }

    size_t CompactExteriorDerivativeTableStorageBytesLowerBound(
        int dim, int degree)
    {
        return GetCompactExteriorDerivativeTable(dim, degree)
            ->OwnedStorageBytesLowerBound();
    }

} // namespace kforms_internal
