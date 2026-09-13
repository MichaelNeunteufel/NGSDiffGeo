// Implementation fragment included only by ../kforms.cpp.
// Construction and recognition of compact form facades.

namespace kforms_internal
{
    shared_ptr<KFormCoefficientFunction> KFormFromIndependentCF(
        shared_ptr<CoefficientFunction> independent_value,
        int degree, int dim)
    {
        independent_value = RequireCompactNonNull(
            std::move(independent_value));
        auto facade = make_shared<AlternatingFormCoefficientFunction>(
            std::move(independent_value), degree, dim);
        return KFormCF(facade, degree, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> DoubleFormFromIndependentCF(
        shared_ptr<CoefficientFunction> independent_value,
        int left_degree, int right_degree, int dim)
    {
        independent_value = RequireCompactNonNull(
            std::move(independent_value));
        auto facade =
            make_shared<AlternatingDoubleFormCoefficientFunction>(
                std::move(independent_value), left_degree,
                right_degree, dim);
        return DoubleFormCF(facade, left_degree, right_degree, dim);
    }

    std::optional<CompactKFormView> TryGetCompactView(
        const KFormCoefficientFunction &form)
    {
        auto full = form.GetFullCoefficient();
        if (auto facade = GetAlternatingForm(full))
            return CompactKFormView{facade->IndependentValue(),
                                    facade->Degree(),
                                    facade->DimensionOfSpace()};
        if (auto operation = GetCompactKFormOperation(full))
            return CompactKFormView{operation->IndependentValue(),
                                    int(form.Degree()),
                                    int(form.DimensionOfSpace())};
        return std::nullopt;
    }

    std::optional<CompactDoubleFormView> TryGetCompactView(
        const DoubleFormCoefficientFunction &form)
    {
        if (auto facade =
                GetAlternatingDoubleForm(form.GetFullCoefficient()))
            return CompactDoubleFormView{
                facade->IndependentValue(), facade->LeftDegree(),
                facade->RightDegree(), facade->DimensionOfSpace()};
        if (auto operation =
                GetCompactDoubleFormOperation(form.GetFullCoefficient()))
            return CompactDoubleFormView{
                operation->IndependentValue(), int(form.LeftDegree()),
                int(form.RightDegree()), int(form.DimensionOfSpace())};
        return std::nullopt;
    }

} // namespace kforms_internal
