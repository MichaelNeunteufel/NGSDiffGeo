// Implementation fragment included only by ../kforms.cpp.
// Compact and dense algebra factories, arithmetic, and slot operations.

namespace kforms_internal
{
    // Representation-specific wedge and slot-operation factories.
    shared_ptr<CoefficientFunction> MakeCompactDoubleWedgeFullEvaluator(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right)
    {
        left = RequireCompactNonNull(std::move(left));
        right = RequireCompactNonNull(std::move(right));
        auto left_view = TryGetProvenCompactInput(left);
        auto right_view = TryGetProvenCompactInput(right);
        if (!left_view || !right_view)
            throw Exception(
                "MakeCompactDoubleWedgeFullEvaluator: operands are not compact");
        return make_shared<CompactDoubleWedgeFullCoefficientFunction>(
            left_view->independent, right_view->independent,
            left_view->left_degree, left_view->right_degree,
            right_view->left_degree, right_view->right_degree,
            left_view->dim);
    }

    shared_ptr<DoubleFormCoefficientFunction>
    WedgeCompactDoubleFormsWithViews(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right,
        CompactDoubleFormView left_view,
        CompactDoubleFormView right_view)
    {
        const int dim = left_view.dim;
        const int output_left =
            left_view.left_degree + right_view.left_degree;
        const int output_right =
            left_view.right_degree + right_view.right_degree;
        shared_ptr<CoefficientFunction> independent;
        shared_ptr<CoefficientFunction> evaluator;
        if (IsConstantZero(left_view.independent) ||
            IsConstantZero(right_view.independent))
        {
            auto left_basis = GetFormBasis(dim, output_left);
            auto right_basis = GetFormBasis(dim, output_right);
            independent = ZeroCF(Array<int>{int(
                left_basis->IndependentSize() *
                right_basis->IndependentSize())});
            evaluator = make_shared<
                AlternatingDoubleFormCoefficientFunction>(
                independent, output_left, output_right, dim);
        }
        else
        {
            auto table = GetCompactWedgeTable(
                dim, left_view.left_degree, left_view.right_degree,
                right_view.left_degree, right_view.right_degree);
            independent = make_shared<
                CompactWedgeIndependentCoefficientFunction>(
                left_view.independent, right_view.independent,
                left_view.left_degree, left_view.right_degree,
                right_view.left_degree, right_view.right_degree, dim,
                table);
            evaluator = make_shared<
                CompactDoubleWedgeFullCoefficientFunction>(
                left_view.independent, right_view.independent,
                left_view.left_degree, left_view.right_degree,
                right_view.left_degree, right_view.right_degree, dim,
                std::move(table));
        }
        auto facade =
            make_shared<CompactDoubleFormOperationCoefficientFunction>(
                int(CompactDoubleFormOperation::Wedge),
                Array<shared_ptr<CoefficientFunction>>{left, right},
                std::move(independent), output_left, output_right, dim,
                std::move(evaluator));
        return DoubleFormCF(facade, output_left, output_right, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> WedgeDenseDoubleForms(
        shared_ptr<DoubleFormCoefficientFunction> left,
        shared_ptr<DoubleFormCoefficientFunction> right)
    {
        left = RequireCompactNonNull(std::move(left));
        right = RequireCompactNonNull(std::move(right));
        if (left->DimensionOfSpace() != right->DimensionOfSpace())
            throw Exception("WedgeDenseDoubleForms: dimension mismatch");
        const int dim = left->DimensionOfSpace();
        const int output_left =
            left->LeftDegree() + right->LeftDegree();
        const int output_right =
            left->RightDegree() + right->RightDegree();
        if (output_left > dim || output_right > dim)
            return ZeroDoubleForm(output_left, output_right, dim);
        auto wedge = make_shared<DoubleFormWedgeCoefficientFunction>(
            left, right);
        return DoubleFormCF(wedge, output_left, output_right, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> SwapCompactDoubleFormSlots(
        shared_ptr<DoubleFormCoefficientFunction> form)
    {
        form = RequireCompactNonNull(std::move(form));
        auto view = TryGetProvenCompactInput(form);
        if (!view)
            throw Exception(
                "SwapCompactDoubleFormSlots: input is not compact");
        auto left_basis = GetFormBasis(
            view->dim, view->left_degree);
        auto right_basis = GetFormBasis(
            view->dim, view->right_degree);
        Array<shared_ptr<CoefficientFunction>> components;
        for (size_t output_left = 0;
             output_left < right_basis->IndependentSize();
             ++output_left)
            for (size_t output_right = 0;
                 output_right < left_basis->IndependentSize();
                 ++output_right)
            {
                const int input_index =
                    int(output_right * right_basis->IndependentSize() +
                        output_left);
                components.Append(MakeComponentCoefficientFunction(
                    view->independent, input_index));
            }
        auto independent =
            MakeVectorialCoefficientFunction(std::move(components));
        auto facade =
            make_shared<CompactDoubleFormOperationCoefficientFunction>(
                int(CompactDoubleFormOperation::Swap),
                Array<shared_ptr<CoefficientFunction>>{form},
                std::move(independent), form->RightDegree(),
                form->LeftDegree(), form->DimensionOfSpace());
        return DoubleFormCF(facade, form->RightDegree(),
                            form->LeftDegree(),
                            form->DimensionOfSpace());
    }

    shared_ptr<DoubleFormCoefficientFunction> SwapDenseDoubleFormSlots(
        shared_ptr<DoubleFormCoefficientFunction> form)
    {
        form = RequireCompactNonNull(std::move(form));
        const int left_degree = form->LeftDegree();
        const int right_degree = form->RightDegree();
        std::vector<int> order;
        order.reserve(size_t(left_degree + right_degree));
        for (int i = 0; i < right_degree; ++i)
            order.push_back(left_degree + i);
        for (int i = 0; i < left_degree; ++i)
            order.push_back(i);
        auto reordered = PermuteTensorCF(form, order);
        return DoubleFormCF(reordered, right_degree, left_degree,
                            form->DimensionOfSpace());
    }

    shared_ptr<CoefficientFunction> CompactWedgeIndependentValue(
        const CompactKFormView &left, const CompactKFormView &right)
    {
        auto table = GetCompactWedgeTable(
            left.dim, left.degree, 0, right.degree, 0);
        return make_shared<CompactWedgeIndependentCoefficientFunction>(
            left.independent, right.independent,
            left.degree, 0, right.degree, 0, left.dim,
            std::move(table));
    }

    shared_ptr<KFormCoefficientFunction> AddKForms(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right,
        bool subtract)
    {
        left = RequireCompactNonNull(std::move(left));
        right = RequireCompactNonNull(std::move(right));
        if (left->Degree() != right->Degree() ||
            left->DimensionOfSpace() != right->DimensionOfSpace())
            throw Exception("AddKForms: form metadata mismatch");
        auto left_view = TryGetProvenCompactInput(left);
        auto right_view = TryGetProvenCompactInput(right);
        if (!left_view || !right_view)
        {
            auto right_value = subtract
                                   ? ScaleCoefficientCF(right, ConstantCF(-1.0))
                                   : static_pointer_cast<CoefficientFunction>(right);
            return KFormCF(SymbolicSumCF(left, right_value),
                           left->Degree(), left->DimensionOfSpace());
        }

        const bool left_zero = IsConstantZero(left_view->independent);
        const bool right_zero = IsConstantZero(right_view->independent);
        shared_ptr<CoefficientFunction> independent;
        if (left_zero && right_zero)
            independent = ZeroCF(left_view->independent->Dimensions());
        else if (left_zero)
            independent = subtract ? -right_view->independent
                                   : right_view->independent;
        else if (right_zero)
            independent = left_view->independent;
        else
            independent = subtract
                              ? left_view->independent -
                                    right_view->independent
                              : left_view->independent +
                                    right_view->independent;
        auto operation = subtract ? CompactKFormOperation::Difference
                                  : CompactKFormOperation::Sum;
        auto facade = make_shared<CompactKFormOperationCoefficientFunction>(
            int(operation),
            Array<shared_ptr<CoefficientFunction>>{left, right},
            std::move(independent), left->Degree(),
            left->DimensionOfSpace());
        return KFormCF(facade, left->Degree(), left->DimensionOfSpace());
    }

    shared_ptr<KFormCoefficientFunction> ScaleKForm(
        shared_ptr<KFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> scalar)
    {
        form = RequireCompactNonNull(std::move(form));
        scalar = RequireCompactNonNull(std::move(scalar));
        if (scalar->Dimensions().Size() != 0)
            throw Exception("ScaleKForm: factor must be scalar");
        auto view = TryGetProvenCompactInput(form);
        if (!view)
            return KFormCF(ScaleCoefficientCF(form, scalar),
                           form->Degree(), form->DimensionOfSpace());
        auto native_scalar = NativeCoefficientValue(scalar);
        auto independent =
            IsConstantZero(view->independent) ||
                    IsProvablyZeroNativeValue(native_scalar)
                ? ZeroCF(view->independent->Dimensions())
                : ScaleCoefficientCF(view->independent, native_scalar);
        auto facade = make_shared<CompactKFormOperationCoefficientFunction>(
            int(CompactKFormOperation::Scale),
            Array<shared_ptr<CoefficientFunction>>{form, scalar},
            std::move(independent), form->Degree(),
            form->DimensionOfSpace());
        return KFormCF(facade, form->Degree(), form->DimensionOfSpace());
    }

    shared_ptr<KFormCoefficientFunction> WedgeCompactKFormsWithViews(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right,
        CompactKFormView left_view,
        CompactKFormView right_view)
    {
        const int output_degree =
            left_view.degree + right_view.degree;
        const int dim = left_view.dim;
        auto independent =
            IsConstantZero(left_view.independent) ||
                    IsConstantZero(right_view.independent)
                ? ZeroCF(Array<int>{int(
                      GetFormBasis(dim, output_degree)
                          ->IndependentSize())})
                : CompactWedgeIndependentValue(left_view, right_view);
        auto facade = make_shared<CompactKFormOperationCoefficientFunction>(
            int(CompactKFormOperation::Wedge),
            Array<shared_ptr<CoefficientFunction>>{left, right},
            std::move(independent), output_degree, dim);
        return KFormCF(facade, output_degree, dim);
    }

    shared_ptr<KFormCoefficientFunction> WedgeDenseKForms(
        shared_ptr<KFormCoefficientFunction> left,
        shared_ptr<KFormCoefficientFunction> right)
    {
        left = RequireCompactNonNull(std::move(left));
        right = RequireCompactNonNull(std::move(right));
        if (left->DimensionOfSpace() != right->DimensionOfSpace())
            throw Exception("WedgeDenseKForms: dimension mismatch");
        const int dim = left->DimensionOfSpace();
        const int left_degree = left->Degree();
        const int right_degree = right->Degree();
        if (left_degree + right_degree > dim)
            return ZeroKForm(left_degree + right_degree, dim);
        if (left_degree == 0 || right_degree == 0)
            return KFormCF(
                ScaleCoefficientCF(left_degree == 0 ? right : left,
                                   left_degree == 0 ? left : right),
                left_degree + right_degree, dim);

        auto product = TensorProduct(left, right);
        const auto &shuffle_data =
            GetSignedWedgeOrders(left_degree, right_degree);
        shared_ptr<CoefficientFunction> result;
        for (size_t permutation = 0;
             permutation < shuffle_data.orders.size(); ++permutation)
        {
            shared_ptr<CoefficientFunction> term = PermuteTensorCF(
                product, shuffle_data.orders[permutation]);
            if (shuffle_data.signs[permutation] < 0)
                term = ScaleCoefficientCF(term, ConstantCF(-1));
            result = result ? SymbolicSumCF(result, term) : term;
        }
        return KFormCF(result, left_degree + right_degree, dim);
    }

} // namespace kforms_internal
