// Implementation fragment included only by ../kforms.cpp.
// Semantic compact-operation nodes and reconstruction rules.

namespace kforms_internal
{
    enum class CompactKFormOperation : int
    {
        Sum = 1,
        Difference = 2,
        Scale = 3,
        Wedge = 4,
        ExteriorDerivative = 5
    };

    class CompactKFormOperationCoefficientFunction
        : public SymbolicExpressionCoefficientFunction
    {
        CompactKFormOperation operation;
        shared_ptr<CoefficientFunction> independent;
        int degree;
        int dim;

        static shared_ptr<CoefficientFunction> MakeEvaluator(
            const shared_ptr<CoefficientFunction> &independent,
            int degree, int dim)
        {
            return make_shared<AlternatingFormCoefficientFunction>(
                independent, degree, dim);
        }

        shared_ptr<CoefficientFunction> Rebuild(
            const Array<shared_ptr<CoefficientFunction>> &inputs) const override
        {
            // Once a semantic operand changes, its replacement is no
            // longer known to be alternating. Preserve every full entry.
            if (operation == CompactKFormOperation::ExteriorDerivative)
            {
                auto original = dynamic_pointer_cast<KFormCoefficientFunction>(
                    operands[0]);
                return ExteriorDerivative(KFormCF(
                    inputs[0], original->Degree(), dim));
            }
            if (operation == CompactKFormOperation::Sum)
                return AddKForms(KFormCF(inputs[0], degree, dim),
                                 KFormCF(inputs[1], degree, dim));
            if (operation == CompactKFormOperation::Difference)
                return AddKForms(KFormCF(inputs[0], degree, dim),
                                 KFormCF(inputs[1], degree, dim), true);
            if (operation == CompactKFormOperation::Scale)
                return ScaleKForm(KFormCF(inputs[0], degree, dim), inputs[1]);

            auto left = dynamic_pointer_cast<KFormCoefficientFunction>(inputs[0]);
            auto right = dynamic_pointer_cast<KFormCoefficientFunction>(inputs[1]);
            if (!left)
                left = KFormCF(inputs[0],
                               dynamic_pointer_cast<KFormCoefficientFunction>(
                                   operands[0])
                                   ->Degree(),
                               dim);
            if (!right)
                right = KFormCF(inputs[1],
                                dynamic_pointer_cast<KFormCoefficientFunction>(
                                    operands[1])
                                    ->Degree(),
                                dim);
            return Wedge(left, right);
        }

    public:
        CompactKFormOperationCoefficientFunction(
            int aoperation,
            const Array<shared_ptr<CoefficientFunction>> &semantic_operands,
            shared_ptr<CoefficientFunction> aindependent,
            int adegree, int adim)
            : SymbolicExpressionCoefficientFunction(
                  semantic_operands,
                  MakeEvaluator(aindependent, adegree, adim)),
              operation(CompactKFormOperation(aoperation)),
              independent(std::move(aindependent)), degree(adegree), dim(adim)
        {
            const size_t expected_operands =
                operation == CompactKFormOperation::ExteriorDerivative
                    ? 1
                    : 2;
            if ((operation != CompactKFormOperation::Sum &&
                 operation != CompactKFormOperation::Difference &&
                 operation != CompactKFormOperation::Scale &&
                 operation != CompactKFormOperation::Wedge &&
                 operation != CompactKFormOperation::ExteriorDerivative) ||
                operands.Size() != expected_operands)
                throw Exception("CompactKFormOperation: invalid operation state");
            if (operation == CompactKFormOperation::ExteriorDerivative)
            {
                auto form = dynamic_pointer_cast<KFormCoefficientFunction>(
                    operands[0]);
                if (!form || form->DimensionOfSpace() != dim ||
                    form->Degree() + 1 != degree)
                    throw Exception(
                        "CompactKFormOperation: exterior derivative operand metadata mismatch");
            }
            else if (operation == CompactKFormOperation::Sum ||
                     operation == CompactKFormOperation::Difference)
            {
                if (operands[0]->Dimensions() != Dimensions() ||
                    operands[1]->Dimensions() != Dimensions())
                    throw Exception("CompactKFormOperation: sum operand shape mismatch");
            }
            else if (operation == CompactKFormOperation::Scale)
            {
                if (operands[0]->Dimensions() != Dimensions() ||
                    operands[1]->Dimensions().Size() != 0)
                    throw Exception("CompactKFormOperation: scale operand shape mismatch");
            }
            else
            {
                auto left = dynamic_pointer_cast<KFormCoefficientFunction>(operands[0]);
                auto right = dynamic_pointer_cast<KFormCoefficientFunction>(operands[1]);
                if (!left || !right ||
                    left->DimensionOfSpace() != dim ||
                    right->DimensionOfSpace() != dim ||
                    left->Degree() + right->Degree() != degree)
                    throw Exception("CompactKFormOperation: wedge operand metadata mismatch");
            }
        }

        auto GetCArgs() const
        {
            return tuple{int(operation),
                         Array<shared_ptr<CoefficientFunction>>(operands),
                         independent, degree, dim};
        }
        string GetDescription() const override
        {
            switch (operation)
            {
            case CompactKFormOperation::Sum:
                return "CompactKFormOperationSum";
            case CompactKFormOperation::Difference:
                return "CompactKFormOperationDifference";
            case CompactKFormOperation::Scale:
                return "CompactKFormOperationScale";
            case CompactKFormOperation::Wedge:
                return "CompactKFormOperationWedge";
            case CompactKFormOperation::ExteriorDerivative:
                return "CompactKFormOperationExteriorDerivative";
            }
            throw Exception("CompactKFormOperation: invalid operation state");
        }
        void CalcEquivalenceKey() override
        {
            SymbolicExpressionCoefficientFunction::CalcEquivalenceKey();
            equivalence_key += "[k=" + ToString(degree) +
                               ",dim=" + ToString(dim) + "]";
        }
        shared_ptr<CoefficientFunction> IndependentValue() const
        {
            return independent;
        }
        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            auto derivative_form = [&](const shared_ptr<CoefficientFunction> &value,
                                       int operand_degree)
            {
                return IsConstantZero(value)
                           ? ZeroKForm(operand_degree, dim)
                           : KFormCF(value, operand_degree, dim);
            };
            if (operation == CompactKFormOperation::ExteriorDerivative)
            {
                auto input = dynamic_pointer_cast<KFormCoefficientFunction>(
                    operands[0]);
                return ExteriorDerivative(derivative_form(
                    input->Diff(var, dir), input->Degree()));
            }
            if (operation == CompactKFormOperation::Sum)
                return AddKForms(
                    derivative_form(operands[0]->Diff(var, dir), degree),
                    derivative_form(operands[1]->Diff(var, dir), degree));
            if (operation == CompactKFormOperation::Difference)
                return AddKForms(
                    derivative_form(operands[0]->Diff(var, dir), degree),
                    derivative_form(operands[1]->Diff(var, dir), degree),
                    true);
            if (operation == CompactKFormOperation::Scale)
            {
                auto form = dynamic_pointer_cast<KFormCoefficientFunction>(
                    operands[0]);
                auto dform = derivative_form(
                    operands[0]->Diff(var, dir), degree);
                auto left_term = ScaleKForm(dform, operands[1]);
                auto right_term = ScaleKForm(
                    form, operands[1]->Diff(var, dir));
                return AddKForms(left_term, right_term);
            }

            auto left = dynamic_pointer_cast<KFormCoefficientFunction>(operands[0]);
            auto right = dynamic_pointer_cast<KFormCoefficientFunction>(operands[1]);
            auto dleft = derivative_form(
                left->Diff(var, dir), left->Degree());
            auto dright = derivative_form(
                right->Diff(var, dir), right->Degree());
            return AddKForms(Wedge(dleft, right), Wedge(left, dright));
        }

        shared_ptr<CoefficientFunction> DiffJacobi(
            const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(
                shared_from_this());
            if (auto found = cache.find(self); found != cache.end())
                return found->second;
            if (this == var)
                return cache[self] = IdentityCF(Dimensions());
            if (var->Dimensions().Size() == 0)
                return cache[self] = Diff(
                           var, make_shared<ConstantCoefficientFunction>(1.0));
            return cache[self] =
                       SymbolicExpressionCoefficientFunction::DiffJacobi(var,
                                                                         cache);
        }
    };

    shared_ptr<DoubleFormCoefficientFunction> HodgeDoubleFormWithMetric(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric,
        shared_ptr<CoefficientFunction> volume, int slot);
    shared_ptr<DoubleFormCoefficientFunction>
    HodgeDenseDoubleFormWithMetric(
        shared_ptr<DoubleFormCoefficientFunction> form,
        shared_ptr<CoefficientFunction> inverse_metric,
        shared_ptr<CoefficientFunction> volume, int slot);

    enum class CompactDoubleFormOperation : int
    {
        Sum = 0,
        Difference = 1,
        Scale = 2,
        Wedge = 3,
        Swap = 4,
        Trace = 5,
        HodgeLeft = 6,
        HodgeRight = 7,
        HodgeBoth = 8
    };

    constexpr bool IsValidCompactDoubleFormOperation(
        CompactDoubleFormOperation operation)
    {
        return operation >= CompactDoubleFormOperation::Sum &&
               operation <= CompactDoubleFormOperation::HodgeBoth;
    }

    constexpr bool IsCompactDoubleFormHodgeOperation(
        CompactDoubleFormOperation operation)
    {
        return operation == CompactDoubleFormOperation::HodgeLeft ||
               operation == CompactDoubleFormOperation::HodgeRight ||
               operation == CompactDoubleFormOperation::HodgeBoth;
    }

    constexpr size_t CompactDoubleFormOperationArity(
        CompactDoubleFormOperation operation)
    {
        return operation == CompactDoubleFormOperation::Swap
                   ? 1
               : IsCompactDoubleFormHodgeOperation(operation) ? 3
                                                              : 2;
    }

    int CompactDoubleFormHodgeSlot(
        CompactDoubleFormOperation operation)
    {
        if (operation == CompactDoubleFormOperation::HodgeLeft)
            return 0;
        if (operation == CompactDoubleFormOperation::HodgeRight)
            return 1;
        if (operation == CompactDoubleFormOperation::HodgeBoth)
            return -1;
        throw Exception(
            "CompactDoubleFormOperation: operation is not a Hodge star");
    }

    const char *CompactDoubleFormOperationDescription(
        CompactDoubleFormOperation operation)
    {
        switch (operation)
        {
        case CompactDoubleFormOperation::Sum:
            return "CompactDoubleFormOperationSum";
        case CompactDoubleFormOperation::Difference:
            return "CompactDoubleFormOperationDifference";
        case CompactDoubleFormOperation::Scale:
            return "CompactDoubleFormOperationScale";
        case CompactDoubleFormOperation::Wedge:
            return "CompactDoubleFormOperationWedge";
        case CompactDoubleFormOperation::Swap:
            return "CompactDoubleFormOperationSwap";
        case CompactDoubleFormOperation::Trace:
            return "CompactDoubleFormOperationTrace";
        case CompactDoubleFormOperation::HodgeLeft:
            return "CompactDoubleFormOperationHodgeLeft";
        case CompactDoubleFormOperation::HodgeRight:
            return "CompactDoubleFormOperationHodgeRight";
        case CompactDoubleFormOperation::HodgeBoth:
            return "CompactDoubleFormOperationHodgeBoth";
        }
        throw Exception(
            "CompactDoubleFormOperation: invalid operation state");
    }

    class CompactDoubleFormOperationCoefficientFunction
        : public SymbolicExpressionCoefficientFunction
    {
        CompactDoubleFormOperation operation;
        shared_ptr<CoefficientFunction> independent;
        int left_degree;
        int right_degree;
        int dim;

        static shared_ptr<CoefficientFunction> MakeEvaluator(
            int aoperation,
            const Array<shared_ptr<CoefficientFunction>> &semantic_operands,
            const shared_ptr<CoefficientFunction> &independent,
            int left_degree, int right_degree, int dim)
        {
            // Fuse the final compact wedge with its full-shaped facade.
            // Downstream form operations consume `independent` instead,
            // so neither path materializes a full intermediate operand.
            const auto operation =
                CompactDoubleFormOperation(aoperation);
            if (operation == CompactDoubleFormOperation::Wedge)
            {
                if (semantic_operands.Size() != 2)
                    throw Exception(
                        "CompactDoubleFormOperation: wedge requires two operands");
                auto left = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(semantic_operands[0]);
                auto right = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(semantic_operands[1]);
                if (!left || !right)
                    throw Exception(
                        "CompactDoubleFormOperation: wedge operands must be double forms");
                return MakeCompactDoubleWedgeFullEvaluator(left, right);
            }
            return make_shared<
                AlternatingDoubleFormCoefficientFunction>(
                independent, left_degree, right_degree, dim);
        }

        shared_ptr<CoefficientFunction> Rebuild(
            const Array<shared_ptr<CoefficientFunction>> &inputs) const override
        {
            if (operation == CompactDoubleFormOperation::Sum)
                return AddDoubleForms(
                    DoubleFormCF(inputs[0], left_degree,
                                 right_degree, dim),
                    DoubleFormCF(inputs[1], left_degree,
                                 right_degree, dim));
            if (operation == CompactDoubleFormOperation::Difference)
                return AddDoubleForms(
                    DoubleFormCF(inputs[0], left_degree,
                                 right_degree, dim),
                    DoubleFormCF(inputs[1], left_degree,
                                 right_degree, dim),
                    true);
            if (operation == CompactDoubleFormOperation::Scale)
                return ScaleDoubleForm(
                    DoubleFormCF(inputs[0], left_degree,
                                 right_degree, dim),
                    inputs[1]);
            if (operation == CompactDoubleFormOperation::Swap)
            {
                auto original = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                return SwapDoubleFormSlots(DoubleFormCF(
                    inputs[0], original->LeftDegree(),
                    original->RightDegree(), dim));
            }
            if (operation == CompactDoubleFormOperation::Trace)
            {
                auto original = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                return TraceDoubleFormWithMetric(
                    DoubleFormCF(inputs[0], original->LeftDegree(),
                                 original->RightDegree(), dim),
                    inputs[1]);
            }
            if (IsCompactDoubleFormHodgeOperation(operation))
            {
                auto original = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                return HodgeDoubleFormWithMetric(
                    DoubleFormCF(inputs[0], original->LeftDegree(),
                                 original->RightDegree(), dim),
                    inputs[1], inputs[2],
                    CompactDoubleFormHodgeSlot(operation));
            }

            auto original_left = dynamic_pointer_cast<
                DoubleFormCoefficientFunction>(operands[0]);
            auto original_right = dynamic_pointer_cast<
                DoubleFormCoefficientFunction>(operands[1]);
            return Wedge(
                DoubleFormCF(inputs[0], original_left->LeftDegree(),
                             original_left->RightDegree(), dim),
                DoubleFormCF(inputs[1], original_right->LeftDegree(),
                             original_right->RightDegree(), dim));
        }

    public:
        CompactDoubleFormOperationCoefficientFunction(
            int aoperation,
            const Array<shared_ptr<CoefficientFunction>> &semantic_operands,
            shared_ptr<CoefficientFunction> aindependent,
            int aleft_degree, int aright_degree, int adim)
            : CompactDoubleFormOperationCoefficientFunction(
                  aoperation, semantic_operands, aindependent,
                  aleft_degree, aright_degree, adim,
                  MakeEvaluator(aoperation, semantic_operands,
                                aindependent, aleft_degree,
                                aright_degree, adim))
        {
        }

        CompactDoubleFormOperationCoefficientFunction(
            int aoperation,
            const Array<shared_ptr<CoefficientFunction>> &semantic_operands,
            shared_ptr<CoefficientFunction> aindependent,
            int aleft_degree, int aright_degree, int adim,
            shared_ptr<CoefficientFunction> aevaluator)
            : SymbolicExpressionCoefficientFunction(
                  semantic_operands,
                  RequireCompactNonNull(std::move(aevaluator))),
              operation(CompactDoubleFormOperation(aoperation)),
              independent(std::move(aindependent)),
              left_degree(aleft_degree),
              right_degree(aright_degree), dim(adim)
        {
            const bool hodge_operation =
                IsCompactDoubleFormHodgeOperation(operation);
            if (!IsValidCompactDoubleFormOperation(operation) ||
                operands.Size() !=
                    CompactDoubleFormOperationArity(operation))
                throw Exception(
                    "CompactDoubleFormOperation: invalid operation state");

            if (operation == CompactDoubleFormOperation::Sum ||
                operation == CompactDoubleFormOperation::Difference)
            {
                if (operands[0]->Dimensions() != Dimensions() ||
                    operands[1]->Dimensions() != Dimensions())
                    throw Exception(
                        "CompactDoubleFormOperation: sum operand shape mismatch");
            }
            else if (operation == CompactDoubleFormOperation::Scale)
            {
                if (operands[0]->Dimensions() != Dimensions() ||
                    operands[1]->Dimensions().Size() != 0)
                    throw Exception(
                        "CompactDoubleFormOperation: scale operand shape mismatch");
            }
            else if (operation == CompactDoubleFormOperation::Swap)
            {
                auto input = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                if (!input || input->DimensionOfSpace() != dim ||
                    input->LeftDegree() != right_degree ||
                    input->RightDegree() != left_degree)
                    throw Exception(
                        "CompactDoubleFormOperation: swap operand metadata mismatch");
            }
            else if (operation == CompactDoubleFormOperation::Trace)
            {
                auto input = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                if (!input || input->DimensionOfSpace() != dim ||
                    input->LeftDegree() != left_degree + 1 ||
                    input->RightDegree() != right_degree + 1 ||
                    operands[1]->Dimensions() != Array<int>{dim, dim})
                    throw Exception(
                        "CompactDoubleFormOperation: trace operand metadata mismatch");
            }
            else if (hodge_operation)
            {
                auto input = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                if (!input || input->DimensionOfSpace() != dim ||
                    operands[1]->Dimensions() != Array<int>{dim, dim} ||
                    operands[2]->Dimensions().Size() != 0)
                    throw Exception(
                        "CompactDoubleFormOperation: hodge operand metadata mismatch");
                const int expected_left =
                    operation == CompactDoubleFormOperation::HodgeRight
                        ? input->LeftDegree()
                        : dim - input->LeftDegree();
                const int expected_right =
                    operation == CompactDoubleFormOperation::HodgeLeft
                        ? input->RightDegree()
                        : dim - input->RightDegree();
                if (left_degree != expected_left ||
                    right_degree != expected_right)
                    throw Exception(
                        "CompactDoubleFormOperation: hodge output metadata mismatch");
            }
            else
            {
                auto left = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                auto right = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[1]);
                if (!left || !right ||
                    left->DimensionOfSpace() != dim ||
                    right->DimensionOfSpace() != dim ||
                    left->LeftDegree() + right->LeftDegree() !=
                        left_degree ||
                    left->RightDegree() + right->RightDegree() !=
                        right_degree)
                    throw Exception(
                        "CompactDoubleFormOperation: wedge operand metadata mismatch");
            }
        }

        auto GetCArgs() const
        {
            return tuple{
                int(operation),
                Array<shared_ptr<CoefficientFunction>>(operands),
                independent, left_degree, right_degree, dim};
        }

        string GetDescription() const override
        {
            return CompactDoubleFormOperationDescription(operation);
        }

        void CalcEquivalenceKey() override
        {
            SymbolicExpressionCoefficientFunction::CalcEquivalenceKey();
            equivalence_key +=
                "[p=" + ToString(left_degree) +
                ",q=" + ToString(right_degree) +
                ",dim=" + ToString(dim) + "]";
        }

        shared_ptr<CoefficientFunction> IndependentValue() const
        {
            return independent;
        }

        shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            auto derivative_form = [&](
                                       const shared_ptr<CoefficientFunction> &value,
                                       int p, int q)
            {
                return IsConstantZero(value)
                           ? ZeroDoubleForm(p, q, dim)
                           : DoubleFormCF(value, p, q, dim);
            };
            if (operation == CompactDoubleFormOperation::Sum ||
                operation == CompactDoubleFormOperation::Difference)
                return AddDoubleForms(
                    derivative_form(operands[0]->Diff(var, dir),
                                    left_degree, right_degree),
                    derivative_form(operands[1]->Diff(var, dir),
                                    left_degree, right_degree),
                    operation ==
                        CompactDoubleFormOperation::Difference);
            if (operation == CompactDoubleFormOperation::Scale)
            {
                auto form = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                auto left_term = ScaleDoubleForm(
                    derivative_form(form->Diff(var, dir),
                                    left_degree, right_degree),
                    operands[1]);
                auto right_term = ScaleDoubleForm(
                    form, operands[1]->Diff(var, dir));
                return AddDoubleForms(left_term, right_term);
            }
            if (operation == CompactDoubleFormOperation::Swap)
            {
                auto input = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                return SwapDoubleFormSlots(derivative_form(
                    input->Diff(var, dir), input->LeftDegree(),
                    input->RightDegree()));
            }
            if (operation == CompactDoubleFormOperation::Trace)
            {
                auto input = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                auto differentiated_input = derivative_form(
                    input->Diff(var, dir), input->LeftDegree(),
                    input->RightDegree());
                auto input_term = TraceDoubleFormWithMetric(
                    differentiated_input, operands[1]);
                auto differentiated_metric =
                    operands[1]->Diff(var, dir);
                auto metric_term = IsConstantZero(differentiated_metric)
                                       ? ZeroDoubleForm(
                                             left_degree, right_degree,
                                             dim)
                                       : TraceDoubleFormWithMetric(
                                             input,
                                             differentiated_metric);
                return AddDoubleForms(input_term, metric_term);
            }
            if (IsCompactDoubleFormHodgeOperation(operation))
            {
                bool evaluator_contains_var = false;
                if (var->Dimensions().Size() == 0)
                    TraverseSemanticDAG(
                        evaluator, [&](CoefficientFunction &node)
                        { evaluator_contains_var |= &node == var; });
                if (evaluator_contains_var)
                    return evaluator->Diff(var, std::move(dir));
                auto input = dynamic_pointer_cast<
                    DoubleFormCoefficientFunction>(operands[0]);
                return HodgeDenseDoubleFormWithMetric(
                           input, operands[1], operands[2],
                           CompactDoubleFormHodgeSlot(operation))
                    ->Diff(var, std::move(dir));
            }

            auto left = dynamic_pointer_cast<
                DoubleFormCoefficientFunction>(operands[0]);
            auto right = dynamic_pointer_cast<
                DoubleFormCoefficientFunction>(operands[1]);
            auto dleft = derivative_form(
                left->Diff(var, dir), left->LeftDegree(),
                left->RightDegree());
            auto dright = derivative_form(
                right->Diff(var, dir), right->LeftDegree(),
                right->RightDegree());
            return AddDoubleForms(Wedge(dleft, right),
                                  Wedge(left, dright));
        }

        shared_ptr<CoefficientFunction> DiffJacobi(
            const CoefficientFunction *var, T_DJC &cache) const override
        {
            auto self = const_pointer_cast<CoefficientFunction>(
                shared_from_this());
            if (auto found = cache.find(self); found != cache.end())
                return found->second;
            if (this == var)
                return cache[self] = IdentityCF(Dimensions());
            if (var->Dimensions().Size() == 0)
                return cache[self] = Diff(
                           var, make_shared<ConstantCoefficientFunction>(1.0));
            return cache[self] =
                       SymbolicExpressionCoefficientFunction::DiffJacobi(var,
                                                                         cache);
        }
    };

    shared_ptr<CompactDoubleFormOperationCoefficientFunction>
    GetCompactDoubleFormOperation(
        const shared_ptr<CoefficientFunction> &value)
    {
        auto full = value;
        while (auto wrapper =
                   dynamic_pointer_cast<TensorFieldCoefficientFunction>(full))
            full = wrapper->GetFullCoefficient();
        return dynamic_pointer_cast<
            CompactDoubleFormOperationCoefficientFunction>(full);
    }

    shared_ptr<CompactKFormOperationCoefficientFunction>
    GetCompactKFormOperation(const shared_ptr<CoefficientFunction> &value)
    {
        auto full = value;
        while (auto wrapper =
                   dynamic_pointer_cast<TensorFieldCoefficientFunction>(full))
            full = wrapper->GetFullCoefficient();
        return dynamic_pointer_cast<CompactKFormOperationCoefficientFunction>(
            full);
    }

    static ngcore::RegisterClassForArchive<
        CompactKFormOperationCoefficientFunction, CoefficientFunction>
        reg_compact_kform_operation_cf;
    static ngcore::RegisterClassForArchive<
        CompactDoubleFormOperationCoefficientFunction,
        CoefficientFunction>
        reg_compact_double_form_operation_cf;

} // namespace kforms_internal
