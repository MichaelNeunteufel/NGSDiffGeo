#include "coefficient_grad.hpp"
#include "tensor_fields.hpp"
#include "symbolic_expression.hpp"
#include <core/register_archive.hpp>
#include <set>

namespace ngfem
{

    using namespace ngcomp;

    class SymbolicDerivativeCoefficientFunction : public SymbolicExpressionCoefficientFunction
    {
        int dim, order;
        bool surface;
        shared_ptr<CoefficientFunction> Rebuild(
            const Array<shared_ptr<CoefficientFunction>> &inputs) const override
        {
            return order == 1 ? GradCF(inputs[0], dim, surface)
                              : HesseCF(inputs[0], dim, surface);
        }

    public:
        SymbolicDerivativeCoefficientFunction(shared_ptr<CoefficientFunction> input,
                                              int adim, int aorder, bool asurface, shared_ptr<CoefficientFunction> value)
            : SymbolicExpressionCoefficientFunction({input}, value, true),
              dim(adim), order(aorder), surface(asurface) {}
        auto GetCArgs() const { return tuple{operands[0], dim, order, surface, evaluator}; }
        string GetDescription() const override
        {
            return "SymbolicDerivativeCF<" + ToString(dim) + "," + ToString(order) + "," + ToString(surface) + ">";
        }
        shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                             shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return Rebuild({operands[0]->Diff(var, dir)});
        }
        shared_ptr<CoefficientFunction> NestedGradient(int next_dim, bool next_surface) const
        {
            if (order == 1 && dim == next_dim && surface == next_surface)
                return HesseCF(operands[0], dim, surface);
            throw Exception("GradCF: this nested symbolic derivative requires a higher native operator");
        }
    };

    static ngcore::RegisterClassForArchive<SymbolicDerivativeCoefficientFunction,
                                           CoefficientFunction>
        reg_symbolic_derivative;

    struct ProxyInfo
    {
        bool has_trial = false;
        bool has_test = false;
    };

    ProxyInfo GetProxyInfo(const shared_ptr<CoefficientFunction> &cf)
    {
        ProxyInfo info;
        TraverseSemanticDAG(cf, [&](CoefficientFunction &nodecf)
                            {
          if (auto proxy = dynamic_cast<ProxyFunction*> (&nodecf))
            {
              if (proxy->IsTestFunction())
                  info.has_test = true;
              else
                  info.has_trial = true;
            } });
        return info;
    }

    bool SameDimensions(FlatArray<int> a, FlatArray<int> b)
    {
        if (a.Size() != b.Size())
            return false;
        for (size_t i = 0; i < a.Size(); i++)
            if (a[i] != b[i])
                return false;
        return true;
    }

    Array<int> DerivativeFirstDimensions(const shared_ptr<CoefficientFunction> &cf,
                                         size_t dim,
                                         size_t difforder)
    {
        Array<int> dims;
        for (size_t i = 0; i < difforder; i++)
            dims.Append(int(dim));
        dims += cf->Dimensions();
        return dims;
    }

    Array<int> DerivativeLastDimensions(const shared_ptr<CoefficientFunction> &cf,
                                        size_t dim,
                                        size_t difforder)
    {
        Array<int> dims;
        dims += cf->Dimensions();
        for (size_t i = 0; i < difforder; i++)
            dims.Append(int(dim));
        return dims;
    }

    shared_ptr<CoefficientFunction> NormalizeDerivativeSlots(shared_ptr<CoefficientFunction> result,
                                                             const shared_ptr<CoefficientFunction> &cf,
                                                             size_t dim,
                                                             size_t difforder,
                                                             bool derivatives_are_last = false)
    {
        if (!result)
            throw Exception("derivative operator returned nullptr");

        auto first = DerivativeFirstDimensions(cf, dim, difforder);
        auto last = DerivativeLastDimensions(cf, dim, difforder);
        auto move_last_to_first = [&]()
        {
            auto transposed = result;
            size_t cfdim = cf->Dimensions().Size();
            for (size_t deriv = 0; deriv < difforder; deriv++)
            {
                for (size_t pos = cfdim + deriv; pos > deriv; pos--)
                    transposed = transposed->TensorTranspose(int(pos - 1), int(pos));
            }
            return transposed;
        };

        if (derivatives_are_last && SameDimensions(result->Dimensions(), last))
            return move_last_to_first();

        if (SameDimensions(result->Dimensions(), first))
            return result;

        if (SameDimensions(result->Dimensions(), last))
            return move_last_to_first();

        size_t expected_dim = 1;
        for (int d : first)
            expected_dim *= d;
        if (result->Dimension() == expected_dim)
        {
            result = result->Reshape(
                derivatives_are_last ? last : first);
            return derivatives_are_last ? move_last_to_first() : result;
        }

        throw Exception(
            "derivative operator returned dimensions " + ToString(result->Dimensions()) + ", expected " + ToString(first));
    }

    shared_ptr<CoefficientFunction> ProxyDirection(ProxyFunction &proxy, const string &opname)
    {
        auto op = proxy.Operator(opname);
        if (!op)
            throw Exception(string("operator \"") + opname + string("\" returned nullptr"));
        return op;
    }

    shared_ptr<CoefficientFunction> DerivativeOfVariable(const shared_ptr<CoefficientFunction> &var,
                                                         size_t dim,
                                                         bool surface)
    {
        if (auto proxy = dynamic_pointer_cast<ProxyFunction>(var))
        {
            // A native gradient proxy is an independent leaf of this spatial
            // chain rule. Its spatial derivative is the primary proxy's Hessian.
            if (auto primary = dynamic_pointer_cast<ProxyFunction>(proxy->Primary()))
            {
                auto gradient = dynamic_pointer_cast<ProxyFunction>(
                    ProxyDirection(*primary, surface ? "Gradboundary" : "Grad"));
                if (gradient && *gradient->Evaluator() == *proxy->Evaluator())
                {
                    auto hessian = HesseCF(primary, dim, surface);
                    // Hessian: (new derivative, old derivative, components).
                    // Native Grad proxy: (components, old derivative).
                    for (int axis = 1; axis < primary->Dimensions().Size() + 1; ++axis)
                        hessian = hessian->TensorTranspose(axis, axis + 1);
                    return hessian;
                }
            }
            return NormalizeDerivativeSlots(
                ProxyDirection(*proxy, surface ? "Gradboundary" : "Grad"),
                var,
                dim,
                1,
                true);
        }
        return GradCF(var, dim, surface);
    }

    shared_ptr<CoefficientFunction> DerivativeDirection(
        const shared_ptr<CoefficientFunction> &var,
        size_t dim,
        bool surface,
        size_t direction)
    {
        auto derivative = DerivativeOfVariable(var, dim, surface);
        if (var->Dimensions().Size() == 0)
            return MakeComponentCoefficientFunction(derivative, int(direction));

        Array<shared_ptr<CoefficientFunction>> components(var->Dimension());
        for (size_t component = 0; component < var->Dimension(); component++)
            components[component] = MakeComponentCoefficientFunction(
                derivative,
                int(direction * var->Dimension() + component));
        return MakeVectorialCoefficientFunction(std::move(components))
            ->Reshape(var->Dimensions());
    }

    shared_ptr<CoefficientFunction> SymbolicGradByChainRule(const shared_ptr<CoefficientFunction> &cf,
                                                            size_t dim,
                                                            bool surface)
    {
        Array<shared_ptr<CoefficientFunction>> vars;
        set<const CoefficientFunction *> seen;

        // A retained differential expression is itself a spatial-chain-rule
        // variable. Descending into its evaluator would lose that identity and
        // differentiate a different graph. Visit every shared node only once.
        function<void(shared_ptr<CoefficientFunction>)> collect = [&](auto node)
        {
            if (!node || !seen.insert(node.get()).second)
                return;
            auto inputs = node->InputCoefficientFunctions();
            if (dynamic_pointer_cast<SymbolicDerivativeCoefficientFunction>(node) || inputs.Size() == 0)
                vars.Append(node);
            else
                for (auto input : inputs)
                    collect(input);
        };
        collect(cf);

        Array<shared_ptr<CoefficientFunction>> comps(dim);
        for (size_t d = 0; d < dim; d++)
        {
            comps[d] = ZeroCF(cf->Dimensions());
            for (auto var : vars)
            {
                auto direction = DerivativeDirection(var, dim, surface, d);
                comps[d] = comps[d] + cf->Diff(var.get(), direction);
            }
        }

        auto result = MakeVectorialCoefficientFunction(std::move(comps));
        return result->Reshape(DerivativeFirstDimensions(cf, dim, 1));
    }

    shared_ptr<CoefficientFunction> ProjectSecondDerivativeSlotToBoundary(
        const shared_ptr<CoefficientFunction> &hessian,
        const shared_ptr<CoefficientFunction> &cf,
        size_t dim)
    {
        auto normal = NormalVectorCF(int(dim));
        auto normal_column = normal->Reshape(Array<int>{int(dim), 1});
        auto projection = IdentityCF(int(dim)) - normal_column * TransposeCF(normal_column);
        const size_t value_dimension = cf->Dimension();
        Array<shared_ptr<CoefficientFunction>> components(
            dim * dim * value_dimension);

        for (size_t first = 0; first < dim; first++)
            for (size_t second = 0; second < dim; second++)
                for (size_t component = 0;
                     component < value_dimension;
                     component++)
                {
                    shared_ptr<CoefficientFunction> projected_component;
                    for (size_t contracted = 0; contracted < dim; contracted++)
                    {
                        auto term = MakeComponentCoefficientFunction(
                                        hessian,
                                        int((first * dim + contracted) * value_dimension + component)) *
                                    MakeComponentCoefficientFunction(
                                        projection,
                                        int(contracted * dim + second));
                        projected_component = projected_component
                                                  ? projected_component + term
                                                  : term;
                    }
                    components[(first * dim + second) * value_dimension + component] = projected_component;
                }

        return MakeVectorialCoefficientFunction(std::move(components))
            ->Reshape(DerivativeFirstDimensions(cf, dim, 2));
    }

    shared_ptr<CoefficientFunction> GradCF(
        const shared_ptr<CoefficientFunction> &cf,
        int dim,
        bool surface)
    {
        if (!cf)
            throw Exception("GradCF: input coefficient is null");
        if (dim < 1 || dim > 3)
            throw Exception("GradCF: only dimensions 1,2,3 supported");
        if (surface && dim < 2)
            throw Exception("GradCF(surface): only dimensions 2,3 supported");

        if (auto derivative = dynamic_pointer_cast<SymbolicDerivativeCoefficientFunction>(cf))
            return derivative->NestedGradient(dim, surface);

        // Keep zero-valued expressions with semantic dependencies.
        if (IsConstantZero(cf))
        {
            Array<int> resultdims = {int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        // Keep the semantic expression as the differentiation operand, but
        // build the numerical derivative from its already optimized evaluator.
        auto native_cf = NativeCoefficientValue(cf);
        if (native_cf != cf)
            return make_shared<SymbolicDerivativeCoefficientFunction>(
                cf, dim, 1, surface, GradCF(native_cf, dim, surface));

        auto proxy_info = GetProxyInfo(cf);

        if (proxy_info.has_trial && proxy_info.has_test)
            throw Exception("GradCF: expressions containing trial and test functions in the same GradCF are not supported yet");

        if (proxy_info.has_trial || proxy_info.has_test)
        {
            try
            {
                cf->SetSpaceDim(int(dim));
                auto value = NormalizeDerivativeSlots(cf->Operator(surface ? "Gradboundary" : "Grad"),
                                                      cf, dim, 1, true);
                // Keep the public GradProxy alias for direct native proxies.
                if (dynamic_pointer_cast<ProxyFunction>(cf))
                    return value;
                return make_shared<SymbolicDerivativeCoefficientFunction>(cf, dim, 1, surface, value);
            }
            catch (const Exception &)
            {
                // Fall back when this proxy type has no native operator.
            }
            auto value = NormalizeDerivativeSlots(SymbolicGradByChainRule(cf, dim, surface), cf, dim, 1);
            return make_shared<SymbolicDerivativeCoefficientFunction>(cf, dim, 1, surface, value);
        }
        else
            switch (dim)
            {
            case 1:
                return make_shared<GradCoefficientFunction<1>>(cf, surface);
            case 2:
                return make_shared<GradCoefficientFunction<2>>(cf, surface);
            default:
                return make_shared<GradCoefficientFunction<3>>(cf, surface);
            }
    }

    shared_ptr<CoefficientFunction> HesseCF(const shared_ptr<CoefficientFunction> &cf, size_t dim, bool boundary)
    {
        if (!cf)
            throw Exception("HesseCF: input coefficient is null");
        if (dim < 1 || dim > 3)
            throw Exception("HesseCF: only dimensions 1,2,3 supported");
        if (boundary && dim < 2)
            throw Exception("HesseCF(boundary): only dimensions 2,3 supported");

        if (IsConstantZero(cf))
        {
            Array<int> resultdims = {int(dim), int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        auto native_cf = NativeCoefficientValue(cf);
        if (native_cf != cf)
            return make_shared<SymbolicDerivativeCoefficientFunction>(
                cf, int(dim), 2, boundary, HesseCF(native_cf, dim, boundary));

        auto proxy_info = GetProxyInfo(cf);

        if (proxy_info.has_trial && proxy_info.has_test)
            throw Exception("HesseCF: expressions containing trial and test functions in the same HesseCF are not supported yet");

        if (proxy_info.has_trial || proxy_info.has_test)
        {
            string opname = boundary ? "hesseboundary" : "hesse";
            auto native_input = cf;
            while (auto tensor = dynamic_pointer_cast<TensorFieldCoefficientFunction>(native_input))
                native_input = tensor->GetFullCoefficient();
            auto wrap = [&](shared_ptr<CoefficientFunction> value) -> shared_ptr<CoefficientFunction>
            {
                if (dynamic_pointer_cast<ProxyFunction>(cf))
                    return value;
                return make_shared<SymbolicDerivativeCoefficientFunction>(cf, dim, 2, boundary, value);
            };
            try
            {
                native_input->SetSpaceDim(int(dim));
                return wrap(NormalizeDerivativeSlots(
                    native_input->Operator(opname), cf, dim, 2, true));
            }
            catch (const Exception &e)
            {
                // H1(dim=...) exposes a scalar Hessian evaluator even though
                // its identity and gradient evaluators are block-valued.
                // Rebuild that direct proxy operator with one scalar Hessian
                // block per component. BlockDifferentialOperator stores the
                // differential-operator slots before the component slot.
                if (auto proxy = dynamic_pointer_cast<ProxyFunction>(native_input))
                {
                    auto evaluator = proxy->GetAdditionalEvaluator(opname);
                    if (evaluator && evaluator->Dim() == dim * dim && cf->Dimension() > 1)
                    {
                        auto block_evaluator =
                            make_shared<BlockDifferentialOperator>(
                                evaluator, cf->Dimension());
                        return wrap(NormalizeDerivativeSlots(
                            proxy->Operator(block_evaluator),
                            cf,
                            dim,
                            2,
                            false));
                    }
                }
                throw Exception(string("HesseCF: symbolic proxy Hessian requires Operator(\"") +
                                opname + string("\") support for the full expression. Original error: ") +
                                e.What());
            }
        }

        auto hessian = NormalizeDerivativeSlots(
            GradCF(GradCF(cf, dim, boundary), dim, boundary),
            cf,
            dim,
            2);
        return boundary
                   ? ProjectSecondDerivativeSlotToBoundary(hessian, cf, dim)
                   : hessian;
    }

    static ngcore::RegisterClassForArchive<
        GradCoefficientFunction<1>, CoefficientFunction>
        reg_grad_cf_1;
    static ngcore::RegisterClassForArchive<
        GradCoefficientFunction<2>, CoefficientFunction>
        reg_grad_cf_2;
    static ngcore::RegisterClassForArchive<
        GradCoefficientFunction<3>, CoefficientFunction>
        reg_grad_cf_3;

}

void ExportGradCF(py::module m)
{
    using namespace ngfem;

    m.attr("GradProxy") =
        py::module_::import("ngsolve.comp").attr("ProxyFunction");
    m.def(
        "GradCF",
        [](shared_ptr<CoefficientFunction> cf, int dim, bool surface)
        {
            return GradCF(cf, dim, surface);
        },
        R"doc(
Differentiate a coefficient function in physical coordinates.

The derivative direction is the first result axis, so the output dimensions
are ``(dim, *cf.dims)``. Proxy expressions use NGSolve's symbolic gradient
operators. Pure coefficient graphs use fourth-order numerical differentiation.
With ``surface=True``, the result is the tangential surface gradient in ambient
coordinates.
)doc",
        py::arg("cf"),
        py::arg("dim"),
        py::arg("surface") = false);
    m.def(
        "HesseCF",
        [](shared_ptr<CoefficientFunction> cf, int dim, bool boundary)
        {
            return HesseCF(cf, dim, boundary);
        },
        R"doc(
Return the physical Hessian with derivative axes first.

Pure coefficient graphs and direct ``VectorH1`` or ``H1(dim=...)`` proxies are
supported. Composite proxy expressions require NGSolve to provide the
corresponding native Hessian operator. With ``boundary=True``, return the
tangential boundary Hessian in ambient coordinates.
)doc",
        py::arg("cf"),
        py::arg("dim"),
        py::arg("boundary") = false);
}
