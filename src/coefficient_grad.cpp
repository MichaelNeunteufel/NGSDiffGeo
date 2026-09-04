#include "coefficient_grad.hpp"
#include <core/register_archive.hpp>
#include <set>

namespace ngfem
{

    using namespace ngcomp;

    struct ProxyInfo
    {
        bool has_trial = false;
        bool has_test = false;
    };

    ProxyInfo GetProxyInfo(const shared_ptr<CoefficientFunction> &cf)
    {
        ProxyInfo info;
        cf->TraverseTree([&](CoefficientFunction &nodecf)
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

    shared_ptr<CoefficientFunction> ProxyVariable(ProxyFunction &proxy)
    {
        if (auto primary = dynamic_pointer_cast<ProxyFunction>(proxy.Primary()))
            return primary;
        return dynamic_pointer_cast<CoefficientFunction>(proxy.shared_from_this());
    }

    shared_ptr<CoefficientFunction> DerivativeOfVariable(const shared_ptr<CoefficientFunction> &var,
                                                         size_t dim,
                                                         bool surface)
    {
        if (auto proxy = dynamic_pointer_cast<ProxyFunction>(var))
            return NormalizeDerivativeSlots(
                ProxyDirection(*proxy, surface ? "Gradboundary" : "Grad"),
                var,
                dim,
                1,
                true);
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

        cf->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          if (nodecf.InputCoefficientFunctions().Size() != 0)
            return;

          shared_ptr<CoefficientFunction> var;
          if (auto proxy = dynamic_cast<ProxyFunction *>(&nodecf))
            var = ProxyVariable(*proxy);
          else
            var = const_pointer_cast<CoefficientFunction>(nodecf.shared_from_this());

          if (var && !seen.count(var.get()))
            {
              seen.insert(var.get());
              vars.Append(var);
            } });

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

        // create new ZeroCF with updated dimensions
        if (cf->IsZeroCF())
        {
            Array<int> resultdims = {int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        auto proxy_info = GetProxyInfo(cf);

        if (proxy_info.has_trial && proxy_info.has_test)
            throw Exception("GradCF: expressions containing trial and test functions in the same GradCF are not supported yet");

        if (proxy_info.has_trial || proxy_info.has_test)
        {
            try
            {
                cf->SetSpaceDim(int(dim));
                return NormalizeDerivativeSlots(cf->Operator(surface ? "Gradboundary" : "Grad"),
                                                cf, dim, 1, true);
            }
            catch (const Exception &)
            {
                // Fall back when this proxy type has no native operator.
            }
            return NormalizeDerivativeSlots(SymbolicGradByChainRule(cf, dim, surface),
                                            cf, dim, 1);
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

        if (cf->IsZeroCF())
        {
            Array<int> resultdims = {int(dim), int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        auto proxy_info = GetProxyInfo(cf);

        if (proxy_info.has_trial && proxy_info.has_test)
            throw Exception("HesseCF: expressions containing trial and test functions in the same HesseCF are not supported yet");

        if (proxy_info.has_trial || proxy_info.has_test)
        {
            string opname = boundary ? "hesseboundary" : "hesse";
            try
            {
                cf->SetSpaceDim(int(dim));
                return NormalizeDerivativeSlots(
                    cf->Operator(opname), cf, dim, 2, true);
            }
            catch (const Exception &e)
            {
                // H1(dim=...) exposes a scalar Hessian evaluator even though
                // its identity and gradient evaluators are block-valued.
                // Rebuild that direct proxy operator with one scalar Hessian
                // block per component. BlockDifferentialOperator stores the
                // differential-operator slots before the component slot.
                if (auto proxy = dynamic_pointer_cast<ProxyFunction>(cf))
                {
                    auto evaluator = proxy->GetAdditionalEvaluator(opname);
                    if (evaluator && evaluator->Dim() == dim * dim && cf->Dimension() > 1)
                    {
                        auto block_evaluator =
                            make_shared<BlockDifferentialOperator>(
                                evaluator, cf->Dimension());
                        return NormalizeDerivativeSlots(
                            proxy->Operator(block_evaluator),
                            cf,
                            dim,
                            2,
                            false);
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
