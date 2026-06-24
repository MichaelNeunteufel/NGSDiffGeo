#include "coefficient_grad.hpp"
#include <hcurlhdiv_dshape.hpp>
#include <fespace.hpp>
#include <gridfunction.hpp>
#include <set>

namespace ngfem
{

    using namespace ngcomp;

    struct ProxyInfo
    {
        bool has_trial = false;
        bool has_test = false;
        ProxyFunction *proxy = nullptr;
    };

    ProxyInfo GetProxyInfo(const shared_ptr<CoefficientFunction> &cf)
    {
        ProxyInfo info;
        cf->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          if (auto proxy = dynamic_cast<ProxyFunction*> (&nodecf))
            {
              if (!info.proxy)
                info.proxy = proxy;
              if (proxy->IsTestFunction())
                  info.has_test = true;
              else
                  info.has_trial = true;
            } });
        return info;
    }

    bool HasDescriptionPrefix(const CoefficientFunction &cf, const string &prefix)
    {
        return cf.GetDescription().rfind(prefix, 0) == 0;
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
                                                             size_t difforder)
    {
        auto first = DerivativeFirstDimensions(cf, dim, difforder);
        if (SameDimensions(result->Dimensions(), first))
            return result;

        auto last = DerivativeLastDimensions(cf, dim, difforder);
        if (SameDimensions(result->Dimensions(), last))
        {
            auto transposed = result;
            size_t cfdim = cf->Dimensions().Size();
            for (size_t deriv = 0; deriv < difforder; deriv++)
            {
                for (size_t pos = cfdim + deriv; pos > deriv; pos--)
                    transposed = transposed->TensorTranspose(int(pos - 1), int(pos));
            }
            return transposed;
        }

        size_t expected_dim = 1;
        for (int d : first)
            expected_dim *= d;
        if (result->Dimension() == expected_dim)
            return result->Reshape(first);

        return result;
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
            return ProxyDirection(*proxy, surface ? "Gradboundary" : "Grad");
        return GradCF(var, dim, surface);
    }

    shared_ptr<CoefficientFunction> SymbolicGradByChainRule(const shared_ptr<CoefficientFunction> &cf,
                                                           size_t dim,
                                                           bool surface)
    {
        Array<shared_ptr<CoefficientFunction>> vars;
        set<const CoefficientFunction *> seen;
        bool has_component_wrapper = false;

        cf->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          if (HasDescriptionPrefix(nodecf, "ComponentCoefficientFunction"))
            has_component_wrapper = true; });

        cf->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          bool is_component_wrapper =
            HasDescriptionPrefix(nodecf, "ComponentCoefficientFunction");
          if (!is_component_wrapper && nodecf.InputCoefficientFunctions().Size() != 0)
            return;
          if (has_component_wrapper && dynamic_cast<ngcomp::GridFunctionCoefficientFunction *>(&nodecf))
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
                auto dvar = DerivativeOfVariable(var, dim, surface);
                auto dvar_comp = MakeComponentCoefficientFunction(dvar, d);
                comps[d] = comps[d] + cf->Diff(var.get(), dvar_comp);
            }
        }

        auto result = MakeVectorialCoefficientFunction(std::move(comps));
        return result->Reshape(DerivativeFirstDimensions(cf, dim, 1));
    }

    shared_ptr<CoefficientFunction> GradCF(const shared_ptr<CoefficientFunction> &cf, size_t dim, bool surface)
    {
        // create new ZeroCF with updated dimensions
        if (cf->IsZeroCF())
        {
            Array<int> resultdims = {int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        auto proxy_info = GetProxyInfo(cf);

        if (dim < 1 || dim > 3)
            throw Exception("GradCF: only dimensions 1,2,3 supported");
        if (surface && dim < 2)
            throw Exception("GradCF(surface): only dimensions 2,3 supported");

        if (proxy_info.has_trial && proxy_info.has_test)
            throw Exception("GradCF: expressions containing trial and test functions in the same GradCF are not supported yet");

        if (proxy_info.has_trial || proxy_info.has_test)
        {
            try
            {
                cf->SetSpaceDim(int(dim));
                return NormalizeDerivativeSlots(cf->Operator(surface ? "Gradboundary" : "Grad"),
                                                cf, dim, 1);
            }
            catch (...)
            {
                ;
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
        if (cf->IsZeroCF())
        {
            Array<int> resultdims = {int(dim), int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        auto proxy_info = GetProxyInfo(cf);

        if (dim < 1 || dim > 3)
            throw Exception("HesseCF: only dimensions 1,2,3 supported");

        if (proxy_info.has_trial && proxy_info.has_test)
            throw Exception("HesseCF: expressions containing trial and test functions in the same HesseCF are not supported yet");

        if (proxy_info.has_trial || proxy_info.has_test)
        {
            string opname = boundary ? "hesseboundary" : "hesse";
            try
            {
                cf->SetSpaceDim(int(dim));
                return NormalizeDerivativeSlots(cf->Operator(opname), cf, dim, 2);
            }
            catch (const Exception &e)
            {
                throw Exception(string("HesseCF: symbolic proxy Hessian requires Operator(\"") +
                                opname + string("\") support for the full expression. Original error: ") +
                                e.What());
            }
        }

        return NormalizeDerivativeSlots(GradCF(GradCF(cf, dim, boundary), dim, boundary),
                                        cf, dim, 2);
    }

};

void ExportGradCF(py::module m)
{
    using namespace ngfem;

    m.def("GradCF", [](shared_ptr<CoefficientFunction> cf, int dim, bool surface)
          { return GradCF(cf, dim, surface); }, "Create a GradientCoefficientFunction. Uses numerical differentiation to compute the gradient of a given CoefficientFunction. Set surface=True for tangential surface gradients.", py::arg("cf"), py::arg("dim"), py::arg("surface") = false);
    m.def("HesseCF", [](shared_ptr<CoefficientFunction> cf, int dim, bool boundary)
          { return HesseCF(cf, dim, boundary); }, "Create a Hessian CoefficientFunction. Uses numerical differentiation for pure coefficient functions and symbolic finite element operators for trial/test functions.", py::arg("cf"), py::arg("dim"), py::arg("boundary") = false);
}
