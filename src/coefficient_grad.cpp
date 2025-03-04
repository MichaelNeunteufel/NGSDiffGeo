#include "coefficient_grad.hpp"
#include <hcurlhdiv_dshape.hpp>
#include <fespace.hpp>

namespace ngfem
{

    using namespace ngcomp;
    shared_ptr<CoefficientFunction> GradCF(const shared_ptr<CoefficientFunction> &cf, size_t dim)
    {
        if (cf->IsZeroCF())
            return cf;

        bool has_trial = false, has_test = false;

        ProxyFunction *proxy = nullptr;
        cf->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          
          if (dynamic_cast<ProxyFunction*> (&nodecf))
            {
            proxy = dynamic_cast<ProxyFunction *>(&nodecf);
              if (proxy->IsTestFunction())
                  has_test = true;
              else
                  has_trial = true;
            } });

        // If the input coefficient function includes a trial or test function, we need to use
        // a proxy function to calculate the gradient via a differential operator.
        // Otherwise, we can directly calculate the gradient using the CoefficientFunction.
        if (has_trial != has_test)
        {
            shared_ptr<GradProxy> grad_proxy;
            switch (dim)
            {
            case 1:
                grad_proxy = make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<1>>(cf, has_test));
                break;
            case 2:
                grad_proxy = make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<2>>(cf, has_test));
                break;
            default:
                grad_proxy = make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<3>>(cf, has_test));
                break;
            }
            return cf->Diff(proxy, grad_proxy);
            // switch (dim)
            // {
            // case 1:
            //     return make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<1>>(cf, has_test));
            // case 2:
            //     return make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<2>>(cf, has_test));
            // default:
            //     return make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<3>>(cf, has_test));
            // }
        }
        else
            switch (dim)
            {
            case 1:
                return make_shared<GradCoefficientFunction<1>>(cf);
            case 2:
                return make_shared<GradCoefficientFunction<2>>(cf);
            default:
                return make_shared<GradCoefficientFunction<3>>(cf);
            }
    }

    template <int D>
    GradDiffOp<D>::GradDiffOp(shared_ptr<CoefficientFunction> afunc, bool atestfunction)
        : DifferentialOperator(D * afunc->Dimension(), 1, VOL, 1), func(afunc), testfunction(atestfunction)
    {
        for (auto cf_dim : func->Dimensions())
            if (cf_dim != D)
                throw Exception("GradDiffOp: all dimensions must be the same and equal to D");

        Array<int> tensor_dims(func->Dimensions().Size() + 1);
        for (size_t i = 0; i < tensor_dims.Size(); i++)
            tensor_dims[i] = D;
        SetDimensions(tensor_dims);

        // Extract proxy function
        func->TraverseTree([&](CoefficientFunction &nodecf)
                           {
            if (dynamic_cast<ProxyFunction *>(&nodecf))
                proxy = dynamic_cast<ProxyFunction *>(&nodecf); });
    }

    template <int D>
    void GradDiffOp<D>::CalcMatrix(const FiniteElement &inner_fel,
                                   const BaseMappedIntegrationRule &bmir,
                                   BareSliceMatrix<double, ColMajor> mat,
                                   LocalHeap &lh) const
    {
        cout << "in GradDiffOp CalcMatrix" << endl
             << "proxy_dim = " << proxy->Dimension() << ", D = " << D << ", ndof = " << inner_fel.GetNDof() << ", mir.size = " << bmir.Size() << endl;
        HeapReset hr(lh);

        auto &mir = static_cast<const MappedIntegrationRule<D, D> &>(bmir);
        auto &ir = mir.IR();

        size_t proxy_dim = proxy->Dimension();
        FlatMatrix<double> bbmat(inner_fel.GetNDof(), 4 * proxy_dim, lh);
        FlatMatrix<double> dshape_ref(inner_fel.GetNDof() * proxy_dim, D, lh);
        FlatMatrix<double> dshape(inner_fel.GetNDof() * proxy_dim, D, lh);

        for (size_t i = 0; i < mir.Size(); i++)
        {
            const IntegrationPoint &ip = ir[i];
            const ElementTransformation &eltrans = mir[i].GetTransformation();
            dshape_ref = 0;
            for (int j = 0; j < D; j++) // d / d t_j
            {
                HeapReset hr(lh);
                IntegrationPoint ipts[4];
                ipts[0] = ip;
                ipts[0](j) -= eps;
                ipts[1] = ip;
                ipts[1](j) += eps;
                ipts[2] = ip;
                ipts[2](j) -= 2 * eps;
                ipts[3] = ip;
                ipts[3](j) += 2 * eps;

                IntegrationRule ir_j(4, ipts);
                MappedIntegrationRule<D, D, double> mir_j(ir_j, eltrans, lh);

                proxy->Evaluator()->CalcMatrix(inner_fel, mir_j, Trans(bbmat), lh);

                // cout << "bbmat = " << bbmat << endl;

                // dshape_ref.Col(j) = (1.0 / (12.0 * eps)) * (8.0 * bbmat.Cols(proxy_dim, 2 * proxy_dim).AsVector() - 8.0 * bbmat.Cols(0, proxy_dim).AsVector() - bbmat.Cols(3 * proxy_dim, 4 * proxy_dim).AsVector() + bbmat.Cols(2 * proxy_dim, 3 * proxy_dim).AsVector());
                dshape_ref.Col(j) = (1.0 / (12.0 * eps)) * (8.0 * bbmat.Col(1) - 8.0 * bbmat.Col(0) - bbmat.Col(3) + bbmat.Col(2));
                //  if (j == D - 1)
                //      cout << "dshape_ref = " << dshape_ref << endl;
            }
            dshape = dshape_ref * mir[i].GetJacobianInverse();
            cout << "dshape = " << dshape << endl;

            for (auto k : Range(dshape.Height()))
                for (auto l : Range(dshape.Width()))
                    mat(i * dshape.Width() + l, k) = dshape(k, l);
        }

        cout << "GradDiffOp CalcMatrix done" << endl;
    }

    shared_ptr<FESpace> FindProxySpace(shared_ptr<CoefficientFunction> func)
    {
        shared_ptr<FESpace> space;

        func->TraverseTree([&](CoefficientFunction &nodecf)
                           {
          if (auto proxy = dynamic_cast<ProxyFunction*> (&nodecf))
            space = proxy->GetFESpace(); });
        return space;
    }

    GradProxy::GradProxy(shared_ptr<CoefficientFunction> afunc, bool atestfunction, int adim, shared_ptr<DifferentialOperator> adiffop)
        : ProxyFunction(FindProxySpace(afunc), atestfunction, false,
                        adiffop, nullptr, nullptr,
                        nullptr, nullptr, nullptr),
          func(afunc), testfunction(atestfunction), dim(adim)
    {
        Array<int> tensor_dims(func->Dimensions().Size() + 1);
        for (size_t i = 0; i < tensor_dims.Size(); i++)
            tensor_dims[i] = dim;
        SetDimensions(tensor_dims);
    }

    shared_ptr<CoefficientFunction> GradProxy::Diff(const CoefficientFunction *var, shared_ptr<CoefficientFunction> dir) const
    {
        if (this == var)
            return dir;
        return make_shared<GradProxy>(func->Diff(var, dir), testfunction, dim, this->Evaluator());
    }
};

void ExportGradCF(py::module m)
{
    using namespace ngfem;

    py::class_<GradProxy, shared_ptr<GradProxy>, ProxyFunction>(m, "GradProxy");
    m.def("GradCF", [](shared_ptr<CoefficientFunction> cf, int dim)
          { return GradCF(cf, dim); }, "Create a GradientCoefficientFunction. Uses numerical differentiation to compute the gradient of a given CoefficientFunction");
}