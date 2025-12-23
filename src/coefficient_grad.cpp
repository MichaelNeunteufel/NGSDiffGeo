#include "coefficient_grad.hpp"
#include <hcurlhdiv_dshape.hpp>
#include <fespace.hpp>

namespace ngfem
{

    using namespace ngcomp;
    shared_ptr<CoefficientFunction> GradCF(const shared_ptr<CoefficientFunction> &cf, size_t dim)
    {
        // create new ZeroCF with updated dimensions
        if (cf->IsZeroCF())
        {
            Array<int> resultdims = {int(dim)};
            resultdims += cf->Dimensions();
            return ZeroCF(resultdims);
        }

        bool has_trial = false, has_test = false;

        cf->TraverseTree([&](CoefficientFunction &nodecf)
                         {
          
          if (auto proxy = dynamic_cast<ProxyFunction*> (&nodecf))
            {
              if (proxy->IsTestFunction())
                  has_test = true;
              else
                  has_trial = true;
            } });

        if (dim < 1 || dim > 3)
            throw Exception("GradCF: only dimensions 1,2,3 supported");

        // If the input coefficient function includes a trial or test function, we need to use
        // a proxy function to calculate the gradient via a differential operator.
        // Otherwise, we can directly calculate the gradient using the CoefficientFunction.
        if (has_trial != has_test)
            switch (dim)
            {
            case 1:
                return make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<1>>(cf, has_test));
            case 2:
                return make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<2>>(cf, has_test));
            default:
                return make_shared<GradProxy>(cf, has_test, dim, make_shared<GradDiffOp<3>>(cf, has_test));
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
        if (!proxy)
            throw Exception("GradDiffOp: no ProxyFunction found");
    }

    template <int D>
    void GradDiffOp<D>::CalcMatrix(const FiniteElement &inner_fel,
                                   const BaseMappedIntegrationRule &bmir,
                                   BareSliceMatrix<double, ColMajor> mat,
                                   LocalHeap &lh) const
    {
        // cout << "in GradDiffOp CalcMatrix" << endl;
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
                ipts[0](j) -= eps();
                ipts[1] = ip;
                ipts[1](j) += eps();
                ipts[2] = ip;
                ipts[2](j) -= 2 * eps();
                ipts[3] = ip;
                ipts[3](j) += 2 * eps();

                IntegrationRule ir_j(4, ipts);
                MappedIntegrationRule<D, D, double> mir_j(ir_j, eltrans, lh);

                proxy->Evaluator()->CalcMatrix(inner_fel, mir_j, Trans(bbmat), lh);

                // cout << "bbmat = " << bbmat << endl;

                // dshape_ref.Col(j) = (1.0 / (12.0 * eps)) * (8.0 * bbmat.Cols(proxy_dim, 2 * proxy_dim).AsVector() - 8.0 * bbmat.Cols(0, proxy_dim).AsVector() - bbmat.Cols(3 * proxy_dim, 4 * proxy_dim).AsVector() + bbmat.Cols(2 * proxy_dim, 3 * proxy_dim).AsVector());
                dshape_ref.Col(j) = (1.0 / (12.0 * eps())) * (8.0 * bbmat.Cols(proxy_dim, 2 * proxy_dim).AsVector() - 8.0 * bbmat.Cols(0, proxy_dim).AsVector() - bbmat.Cols(3 * proxy_dim, 4 * proxy_dim).AsVector() + bbmat.Cols(2 * proxy_dim, 3 * proxy_dim).AsVector());
                // if (j == D - 1)
                //     cout << "dshape_ref = " << dshape_ref << endl;
            }
            dshape = dshape_ref * mir[i].GetJacobianInverse();
            // cout << "dshape = " << dshape << endl;

            for (auto k : Range(dshape.Height()))
                for (auto l : Range(dshape.Width()))
                    mat(k * dshape.Width() + l, i) = dshape(k, l);
            // mat.Col(i) = dshape.AsVector();
        }

        // int nd_u = inner_fel.GetNDof();

        // FlatMatrix<double> shape_ul(nd_u, proxy_dim, lh);
        // FlatMatrix<double> shape_ur(nd_u, proxy_dim, lh);
        // FlatMatrix<double> shape_ull(nd_u, proxy_dim, lh);
        // FlatMatrix<double> shape_urr(nd_u, proxy_dim, lh);
        // FlatMatrix<double> dshape_u_ref(nd_u, proxy_dim, lh);
        // FlatMatrix<double> bmatu(nd_u, D * proxy_dim, lh);

        // FlatMatrix<double> dshape_u_ref_comp(nd_u, D, lh);
        // FlatMatrix<double> dshape_u(nd_u, D, lh); //(shape_ul);///saves "reserved lh-memory"

        // for (size_t i = 0; i < mir.Size(); i++)
        // {
        //     bmatu = 0;
        //     const IntegrationPoint &ip = mir[i].IP(); // volume_ir[i];
        //     const ElementTransformation &eltrans = mir[i].GetTransformation();
        //     for (int j = 0; j < D; j++) // d / dxj
        //     {
        //         IntegrationPoint ipl(ip);
        //         ipl(j) -= eps;
        //         IntegrationPoint ipr(ip);
        //         ipr(j) += eps;
        //         IntegrationPoint ipll(ip);
        //         ipll(j) -= 2 * eps;
        //         IntegrationPoint iprr(ip);
        //         iprr(j) += 2 * eps;

        //         MappedIntegrationPoint<D, D> mipl(ipl, eltrans);
        //         MappedIntegrationPoint<D, D> mipr(ipr, eltrans);
        //         MappedIntegrationPoint<D, D> mipll(ipll, eltrans);
        //         MappedIntegrationPoint<D, D> miprr(iprr, eltrans);

        //         proxy->Evaluator()->CalcMatrix(inner_fel, mipl, Trans(shape_ul), lh);
        //         proxy->Evaluator()->CalcMatrix(inner_fel, mipr, Trans(shape_ur), lh);
        //         proxy->Evaluator()->CalcMatrix(inner_fel, mipll, Trans(shape_ull), lh);
        //         proxy->Evaluator()->CalcMatrix(inner_fel, miprr, Trans(shape_urr), lh);

        //         dshape_u_ref = (1.0 / (12.0 * eps)) * (8.0 * shape_ur - 8.0 * shape_ul - shape_urr + shape_ull);
        //         for (int l = 0; l < proxy_dim; l++)
        //             bmatu.Col(j * proxy_dim + l) = dshape_u_ref.Col(l);
        //         // if (j == D - 1)
        //         //     cout << "bmatu = " << bmatu << endl;
        //     }

        //     for (int j = 0; j < proxy_dim; j++)
        //     {
        //         for (int k = 0; k < nd_u; k++)
        //             for (int l = 0; l < D; l++)
        //                 dshape_u_ref_comp(k, l) = bmatu(k, l * proxy_dim + j);

        //         dshape_u = dshape_u_ref_comp * mir[i].GetJacobianInverse();

        //         for (int k = 0; k < nd_u; k++)
        //             for (int l = 0; l < D; l++)
        //                 bmatu(k, l * proxy_dim + j) = dshape_u(k, l);
        //     }
        //     cout << "bmatu final = " << bmatu << endl;
        // }
        // cout << "mat = " << mat << endl;
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
