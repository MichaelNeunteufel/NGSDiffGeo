#include "riemannian_manifold.hpp"

#include <coefficient_stdmath.hpp>

namespace ngfem
{

    RiemannianManifold::RiemannianManifold(shared_ptr<CoefficientFunction> _g)
        : g(_g)
    {
        if (_g->Dimensions().Size() != 2 || _g->Dimensions()[0] != _g->Dimensions()[1])
            throw Exception("In RMF: input must be a square matrix");

        dim = _g->Dimensions()[0];
        g_inv = InverseCF(g);

        // Volume forms on VOL, BND, BBND, and BBBND
        vol[VOL] = sqrt(DeterminantCF(g));
        auto one_cf = make_shared<ConstantCoefficientFunction>(1.0);
        auto tv = TangentialVectorCF(dim, false);
        vol[BND] = one_cf;
        vol[BBND] = one_cf;
        vol[BBBND] = one_cf;
        if (dim == 2)
        {
            vol[BND] = sqrt(InnerProduct(g * tv, tv));
        }
        else if (dim == 3)
        {
            auto nv = NormalVectorCF(dim);
            vol[BND] = sqrt(InnerProduct(CofactorCF(g) * nv, nv));
            vol[BBND] = sqrt(InnerProduct(g * tv, tv));
        }
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetMetric() const
    {
        return g;
    }

    shared_ptr<CoefficientFunction> RiemannianManifold::GetVolumeForm(VorB vb) const
    {
        return vol[int(vb)];
    }
}

void ExportRiemannianManifold(py::module m)
{
    using namespace ngfem;

    py::class_<RiemannianManifold, shared_ptr<RiemannianManifold>>(m, "RiemannianManifold")
        .def(py::init<shared_ptr<CoefficientFunction>>(), "constructor", py::arg("metric"))
        .def("GetMetric", &RiemannianManifold::GetMetric, "return the metric")
        .def("VolumeForm", &RiemannianManifold::GetVolumeForm, "return the volume form of given dimension", py::arg("vb"))
        .def_property_readonly("dx", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetVolumeForm(VOL); }, "return the volume form")
        .def_property_readonly("ds", [](shared_ptr<RiemannianManifold> self)
                               { return self->GetVolumeForm(BND); }, "return the boundary volume form");
}
