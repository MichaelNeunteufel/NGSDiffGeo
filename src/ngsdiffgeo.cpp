#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"
#include "kforms.hpp"

PYBIND11_MODULE(ngsdiffgeo, m)
{
    ExportRiemannianManifold(m);
    ExportTensorFields(m);
    ExportGradCF(m);
    ExportKForms(m);
}
