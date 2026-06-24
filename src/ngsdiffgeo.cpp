#include "riemannian_manifold.hpp"
#include "tensor_fields.hpp"
#include "coefficient_grad.hpp"
#include "kforms.hpp"

#include <coefficient.hpp>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <typeinfo>
#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace
{
    std::string Demangle(const char *name)
    {
#ifdef __GNUG__
        int status = 0;
        char *demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        std::string result = (status == 0 && demangled) ? demangled : name;
        free(demangled);
        return result;
#else
        return name;
#endif
    }

    py::dict CoefficientFunctionStats(std::shared_ptr<ngfem::CoefficientFunction> cf)
    {
        if (!cf)
            throw ngfem::Exception("CFStats: input must be non-null");

        size_t visits = 0;
        std::set<ngfem::CoefficientFunction *> unique_nodes;
        std::map<std::string, size_t> type_counts;

        cf->TraverseTree([&](ngfem::CoefficientFunction &node)
                         {
                             visits++;
                             unique_nodes.insert(&node);
                             type_counts[Demangle(typeid(node).name())]++;
                         });

        py::dict types;
        for (const auto &[name, count] : type_counts)
            types[py::str(name)] = py::int_(count);

        py::dict out;
        out["visits"] = py::int_(visits);
        out["unique_nodes"] = py::int_(unique_nodes.size());
        out["types"] = types;
        out["dim"] = py::int_(cf->Dimension());
        py::list dims;
        for (auto d : cf->Dimensions())
            dims.append(d);
        out["dims"] = dims;
        return out;
    }
}

PYBIND11_MODULE(ngsdiffgeo, m)
{
    m.def("CFStats", &CoefficientFunctionStats,
          "Return debug statistics for a CoefficientFunction expression tree. "
          "This traverses the tree only when called and has no effect on normal evaluation.",
          py::arg("cf"));
    ExportRiemannianManifold(m);
    ExportTensorFields(m);
    ExportGradCF(m);
    ExportKForms(m);
}
