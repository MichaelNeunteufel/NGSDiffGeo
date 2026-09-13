#ifndef NGSDIFFGEO_KFORMS_BINDINGS_HPP
#define NGSDIFFGEO_KFORMS_BINDINGS_HPP

#include <python_ngstd.hpp>

void ExportKForms(py::module m);

namespace ngfem
{
    /// Register the concrete dense alternation node without exposing its type
    void ExportAlternationBinding(py::module m);
}

#endif
