#include "kforms.hpp"
#include "kforms_bindings.hpp"
#include "kforms_detail.hpp"
#include "kforms_diagnostics.hpp"
#include "coefficient_grad.hpp"
#include "riemannian_manifold.hpp"
#include "symbolic_expression.hpp"

#include <algorithm>
#include <array>
#include <core/register_archive.hpp>
#include <limits>
#include <numeric>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ngfem
{
// Shared validation precedes every implementation fragment
#include "kforms/kforms_common.inc"

// Representation and semantic-node ownership
#include "kforms/kforms_storage_facades.inc"
#include "kforms/kforms_symbolic_operations.inc"

// Public wrappers and exact dense/reference machinery
#include "kforms/kforms_wrappers.inc"
#include "kforms/kforms_hodge_helpers.inc"
#include "kforms/kforms_dense_nodes.inc"

// Compact factories and metric/algebra consumers. These remain in one
// translation unit so their hot lookup/evaluation helpers stay inlineable
#include "kforms/kforms_compact_facade.inc"
#include "kforms/kforms_metric.inc"
#include "kforms/kforms_trace_hodge.inc"
#include "kforms/kforms_compact_algebra.inc"
#include "kforms/kforms_compact_exterior.inc"
#include "kforms/kforms_algebra_operations.inc"
#include "kforms/kforms_exterior_factories.inc"

// Stable public dispatch is deliberately last.
#include "kforms/kforms_dispatch.inc"
} // namespace ngfem
