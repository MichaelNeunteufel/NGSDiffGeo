"""
Python-side wrappers for ngsdiffgeo pybind classes.
"""

from __future__ import annotations

import importlib
import numbers
import ngsolve

_cpp = importlib.import_module(".ngsdiffgeo", __package__)

# ---- references to the C++/pybind classes ----
_CPP_ScalarField = _cpp.ScalarField
_CPP_OneForm = _cpp.OneForm
_CPP_TwoForm = _cpp.TwoForm
_CPP_ThreeForm = _cpp.ThreeForm
_CPP_KForm = _cpp.KForm
_CPP_DoubleForm = _cpp.DoubleForm
_CPP_VectorField = _cpp.VectorField
_CPP_TensorField = _cpp.TensorField
_CPP_RiemannianManifold = _cpp.RiemannianManifold
_MAX_CONCRETE_FORM_RANK = int(_cpp._MAX_FORM_RANK)
_MAX_SPACE_DIM = int(_cpp._MAX_SPACE_DIM)


# ---------------- helpers ----------------


def _call_if_callable(x):
    return x() if callable(x) else x


def _unwrap_cf(obj):
    try:
        return obj.coef
    except Exception:
        return obj


def _infer_dim(obj):
    """Try to infer dimension. Returns int or None."""
    for attr in ("dim_space", "dim"):
        if hasattr(obj, attr):
            try:
                val = _call_if_callable(getattr(obj, attr))
                if isinstance(val, int) and val > 0:
                    return val
            except Exception:
                pass
    if hasattr(obj, "dims"):
        try:
            dims = obj.dims
            if hasattr(dims, "__len__") and len(dims) > 0 and isinstance(dims[0], int):
                return int(dims[0])
        except Exception:
            pass
    return None


def _tensorfield_dim(tf):
    if hasattr(tf, "dims"):
        try:
            dims = tf.dims
            if hasattr(dims, "__len__") and len(dims) > 0 and isinstance(dims[0], int):
                return int(dims[0])
        except Exception:
            pass
    if hasattr(tf, "dim"):
        try:
            total = int(tf.dim)
            root = int(total**0.5 + 0.5)
            if root * root == total and root > 0:
                return root
        except Exception:
            pass
    return None


def _tensorfield_covariance(tf):
    return getattr(tf, "covariant_indices", "")


def _is_doubleform_like(obj):
    if isinstance(obj, (DoubleForm, _CPP_DoubleForm)):
        return True
    if isinstance(obj, TensorField):
        return _tensorfield_covariance(obj) == "11"
    return False


def _is_scalarfield_like(obj):
    if isinstance(obj, ScalarField):
        return True
    if isinstance(obj, ngsolve.CoefficientFunction) and getattr(obj, "dim", None) == 1:
        return True
    return False


def _is_raw_scalar_coefficient(obj):
    return (
        isinstance(obj, ngsolve.CoefficientFunction)
        and getattr(obj, "dim", None) == 1
        and not isinstance(
            obj,
            (
                _CPP_KForm,
                _CPP_DoubleForm,
                _CPP_VectorField,
                _CPP_TensorField,
            ),
        )
    )


def _raise_doubleform_scalar_op_error(op, other):
    raise TypeError(
        f"DoubleForm '{op}' only supports scalar operands. "
        "Use dg.Wedge(...) for double-form products, or multiply coefficient matrices via '.coef' "
        "(for example 'A * df.coef * B') if you intended plain matrix multiplication. "
        f"Received operand of type {type(other)}."
    )


def _as_doubleform_like(obj, *, dim=None):
    if isinstance(obj, (DoubleForm, _CPP_DoubleForm)):
        return obj
    if isinstance(obj, TensorField) and _tensorfield_covariance(obj) == "11":
        inferred = _tensorfield_dim(obj)
        if dim is None or (inferred is not None and dim != inferred):
            dim = inferred
        if dim is None:
            raise TypeError(
                "Wedge: dim must be provided or inferable for covariant (2,0) tensor"
            )
        return DoubleForm(obj, p=1, q=1, dim=dim)
    raise TypeError("Wedge: expected DoubleForm or covariant (2,0) TensorField, but received type {}".format(type(obj)))


def _vb_dimension(M, vb):
    if vb == ngsolve.VOL:
        return M.dim
    if vb == ngsolve.BND:
        return M.dim - 1
    if vb == ngsolve.BBND:
        return M.dim - 2
    raise ValueError("vb must be VOL, BND, or BBND")


def _parse_slot(slot):
    if isinstance(slot, str):
        s = slot.lower()
        if s == "left":
            return 0
        if s == "right":
            return 1
        if s in ("both", "-1"):
            return -1
    if isinstance(slot, numbers.Integral):
        val = int(slot)
        if val in (-1, 0, 1):
            return val
    raise ValueError("slot must be -1/0/1 or 'left'/'right'/'both'")


def _parse_proj_mode(mode):
    if isinstance(mode, str):
        s = mode.lower()
        if s in ("f", "tangent", "tan", "1"):
            return 1
        if s in ("n", "normal", "2"):
            return 2
        if s in ("e", "edge", "3"):
            return 3
        if s in ("m", "conormal", "4"):
            return 4
        if s in ("none", "0"):
            return 0
        raise ValueError(
            "ProjectDoubleForm: mode must be 'F'/'tangent', 'n'/'normal', 'E'/'edge', 'm'/'conormal', or 'none'"
        )
    if isinstance(mode, numbers.Integral):
        if int(mode) in (0, 1, 2, 3, 4):
            return int(mode)
        raise ValueError(
            "ProjectDoubleForm: mode must be 0 (none), 1 (F/tangent), 2 (n/normal), 3 (E/edge), or 4 (m/conormal)"
        )
    raise ValueError("ProjectDoubleForm: mode must be string or int")


def _projected_doubleform_degrees(p, q, left_mode, right_mode):
    p_out = int(p)
    q_out = int(q)
    if left_mode in (2, 4):
        p_out -= 1
    if right_mode in (2, 4):
        q_out -= 1
    return p_out, q_out


def _zero_tensor_cf(rank, dim):
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank == 0:
        return 0
    dims = tuple(int(dim) for _ in range(rank))
    size = int(dim) ** int(rank)
    return ngsolve.CF(tuple(0 for _ in range(size)), dims=dims)


def _validate_materialized_zero_shape(rank, dim, name):
    if rank > _MAX_CONCRETE_FORM_RANK:
        raise ValueError(
            f"cannot materialize {name} above the supported concrete rank "
            f"{_MAX_CONCRETE_FORM_RANK}"
        )
    if not (1 <= dim <= _MAX_SPACE_DIM or (rank == 0 and dim == 0)):
        raise ValueError(f"cannot materialize {name} with dim {dim}")


def _star_requires_formal(a, n, slot_id, *, double=False):
    if isinstance(a, FormalZeroDoubleForm) or isinstance(a, FormalZeroKForm):
        return True

    if double or isinstance(a, (DoubleForm, _CPP_DoubleForm)):
        degs = _doubleform_degrees(a)
        if degs is None:
            return False
        p, q, _ = degs
        if slot_id == -1:
            return p > n or q > n
        if slot_id == 0:
            return p > n
        return q > n

    deg = _kform_degree(a)
    if deg is None:
        return False
    return deg[0] > n


def _doubleform_degrees(obj):
    if isinstance(obj, FormalZeroDoubleForm):
        return obj.degree_left, obj.degree_right, obj.dim_space
    if isinstance(obj, (DoubleForm, _CPP_DoubleForm)):
        return int(obj.degree_left), int(obj.degree_right), int(obj.dim_space)
    if _is_scalarfield_like(obj):
        dim = _infer_dim(obj)
        if dim is None:
            raise TypeError("double-form scalar input requires inferable dim")
        return 0, 0, int(dim)
    return None


def _is_zero_doubleform_like(obj):
    if isinstance(obj, FormalZeroDoubleForm):
        return True
    if isinstance(obj, (DoubleForm, _CPP_DoubleForm)):
        return bool(getattr(obj, "is_zero", False))
    return False


def _raise_doubleform_degree_error(op):
    raise TypeError(
        f"cannot {op} double-forms of different left/right degree or dimension"
    )


def _zero_neutral_doubleform_result(left, right, op):
    if _same_doubleform_degree(left, right):
        return None
    if not (_is_zero_doubleform_like(left) or _is_zero_doubleform_like(right)):
        _raise_doubleform_degree_error(op)

    if _is_zero_doubleform_like(left) and _is_zero_doubleform_like(right):
        return left
    if op == "add":
        return right if _is_zero_doubleform_like(left) else left
    if op == "subtract":
        return -right if _is_zero_doubleform_like(left) else left
    raise ValueError("unsupported double-form operation")


def _kform_degree(obj):
    if isinstance(obj, FormalZeroKForm):
        return int(obj.degree), int(obj.dim_space)
    if isinstance(obj, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm, _CPP_KForm)):
        return int(obj.degree), int(obj.dim_space)
    if _is_scalarfield_like(obj):
        dim = _infer_dim(obj)
        if dim is None:
            raise TypeError("scalar input requires inferable dim")
        return 0, int(dim)
    return None


def _same_kform_degree(a, b):
    da = _kform_degree(a)
    db = _kform_degree(b)
    if da is None or db is None:
        return False
    return da[0] == db[0] and da[1] == db[1]


def _same_doubleform_degree(a, b):
    da = _doubleform_degrees(a)
    db = _doubleform_degrees(b)
    if da is None or db is None:
        return False
    return da == db


# ---------------- KForm factory + isinstance ----------------


class _KFormMeta(type):
    def __instancecheck__(cls, obj):
        # All pybind forms (ScalarField/OneForm/...) are subclasses of _CPP_KForm.
        return isinstance(obj, _CPP_KForm)


class KForm(metaclass=_KFormMeta):
    """
    Public Python 'KForm' wrapper/factory.

    - `isinstance(x, dg.KForm)` is True for any pybind k-form (including wrappers).
    - Calling `dg.KForm(cf, k=..., dim=...)` returns a typed wrapper instance.
    """

    def __new__(cls, cf, *args, k=None, dim=None, **kwargs):
        if k is None and len(args) >= 1:
            k = args[0]
        if dim is None and len(args) >= 2:
            dim = args[1]
        if k is None:
            raise TypeError("KForm: missing required argument k")
        k = int(k)
        if dim is None:
            dim = _infer_dim(cf)
        if dim is None and k == 0:
            dim = 0
        elif dim is None:
            raise TypeError("KForm: dim must be provided or inferable")
        return as_kform(cf, k=k, dim=dim)


# ----------------  wrappers  ----------------


class ScalarField(_CPP_ScalarField):
    def __init__(self, cf, *, dim=-1):
        _CPP_ScalarField.__init__(self, cf, dim=dim)

    def _wrap(self, cf, k=0):
        return as_kform(cf, k=k, dim=self.dim_space)

    def __add__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return self
        if is_formal_zero_doubleform(other):
            if not _same_doubleform_degree(self, other):
                raise TypeError(
                    "cannot add double-forms of different left/right degree or dimension"
                )
            return as_doubleform(self, p=0, q=0, dim=self.dim_space)
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return self
        if is_formal_zero_doubleform(other):
            if not _same_doubleform_degree(self, other):
                raise TypeError(
                    "cannot subtract double-forms of different left/right degree or dimension"
                )
            return as_doubleform(self, p=0, q=0, dim=self.dim_space)
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        if isinstance(other, KForm) or (
            isinstance(other, ngsolve.CoefficientFunction) and other.dim == 1
        ):
            k = 0
            if hasattr(other, "degree"):
                k = other.degree
            return self._wrap(self.coef * _unwrap_cf(other), k=k)
        elif isinstance(other, (int, float, complex)):
            # keep scalar * ScalarField results typed as ScalarField
            return self._wrap(self.coef * other)
        elif isinstance(other, (VectorField, TensorField)):
            return as_tensorfield(
                self.coef * _unwrap_cf(other),
                covariant_indices=_tensorfield_covariance(other),
            )
        return _CPP_ScalarField.__mul__(self, other)

    # if isinstance(
    #         other, (KForm, TensorField, VectorField, ngsolve.CoefficientFunction)
    #     ):
    #         k = 0
    #         if hasattr(other, "degree"):
    #             k = other.degree
    #         return self._wrap(_CPP_ScalarField.__mul__(self, other), k=k)
    #     return _CPP_ScalarField.__mul__(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))


class OneForm(_CPP_OneForm):
    def __init__(self, cf):
        _CPP_OneForm.__init__(self, cf)

    def _wrap(self, cf):
        return as_kform(cf, k=1, dim=self.dim_space)

    def __add__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))


class TwoForm(_CPP_TwoForm):
    def __init__(self, cf, *, dim=-1):
        _CPP_TwoForm.__init__(self, cf, dim=dim)

    def _wrap(self, cf):
        return as_kform(cf, k=2, dim=self.dim_space)

    def __add__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))


class ThreeForm(_CPP_ThreeForm):
    def __init__(self, cf, *, dim=-1):
        _CPP_ThreeForm.__init__(self, cf, dim=dim)

    def _wrap(self, cf):
        return as_kform(cf, k=3, dim=self.dim_space)

    def __add__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))


class GenericKForm(_CPP_KForm):
    def __init__(self, cf, *, k, dim):
        _CPP_KForm.__init__(self, cf, k=int(k), dim=int(dim))

    def _wrap(self, cf):
        return as_kform(cf, k=self.degree, dim=self.dim_space)

    def __add__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return self
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))


class DoubleForm(_CPP_DoubleForm):
    def __init__(self, cf, *, p, q, dim):
        _CPP_DoubleForm.__init__(self, cf, p=int(p), q=int(q), dim=int(dim))

    def _wrap(self, cf):
        return as_doubleform(
            cf,
            p=self.degree_left,
            q=self.degree_right,
            dim=self.dim_space,
        )

    def _is_overflow_zero_degree(self):
        return (
            self.degree_left > self.dim_space
            or self.degree_right > self.dim_space
        )

    def _formal_zero(self, *, reason):
        return FormalZeroDoubleForm(
            self.degree_left,
            self.degree_right,
            self.dim_space,
            reason=reason,
        )

    def _can_add_scalar(self, other):
        if not _is_scalarfield_like(other):
            return False
        if self.degree_left != 0 or self.degree_right != 0:
            return False
        if _is_raw_scalar_coefficient(other):
            return True
        other_dim = _infer_dim(other)
        return other_dim is None or other_dim == self.dim_space

    def __add__(self, other):
        if is_formal_zero_doubleform(other):
            neutral = _zero_neutral_doubleform_result(self, other, "add")
            if neutral is not None:
                return neutral
            return self._formal_zero(reason="add") if self._is_overflow_zero_degree() else self
        if self._can_add_scalar(other):
            return self._wrap(self.coef + _unwrap_cf(other))
        other_degrees = _doubleform_degrees(other)
        if other_degrees is not None:
            neutral = _zero_neutral_doubleform_result(self, other, "add")
            if neutral is not None:
                return neutral
            if self._is_overflow_zero_degree():
                return self._formal_zero(reason="add")
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_doubleform(other):
            neutral = _zero_neutral_doubleform_result(self, other, "subtract")
            if neutral is not None:
                return neutral
            return self._formal_zero(reason="sub") if self._is_overflow_zero_degree() else self
        if self._can_add_scalar(other):
            return self._wrap(self.coef - _unwrap_cf(other))
        other_degrees = _doubleform_degrees(other)
        if other_degrees is not None:
            neutral = _zero_neutral_doubleform_result(self, other, "subtract")
            if neutral is not None:
                return neutral
            if self._is_overflow_zero_degree():
                return self._formal_zero(reason="sub")
        return self._wrap(self.coef - _unwrap_cf(other))

    def __rsub__(self, other):
        if self._can_add_scalar(other):
            return self._wrap(_unwrap_cf(other) - self.coef)
        return NotImplemented

    def __neg__(self):
        if self._is_overflow_zero_degree():
            return self._formal_zero(reason="neg")
        return self._wrap(-self.coef)

    def __mul__(self, other):
        if not (
            isinstance(other, numbers.Number) or _is_scalarfield_like(other)
        ):
            _raise_doubleform_scalar_op_error("*", other)
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        if not (
            isinstance(other, numbers.Number) or _is_scalarfield_like(other)
        ):
            _raise_doubleform_scalar_op_error("*", other)
        return self.__mul__(other)

    def __truediv__(self, other):
        if not (
            isinstance(other, numbers.Number) or _is_scalarfield_like(other)
        ):
            _raise_doubleform_scalar_op_error("/", other)
        return self._wrap(self.coef / _unwrap_cf(other))

    def __pow__(self, power):
        return WedgePower(self, power)


class FormalZeroBase:
    is_formal_zero = True

    def __init__(self, *, reason=None):
        self.reason = reason

    def __repr__(self):
        attrs = []
        for key in ("degree", "degree_left", "degree_right", "dim_space", "reason"):
            if hasattr(self, key):
                attrs.append(f"{key}={getattr(self, key)!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"


class FormalZeroKForm(FormalZeroBase):
    def __init__(self, k, dim, *, reason=None):
        super().__init__(reason=reason)
        self.degree = int(k)
        self.dim_space = int(dim)

    def __add__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return self
        if _kform_degree(other) is not None:
            if not _same_kform_degree(self, other):
                raise TypeError("cannot add k-forms of different degree/dimension")
            return other
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_kform(other):
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return self
        if _kform_degree(other) is not None:
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return (-1) * other
        return NotImplemented

    def __rsub__(self, other):
        if _kform_degree(other) is not None:
            if not _same_kform_degree(self, other):
                raise TypeError("cannot subtract k-forms of different degree/dimension")
            return other
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) or _is_scalarfield_like(other):
            return self
        _raise_doubleform_scalar_op_error("*", other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self

    def InnerProduct(self, other):
        _validate_inner_product_compatibility(self, other, forms=False)
        return ngsolve.CF(0)

    def Norm(self):
        return ngsolve.CF(0)


class FormalZeroDoubleForm(FormalZeroBase):
    def __init__(self, p, q, dim, *, reason=None):
        super().__init__(reason=reason)
        self.degree_left = int(p)
        self.degree_right = int(q)
        self.dim_space = int(dim)

    def __add__(self, other):
        if is_formal_zero_doubleform(other):
            neutral = _zero_neutral_doubleform_result(self, other, "add")
            if neutral is not None:
                return neutral
            return self
        if _doubleform_degrees(other) is not None:
            neutral = _zero_neutral_doubleform_result(self, other, "add")
            if neutral is not None:
                return neutral
            return other
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_formal_zero_doubleform(other):
            neutral = _zero_neutral_doubleform_result(self, other, "subtract")
            if neutral is not None:
                return neutral
            return self
        if _doubleform_degrees(other) is not None:
            neutral = _zero_neutral_doubleform_result(self, other, "subtract")
            if neutral is not None:
                return neutral
            return (-1) * other
        return NotImplemented

    def __rsub__(self, other):
        if _doubleform_degrees(other) is not None:
            neutral = _zero_neutral_doubleform_result(other, self, "subtract")
            if neutral is not None:
                return neutral
            return other
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) or _is_scalarfield_like(other):
            return self
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self

    def InnerProduct(self, other):
        _validate_inner_product_compatibility(self, other, forms=True)
        return ngsolve.CF(0)

    def Norm(self):
        return ngsolve.CF(0)

    @property
    def trans(self):
        return FormalZeroDoubleForm(
            self.degree_right,
            self.degree_left,
            self.dim_space,
            reason="trans",
        )


# ---------------- as_* functions ----------------


def as_scalarfield(cf, *, dim=-1):
    if isinstance(cf, ScalarField):
        if dim is None or int(dim) < 1 or int(cf.dim_space) == int(dim):
            return cf
    if isinstance(cf, _CPP_ScalarField):
        return ScalarField(cf, dim=dim)
    return ScalarField(cf, dim=dim)


def as_oneform(cf):
    if isinstance(cf, OneForm):
        return cf
    if isinstance(cf, _CPP_OneForm):
        return OneForm(cf)
    return OneForm(cf)


def as_twoform(cf, *, dim):
    if isinstance(cf, TwoForm):
        return cf
    if isinstance(cf, _CPP_TwoForm):
        return TwoForm(cf, dim=dim)
    return TwoForm(cf, dim=dim)


def as_threeform(cf, *, dim):
    if isinstance(cf, ThreeForm):
        return cf
    if isinstance(cf, _CPP_ThreeForm):
        return ThreeForm(cf, dim=dim)
    return ThreeForm(cf, dim=dim)


def as_kform(cf, *, k, dim=None):
    k = int(k)
    if dim is None:
        dim = _infer_dim(cf)
    if dim is None and k == 0:
        dim = 0
    elif dim is None:
        raise TypeError("as_kform: dim must be provided or inferable")

    dim = int(dim)
    if isinstance(cf, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm)):
        if int(cf.degree) == k and int(cf.dim_space) == dim:
            return cf

    if k == 0:
        return as_scalarfield(cf, dim=dim)
    if k == 1:
        return as_oneform(cf)
    if k == 2:
        return as_twoform(cf, dim=dim)
    if k == 3:
        return as_threeform(cf, dim=dim)

    return GenericKForm(cf, k=k, dim=dim)


def as_doubleform(cf, *, p=None, q=None, dim=None):
    if isinstance(cf, DoubleForm):
        return cf

    if p is None and hasattr(cf, "degree_left"):
        try:
            p = int(cf.degree_left)
        except Exception:
            p = None
    if q is None and hasattr(cf, "degree_right"):
        try:
            q = int(cf.degree_right)
        except Exception:
            q = None

    if p is None or q is None:
        raise TypeError("as_doubleform: p and q must be provided or inferable")

    if dim is None:
        dim = _infer_dim(cf)
    if dim is None:
        raise TypeError("as_doubleform: dim must be provided or inferable")

    return DoubleForm(cf, p=p, q=q, dim=dim)


def is_formal_zero(obj):
    return bool(getattr(obj, "is_formal_zero", False))


def is_formal_zero_kform(obj):
    return isinstance(obj, FormalZeroKForm)


def is_formal_zero_doubleform(obj):
    return isinstance(obj, FormalZeroDoubleForm)


def materialize_zero(obj):
    if isinstance(obj, FormalZeroKForm):
        if obj.degree < 0:
            raise ValueError("cannot materialize FormalZeroKForm with negative degree")
        _validate_materialized_zero_shape(
            obj.degree, obj.dim_space, "FormalZeroKForm"
        )
        return as_kform(_zero_tensor_cf(obj.degree, obj.dim_space), k=obj.degree, dim=obj.dim_space)
    if isinstance(obj, FormalZeroDoubleForm):
        if obj.degree_left < 0 or obj.degree_right < 0:
            raise ValueError(
                "cannot materialize FormalZeroDoubleForm with negative degree"
            )
        rank = obj.degree_left + obj.degree_right
        _validate_materialized_zero_shape(
            rank, obj.dim_space, "FormalZeroDoubleForm"
        )
        return as_doubleform(
            _zero_tensor_cf(rank, obj.dim_space),
            p=obj.degree_left,
            q=obj.degree_right,
            dim=obj.dim_space,
        )
    return obj


def _contract_slot_formal(M, tf, vf, slot="left"):
    slot_id = _parse_slot(slot)
    if slot_id not in (0, 1):
        raise ValueError("ContractSlot: slot must be 'left' or 'right'")

    if isinstance(tf, FormalZeroDoubleForm):
        p = tf.degree_left - (1 if slot_id == 0 else 0)
        q = tf.degree_right - (1 if slot_id == 1 else 0)
        return FormalZeroDoubleForm(p, q, tf.dim_space, reason="ContractSlot")

    if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        raise TypeError(
            "ContractSlot expects a DoubleForm, but received type {}".format(
                type(tf)
            )
        )

    if (slot_id == 0 and tf.degree_left == 0) or (
        slot_id == 1 and tf.degree_right == 0
    ):
        return FormalZeroDoubleForm(
            tf.degree_left - (1 if slot_id == 0 else 0),
            tf.degree_right - (1 if slot_id == 1 else 0),
            tf.dim_space,
            reason="ContractSlot",
        )

    vf_wrapped = as_vectorfield(vf)
    out = _CPP_RiemannianManifold.ContractSlot(M, tf, vf_wrapped, slot_id)
    return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)


def _project_doubleform_formal(
    M,
    tf,
    left="none",
    right="none",
    normal=None,
    conormal=None,
    project_remaining=True,
):
    left_mode = _parse_proj_mode(left)
    right_mode = _parse_proj_mode(right)

    if isinstance(tf, FormalZeroDoubleForm):
        p_out, q_out = _projected_doubleform_degrees(
            tf.degree_left, tf.degree_right, left_mode, right_mode
        )
        return FormalZeroDoubleForm(
            p_out, q_out, tf.dim_space, reason="ProjectDoubleForm"
        )

    if isinstance(tf, (ScalarField, _CPP_ScalarField)):
        p_out, q_out = _projected_doubleform_degrees(0, 0, left_mode, right_mode)
        if left_mode in (2, 4) or right_mode in (2, 4):
            return FormalZeroDoubleForm(
                p_out, q_out, M.dim, reason="ProjectDoubleForm"
            )
        return as_scalarfield(tf, dim=M.dim)

    if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        raise TypeError(
            "ProjectDoubleForm expects a DoubleForm, but received type {}".format(
                type(tf)
            )
        )

    p_out, q_out = _projected_doubleform_degrees(
        tf.degree_left, tf.degree_right, left_mode, right_mode
    )
    if (left_mode in (2, 4) and tf.degree_left == 0) or (
        right_mode in (2, 4) and tf.degree_right == 0
    ):
        return FormalZeroDoubleForm(
            p_out, q_out, tf.dim_space, reason="ProjectDoubleForm"
        )

    out = _CPP_RiemannianManifold.ProjectDoubleForm(
        M, tf, left, right, normal, conormal, project_remaining
    )
    return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)


def _star_formal(a, M, vb=ngsolve.VOL, double=False, slot="both"):
    slot_id = _parse_slot(slot)
    n = _vb_dimension(M, vb)

    if isinstance(a, FormalZeroDoubleForm):
        if slot_id == -1:
            return FormalZeroDoubleForm(
                n - a.degree_left,
                n - a.degree_right,
                a.dim_space,
                reason="star",
            )
        if slot_id == 0:
            return FormalZeroDoubleForm(
                n - a.degree_left, a.degree_right, a.dim_space, reason="star"
            )
        return FormalZeroDoubleForm(
            a.degree_left, n - a.degree_right, a.dim_space, reason="star"
        )

    if isinstance(a, FormalZeroKForm):
        return FormalZeroKForm(n - a.degree, a.dim_space, reason="star")

    if _star_requires_formal(a, n, slot_id, double=double):
        if double or isinstance(a, (DoubleForm, _CPP_DoubleForm)):
            degs = _doubleform_degrees(a)
            if degs is not None:
                p, q, dim = degs
                if slot_id == -1:
                    return FormalZeroDoubleForm(
                        n - p, n - q, dim, reason="star"
                    )
                if slot_id == 0:
                    return FormalZeroDoubleForm(n - p, q, dim, reason="star")
                return FormalZeroDoubleForm(p, n - q, dim, reason="star")

        deg = _kform_degree(a)
        if deg is not None:
            k, dim = deg
            return FormalZeroKForm(n - k, dim, reason="star")

    return star(a, M, vb=vb, double=double, slot=slot)


def _inv_star_formal(a, M, vb=ngsolve.VOL, double=False, slot="both"):
    slot_id = _parse_slot(slot)
    n = _vb_dimension(M, vb)

    if isinstance(a, FormalZeroDoubleForm):
        if slot_id == -1:
            return FormalZeroDoubleForm(
                n - a.degree_left,
                n - a.degree_right,
                a.dim_space,
                reason="inv_star",
            )
        if slot_id == 0:
            return FormalZeroDoubleForm(
                n - a.degree_left,
                a.degree_right,
                a.dim_space,
                reason="inv_star",
            )
        return FormalZeroDoubleForm(
            a.degree_left,
            n - a.degree_right,
            a.dim_space,
            reason="inv_star",
        )

    if isinstance(a, FormalZeroKForm):
        return FormalZeroKForm(n - a.degree, a.dim_space, reason="inv_star")

    if _star_requires_formal(a, n, slot_id, double=double):
        if double or isinstance(a, (DoubleForm, _CPP_DoubleForm)):
            degs = _doubleform_degrees(a)
            if degs is not None:
                p, q, dim = degs
                if slot_id == -1:
                    return FormalZeroDoubleForm(
                        n - p, n - q, dim, reason="inv_star"
                    )
                if slot_id == 0:
                    return FormalZeroDoubleForm(
                        n - p, q, dim, reason="inv_star"
                    )
                return FormalZeroDoubleForm(p, n - q, dim, reason="inv_star")

        deg = _kform_degree(a)
        if deg is not None:
            k, dim = deg
            return FormalZeroKForm(n - k, dim, reason="inv_star")

    return inv_star(a, M, vb=vb, double=double, slot=slot)


def _wedge_formal(a, b):
    dim = _infer_dim(a) or _infer_dim(b)

    if _doubleform_degrees(a) is None and _is_doubleform_like(a):
        a = _as_doubleform_like(a, dim=dim)
    if _doubleform_degrees(b) is None and _is_doubleform_like(b):
        b = _as_doubleform_like(b, dim=dim)

    da = _doubleform_degrees(a)
    db = _doubleform_degrees(b)
    if da is not None and db is not None:
        p1, q1, dim1 = da
        p2, q2, dim2 = db
        dim = dim1 if dim1 is not None else dim2
        if dim1 is not None and dim2 is not None and dim1 != dim2:
            raise ValueError("Wedge: double-form dimensions must match")
        if is_formal_zero(a) or is_formal_zero(b):
            return FormalZeroDoubleForm(p1 + p2, q1 + q2, dim, reason="Wedge")
        return Wedge(a, b)

    ka = _kform_degree(a)
    kb = _kform_degree(b)
    if ka is not None and kb is not None:
        k1, dim1 = ka
        k2, dim2 = kb
        dim = dim1 if dim1 is not None else dim2
        if dim1 is not None and dim2 is not None and dim1 != dim2:
            raise ValueError("Wedge: k-form dimensions must match")
        if is_formal_zero(a) or is_formal_zero(b):
            return FormalZeroKForm(k1 + k2, dim, reason="Wedge")
        return Wedge(a, b)

    raise TypeError(
        "Wedge expects k-form-like or double-form-like operands, but received {} and {}".format(
            type(a), type(b)
        )
    )


def compare_formal_zero(a, b):
    return is_formal_zero(a) and is_formal_zero(b)


def _inner_product_mode(a, b, forms):
    if not forms:
        return "kform"

    a_df = isinstance(a, (FormalZeroDoubleForm, DoubleForm, _CPP_DoubleForm))
    b_df = isinstance(b, (FormalZeroDoubleForm, DoubleForm, _CPP_DoubleForm))
    a_sf = _is_scalarfield_like(a)
    b_sf = _is_scalarfield_like(b)
    if a_df or b_df or (a_sf and b_sf):
        return "doubleform"
    return "kform"


def _validate_inner_product_compatibility(a, b, *, forms=False):
    mode = _inner_product_mode(a, b, forms)
    if mode == "doubleform":
        da = _doubleform_degrees(a)
        db = _doubleform_degrees(b)
        if da is None or db is None:
            raise TypeError(
                "InnerProduct expects double-form compatible operands when forms=True, but received {} and {}".format(
                    type(a), type(b)
                )
            )
        if da != db:
            raise TypeError(
                "InnerProduct requires matching double-form degrees and dimension, but received {} and {}".format(
                    da, db
                )
            )
        return

    ka = _kform_degree(a)
    kb = _kform_degree(b)
    if ka is None or kb is None:
        if is_formal_zero(a) or is_formal_zero(b):
            raise TypeError(
                "InnerProduct expects form-compatible operands with formal zero input, but received {} and {}".format(
                    type(a), type(b)
                )
            )
        return
    if ka != kb:
        raise TypeError(
            "InnerProduct requires matching form degrees and dimension, but received {} and {}".format(
                ka, kb
            )
        )


def _prepare_inner_product_operands(a, b, *, dim, forms=False):
    _validate_inner_product_compatibility(a, b, forms=forms)
    if forms and _inner_product_mode(a, b, forms) == "doubleform":
        if _is_scalarfield_like(a):
            a = DoubleForm(a, p=0, q=0, dim=dim)
        if _is_scalarfield_like(b):
            b = DoubleForm(b, p=0, q=0, dim=dim)
    return a, b


def _normalize_trace_sigma_sigma(sigma, *, dim):
    try:
        return _as_doubleform_like(sigma, dim=dim)
    except TypeError:
        return DoubleForm(sigma, p=1, q=1, dim=dim)


def _d_formal(a):
    if isinstance(a, FormalZeroKForm):
        return FormalZeroKForm(a.degree + 1, a.dim_space, reason="d")
    raise TypeError(
        "d expects a FormalZeroKForm, but received type {}".format(type(a))
    )


def _delta_formal(a, M):
    if isinstance(a, FormalZeroKForm):
        return FormalZeroKForm(a.degree - 1, a.dim_space, reason="delta")
    raise TypeError(
        "delta expects a FormalZeroKForm, but received type {}".format(type(a))
    )


def _parse_compile_inner(compile_inner):
    if compile_inner is None or compile_inner is False:
        return False
    if isinstance(compile_inner, str) and compile_inner == "graph":
        return True
    raise ValueError("compile_inner must be False, None, or 'graph'")


def _d_cov_formal(
    M, tf, slot="left", vb=ngsolve.VOL, *, compile_inner_graph=False
):
    slot_id = _parse_slot(slot)
    if slot_id not in (0, 1):
        raise ValueError("d_cov: slot must be 'left' or 'right'")

    if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        out = _CPP_RiemannianManifold.d_cov(
            M, tf, slot, vb, compile_inner_graph
        )
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)

    if isinstance(tf, FormalZeroDoubleForm):
        return FormalZeroDoubleForm(
            tf.degree_left + (1 if slot_id == 0 else 0),
            tf.degree_right + (1 if slot_id == 1 else 0),
            tf.dim_space,
            reason="d_cov",
        )

    return M.d_cov(
        tf,
        slot=slot,
        vb=vb,
        compile_inner="graph" if compile_inner_graph else False,
    )


def _delta_cov_formal(
    M, tf, slot="left", vb=ngsolve.VOL, *, compile_inner_graph=False
):
    slot_id = _parse_slot(slot)
    if slot_id not in (0, 1):
        raise ValueError("delta_cov: slot must be 'left' or 'right'")

    if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        if (slot_id == 0 and tf.degree_left == 0) or (
            slot_id == 1 and tf.degree_right == 0
        ):
            return FormalZeroDoubleForm(
                tf.degree_left - (1 if slot_id == 0 else 0),
                tf.degree_right - (1 if slot_id == 1 else 0),
                tf.dim_space,
                reason="delta_cov",
            )
        out = _CPP_RiemannianManifold.delta_cov(
            M, tf, slot, vb, compile_inner_graph
        )
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)

    if isinstance(tf, FormalZeroDoubleForm):
        return FormalZeroDoubleForm(
            tf.degree_left - (1 if slot_id == 0 else 0),
            tf.degree_right - (1 if slot_id == 1 else 0),
            tf.dim_space,
            reason="delta_cov",
        )

    return M.delta_cov(
        tf,
        slot=slot,
        vb=vb,
        compile_inner="graph" if compile_inner_graph else False,
    )


def _covdiv_formal(M, tf, slot="left", vb=ngsolve.VOL):
    slot_id = _parse_slot(slot)
    if slot_id not in (0, 1):
        raise ValueError("CovDiv: slot must be 'left' or 'right'")

    if isinstance(tf, FormalZeroDoubleForm):
        return _delta_cov_formal(M, tf, slot=slot, vb=vb)
    if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        if (slot_id == 0 and tf.degree_left == 0) or (
            slot_id == 1 and tf.degree_right == 0
        ):
            return FormalZeroDoubleForm(
                tf.degree_left - (1 if slot_id == 0 else 0),
                tf.degree_right - (1 if slot_id == 1 else 0),
                tf.dim_space,
                reason="CovDiv",
            )
        out = _CPP_RiemannianManifold.CovDiv(M, tf, slot, vb)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
    raise TypeError(
        "CovDiv expects a DoubleForm, but received type {}".format(
            type(tf)
        )
    )


def _trace_formal(M, tf, vb=None, index1=0, index2=1, l=None):
    if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        if index1 != 0 or index2 != 1:
            raise ValueError(
                "Trace only supports the double-form trace convention; use l for repeated traces"
            )
        if l is None:
            l = 1
        if not isinstance(l, numbers.Integral):
            raise TypeError("Trace: l must be an integer")
        l = int(l)
        p_out = tf.degree_left - l
        q_out = tf.degree_right - l
        if p_out < 0 or q_out < 0:
            return FormalZeroDoubleForm(p_out, q_out, tf.dim_space, reason="Trace")
        if vb is None:
            out = _CPP_RiemannianManifold.Trace(M, tf, l)
        else:
            out = _CPP_RiemannianManifold.Trace(M, tf, l, vb)
        if isinstance(out, _CPP_DoubleForm):
            return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
        return as_scalarfield(out, dim=M.dim)

    if isinstance(tf, FormalZeroDoubleForm):
        if index1 != 0 or index2 != 1:
            raise ValueError(
                "Trace only supports the double-form trace convention; use l for repeated traces"
            )
        if l is None:
            l = 1
        if not isinstance(l, numbers.Integral):
            raise TypeError("Trace: l must be an integer")
        l = int(l)
        return FormalZeroDoubleForm(
            tf.degree_left - l,
            tf.degree_right - l,
            tf.dim_space,
            reason="Trace",
        )

    if isinstance(tf, FormalZeroKForm):
        raise TypeError("Trace does not support FormalZeroKForm inputs")

    raise TypeError(
        "Trace expects a FormalZeroDoubleForm, but received type {}".format(
            type(tf)
        )
    )


def _trace_sigma_formal(M, tf, sigma, vb=ngsolve.VOL):
    sigma_df = _normalize_trace_sigma_sigma(sigma, dim=M.dim)

    if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        if tf.degree_left == 0 or tf.degree_right == 0:
            return FormalZeroDoubleForm(
                tf.degree_left - 1,
                tf.degree_right - 1,
                tf.dim_space,
                reason="TraceSigma",
            )
        out = _CPP_RiemannianManifold.TraceSigma(M, tf, sigma_df, vb)
        if isinstance(out, _CPP_DoubleForm):
            return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
        return as_scalarfield(out, dim=M.dim)

    if isinstance(tf, FormalZeroDoubleForm):
        return FormalZeroDoubleForm(
            tf.degree_left - 1,
            tf.degree_right - 1,
            tf.dim_space,
            reason="TraceSigma",
        )
    raise TypeError(
        "TraceSigma expects a FormalZeroDoubleForm, but received type {}".format(
            type(tf)
        )
    )


def _s_formal(M, tf, vb=None):
    if vb is None:
        vb = ngsolve.VOL

    if vb not in (ngsolve.VOL, ngsolve.BND):
        raise ValueError("s: vb must be VOL or BND")

    if isinstance(tf, FormalZeroDoubleForm):
        return FormalZeroDoubleForm(
            tf.degree_left + 1,
            tf.degree_right - 1,
            tf.dim_space,
            reason="s",
        )

    if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
        n = _vb_dimension(M, vb)
        p_out = tf.degree_left + 1
        q_out = tf.degree_right - 1
        if q_out < 0 or p_out > n or q_out > n:
            return FormalZeroDoubleForm(p_out, q_out, tf.dim_space, reason="s")
        out = _CPP_RiemannianManifold.s(M, tf, vb)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)

    raise TypeError(
        "s expects a DoubleForm or FormalZeroDoubleForm, but received type {}".format(
            type(tf)
        )
    )


def _contraction_formal(M, tf, vf, slot=0):
    if slot is None:
        slot = 0

    tf_is_vector = isinstance(tf, VectorField)
    vf_is_vector = isinstance(vf, VectorField)
    tf_is_zero_kform = isinstance(tf, FormalZeroKForm)
    vf_is_zero_kform = isinstance(vf, FormalZeroKForm)

    if tf_is_vector and isinstance(vf, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm)):
        k = int(vf.degree)
        if k == 0:
            return FormalZeroKForm(-1, _infer_dim(vf), reason="Contraction")
    if vf_is_vector and isinstance(tf, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm)):
        k = int(tf.degree)
        if k == 0:
            return FormalZeroKForm(-1, _infer_dim(tf), reason="Contraction")

    if tf_is_vector and vf_is_zero_kform:
        return FormalZeroKForm(vf.degree - 1, vf.dim_space, reason="Contraction")
    if vf_is_vector and tf_is_zero_kform:
        return FormalZeroKForm(tf.degree - 1, tf.dim_space, reason="Contraction")

    raise TypeError(
        "Contraction expects exactly one VectorField and one FormalZeroKForm, but received {} and {}".format(
            type(tf), type(vf)
        )
    )


# ---------------- VectorField / TensorField ----------------


class VectorField(_CPP_VectorField):
    def __init__(self, cf):
        _CPP_VectorField.__init__(self, cf)

    def _wrap(self, cf):
        return as_vectorfield(cf)

    def __add__(self, other):
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))


def as_vectorfield(cf):
    if isinstance(cf, VectorField):
        return cf
    if isinstance(cf, _CPP_VectorField):
        return VectorField(cf)
    return VectorField(cf)


class TensorField(_CPP_TensorField):
    def __init__(self, cf, covariant_indices):
        _CPP_TensorField.__init__(self, cf, covariant_indices=covariant_indices)

    def _wrap(self, cf):
        return as_tensorfield(cf, covariant_indices=self.covariant_indices)

    def __add__(self, other):
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(self.coef - _unwrap_cf(other))

    def __neg__(self):
        return self._wrap(-self.coef)

    def __mul__(self, other):
        return self._wrap(self.coef * _unwrap_cf(other))

    def __rmul__(self, other):
        # if other is a number or a ngsolve CoefficientFunction with dim=1:
        if isinstance(other, (int, float, ngsolve.CoefficientFunction)) and (
            not hasattr(other, "dim") or other.dim == 1
        ):
            return self._wrap(self.coef * _unwrap_cf(other))
            # return self.__mul__(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self._wrap(self.coef / _unwrap_cf(other))

    def __pow__(self, power):
        if self.covariant_indices != "11":
            return NotImplemented
        return WedgePower(self, power)


def as_tensorfield(cf, *, covariant_indices=None, dim=-1):
    if isinstance(cf, TensorField):
        return cf
    if covariant_indices is None:
        try:
            covariant_indices = cf.covariant_indices
        except Exception:
            covariant_indices = ""
    # if isinstance(cf, _CPP_TensorField):
    #     return TensorField(cf, covariant_indices=covariant_indices)
    # return TensorField(cf, covariant_indices=covariant_indices)
    if covariant_indices == "":
        if dim is None or dim < 1:
            dim = _infer_dim(cf)
        if dim is None:
            dim = 0
        return ScalarField(cf, dim=dim)
    elif covariant_indices == "0":
        return VectorField(cf)
    elif covariant_indices == "1":
        return OneForm(cf)
    return TensorField(cf, covariant_indices=covariant_indices)


# ---------------- wrapping of exported C++ functions ----------------


def Wedge(a, b):
    if is_formal_zero(a) or is_formal_zero(b):
        return _wedge_formal(a, b)
    if _is_scalarfield_like(a) and _is_scalarfield_like(b):
        return as_scalarfield(a * b, dim=_infer_dim(a) or _infer_dim(b))
    if _is_scalarfield_like(a) and _is_doubleform_like(b):
        db = _as_doubleform_like(b, dim=_infer_dim(b))
        return as_doubleform(
            db * a, p=db.degree_left, q=db.degree_right, dim=db.dim_space
        )
    if _is_scalarfield_like(b) and _is_doubleform_like(a):
        da = _as_doubleform_like(a, dim=_infer_dim(a))
        return as_doubleform(
            da * b, p=da.degree_left, q=da.degree_right, dim=da.dim_space
        )
    if _is_scalarfield_like(a):
        other_dim = _infer_dim(b)
        if other_dim is not None:
            a = as_scalarfield(a, dim=other_dim)
        out = _cpp.Wedge(a, b)
        return as_kform(out, k=out.degree, dim=out.dim_space)
    if _is_scalarfield_like(b):
        other_dim = _infer_dim(a)
        if other_dim is not None:
            b = as_scalarfield(b, dim=other_dim)
        out = _cpp.Wedge(a, b)
        return as_kform(out, k=out.degree, dim=out.dim_space)
    if _is_doubleform_like(a) or _is_doubleform_like(b):
        dim = _infer_dim(a) or _infer_dim(b)
        da = _as_doubleform_like(a, dim=dim)
        db = _as_doubleform_like(b, dim=dim)
        if da.degree_left + db.degree_left > dim or da.degree_right + db.degree_right > dim:
            return FormalZeroDoubleForm(
                da.degree_left + db.degree_left,
                da.degree_right + db.degree_right,
                dim,
                reason="Wedge",
            )
        out = _cpp.Wedge(da, db)
        return as_doubleform(
            out, p=out.degree_left, q=out.degree_right, dim=out.dim_space
        )
    out = _cpp.Wedge(a, b)
    if isinstance(out, _CPP_DoubleForm):
        return as_doubleform(
            out, p=out.degree_left, q=out.degree_right, dim=out.dim_space
        )
    return as_kform(out, k=out.degree, dim=out.dim_space)


def WedgePower(df, l):
    if not isinstance(l, numbers.Integral):
        raise TypeError("WedgePower: l must be a non-negative integer")
    if l < 0:
        raise ValueError("WedgePower: l must be non-negative")

    dim = _infer_dim(df)
    if isinstance(df, TensorField):
        tf_dim = _tensorfield_dim(df)
        if tf_dim is not None:
            dim = tf_dim
    if l == 0:
        if dim is None:
            raise TypeError("WedgePower: dim must be provided or inferable for l=0")
        return as_scalarfield(1, dim=dim)

    df = _as_doubleform_like(df, dim=dim)
    if df.degree_left != 1 or df.degree_right != 1:
        raise ValueError("WedgePower: expected a (1,1) double form")

    out = df
    for _ in range(1, int(l)):
        out = Wedge(out, df)
    return out


def Sym(a):
    df = _as_doubleform_like(a, dim=_infer_dim(a))
    if df.degree_left != df.degree_right:
        raise ValueError("Sym: expected a (k,k) double form")

    tr = df.trans
    out = 0.5 * (df.coef + tr.coef)
    return as_doubleform(out, p=df.degree_left, q=df.degree_right, dim=df.dim_space)


def d(a):
    if is_formal_zero(a):
        return _d_formal(a)
    out = _cpp.d(a)
    return as_kform(out, k=out.degree, dim=out.dim_space)


def star(a, M, vb=ngsolve.VOL, double=False, slot="both"):
    slot_id = _parse_slot(slot)
    n = _vb_dimension(M, vb)
    if is_formal_zero(a) or _star_requires_formal(a, n, slot_id, double=double):
        return _star_formal(a, M, vb=vb, double=double, slot=slot)
    if double or isinstance(a, (DoubleForm, _CPP_DoubleForm)):
        if not isinstance(a, (DoubleForm, _CPP_DoubleForm)):
            a = DoubleForm(a, p=0, q=0, dim=M.dim)
        out = _cpp.star(a, M, vb, slot)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
    out = _cpp.star(a, M, vb)
    if isinstance(out, _CPP_DoubleForm):
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
    return as_kform(out, k=out.degree, dim=M.dim)


def inv_star(a, M, vb=ngsolve.VOL, double=False, slot="both"):
    slot_id = _parse_slot(slot)
    n = _vb_dimension(M, vb)
    if is_formal_zero(a) or _star_requires_formal(a, n, slot_id, double=double):
        return _inv_star_formal(a, M, vb=vb, double=double, slot=slot)
    if double or isinstance(a, (DoubleForm, _CPP_DoubleForm)):
        if not isinstance(a, (DoubleForm, _CPP_DoubleForm)):
            a = DoubleForm(a, p=0, q=0, dim=M.dim)
        out = _cpp.inv_star(a, M, vb, slot)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
    out = _cpp.inv_star(a, M, vb)
    if isinstance(out, _CPP_DoubleForm):
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=M.dim)
    return as_kform(out, k=out.degree, dim=M.dim)


def slot_inner_product(a, M, vb=ngsolve.VOL, forms=True):
    if is_formal_zero(a):
        if forms:
            _validate_inner_product_compatibility(a, a, forms=True)
        return as_scalarfield(0, dim=M.dim)
    if _is_scalarfield_like(a):
        return as_scalarfield(a, dim=M.dim)
    out = _cpp.slot_inner_product(a, M, vb, forms)
    return as_scalarfield(out, dim=M.dim)


def delta(a, M):
    if is_formal_zero(a):
        return _delta_formal(a, M)
    deg = _kform_degree(a)
    if deg is not None and deg[0] == 0:
        return FormalZeroKForm(-1, deg[1], reason="delta")
    out = _cpp.delta(a, M)
    return as_kform(out, k=out.degree, dim=M.dim)


# ---------------- RiemannianManifold wrapper ----------------


class RiemannianManifold(_CPP_RiemannianManifold):
    def __init__(self, metric, normal_sign=1.0, change_riemann_sign=False):
        super().__init__(metric, normal_sign, change_riemann_sign)
        self._property_cache = {}

    def _cached_property(self, name, factory):
        cache = self._property_cache
        if name not in cache:
            cache[name] = factory()
        return cache[name]

    # properties
    @property
    def G(self):
        return self._cached_property(
            "G",
            lambda: as_tensorfield(_CPP_RiemannianManifold.G.__get__(self)),
        )

    @property
    def G_F(self):
        return self._cached_property(
            "G_F",
            lambda: as_tensorfield(_CPP_RiemannianManifold.G_F.__get__(self)),
        )

    @property
    def G_F_inv(self):
        return self._cached_property(
            "G_F_inv",
            lambda: as_tensorfield(_CPP_RiemannianManifold.G_F_inv.__get__(self)),
        )

    @property
    def G_E(self):
        return self._cached_property(
            "G_E",
            lambda: as_tensorfield(_CPP_RiemannianManifold.G_E.__get__(self)),
        )

    @property
    def G_E_inv(self):
        return self._cached_property(
            "G_E_inv",
            lambda: as_tensorfield(_CPP_RiemannianManifold.G_E_inv.__get__(self)),
        )

    @property
    def G_inv(self):
        return self._cached_property(
            "G_inv",
            lambda: as_tensorfield(_CPP_RiemannianManifold.G_inv.__get__(self)),
        )

    @property
    def normal(self):
        return self._cached_property(
            "normal",
            lambda: as_vectorfield(_CPP_RiemannianManifold.normal.__get__(self)),
        )

    @property
    def tangent(self):
        return self._cached_property(
            "tangent",
            lambda: as_tensorfield(_CPP_RiemannianManifold.tangent.__get__(self)),
        )

    def edge_conormal(self, i):
        return self._cached_property(
            f"edge_conormal_{i}",
            lambda: as_vectorfield(_CPP_RiemannianManifold.EdgeConormal(self, i)),
        )

    def edge_normal(self, i):
        return self._cached_property(
            f"edge_normal_{i}",
            lambda: as_vectorfield(_CPP_RiemannianManifold.EdgeNormal(self, i)),
        )

    @property
    def edge_conormals(self):
        return self._cached_property(
            "edge_conormals",
            lambda: [self.edge_conormal(0), self.edge_conormal(1)],
        )

    @property
    def edge_normals(self):
        return self._cached_property(
            "edge_normals",
            lambda: [self.edge_normal(0), self.edge_normal(1)],
        )

    @property
    def G_deriv(self):
        return self._cached_property(
            "G_deriv",
            lambda: _CPP_RiemannianManifold.G_deriv.__get__(self),
        )

    @property
    def Riemann(self):
        return self._cached_property(
            "Riemann",
            lambda: as_doubleform(
                _CPP_RiemannianManifold.Riemann.__get__(self),
                p=2,
                q=2,
                dim=self.dim,
            ),
        )

    @property
    def Curvature(self):
        return self._cached_property(
            "Curvature",
            lambda: as_tensorfield(
                _CPP_RiemannianManifold.Curvature.__get__(self),
                dim=self.dim,
            ),
        )

    @property
    def Gauss(self):
        return self._cached_property(
            "Gauss",
            lambda: as_scalarfield(
                _CPP_RiemannianManifold.Gauss.__get__(self),
                dim=self.dim,
            ),
        )

    @property
    def Ricci(self):
        return self._cached_property(
            "Ricci",
            lambda: as_doubleform(
                _CPP_RiemannianManifold.Ricci.__get__(self),
                p=1,
                q=1,
                dim=self.dim,
            ),
        )

    @property
    def Einstein(self):
        return self._cached_property(
            "Einstein",
            lambda: as_doubleform(
                _CPP_RiemannianManifold.Einstein.__get__(self),
                p=1,
                q=1,
                dim=self.dim,
            ),
        )

    @property
    def Scalar(self):
        return self._cached_property(
            "Scalar",
            lambda: as_scalarfield(
                _CPP_RiemannianManifold.Scalar.__get__(self),
                dim=self.dim,
            ),
        )

    @property
    def SFF(self):
        return self._cached_property(
            "SFF",
            lambda: as_doubleform(
                _CPP_RiemannianManifold.SFF.__get__(self),
                p=1,
                q=1,
                dim=self.dim,
            ),
        )

    def Raise(self, tf, index=0, vb=ngsolve.VOL):
        if isinstance(index, (list, tuple)):
            out = _CPP_RiemannianManifold.Raise(self, tf, list(index), vb)
        else:
            out = _CPP_RiemannianManifold.Raise(self, tf, index, vb)
        return as_tensorfield(out)

    def Lower(self, tf, index=0, vb=ngsolve.VOL):
        if isinstance(index, (list, tuple)):
            out = _CPP_RiemannianManifold.Lower(self, tf, list(index), vb)
        else:
            out = _CPP_RiemannianManifold.Lower(self, tf, index, vb)
        return as_tensorfield(out)

    @property
    def GeodesicCurvature(self):
        return self._cached_property(
            "GeodesicCurvature",
            lambda: as_scalarfield(
                _CPP_RiemannianManifold.GeodesicCurvature.__get__(self),
                dim=self.dim,
            ),
        )

    @property
    def MeanCurvature(self):
        return self._cached_property(
            "MeanCurvature",
            lambda: as_scalarfield(
                _CPP_RiemannianManifold.MeanCurvature.__get__(self),
                dim=self.dim,
            ),
        )

    @property
    def AngleDefect(self):
        return self._cached_property(
            "AngleDefect",
            lambda: as_scalarfield(
                _CPP_RiemannianManifold.AngleDefect.__get__(self),
                dim=self.dim,
            ),
        )

    def KForm(self, cf, k):
        out = _CPP_RiemannianManifold.KForm(self, cf, k)
        return as_kform(out, k=k, dim=self.dim)

    def star(self, a, vb=ngsolve.VOL, double=False, slot="both"):
        return star(a, self, vb=vb, double=double, slot=slot)

    def inv_star(self, a, vb=ngsolve.VOL, double=False, slot="both"):
        return inv_star(a, self, vb=vb, double=double, slot=slot)

    def delta(self, a):
        if is_formal_zero(a):
            return _delta_formal(a, self)
        deg = _kform_degree(a)
        if deg is not None and deg[0] == 0:
            return FormalZeroKForm(-1, deg[1], reason="delta")
        out = _CPP_RiemannianManifold.delta(self, a)
        return as_kform(out, k=out.degree, dim=self.dim)

    def d_cov(self, tf, slot="left", vb=ngsolve.VOL, compile_inner=False):
        """Apply the exterior covariant derivative in one double-form slot.

        ``compile_inner="graph"`` compiles the input expression graph shared by
        the gradient and connection terms. The default keeps the original path.
        """
        compile_inner_graph = _parse_compile_inner(compile_inner)
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            return _d_cov_formal(
                self,
                tf,
                slot=slot,
                vb=vb,
                compile_inner_graph=compile_inner_graph,
            )
        out = _CPP_RiemannianManifold.d_cov(
            self, tf, slot, vb, compile_inner_graph
        )
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def delta_cov(self, tf, slot="left", vb=ngsolve.VOL, compile_inner=False):
        """Apply the covariant codifferential in one double-form slot.

        ``compile_inner`` has the same opt-in graph mode as :meth:`d_cov`.
        """
        compile_inner_graph = _parse_compile_inner(compile_inner)
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            return _delta_cov_formal(
                self,
                tf,
                slot=slot,
                vb=vb,
                compile_inner_graph=compile_inner_graph,
            )
        out = _CPP_RiemannianManifold.delta_cov(
            self, tf, slot, vb, compile_inner_graph
        )
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def ProjectDoubleForm(
        self,
        tf,
        left="none",
        right="none",
        normal=None,
        conormal=None,
        project_remaining=True,
    ):
        if is_formal_zero(tf) or isinstance(
            tf, (ScalarField, _CPP_ScalarField, DoubleForm, _CPP_DoubleForm)
        ):
            return _project_doubleform_formal(
                self,
                tf,
                left=left,
                right=right,
                normal=normal,
                conormal=conormal,
                project_remaining=project_remaining,
            )

        if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            raise TypeError(
                "ProjectDoubleForm expects a DoubleForm, but received type {}".format(
                    type(tf)
                )
            )
        return _project_doubleform_formal(
            self,
            tf,
            left=left,
            right=right,
            normal=normal,
            conormal=conormal,
            project_remaining=project_remaining,
        )

    def ProjectTensor(self, tf, mode="none"):
        tf_wrapped = as_tensorfield(tf, dim=self.dim)
        out = _CPP_RiemannianManifold.ProjectTensor(self, tf_wrapped, mode)
        mode_str = str(mode).lower()

        if isinstance(
            tf,
            (
                ScalarField,
                OneForm,
                TwoForm,
                ThreeForm,
                GenericKForm,
                _CPP_ScalarField,
                _CPP_OneForm,
                _CPP_TwoForm,
                _CPP_ThreeForm,
                _CPP_KForm,
            ),
        ):
            k_in = getattr(tf, "degree", None)
            if k_in is None:
                return as_tensorfield(out, dim=self.dim)
            k_out = k_in - 1 if mode_str in ("n", "normal", "2") else k_in
            return as_kform(out, k=k_out, dim=self.dim)

        if isinstance(tf, (VectorField, _CPP_VectorField)):
            if mode_str in ("n", "normal", "2"):
                return as_scalarfield(out, dim=self.dim)
            return as_vectorfield(out)

        if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            p_in = getattr(tf, "degree_left", None)
            q_in = getattr(tf, "degree_right", None)
            if p_in is None or q_in is None:
                return as_tensorfield(out, dim=self.dim)
            if mode_str in ("n", "normal", "2"):
                p_out, q_out = p_in, q_in
                if p_in > 0:
                    p_out = p_in - 1
                elif q_in > 0:
                    q_out = q_in - 1
                return as_doubleform(out, p=p_out, q=q_out, dim=self.dim)
            return as_doubleform(out, p=p_in, q=q_in, dim=self.dim)

        return as_tensorfield(out, dim=self.dim)

    def ContractSlot(self, tf, vf, slot="left"):
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            return _contract_slot_formal(self, tf, vf, slot=slot)
        if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            raise TypeError("ContractSlot expects a DoubleForm, but received type {}".format(type(tf)))
        vf_wrapped = as_vectorfield(vf)
        out = _CPP_RiemannianManifold.ContractSlot(self, tf, vf_wrapped, slot)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def InnerProduct(self, tf1, tf2, vb=None, forms=False):
        tf1, tf2 = _prepare_inner_product_operands(
            tf1, tf2, dim=self.dim, forms=forms
        )
        if is_formal_zero(tf1) or is_formal_zero(tf2):
            return as_scalarfield(0, dim=self.dim)

        if vb is None:
            out = _CPP_RiemannianManifold.InnerProduct(self, tf1, tf2, forms=forms)
        else:
            out = _CPP_RiemannianManifold.InnerProduct(self, tf1, tf2, vb, forms)
        return as_scalarfield(out, dim=self.dim)

    def Cross(self, tf1, tf2):
        out = _CPP_RiemannianManifold.Cross(self, tf1, tf2)
        return as_vectorfield(out)

    def CovDeriv(self, tf, vb=None, compile_inner=False):
        """Apply the covariant derivative to a tensor field.

        Use ``compile_inner="graph"`` to opt into NGSolve graph compilation of
        the input used by the gradient and connection terms.
        """
        compile_inner_graph = _parse_compile_inner(compile_inner)
        vb = ngsolve.VOL if vb is None else vb
        out = _CPP_RiemannianManifold.CovDeriv(
            self, tf, vb, compile_inner_graph
        )
        return as_tensorfield(out)

    def CovHesse(self, tf):
        out = _CPP_RiemannianManifold.CovHesse(self, tf)
        return as_tensorfield(out)

    def CovCurl(self, tf):
        out = _CPP_RiemannianManifold.CovCurl(self, tf)
        return as_tensorfield(out, dim=self.dim)

    def CovInc(self, tf, matrix=False):
        out = _CPP_RiemannianManifold.CovInc(self, tf, matrix)
        return as_tensorfield(out, dim=self.dim)

    def CovEin(self, tf):
        out = _CPP_RiemannianManifold.CovEin(self, tf)
        return as_tensorfield(out)

    def CovLaplace(self, tf):
        out = _CPP_RiemannianManifold.CovLaplace(self, tf)
        return as_tensorfield(out, dim=self.dim)

    def LichnerowiczLaplacian(self, tf):
        out = _CPP_RiemannianManifold.LichnerowiczLaplacian(self, tf)
        return as_tensorfield(out, dim=self.dim)

    def CovDef(self, tf):
        out = _CPP_RiemannianManifold.CovDef(self, tf)
        return as_tensorfield(out)

    def CovRot(self, tf):
        out = _CPP_RiemannianManifold.CovRot(self, tf)
        return as_tensorfield(out)

    def CovDiv(self, tf, slot="left", vb=None):
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            effective_vb = ngsolve.VOL if vb is None else vb
            return _covdiv_formal(self, tf, slot=slot, vb=effective_vb)

        # Backward compatibility for tensors: CovDiv(tf, vb=...) and CovDiv(tf, vb_positional)
        if vb is None:
            if slot != "left":
                vb = slot
            else:
                vb = ngsolve.VOL
        elif slot != "left":
            raise ValueError("CovDiv: 'slot' is only supported for DoubleForm inputs")

        out = _CPP_RiemannianManifold.CovDiv(self, tf, vb)
        return as_tensorfield(out, dim=self.dim)

    def Trace(self, tf, vb=None, index1=0, index2=1, l=None):
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            return _trace_formal(self, tf, vb=vb, index1=index1, index2=index2, l=l)

        if _is_scalarfield_like(tf):
            if l is not None:
                if l == 0:
                    return as_scalarfield(tf, dim=self.dim)
                raise ValueError(
                    "Trace: l is only supported for double-forms; use index1/index2 for tensor fields"
                )
            return as_scalarfield(0, dim=self.dim)

        if l is not None:
            if l == 0:
                return as_tensorfield(tf, dim=self.dim)
            raise ValueError(
                "Trace: l is only supported for double-forms; use index1/index2 for tensor fields"
            )

        if vb is None:
            out = _CPP_RiemannianManifold.Trace(self, tf, index1=index1, index2=index2)
        else:
            out = _CPP_RiemannianManifold.Trace(self, tf, vb, index1, index2)
        return as_tensorfield(out, dim=self.dim)

    def TraceSigma(self, tf, sigma, vb=ngsolve.VOL):
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            return _trace_sigma_formal(self, tf, sigma, vb=vb)
        if _is_scalarfield_like(tf):
            return as_scalarfield(0, dim=self.dim)
        if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            raise TypeError("TraceSigma expects a DoubleForm or ScalarField, but received type {}".format(type(tf)))
        sigma_df = _normalize_trace_sigma_sigma(sigma, dim=self.dim)
        out = _CPP_RiemannianManifold.TraceSigma(self, tf, sigma_df, vb)
        if isinstance(out, _CPP_DoubleForm):
            return as_doubleform(
                out, p=out.degree_left, q=out.degree_right, dim=self.dim
            )
        return as_scalarfield(out, dim=self.dim)

    def SlotInnerProduct(self, tf, vb=ngsolve.VOL, forms=True):
        if is_formal_zero(tf):
            return as_scalarfield(0, dim=self.dim)
        if _is_scalarfield_like(tf):
            return as_scalarfield(tf, dim=self.dim)
        out = _CPP_RiemannianManifold.SlotInnerProduct(self, tf, vb, forms)
        return as_scalarfield(out, dim=self.dim)

    def Contraction(self, tf, vf, slot=0):
        if is_formal_zero(tf) or is_formal_zero(vf):
            return _contraction_formal(self, tf, vf, slot=slot)

        tf_wrapped = as_tensorfield(tf)
        vf_wrapped = as_tensorfield(vf)
        if (
            isinstance(tf_wrapped, VectorField)
            and isinstance(vf_wrapped, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm))
            and int(vf_wrapped.degree) == 0
        ) or (
            isinstance(vf_wrapped, VectorField)
            and isinstance(tf_wrapped, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm))
            and int(tf_wrapped.degree) == 0
        ):
            return _contraction_formal(self, tf, vf, slot=slot)

        # Accept inputs where exactly one argument is a vector field; the other can be any tensor (including k-forms).

        if isinstance(tf_wrapped, VectorField) and not isinstance(
            vf_wrapped, VectorField
        ):
            tensor_arg, vector_arg = vf_wrapped, tf_wrapped
        elif isinstance(vf_wrapped, VectorField) and not isinstance(
            tf_wrapped, VectorField
        ):
            tensor_arg, vector_arg = tf_wrapped, vf_wrapped
        else:
            raise TypeError(
                f"Contraction expects exactly one vector field and one tensor field, but received {type(tf)} and {type(vf)}"
            )

        out = _CPP_RiemannianManifold.Contraction(self, tensor_arg, vector_arg, slot)

        # Preserve k-form typing/dimension when the tensor argument was a form.
        if isinstance(
            tensor_arg, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm)
        ):
            k_in = getattr(tensor_arg, "degree", None)
            if k_in is not None and k_in > 0:
                return as_kform(out, k=k_in - 1, dim=self.dim)
        return as_tensorfield(out, dim=self.dim)

    def Transpose(self, tf, index1=0, index2=1):
        out = _CPP_RiemannianManifold.Transpose(self, tf, index1, index2)
        return as_tensorfield(out)

    def S(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.S(self, tf)
        else:
            out = _CPP_RiemannianManifold.S(self, tf, vb)
        if isinstance(out, _CPP_DoubleForm):
            return as_doubleform(
                out, p=out.degree_left, q=out.degree_right, dim=self.dim
            )
        if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            if tf.degree_left == 1 and tf.degree_right == 1:
                return as_doubleform(out, p=1, q=1, dim=self.dim)
        return as_tensorfield(out)

    def s(self, tf, vb=None):
        if is_formal_zero(tf) or isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            return _s_formal(self, tf, vb=vb)
        raise TypeError(
            "s expects a DoubleForm or FormalZeroDoubleForm, but received type {}".format(
                type(tf)
            )
        )

    def J(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.J(self, tf)
        else:
            out = _CPP_RiemannianManifold.J(self, tf, vb)
        if isinstance(out, _CPP_DoubleForm):
            return as_doubleform(
                out, p=out.degree_left, q=out.degree_right, dim=self.dim
            )
        if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            if tf.degree_left == 1 and tf.degree_right == 1:
                return as_doubleform(out, p=1, q=1, dim=self.dim)
        return as_tensorfield(out)


__all__ = [
    "KForm",
    "GenericKForm",
    "DoubleForm",
    "FormalZeroBase",
    "FormalZeroKForm",
    "FormalZeroDoubleForm",
    "ScalarField",
    "OneForm",
    "TwoForm",
    "ThreeForm",
    "as_scalarfield",
    "as_oneform",
    "as_twoform",
    "as_threeform",
    "as_kform",
    "as_doubleform",
    "is_formal_zero",
    "is_formal_zero_kform",
    "is_formal_zero_doubleform",
    "materialize_zero",
    "VectorField",
    "TensorField",
    "as_vectorfield",
    "as_tensorfield",
    "Wedge",
    "Sym",
    "d",
    "star",
    "inv_star",
    "slot_inner_product",
    "delta",
    "compare_formal_zero",
    "RiemannianManifold",
]
