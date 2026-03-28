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
    if hasattr(obj, "_dim") and isinstance(obj._dim, int) and obj._dim > 0:
        return obj._dim
    for attr in ("dim_space", "_dim", "dim"):
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


def _is_doubleform_like(obj):
    if isinstance(obj, (DoubleForm, _CPP_DoubleForm)):
        return True
    if isinstance(obj, TensorField):
        return getattr(obj, "_covariant_indices", "") == "11"
    return False


def _is_scalarfield_like(obj):
    if isinstance(obj, ScalarField):
        return True
    if isinstance(obj, ngsolve.CoefficientFunction) and getattr(obj, "dim", None) == 1:
        return True
    return False


def _as_doubleform_like(obj, *, dim=None):
    if isinstance(obj, (DoubleForm, _CPP_DoubleForm)):
        return obj
    if isinstance(obj, TensorField) and getattr(obj, "_covariant_indices", "") == "11":
        inferred = _tensorfield_dim(obj)
        if dim is None or (inferred is not None and dim != inferred):
            dim = inferred
        if dim is None:
            raise TypeError(
                "Wedge: dim must be provided or inferable for covariant (2,0) tensor"
            )
        return DoubleForm(obj, p=1, q=1, dim=dim)
    raise TypeError("Wedge: expected DoubleForm or covariant (2,0) TensorField")


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
        if dim is None:
            dim = _infer_dim(cf)
        if dim is None:
            raise TypeError("KForm: dim must be provided or inferable")
        return as_kform(cf, k=int(k), dim=dim)


# ----------------  wrappers  ----------------


class ScalarField(_CPP_ScalarField):
    def __init__(self, cf, *, dim=-1):
        _CPP_ScalarField.__init__(self, cf, dim=dim)
        self._k = 0
        self._dim = dim

    def _wrap(self, cf, k=0):
        return as_kform(cf, k=k, dim=self._dim)

    def __add__(self, other):
        return self._wrap(self.coef + _unwrap_cf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
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
                covariant_indices=other._covariant_indices,
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
        self._k = 1
        self._dim = cf.dim

    def _wrap(self, cf):
        return as_kform(cf, k=1, dim=self._dim)

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


class TwoForm(_CPP_TwoForm):
    def __init__(self, cf, *, dim=-1):
        _CPP_TwoForm.__init__(self, cf, dim=dim)
        self._k = 2
        self._dim = dim

    def _wrap(self, cf):
        return as_kform(cf, k=2, dim=self._dim)

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


class ThreeForm(_CPP_ThreeForm):
    def __init__(self, cf, *, dim=-1):
        _CPP_ThreeForm.__init__(self, cf, dim=dim)
        self._k = 3
        self._dim = dim

    def _wrap(self, cf):
        return as_kform(cf, k=3, dim=self._dim)

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


class GenericKForm(_CPP_KForm):
    def __init__(self, cf, *, k, dim):
        _CPP_KForm.__init__(self, cf, k=int(k), dim=int(dim))
        self._k = int(k)
        self._dim = int(dim)

    def _wrap(self, cf):
        return as_kform(cf, k=self._k, dim=self._dim)

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


class DoubleForm(_CPP_DoubleForm):
    def __init__(self, cf, *, p, q, dim):
        _CPP_DoubleForm.__init__(self, cf, p=int(p), q=int(q), dim=int(dim))
        self._p = int(p)
        self._q = int(q)
        self._dim = int(dim)

    def _wrap(self, cf):
        return as_doubleform(cf, p=self._p, q=self._q, dim=self._dim)

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

    def __pow__(self, power):
        return WedgePower(self, power)


# ---------------- as_* functions ----------------


def as_scalarfield(cf, *, dim=-1):
    if isinstance(cf, ScalarField):
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
    if isinstance(cf, (ScalarField, OneForm, TwoForm, ThreeForm, GenericKForm)):
        return cf

    if dim is None:
        dim = _infer_dim(cf)
    if dim is None:
        raise TypeError("as_kform: dim must be provided or inferable")

    if hasattr(cf, "_k") and hasattr(cf, "_dim") and cf._k == k and cf._dim == dim:
        return cf

    k = int(k)
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
        self._covariant_indices = covariant_indices

    def _wrap(self, cf):
        return as_tensorfield(cf, covariant_indices=self._covariant_indices)

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
        if self._covariant_indices != "11":
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
            raise TypeError(
                "as_tensorfield: dim must be provided or inferable for scalars"
            )
        return ScalarField(cf, dim=dim)
    elif covariant_indices == "0":
        return VectorField(cf)
    elif covariant_indices == "1":
        return OneForm(cf)
    return TensorField(cf, covariant_indices=covariant_indices)


# ---------------- wrapping of exported C++ functions ----------------


def Wedge(a, b):
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
        out = _cpp.Wedge(a, b)
        return as_kform(out, k=out.degree, dim=out.dim_space)
    if _is_scalarfield_like(b):
        out = _cpp.Wedge(a, b)
        return as_kform(out, k=out.degree, dim=out.dim_space)
    if _is_doubleform_like(a) or _is_doubleform_like(b):
        dim = _infer_dim(a) or _infer_dim(b)
        da = _as_doubleform_like(a, dim=dim)
        db = _as_doubleform_like(b, dim=dim)
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
    out = _cpp.d(a)
    return as_kform(out, k=out.degree, dim=out.dim_space)


def star(a, M, vb=ngsolve.VOL, double=False, slot="both"):
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
    if _is_scalarfield_like(a):
        return as_scalarfield(a, dim=M.dim)
    out = _cpp.slot_inner_product(a, M, vb, forms)
    return as_scalarfield(out, dim=M.dim)


def delta(a, M):
    out = _cpp.delta(a, M)
    return as_kform(out, k=out.degree, dim=M.dim)


# ---------------- RiemannianManifold wrapper ----------------


class RiemannianManifold(_CPP_RiemannianManifold):
    def __init__(self, metric, normal_sign=1.0, change_riemann_sign=False):
        super().__init__(metric, normal_sign, change_riemann_sign)

    # properties
    @property
    def G(self):
        out = _CPP_RiemannianManifold.G.__get__(self)
        return as_tensorfield(out)

    @property
    def G_F(self):
        out = _CPP_RiemannianManifold.G_F.__get__(self)
        return as_tensorfield(out)

    @property
    def G_F_inv(self):
        out = _CPP_RiemannianManifold.G_F_inv.__get__(self)
        return as_tensorfield(out)

    @property
    def G_E(self):
        out = _CPP_RiemannianManifold.G_E.__get__(self)
        return as_tensorfield(out)

    @property
    def G_E_inv(self):
        out = _CPP_RiemannianManifold.G_E_inv.__get__(self)
        return as_tensorfield(out)

    @property
    def G_inv(self):
        out = _CPP_RiemannianManifold.G_inv.__get__(self)
        return as_tensorfield(out)

    @property
    def normal(self):
        out = _CPP_RiemannianManifold.normal.__get__(self)
        return as_vectorfield(out)

    @property
    def tangent(self):
        out = _CPP_RiemannianManifold.tangent.__get__(self)
        return as_tensorfield(out)

    def edge_conormal(self, i):
        out = _CPP_RiemannianManifold.EdgeConormal(self, i)
        return as_vectorfield(out)

    def edge_normal(self, i):
        out = _CPP_RiemannianManifold.EdgeNormal(self, i)
        return as_vectorfield(out)

    @property
    def edge_conormals(self):
        return [self.edge_conormal(0), self.edge_conormal(1)]

    @property
    def edge_normals(self):
        return [self.edge_normal(0), self.edge_normal(1)]

    @property
    def G_deriv(self):
        return _CPP_RiemannianManifold.G_deriv.__get__(self)

    @property
    def Riemann(self):
        out = _CPP_RiemannianManifold.Riemann.__get__(self)
        return as_doubleform(out, p=2, q=2, dim=self.dim)

    @property
    def Curvature(self):
        out = _CPP_RiemannianManifold.Curvature.__get__(self)
        return as_tensorfield(out, dim=self.dim)

    @property
    def Gauss(self):
        out = _CPP_RiemannianManifold.Gauss.__get__(self)
        return as_scalarfield(out, dim=self.dim)

    @property
    def Ricci(self):
        out = _CPP_RiemannianManifold.Ricci.__get__(self)
        return as_doubleform(out, p=1, q=1, dim=self.dim)

    @property
    def Einstein(self):
        out = _CPP_RiemannianManifold.Einstein.__get__(self)
        return as_doubleform(out, p=1, q=1, dim=self.dim)

    @property
    def Scalar(self):
        out = _CPP_RiemannianManifold.Scalar.__get__(self)
        return as_scalarfield(out, dim=self.dim)

    @property
    def SFF(self):
        out = _CPP_RiemannianManifold.SFF.__get__(self)
        return as_doubleform(out, p=1, q=1, dim=self.dim)

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
        out = _CPP_RiemannianManifold.GeodesicCurvature.__get__(self)
        return as_scalarfield(out, dim=self.dim)

    @property
    def MeanCurvature(self):
        out = _CPP_RiemannianManifold.MeanCurvature.__get__(self)
        return as_scalarfield(out, dim=self.dim)

    def KForm(self, cf, k):
        out = _CPP_RiemannianManifold.KForm(self, cf, k)
        return as_kform(out, k=k, dim=self.dim)

    def star(self, a, vb=ngsolve.VOL, double=False, slot="both"):
        return star(a, self, vb=vb, double=double, slot=slot)

    def inv_star(self, a, vb=ngsolve.VOL, double=False, slot="both"):
        return inv_star(a, self, vb=vb, double=double, slot=slot)

    def delta(self, a):
        out = _CPP_RiemannianManifold.delta(self, a)
        return as_kform(out, k=out.degree, dim=self.dim)

    def d_cov(self, tf, slot="left", vb=ngsolve.VOL):
        out = _CPP_RiemannianManifold.d_cov(self, tf, slot, vb)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def delta_cov(self, tf, slot="left", vb=ngsolve.VOL):
        out = _CPP_RiemannianManifold.delta_cov(self, tf, slot, vb)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def ProjectDoubleForm(
        self, tf, left="none", right="none", normal=None, conormal=None
    ):
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
                    "ProjectDoubleForm: mode must be 'F'/'tangent', 'n'/'normal', "
                    "'E'/'edge', 'm'/'conormal', or 'none'"
                )
            if isinstance(mode, int):
                if mode in (0, 1, 2, 3, 4):
                    return mode
                raise ValueError(
                    "ProjectDoubleForm: mode must be 0 (none), 1 (F/tangent), "
                    "2 (n/normal), 3 (E/edge), or 4 (m/conormal)"
                )
            raise ValueError("ProjectDoubleForm: mode must be string or int")

        left_mode = _parse_proj_mode(left)
        right_mode = _parse_proj_mode(right)

        if isinstance(tf, (ScalarField, _CPP_ScalarField)):
            if left_mode in (2, 4) or right_mode in (2, 4):
                return as_scalarfield(0, dim=self.dim)
            return as_scalarfield(tf, dim=self.dim)

        if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            raise TypeError("ProjectDoubleForm expects a DoubleForm")

        if (
            getattr(tf, "degree_left", None) == 0
            and getattr(tf, "degree_right", None) == 0
            and (left_mode in (2, 4) or right_mode in (2, 4))
        ):
            return as_doubleform(0, p=0, q=0, dim=self.dim)

        out = _CPP_RiemannianManifold.ProjectDoubleForm(
            self, tf, left, right, normal, conormal
        )
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

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
        if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            raise TypeError("ContractSlot expects a DoubleForm")
        vf_wrapped = as_vectorfield(vf)
        out = _CPP_RiemannianManifold.ContractSlot(self, tf, vf_wrapped, slot)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def InnerProduct(self, tf1, tf2, vb=None, forms=False):
        if forms:
            dim = self.dim
            tf1_df = isinstance(tf1, (DoubleForm, _CPP_DoubleForm))
            tf2_df = isinstance(tf2, (DoubleForm, _CPP_DoubleForm))
            tf1_sf = _is_scalarfield_like(tf1)
            tf2_sf = _is_scalarfield_like(tf2)

            # Treat scalar inputs as (0,0) double-forms so forms=True also works
            # after complete traces such as Trace(star(phi), l=p).
            if tf1_df or tf2_df or (tf1_sf and tf2_sf):
                if tf1_sf:
                    tf1 = DoubleForm(tf1, p=0, q=0, dim=dim)
                if tf2_sf:
                    tf2 = DoubleForm(tf2, p=0, q=0, dim=dim)

        if vb is None:
            out = _CPP_RiemannianManifold.InnerProduct(self, tf1, tf2, forms=forms)
        else:
            out = _CPP_RiemannianManifold.InnerProduct(self, tf1, tf2, vb, forms)
        return as_scalarfield(out, dim=self.dim)

    def Cross(self, tf1, tf2):
        out = _CPP_RiemannianManifold.Cross(self, tf1, tf2)
        return as_vectorfield(out)

    def CovDeriv(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.CovDeriv(self, tf)
        else:
            out = _CPP_RiemannianManifold.CovDeriv(self, tf, vb)
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
        if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            if vb is None:
                vb = ngsolve.VOL
            out = _CPP_RiemannianManifold.CovDiv(self, tf, slot, vb)
            return as_doubleform(
                out, p=out.degree_left, q=out.degree_right, dim=self.dim
            )

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
        if isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            if l is None:
                l = 1
            if vb is None:
                out = _CPP_RiemannianManifold.Trace(self, tf, l)
            else:
                out = _CPP_RiemannianManifold.Trace(self, tf, l, vb)
            if isinstance(out, _CPP_DoubleForm):
                return as_doubleform(
                    out, p=out.degree_left, q=out.degree_right, dim=self.dim
                )
            return as_scalarfield(out, dim=self.dim)

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
        if not isinstance(tf, (DoubleForm, _CPP_DoubleForm)):
            raise TypeError("TraceSigma expects a DoubleForm")
        try:
            sigma_df = _as_doubleform_like(sigma, dim=self.dim)
        except TypeError:
            sigma_df = DoubleForm(sigma, p=1, q=1, dim=self.dim)
        out = _CPP_RiemannianManifold.TraceSigma(self, tf, sigma_df, vb)
        if isinstance(out, _CPP_DoubleForm):
            return as_doubleform(
                out, p=out.degree_left, q=out.degree_right, dim=self.dim
            )
        return as_scalarfield(out, dim=self.dim)

    def SlotInnerProduct(self, tf, vb=ngsolve.VOL, forms=True):
        if _is_scalarfield_like(tf):
            return as_scalarfield(tf, dim=self.dim)
        out = _CPP_RiemannianManifold.SlotInnerProduct(self, tf, vb, forms)
        return as_scalarfield(out, dim=self.dim)

    def Contraction(self, tf, vf, slot=0):
        # Accept inputs where exactly one argument is a vector field; the other can be any tensor (including k-forms).
        tf_wrapped = as_tensorfield(tf)
        vf_wrapped = as_tensorfield(vf)

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
        return as_tensorfield(out)

    def s(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.s(self, tf)
        else:
            out = _CPP_RiemannianManifold.s(self, tf, vb)
        return as_doubleform(out, p=out.degree_left, q=out.degree_right, dim=self.dim)

    def J(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.J(self, tf)
        else:
            out = _CPP_RiemannianManifold.J(self, tf, vb)
        return as_tensorfield(out)


__all__ = [
    "KForm",
    "GenericKForm",
    "DoubleForm",
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
    "RiemannianManifold",
]
