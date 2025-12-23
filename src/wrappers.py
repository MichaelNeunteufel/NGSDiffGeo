"""
Python-side wrappers for ngsdiffgeo pybind classes.
"""

from __future__ import annotations

import importlib
import ngsolve

_cpp = importlib.import_module(".ngsdiffgeo", __package__)

# ---- references to the C++/pybind classes ----
_CPP_ScalarField = _cpp.ScalarField
_CPP_OneForm = _cpp.OneForm
_CPP_TwoForm = _cpp.TwoForm
_CPP_ThreeForm = _cpp.ThreeForm
_CPP_KForm = _cpp.KForm
_CPP_VectorField = _cpp.VectorField
_CPP_TensorField = _cpp.TensorField
_CPP_RiemannianManifold = _cpp.RiemannianManifold


# ---------------- helpers ----------------


def _call_if_callable(x):
    return x() if callable(x) else x


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
        return self._wrap(_CPP_ScalarField.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_ScalarField.__sub__(self, other))

    def __mul__(self, other):
        if isinstance(other, KForm) or (
            isinstance(other, ngsolve.CoefficientFunction) and other.dim == 1
        ):
            k = 0
            if hasattr(other, "degree"):
                k = other.degree
            return self._wrap(_CPP_ScalarField.__mul__(self, other), k=k)
        elif isinstance(other, (int, float, complex)):
            # keep scalar * ScalarField results typed as ScalarField
            return self._wrap(_CPP_ScalarField.__mul__(self, other))
        elif isinstance(other, (VectorField, TensorField)):
            return as_tensorfield(
                _CPP_ScalarField.__mul__(self, other),
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
        return self._wrap(_CPP_ScalarField.__truediv__(self, other))


class OneForm(_CPP_OneForm):
    def __init__(self, cf):
        _CPP_OneForm.__init__(self, cf)
        self._k = 1
        self._dim = cf.dim

    def _wrap(self, cf):
        return as_kform(cf, k=1, dim=self._dim)

    def __add__(self, other):
        return self._wrap(_CPP_OneForm.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_OneForm.__sub__(self, other))

    def __mul__(self, other):
        return self._wrap(_CPP_OneForm.__mul__(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(_CPP_OneForm.__truediv__(self, other))


class TwoForm(_CPP_TwoForm):
    def __init__(self, cf, *, dim=-1):
        _CPP_TwoForm.__init__(self, cf, dim=dim)
        self._k = 2
        self._dim = dim

    def _wrap(self, cf):
        return as_kform(cf, k=2, dim=self._dim)

    def __add__(self, other):
        return self._wrap(_CPP_TwoForm.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_TwoForm.__sub__(self, other))

    def __mul__(self, other):
        return self._wrap(_CPP_TwoForm.__mul__(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(_CPP_TwoForm.__truediv__(self, other))


class ThreeForm(_CPP_ThreeForm):
    def __init__(self, cf, *, dim=-1):
        _CPP_ThreeForm.__init__(self, cf, dim=dim)
        self._k = 3
        self._dim = dim

    def _wrap(self, cf):
        return as_kform(cf, k=3, dim=self._dim)

    def __add__(self, other):
        return self._wrap(_CPP_ThreeForm.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_ThreeForm.__sub__(self, other))

    def __mul__(self, other):
        return self._wrap(_CPP_ThreeForm.__mul__(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(_CPP_ThreeForm.__truediv__(self, other))


class GenericKForm(_CPP_KForm):
    def __init__(self, cf, *, k, dim):
        _CPP_KForm.__init__(self, cf, k=int(k), dim=int(dim))
        self._k = int(k)
        self._dim = int(dim)

    def _wrap(self, cf):
        return as_kform(cf, k=self._k, dim=self._dim)

    def __add__(self, other):
        return self._wrap(_CPP_KForm.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_KForm.__sub__(self, other))

    def __mul__(self, other):
        return self._wrap(_CPP_KForm.__mul__(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(_CPP_KForm.__truediv__(self, other))


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


# ---------------- VectorField / TensorField ----------------


class VectorField(_CPP_VectorField):
    def __init__(self, cf):
        _CPP_VectorField.__init__(self, cf)

    def _wrap(self, cf):
        return as_vectorfield(cf)

    def __add__(self, other):
        return self._wrap(_CPP_VectorField.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_VectorField.__sub__(self, other))

    def __mul__(self, other):
        return self._wrap(_CPP_VectorField.__mul__(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(_CPP_VectorField.__truediv__(self, other))


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
        return self._wrap(_CPP_TensorField.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(_CPP_TensorField.__sub__(self, other))

    def __mul__(self, other):
        return self._wrap(_CPP_TensorField.__mul__(self, other))

    def __rmul__(self, other):
        # if other is a number or a ngsolve CoefficientFunction with dim=1:
        if isinstance(other, (int, float, ngsolve.CoefficientFunction)) and (
            not hasattr(other, "dim") or other.dim == 1
        ):
            return self._wrap(_CPP_TensorField.__rmul__(self, other))
            # return self.__mul__(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self._wrap(_CPP_TensorField.__truediv__(self, other))


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
            raise TypeError("as_tensorfield: dim must be provided or inferable for scalars")
        return ScalarField(cf, dim=dim)
    elif covariant_indices == "0":
        return VectorField(cf)
    elif covariant_indices == "1":
        return OneForm(cf)
    return TensorField(cf, covariant_indices=covariant_indices)


# ---------------- wrapping of exported C++ functions ----------------


def Wedge(a, b):
    out = _cpp.Wedge(a, b)
    return as_kform(out, k=out.degree, dim=out.dim_space)


def d(a):
    out = _cpp.d(a)
    return as_kform(out, k=out.degree, dim=out.dim_space)


def star(a, M, vb=ngsolve.VOL):
    out = _cpp.star(a, M, vb)
    return as_kform(out, k=out.degree, dim=M.dim)


def delta(a, M):
    out = _cpp.delta(a, M)
    return as_kform(out, k=out.degree, dim=M.dim)


# ---------------- RiemannianManifold wrapper ----------------


class RiemannianManifold(_CPP_RiemannianManifold):
    def __init__(self, metric):
        super().__init__(metric)

    # properties
    @property
    def G(self):
        out = _CPP_RiemannianManifold.G.__get__(self)
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

    @property
    def G_deriv(self):
        return _CPP_RiemannianManifold.G_deriv.__get__(self)

    @property
    def Riemann(self):
        out = _CPP_RiemannianManifold.Riemann.__get__(self)
        return as_tensorfield(out)

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
        return as_tensorfield(out)

    @property
    def Einstein(self):
        out = _CPP_RiemannianManifold.Einstein.__get__(self)
        return as_tensorfield(out)

    @property
    def Scalar(self):
        out = _CPP_RiemannianManifold.Scalar.__get__(self)
        return as_scalarfield(out, dim=self.dim)

    @property
    def SFF(self):
        out = _CPP_RiemannianManifold.SFF.__get__(self)
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

    def star(self, a, vb=ngsolve.VOL):
        out = _CPP_RiemannianManifold.star(self, a, vb)
        return as_kform(out, k=out.degree, dim=self.dim)

    def delta(self, a):
        out = _CPP_RiemannianManifold.delta(self, a)
        return as_kform(out, k=out.degree, dim=self.dim)

    def InnerProduct(self, tf1, tf2, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.InnerProduct(self, tf1, tf2)
        else:
            out = _CPP_RiemannianManifold.InnerProduct(self, tf1, tf2, vb)
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

    def CovDiv(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.CovDiv(self, tf)
        else:
            out = _CPP_RiemannianManifold.CovDiv(self, tf, vb)
        return as_tensorfield(out, dim=self.dim)

    def Trace(self, tf, vb=None, index1=0, index2=1):
        if vb is None:
            out = _CPP_RiemannianManifold.Trace(self, tf, index1=index1, index2=index2)
        else:
            out = _CPP_RiemannianManifold.Trace(self, tf, vb, index1, index2)
        return as_tensorfield(out, dim=self.dim)

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

    def J(self, tf, vb=None):
        if vb is None:
            out = _CPP_RiemannianManifold.J(self, tf)
        else:
            out = _CPP_RiemannianManifold.J(self, tf, vb)
        return as_tensorfield(out)


__all__ = [
    "KForm",
    "GenericKForm",
    "ScalarField",
    "OneForm",
    "TwoForm",
    "ThreeForm",
    "as_scalarfield",
    "as_oneform",
    "as_twoform",
    "as_threeform",
    "as_kform",
    "VectorField",
    "TensorField",
    "as_vectorfield",
    "as_tensorfield",
    "Wedge",
    "d",
    "star",
    "delta",
    "RiemannianManifold",
]
