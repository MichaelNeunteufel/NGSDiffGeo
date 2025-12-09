import importlib
import ngsolve

_cpp = importlib.import_module(".ngsdiffgeo", __package__)

# ---- references to the C++ classes----
_CPP_OneForm = _cpp.OneForm
_CPP_TwoForm = _cpp.TwoForm
_CPP_ThreeForm = _cpp.ThreeForm
_CPP_KForm = _cpp.KForm
_CPP_VectorField = _cpp.VectorField
_CPP_TensorField = _cpp.TensorField


def _infer_dim(obj):
    for attr in ("_dim", "dim", "Dim"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if callable(val):
                try:
                    val = val()
                except TypeError:
                    continue
            if isinstance(val, int) and val > 0:
                return val
    return None


def as_oneform(cf, *, dim):
    if isinstance(cf, OneForm):  # wrapper
        return cf
    if isinstance(cf, _CPP_OneForm):  # pybind
        return OneForm(cf, dim=dim)  # rewrap into wrapper
    return OneForm(cf, dim=dim)


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


def as_kform(cf, *, k, dim):
    if dim is None:
        dim = _infer_dim(cf)
        if dim is None:
            return cf
    if isinstance(cf, KForm):
        return cf
    if isinstance(cf, _CPP_KForm):
        return KForm(cf, k=k, dim=dim)
    if k == 1:
        return as_oneform(cf, dim=dim)
    if k == 2:
        return as_twoform(cf, dim=dim)
    if k == 3:
        return as_threeform(cf, dim=dim)
    return KForm(cf, k=k, dim=dim)


def as_vectorfield(cf, *, dim):
    if dim is None:
        dim = _infer_dim(cf)
        if dim is None:
            return cf
    if isinstance(cf, VectorField):
        return cf
    if isinstance(cf, _CPP_VectorField):
        return VectorField(cf, dim=dim)
    return VectorField(cf, dim=dim)


def as_tensorfield(cf, *, covariant_indices):
    if isinstance(cf, TensorField):
        return cf
    if isinstance(cf, _CPP_TensorField):
        return TensorField(cf, covariant_indices=covariant_indices)
    return TensorField(cf, covariant_indices=covariant_indices)


class OneForm(_CPP_OneForm):
    def __init__(self, cf, *, dim=-1):
        super().__init__(cf, dim=dim)
        self._dim = dim

    def _wrap(self, cf):
        return as_oneform(cf, dim=self._dim)

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
    def __init__(self, cf, *, dim):
        super().__init__(cf, dim=dim)
        self._dim = dim

    def _wrap(self, cf):
        return as_twoform(cf, dim=self._dim)

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
    def __init__(self, cf, *, dim):
        super().__init__(cf, dim=dim)
        self._dim = dim

    def _wrap(self, cf):
        return as_threeform(cf, dim=self._dim)

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


class KForm(_CPP_KForm):
    def __init__(self, cf, *, k, dim):
        super().__init__(cf, k=k, dim=dim)
        self._k = k
        self._dim = dim

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


class VectorField(_CPP_VectorField):
    def __init__(self, cf, *, dim=None):
        super().__init__(cf)
        self._dim = dim or _infer_dim(cf)

    def _wrap(self, cf):
        return as_vectorfield(cf, dim=self._dim)

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


class TensorField(_CPP_TensorField):
    def __init__(self, cf, covariant_indices):
        super().__init__(cf, covariant_indices=covariant_indices)
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
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(_CPP_TensorField.__truediv__(self, other))


def Wedge(a, b, dim):
    out = _cpp.Wedge(a, b, dim)
    # wrap based on degree (works even if out is pybind TwoForm/KForm)
    k = out.Degree() if hasattr(out, "Degree") else out.degree
    return as_kform(out, k=k, dim=dim)


def d(a, dim):
    out = _cpp.d(a, dim)
    k = out.Degree() if hasattr(out, "Degree") else out.degree
    return as_kform(out, k=k, dim=dim)


def star(a, M):
    out = _cpp.star(a, M)
    dim = _infer_dim(M) or _infer_dim(a)
    if dim is None:
        return out
    k = out.Degree() if hasattr(out, "Degree") else out.degree
    return as_kform(out, k=k, dim=dim)


def delta(a, M):
    out = _cpp.delta(a, M)
    dim = _infer_dim(M) or _infer_dim(a)
    if dim is None:
        return out
    k = out.Degree() if hasattr(out, "Degree") else out.degree
    return as_kform(out, k=k, dim=dim)
