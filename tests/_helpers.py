from ngsolve import BBND, CF, InnerProduct, Integrate, Inv, dx, sqrt, x, y, z

XVEC = [x, y, z]


def l2_error(a, b, mesh, *, bonus_intorder=None):
    measure = dx(bonus_intorder=bonus_intorder) if bonus_intorder is not None else dx
    return sqrt(Integrate(InnerProduct(a - b, a - b) * measure, mesh))


def l2_norm(a, mesh, *, vb=None, bonus_intorder=None):
    dx_kwargs = {}
    if vb is not None:
        dx_kwargs["element_vb"] = vb
    if bonus_intorder is not None:
        dx_kwargs["bonus_intorder"] = bonus_intorder
    measure = dx(**dx_kwargs) if dx_kwargs else dx
    return sqrt(Integrate(InnerProduct(a, a) * measure, mesh))


def l2_error_bnd(a, b, mesh, *, bonus_intorder=None):
    dx_kwargs = {"element_boundary": True}
    if bonus_intorder is not None:
        dx_kwargs["bonus_intorder"] = bonus_intorder
    return sqrt(Integrate(InnerProduct(a - b, a - b) * dx(**dx_kwargs), mesh))


def l2_norm_bnd(a, mesh, *, bonus_intorder=None):
    dx_kwargs = {"element_boundary": True}
    if bonus_intorder is not None:
        dx_kwargs["bonus_intorder"] = bonus_intorder
    return sqrt(Integrate(InnerProduct(a, a) * dx(**dx_kwargs), mesh))


def l2_error_bbnd(a, b, mesh, *, bonus_intorder=None):
    dx_kwargs = {"element_vb": BBND}
    if bonus_intorder is not None:
        dx_kwargs["bonus_intorder"] = bonus_intorder
    return sqrt(Integrate(InnerProduct(a - b, a - b) * dx(**dx_kwargs), mesh))


def l2_inner(a, b, mesh, *, bonus_intorder=None):
    return l2_error(a, b, mesh, bonus_intorder=bonus_intorder)


def assert_l2_close(a, b, mesh, *, tol, bonus_intorder=None):
    assert l2_error(a, b, mesh, bonus_intorder=bonus_intorder) < tol


def assert_l2_zero(a, mesh, *, tol, vb=None, bonus_intorder=None):
    assert l2_norm(a, mesh, vb=vb, bonus_intorder=bonus_intorder) < tol


def get_diff_op(name, cf):
    dims = cf.dims
    dim = cf.dims[0]
    if name == "grad":
        if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
            return CF(tuple(cf.Diff(XVEC[i]) for i in range(dim)))
        if len(cf.dims) == 1:
            return CF(
                tuple(
                    cf[i].Diff(XVEC[j])
                    for i in range(cf.dims[0])
                    for j in range(cf.dims[0])
                ),
                dims=(cf.dims[0], cf.dims[0]),
            )
        return CF(
            tuple(
                cf[j, k].Diff(XVEC[i])
                for i in range(cf.dims[0])
                for j in range(cf.dims[0])
                for k in range(cf.dims[0])
            ),
            dims=(cf.dims[0], cf.dims[0] ** 2),
        )
    if name == "christoffel":
        cf_grad = get_diff_op("grad", cf)
        return 0.5 * CF(
            tuple(
                cf_grad[i, j + dim * k]
                + cf_grad[j, i + dim * k]
                - cf_grad[k, i + dim * j]
                for i in range(dim)
                for j in range(dim)
                for k in range(dim)
            ),
            dims=(dim, dim, dim),
        )
    if name == "christoffel2":
        chr1 = get_diff_op("christoffel", cf)
        return CF((CF((chr1), dims=(dim**2, dim)) * Inv(cf)), dims=(dim, dim, dim))
    raise ValueError(f"unsupported differential operator: {name}")
