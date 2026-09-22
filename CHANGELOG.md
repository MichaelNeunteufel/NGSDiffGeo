# Changelog

## 0.3.0 — 2026-09-22

This release expands the differential-form and tensor APIs, improves symbolic
derivatives and evaluation, and substantially revises the documentation. It
requires NGSolve 6.2.2607 or newer.

### Highlights

- Added compact-backed operations for differential forms, including native
  exterior-algebra, Hodge, trace, and metric-dependent operations. The public
  `.coef` representation remains a full tensor.
- Improved coefficient gradients and Hessians, including proxy-function
  handling, symbolic differentiation, replacement, and optional compilation
  of inner covariant-derivative graphs.
- Strengthened tensor and form arithmetic contracts, with explicit checks of
  degrees, dimensions, and operand types. Added regression tests and
  reproducible benchmarks for these operations.
- Expanded the tutorials with exterior covariant derivatives of double forms
  and hypersurface geometry, and added a developer guide for the main
  implementation components.

### Upgrading from v0.2.0

- The minimum NGSolve version is now `6.2.2607`. Upgrade NGSolve when upgrading
  NGSDiffGeo.
- `mf.Curvature` now returns the normalized geometric curvature operator for
  Regge metrics as well as ordinary coefficient-function metrics. If your code
  divided this result by `Det(g)` for a Regge metric, remove that extra
  division. NGSolve's native `g.Operator("curvature")` remains unnormalized.
- Use `CovDerivative` and `CovHessian` in new code. `CovDeriv` and `CovHesse`
  remain compatibility aliases but emit `DeprecationWarning` through the Python
  wrapper.
- Form arithmetic now rejects incompatible degrees, dimensions, and operand
  types more consistently. In particular, use `Wedge(alpha, beta)` for the
  exterior product of positive-degree forms rather than `alpha * beta`. Use
  `.coef` for plain coefficient-function tensor arithmetic.

Install or upgrade with `python -m pip install --upgrade ngsdiffgeo`. See the
[documentation](https://michaelneunteufel.github.io/NGSDiffGeo/) for examples
and installation details.
