## Test Layout

- `conftest.py` contains shared mesh and manifold fixtures.
- `_helpers.py` contains shared numeric assertions and reference differential operators.
- `_kforms_support.py` contains common imports and quadrature-tuned helpers for k-form tests.

## Naming

- Use snake_case for helper and test names.
- Use `2d` and `3d` for dimension-specific names.

## Structure

- Keep API and wrapper-behavior tests separate from geometry/operator identity tests.
- Prefer focused files over broad mixed-responsibility files.
- Use fixtures for common meshes, metrics, and Euclidean manifolds.

## Numeric Checks

- Prefer `l2_error`, `l2_norm`, `assert_l2_close`, and `assert_l2_zero` from `_helpers.py`.
- Only inline `Integrate(...)` expressions when the integral itself is the behavior under test.
