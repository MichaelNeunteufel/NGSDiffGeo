import subprocess
import sys
import textwrap
import math

import pytest


def script(*parts):
    return "\n".join(textwrap.dedent(part).strip("\n") for part in parts)


BASE_PRELUDE = script(
    """
from netgen.occ import unit_cube
from ngsolve import *
import ngsdiffgeo as dg

mesh = Mesh(unit_cube.GenerateMesh(maxh=2))
"""
)

EUCLIDEAN_WEDGE_SETUP = script(
    BASE_PRELUDE,
    """
e0 = dg.OneForm(CF((1, 0, 0)))
e1 = dg.OneForm(CF((0, 1, 0)))
twoform = dg.TwoForm(CF((0, 1, 0, -1, 0, 0, 0, 0, 0), dims=(3, 3)))

phi = dg.DoubleForm(dg.Einsum("ij,k->ijk", twoform, e0), p=2, q=1, dim=3)
psi = dg.DoubleForm(dg.Einsum("i,j->ij", e1, e0), p=1, q=1, dim=3)
"""
)

DERIVATIVE_PART3_SETUP = script(
    BASE_PRELUDE,
    """
alpha1 = dg.OneForm(CF((0.3 * x * y, z**2, -0.1 * x)))
beta1 = dg.OneForm(CF((0.3 * z * y, x * z**2, y**2)))
gamma1 = dg.OneForm(CF((0.3 * y * z - x * y, x**2 * z + 0.34 * y**3, -x * y * z)))

B11 = dg.DoubleForm(dg.Einsum("i,j->ij", alpha1, beta1), p=1, q=1, dim=3)
C11 = dg.DoubleForm(dg.Einsum("i,j->ij", beta1, gamma1), p=1, q=1, dim=3)

gfG = GridFunction(HCurlCurl(mesh, order=2))
gfG.Set(dg.Heisenberg().metric)
mf = dg.RiemannianManifold(gfG, normal_sign=-1, change_riemann_sign=True)
"""
)


def run_python_reproducer(code):
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"reproducer crashed with return code {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return float(line.split(maxsplit=1)[1])
    pytest.fail(f"reproducer did not print a RESULT line\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


@pytest.mark.parametrize(
    "code",
    [
        script(
            BASE_PRELUDE,
            """
        phi = dg.DoubleForm(CF((1,) + (0,) * 242, dims=(3, 3, 3, 3, 3)), p=3, q=2, dim=3)
        cf = dg.RiemannianManifold(Id(3)).d_cov(phi, slot="right").coef

        with TaskManager():
            value = Integrate(InnerProduct(cf, cf) * dx, mesh)
        print("RESULT", value)
        """),
        script(
            EUCLIDEAN_WEDGE_SETUP,
            """
        cf = dg.RiemannianManifold(Id(3)).d_cov(dg.Wedge(phi, psi), slot="right")

        with TaskManager():
            value = Integrate(InnerProduct(cf, cf) * dx, mesh)
        print("RESULT", value)
        """),
        script(
            EUCLIDEAN_WEDGE_SETUP,
            """
        metric = CF((1, 0, 0, 0, 1 + x**2, -x, 0, -x, 1), dims=(3, 3))
        cf = dg.RiemannianManifold(metric).d_cov(dg.Wedge(phi, psi), slot="right")

        with TaskManager():
            value = Integrate(InnerProduct(cf, cf) * dx, mesh)
        print("RESULT", value)
        """),
        script(
            EUCLIDEAN_WEDGE_SETUP,
            """
        metric = CF((1, 0, 0, 0, 1 + x**2, -x, 0, -x, 1), dims=(3, 3))
        mf = dg.RiemannianManifold(metric, normal_sign=-1)

        left = mf.d_cov(dg.Wedge(phi, psi), slot="right")
        right = dg.Wedge(mf.d_cov(phi, slot="right"), psi) - dg.Wedge(
            phi, mf.d_cov(psi, slot="right")
        )
        cf = left - right

        with TaskManager():
            value = Integrate(InnerProduct(cf, cf) * dx, mesh)
        print("RESULT", value)
        """),
    ],
)
def test_taskmanager_double_form_regressions_do_not_crash(code):
    assert abs(run_python_reproducer(code)) < 1e-12


def test_taskmanager_high_rank_double_form_sum_does_not_crash():
    value = run_python_reproducer(
        script(
            BASE_PRELUDE,
            """
        e0 = dg.OneForm(CF((1, 0, 0)))
        e1 = dg.OneForm(CF((0, 1, 0)))
        phi = dg.DoubleForm(dg.Einsum("i,j->ij", e0, e1), p=1, q=1, dim=3)

        metric = CF((1, 0, 0, 0, 1 + x**2, -x, 0, -x, 1), dims=(3, 3))
        gf_metric = GridFunction(HCurlCurl(mesh, order=1))
        gf_metric.Set(metric)
        mf = dg.RiemannianManifold(gf_metric)

        star_phi = mf.star(phi)
        cf = mf.d_cov(mf.d_cov(star_phi, slot="right"), slot="left") + mf.d_cov(
            mf.d_cov(star_phi, slot="left"), slot="right"
        )

        serial = Integrate(InnerProduct(cf, cf) * dx(bonus_intorder=1), mesh)
        with TaskManager():
            parallel = Integrate(InnerProduct(cf, cf) * dx(bonus_intorder=1), mesh)
        print("RESULT", abs(parallel - serial) / max(1.0, abs(serial)))
        """)
    )

    assert value < 1e-12


def test_component_and_composite_proxy_gradients_do_not_crash_and_are_correct():
    value = run_python_reproducer(
        script(
            """
        from netgen.occ import unit_square
        from ngsolve import *
        import ngsdiffgeo as dg

        mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
        space = VectorH1(mesh, order=2)
        trial, test = space.TnT()
        native = Grad(trial)

        component = dg.GradCF(trial[0], 2)
        scaled = dg.GradCF(2 * trial, 2)
        rebuilt = dg.GradCF(CF((trial[0], trial[1]), dims=(2,)), 2)

        errors = []
        for direction in range(2):
            form = BilinearForm(space)
            form += SymbolicBFI(
                (component[direction] - native[0, direction]) * test[0],
                simd_evaluate=False,
            )
            form.Assemble()
            errors.append(Norm(form.mat.AsVector()))

            for candidate, scale in ((scaled, 2), (rebuilt, 1)):
                for tensor_component in range(2):
                    form = BilinearForm(space)
                    form += SymbolicBFI(
                        (
                            candidate[direction, tensor_component]
                            - scale * native[tensor_component, direction]
                        )
                        * test[tensor_component],
                        simd_evaluate=False,
                    )
                    form.Assemble()
                    errors.append(Norm(form.mat.AsVector()))

        print("RESULT", max(errors))
        """
        )
    )

    assert value < 2e-10


def test_taskmanager_graph_compiled_boundary_covariant_derivative_matches_default():
    value = run_python_reproducer(
        script(
            DERIVATIVE_PART3_SETUP,
            """
        projected = mf.ProjectDoubleForm(B11, left="F", right="F")
        default = mf.d_cov(projected, slot="right", vb=BND)
        compiled = mf.d_cov(
            projected,
            slot="right",
            vb=BND,
            compile_inner="graph",
        )
        difference = default - compiled

        with TaskManager():
            value = Integrate(
                InnerProduct(difference, difference) * dx(element_vb=BND),
                mesh,
            )
        print("RESULT", value)
        """,
        )
    )
    assert abs(value) < 1e-12


def test_derivative_part3_taskmanager_regressions_do_not_crash():
    value = run_python_reproducer(
        script(
            DERIVATIVE_PART3_SETUP,
            """
        def H_T_star(phi):
            return 0.5 * (
                mf.delta_cov(mf.delta_cov(phi, slot="right"), slot="left")
                + mf.delta_cov(mf.delta_cov(phi, slot="left"), slot="right")
            )

        sigma = dg.Sym(B11)
        A = dg.DoubleForm(
            dg.Einsum(
                "ija,klb,ab->ijkl",
                fem.LeviCivitaSymbol(3),
                fem.LeviCivitaSymbol(3),
                sigma,
            ),
            p=2,
            q=2,
            dim=3,
        )
        star_A = mf.star(dg.Sym(C11), slot="both")

        with TaskManager():
            nested_left = H_T_star(A)
            nested_right = mf.CovDiv(mf.CovDiv(A, slot="right"))
            nested_diff = nested_left - nested_right
            nested_value = Integrate(
                InnerProduct(nested_diff, nested_diff) * dx(bonus_intorder=3), mesh
            )

            inner_value = Integrate(
                mf.InnerProduct(
                    dg.DoubleForm(4 * mf.CovInc(sigma), p=2, q=2, dim=3),
                    star_A,
                    forms=True,
                )
                * mf.VolumeForm(VOL)
                * dx(bonus_intorder=3),
                mesh,
            )

        print("RESULT", nested_value + abs(inner_value))
        """)
    )

    assert math.isfinite(value)
