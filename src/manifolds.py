import ngsolve
from ngsolve.fem import Einsum
from .wrappers import TensorField, ScalarField, DoubleForm

__all__ = [
    "EuclideanMetric",
    "Sphere2",
    "Sphere3",
    "PoincareDisk",
    "HyperbolicH2",
    "HyperbolicH3",
    "Heisenberg",
    "CigarSoliton",
    "WarpedProduct",
    "TestMetric",
]


class EuclideanMetric:
    """
    Euclidean metric on R^dim.
    """

    def __init__(self, dim=2):
        self.metric = TensorField(ngsolve.Id(dim), "11")

        # flat manifold
        self.chr1 = ngsolve.CF((0,) * dim**3, dims=(dim, dim, dim))
        self.chr2 = ngsolve.CF((0,) * dim**3, dims=(dim, dim, dim))
        self.Riemann = DoubleForm(
            ngsolve.CF((0,) * dim**4, dims=(dim, dim, dim, dim)), dim=dim, p=2, q=2
        )
        self.Ricci = TensorField(ngsolve.CF((0,) * dim**2, dims=(dim, dim)), "11")
        self.scalar = ScalarField(ngsolve.CF(0), dim=dim)
        self.Einstein = TensorField(ngsolve.CF((0,) * dim**2, dims=(dim, dim)), "11")
        if dim == 2:
            self.curvature = ScalarField(ngsolve.CF(0), dim=dim)
        else:
            self.curvature = TensorField(
                ngsolve.CF((0,) * dim**2, dims=(dim, dim)), "00"
            )
        return


class Sphere2:
    """
    Standard metric on sphere S^2. x and y are interpreted as angles; x in [0,pi], y in [0,2*pi). Has constant positive curvature.
    """

    def __init__(self):
        # metric tensor
        self.metric = TensorField(
            ngsolve.CF((1, 0, 0, ngsolve.sin(ngsolve.x) ** 2), dims=(2, 2)), "11"
        )
        # Christoffel symbols of the first kind Gamma_{ijk}=0.5*(d_ig_jk+d_jg_ik-d_kg_ij)
        self.chr1 = ngsolve.CF(
            (
                0,
                0,
                0,
                ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
                ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                -ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
            ),
            dims=(2, 2, 2),
        )
        # Christoffel symbols of the second kind Gamma_{ij}^k=g^{kl}Gamma_{ijl}
        self.chr2 = ngsolve.CF(
            (
                0,
                0,
                0,
                ngsolve.cos(ngsolve.x) / ngsolve.sin(ngsolve.x),
                0,
                ngsolve.cos(ngsolve.x) / ngsolve.sin(ngsolve.x),
                -ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
            ),
            dims=(2, 2, 2),
        )
        # Riemann curvature tensor R_{ijkl}=d_iGamma_{jkl}-d_jGamma_{ikl}+Gamma_{jlm}Gamma_{ik}^m-Gamma_{ilm}Gamma_{jk}^m
        self.Riemann = DoubleForm(
            ngsolve.CF(
                (
                    0,
                    0,
                    0,
                    0,
                    0,
                    -(ngsolve.sin(ngsolve.x) ** 2),
                    ngsolve.sin(ngsolve.x) ** 2,
                    0,
                    0,
                    ngsolve.sin(ngsolve.x) ** 2,
                    -(ngsolve.sin(ngsolve.x) ** 2),
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
                dims=(2, 2, 2, 2),
            ),
            dim=2,
            p=2,
            q=2,
        )
        # Ricci curvature tensor R_{ij}=g^{kl}R_{kijl}
        self.Ricci = TensorField(
            ngsolve.CF((1, 0, 0, ngsolve.sin(ngsolve.x) ** 2), dims=(2, 2)), "11"
        )
        # Scalar curvature R=g^{ij}R_{ij}
        self.scalar = ScalarField(ngsolve.CF(2), dim=2)
        # Einstein tensor G_{ij}=R_{ij}-0.5*g_{ij}R
        self.Einstein = TensorField(ngsolve.CF((0, 0, 0, 0), dims=(2, 2)), "11")
        # Curvature operator is the Gauss curvature in 2D
        self.curvature = ScalarField(ngsolve.CF(1), dim=2)
        return


#
class Sphere3:
    """
    Standard metric on sphere S^3. x, y, and z are interpreted as angles; x in [0,pi], y in [0,pi], z in [0,2*pi). Has constant positive curvature.
    """

    def __init__(self):
        # metric
        self.metric = TensorField(
            ngsolve.CF(
                (
                    1,
                    0,
                    0,
                    0,
                    ngsolve.sin(ngsolve.x) ** 2,
                    0,
                    0,
                    0,
                    ngsolve.sin(ngsolve.x) ** 2 * ngsolve.sin(ngsolve.y) ** 2,
                ),
                dims=(3, 3),
            ),
            "11",
        )
        # Christoffel symbols of the first kind Gamma_{ijk}=0.5*(d_ig_jk+d_jg_ik-d_kg_ij)
        self.chr1 = ngsolve.CF(
            (
                0,
                0,
                0,
                0,
                ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
                0,
                0,
                ngsolve.sin(ngsolve.x)
                * ngsolve.sin(ngsolve.y) ** 2
                * ngsolve.cos(ngsolve.x),
                0,
                ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
                -ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
                0,
                0,
                0,
                ngsolve.sin(ngsolve.x) ** 2
                * ngsolve.sin(ngsolve.y)
                * ngsolve.cos(ngsolve.y),
                0,
                0,
                ngsolve.sin(ngsolve.x)
                * ngsolve.sin(ngsolve.y) ** 2
                * ngsolve.cos(ngsolve.x),
                0,
                0,
                ngsolve.sin(ngsolve.x) ** 2
                * ngsolve.sin(ngsolve.y)
                * ngsolve.cos(ngsolve.y),
                -ngsolve.sin(ngsolve.x)
                * ngsolve.cos(ngsolve.x)
                * ngsolve.sin(ngsolve.y) ** 2,
                -(ngsolve.sin(ngsolve.x) ** 2)
                * ngsolve.sin(ngsolve.y)
                * ngsolve.cos(ngsolve.y),
                0,
            ),
            dims=(3, 3, 3),
        )

        self.chr2 = ngsolve.CF(
            (
                0,
                0,
                0,
                0,
                ngsolve.cos(ngsolve.x) / ngsolve.sin(ngsolve.x),
                0,
                0,
                0,
                ngsolve.cos(ngsolve.x) / ngsolve.sin(ngsolve.x),
                0,
                ngsolve.cos(ngsolve.x) / ngsolve.sin(ngsolve.x),
                0,
                -ngsolve.sin(ngsolve.x) * ngsolve.cos(ngsolve.x),
                0,
                0,
                0,
                0,
                ngsolve.cos(ngsolve.y) / ngsolve.sin(ngsolve.y),
                0,
                0,
                ngsolve.cos(ngsolve.x) / ngsolve.sin(ngsolve.x),
                0,
                0,
                ngsolve.cos(ngsolve.y) / ngsolve.sin(ngsolve.y),
                -ngsolve.sin(ngsolve.x)
                * ngsolve.cos(ngsolve.x)
                * ngsolve.sin(ngsolve.y) ** 2,
                -ngsolve.sin(ngsolve.y) * ngsolve.cos(ngsolve.y),
                0,
            ),
            dims=(3, 3, 3),
        )

        self.Ricci = TensorField(2 * self.metric, "11")
        self.scalar = ScalarField(ngsolve.CF(6), dim=3)
        self.Einstein = TensorField(-self.metric, "11")
        # curvature = g^{-1}
        self.curvature = TensorField(
            ngsolve.CF(
                (
                    1,
                    0,
                    0,
                    0,
                    1 / ngsolve.sin(ngsolve.x) ** 2,
                    0,
                    0,
                    0,
                    1 / (ngsolve.sin(ngsolve.x) ** 2 * ngsolve.sin(ngsolve.y) ** 2),
                ),
                dims=(3, 3),
            ),
            "00",
        )
        # Riemann curvature tensor R_{ijkl}=d_iGamma_{jkl}-d_jGamma_{ikl}+Gamma_{jlm}Gamma_{ik}^m-Gamma_{ilm}Gamma_{jk}^m
        self.Riemann = DoubleForm(
            -ngsolve.Det(self.metric)
            * Einsum(
                "ija,klb,ab->ijkl",
                ngsolve.fem.LeviCivitaSymbol(3),
                ngsolve.fem.LeviCivitaSymbol(3),
                self.curvature,
            ),
            dim=3,
            p=2,
            q=2,
        )
        return


class PoincareDisk:
    """
    Hyperbolic metric on the Poincare Disk B_1(0)= {(x,y) in R^2 : x^2+y^2 < 1}. Has constant negative curvature.
    """

    def __init__(self):
        self.metric = TensorField(
            4 / (1 - ngsolve.x**2 - ngsolve.y**2) ** 2 * ngsolve.Id(2), "11"
        )

        self.chr1 = (
            8
            / (1 - (ngsolve.x**2 + ngsolve.y**2)) ** 3
            * ngsolve.CF(
                (
                    ngsolve.x,
                    -ngsolve.y,
                    ngsolve.y,
                    ngsolve.x,
                    ngsolve.y,
                    ngsolve.x,
                    -ngsolve.x,
                    ngsolve.y,
                ),
                dims=(2, 2, 2),
            )
        )
        self.chr2 = (
            2
            / (1 - ngsolve.x**2 - ngsolve.y**2)
            * ngsolve.CF(
                (
                    ngsolve.x,
                    -ngsolve.y,
                    ngsolve.y,
                    ngsolve.x,
                    ngsolve.y,
                    ngsolve.x,
                    -ngsolve.x,
                    ngsolve.y,
                ),
                dims=(2, 2, 2),
            )
        )

        self.Riemann = DoubleForm(
            16
            / (1 - ngsolve.x**2 - ngsolve.y**2) ** 4
            * ngsolve.CF(
                (0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0), dims=(2, 2, 2, 2)
            ),
            dim=2,
            p=2,
            q=2,
        )
        self.Ricci = TensorField(
            -4 / (1 - ngsolve.x**2 - ngsolve.y**2) ** 2 * ngsolve.Id(2), "11"
        )
        self.scalar = ScalarField(ngsolve.CF(-2), dim=2)
        self.Einstein = TensorField(ngsolve.CF((0, 0, 0, 0), dims=(2, 2)), "11")
        self.curvature = ScalarField(ngsolve.CF(-1), dim=2)
        return


#
class HyperbolicH2:
    """
    Hyperbolic metric on H2={(x,y) in R^2 : y > 0}. Has constant negative curvature.
    """

    def __init__(self):
        self.metric = TensorField(1 / ngsolve.y**2 * ngsolve.Id(2), "11")
        self.chr1 = (
            -1 / ngsolve.y**3 * ngsolve.CF((0, -1, 1, 0, 1, 0, 0, -1), dims=(2, 2, 2))
        )
        self.chr2 = (
            -1 / ngsolve.y * ngsolve.CF((0, -1, 1, 0, 1, 0, 0, 1), dims=(2, 2, 2))
        )
        self.Riemann = DoubleForm(
            1
            / ngsolve.y**4
            * ngsolve.CF(
                (0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0), dims=(2, 2, 2, 2)
            ),
            dim=2,
            p=2,
            q=2,
        )
        self.Ricci = TensorField(-1 / ngsolve.y**2 * ngsolve.Id(2), "11")
        self.scalar = ScalarField(ngsolve.CF(-2), dim=2)
        self.Einstein = TensorField(ngsolve.CF((0, 0, 0, 0), dims=(2, 2)), "11")
        self.curvature = ScalarField(ngsolve.CF(-1), dim=2)
        return


#
class HyperbolicH3:
    """
    Hyperbolic metric on H3={(x,y,z) in R^3 : z > 0}. Has constant negative curvature.
    """

    def __init__(self):
        self.metric = TensorField(1 / ngsolve.z**2 * ngsolve.Id(3), "11")
        self.chr1 = (
            -1
            / ngsolve.z**3
            * ngsolve.CF(
                (
                    0,
                    0,
                    -1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                ),
                dims=(3, 3, 3),
            )
        )
        self.chr2 = (
            -1
            / ngsolve.z
            * ngsolve.CF(
                (
                    0,
                    0,
                    -1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                ),
                dims=(3, 3, 3),
            )
        )
        self.Ricci = TensorField(-2 / ngsolve.z**2 * ngsolve.Id(3), "11")
        self.scalar = ScalarField(ngsolve.CF(-6), dim=3)
        # G_{ij}=R_{ij}-0.5*g_{ij}R
        self.Einstein = TensorField(1 / ngsolve.z**2 * ngsolve.Id(3), "11")
        self.Riemann = DoubleForm(
            1
            / ngsolve.z**4
            * (
                Einsum("ik,jl->ijkl", self.metric, self.metric)
                - Einsum("il,jk->ijkl", self.metric, self.metric)
            ),
            dim=3,
            p=2,
            q=2,
        )

        self.curvature = TensorField(-(ngsolve.z**2) * ngsolve.Id(3), "00")
        return


class Heisenberg:
    """
    Heisenberg metric on R^3. Has non-zero Ricci curvature.
    """

    def __init__(self):
        self.metric = TensorField(
            ngsolve.CF(
                (1, 0, 0, 0, 1 + ngsolve.x**2, -ngsolve.x, 0, -ngsolve.x, 1),
                dims=(3, 3),
            ),
            "11",
        )
        self.chr1 = ngsolve.CF(
            (
                0,
                0,
                0,
                0,
                ngsolve.x,
                -1 / 2,
                0,
                -1 / 2,
                0,
                0,
                ngsolve.x,
                -1 / 2,
                -ngsolve.x,
                0,
                0,
                1 / 2,
                0,
                0,
                0,
                -1 / 2,
                0,
                1 / 2,
                0,
                0,
                0,
                0,
                0,
            ),
            dims=(3, 3, 3),
        )
        self.chr2 = ngsolve.CF(
            (
                0,
                0,
                0,
                0,
                ngsolve.x / 2,
                ngsolve.x**2 / 2 - 1 / 2,
                0,
                -1 / 2,
                -ngsolve.x / 2,
                0,
                ngsolve.x / 2,
                ngsolve.x**2 / 2 - 1 / 2,
                -ngsolve.x,
                0,
                0,
                1 / 2,
                0,
                0,
                0,
                -1 / 2,
                -ngsolve.x / 2,
                1 / 2,
                0,
                0,
                0,
                0,
                0,
            ),
            dims=(3, 3, 3),
        )
        self.Ricci = TensorField(
            ngsolve.CF(
                (
                    -0.5,
                    0,
                    0,
                    0,
                    0.5 * (ngsolve.x**2 - 1),
                    -ngsolve.x / 2,
                    0,
                    -ngsolve.x / 2,
                    0.5,  # 0?
                ),
                dims=(3, 3),
            ),
            "11",
        )
        self.scalar = ScalarField(ngsolve.CF(-0.5), dim=3)
        self.Einstein = TensorField(self.Ricci - 0.5 * self.scalar * self.metric, "11")
        self.Riemann = DoubleForm(
            ngsolve.CF(
                (
                    # i=0, j=0  (k,l block)
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    # i=0, j=1
                    0,
                    (3 - ngsolve.x**2) / 4,
                    ngsolve.x / 4,
                    (ngsolve.x**2 - 3) / 4,
                    0,
                    0,
                    -ngsolve.x / 4,
                    0,
                    0,
                    # i=0, j=2
                    0,
                    ngsolve.x / 4,
                    -1 / 4,
                    -ngsolve.x / 4,
                    0,
                    0,
                    1 / 4,
                    0,
                    0,
                    # i=1, j=0
                    0,
                    (ngsolve.x**2 - 3) / 4,
                    -ngsolve.x / 4,
                    (3 - ngsolve.x**2) / 4,
                    0,
                    0,
                    ngsolve.x / 4,
                    0,
                    0,
                    # i=1, j=1
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    # i=1, j=2
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1 / 4,
                    0,
                    1 / 4,
                    0,
                    # i=2, j=0
                    0,
                    -ngsolve.x / 4,
                    1 / 4,
                    ngsolve.x / 4,
                    0,
                    0,
                    -1 / 4,
                    0,
                    0,
                    # i=2, j=1
                    0,
                    0,
                    0,
                    0,
                    0,
                    1 / 4,
                    0,
                    -1 / 4,
                    0,
                    # i=2, j=2
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
                dims=(3, 3, 3, 3),
            ),
            dim=3,
            p=2,
            q=2,
        )

        self.curvature = TensorField(
            ngsolve.CF(
                (
                    1 / 4,
                    0,
                    0,
                    0,
                    1 / 4,
                    ngsolve.x / 4,
                    0,
                    ngsolve.x / 4,
                    ngsolve.x**2 / 4 - 3 / 4,
                ),
                dims=(3, 3),
            ),
            "00",
        )

        return


class CigarSoliton:
    """
    Cigar soliton metric on R^2.
    """

    def __init__(self, t=0):
        self.metric = TensorField(
            1 / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2) * ngsolve.Id(2), "11"
        )
        self.chr1 = (
            -1
            / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2) ** 2
            * ngsolve.CF(
                (
                    ngsolve.x,
                    -ngsolve.y,
                    ngsolve.y,
                    ngsolve.x,
                    ngsolve.y,
                    ngsolve.x,
                    -ngsolve.x,
                    ngsolve.y,
                ),
                dims=(2, 2, 2),
            )
        )
        self.chr2 = (
            -1
            / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2)
            * ngsolve.CF(
                (
                    ngsolve.x,
                    -ngsolve.y,
                    ngsolve.y,
                    ngsolve.x,
                    ngsolve.y,
                    ngsolve.x,
                    -ngsolve.x,
                    ngsolve.y,
                ),
                dims=(2, 2, 2),
            )
        )
        self.Riemann = DoubleForm(
            2
            * ngsolve.exp(4 * t)
            / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2) ** 3
            * ngsolve.CF(
                (0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0), dims=(2, 2, 2, 2)
            ),
            dim=2,
            p=2,
            q=2,
        )
        self.Ricci = TensorField(
            2
            * ngsolve.exp(4 * t)
            / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2) ** 2
            * ngsolve.Id(2),
            "11",
        )
        self.scalar = ScalarField(
            4 * ngsolve.exp(4 * t) / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2),
            dim=2,
        )
        self.Einstein = TensorField(ngsolve.CF((0, 0, 0, 0), dims=(2, 2)), "11")
        self.curvature = ScalarField(
            2 * ngsolve.exp(4 * t) / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2),
            dim=2,
        )
        return


class WarpedProduct:
    """
    Warped product metric on R^3.
    """

    def __init__(self):
        self.metric = TensorField(
            ngsolve.CF(
                (
                    ngsolve.exp(2 * ngsolve.z),
                    0,
                    0,
                    0,
                    ngsolve.exp(2 * ngsolve.z),
                    0,
                    0,
                    0,
                    1,
                ),
                dims=(3, 3),
            ),
            "11",
        )
        self.chr1 = ngsolve.exp(2 * ngsolve.z) * ngsolve.CF(
            (
                0,
                0,
                -1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ),
            dims=(3, 3, 3),
        )
        self.chr2 = ngsolve.CF(
            (
                0,
                0,
                -ngsolve.exp(2 * ngsolve.z),
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -ngsolve.exp(2 * ngsolve.z),
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ),
            dims=(3, 3, 3),
        )
        self.Riemann = DoubleForm(
            Einsum("ik,jl->ijkl", self.metric, self.metric)
            - Einsum("il,jk->ijkl", self.metric, self.metric),
            dim=3,
            p=2,
            q=2,
        )
        self.Ricci = TensorField(
            -2
            * ngsolve.CF(
                (
                    ngsolve.exp(2 * ngsolve.z),
                    0,
                    0,
                    0,
                    ngsolve.exp(2 * ngsolve.z),
                    0,
                    0,
                    0,
                    1,
                ),
                dims=(3, 3),
            ),
            "11",
        )
        self.scalar = ScalarField(ngsolve.CF(-6), dim=3)
        self.curvature = TensorField(-ngsolve.Inv(self.metric), "00")
        self.Einstein = TensorField(self.Ricci - 0.5 * self.scalar * self.metric, "11")
        return


def TestMetric(dim, order=4):
    xvec = [ngsolve.x, ngsolve.y, ngsolve.z]
    return 10 * ngsolve.Id(dim) + 0.1 * ngsolve.CF(
        tuple(
            [
                xvec[i] ** order
                - 3 * xvec[j] ** order
                + 5 * (xvec[(i + 1) % dim] * xvec[(j + 2) % dim]) ** int(order / 2)
                + (4 if i == j else 0)
                + (1 / 3 * xvec[(i + 1) % dim] ** 2 if i == 0 and j == 0 else 0)
                for i in range(dim)
                for j in range(dim)
            ]
        ),
        dims=(dim, dim),
    )
