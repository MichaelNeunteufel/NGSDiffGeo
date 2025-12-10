import ngsolve
from ngsolve.fem import Einsum
from .wrappers import TensorField

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
        self.metric = ngsolve.Id(dim)

        # flat manifold
        self.chr1 = ngsolve.CF((0,) * dim**3, dims=(dim, dim, dim))
        self.chr2 = ngsolve.CF((0,) * dim**3, dims=(dim, dim, dim))
        self.Riemann = ngsolve.CF((0,) * dim**4, dims=(dim, dim, dim, dim))
        self.Ricci = ngsolve.CF((0,) * dim**2, dims=(dim, dim))
        self.scalar = ngsolve.CF(0)
        self.Einstein = ngsolve.CF((0,) * dim**2, dims=(dim, dim))
        self.curvature = ngsolve.CF(0)
        return


class Sphere2:
    """
    Standard metric on sphere S^2. x and y are interpreted as angles; x in [0,pi], y in [0,2*pi). Has constant positive curvature.
    """

    def __init__(self):
        # metric tensor
        self.metric = ngsolve.CF((1, 0, 0, ngsolve.sin(ngsolve.x) ** 2), dims=(2, 2))
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
        # Riemann curvature tensor R_{ijkl}=d_jGamma_{ikl}-d_kGamma_{ijl}+Gamma_{ijm}Gamma_{mkl}-Gamma_{ikm}Gamma_{mjl}
        self.Riemann = ngsolve.CF(
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
        )
        # Ricci curvature tensor R_{ij}=g^{kl}R_{kilj}=-g^{kl}R_{ikjl}
        self.Ricci = ngsolve.CF((1, 0, 0, ngsolve.sin(ngsolve.x) ** 2), dims=(2, 2))
        # Scalar curvature R=g^{ij}R_{ij}
        self.scalar = ngsolve.CF(2)
        # Einstein tensor G_{ij}=R_{ij}-0.5*g_{ij}R
        self.Einstein = ngsolve.CF((0, 0, 0, 0), dims=(2, 2))
        # Curvature operator is the Gauss curvature in 2D
        self.curvature = ngsolve.CF(1)
        return


#
class Sphere3:
    """
    Standard metric on sphere S^3. x, y, and z are interpreted as angles; x in [0,pi], y in [0,pi], z in [0,2*pi). Has constant positive curvature.
    """

    def __init__(self):
        # metric
        self.metric = ngsolve.CF(
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
                -(ngsolve.sin(ngsolve.x) ** 3) * ngsolve.cos(ngsolve.y),
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

        # is correct, but with Einsum it's faster
        # Q00 = (ngsolve.sin(ngsolve.x) ** 4) * ngsolve.sin(ngsolve.y) ** 2
        # Q11 = ngsolve.sin(ngsolve.x) ** 2 * ngsolve.sin(ngsolve.y) ** 2
        # Q22 = ngsolve.sin(ngsolve.x) ** 2
        # self.Riemann = ngsolve.CF(
        #     (
        #         0,  # 0000
        #         0,  # 0001
        #         0,  # 0002
        #         0,  # 0010
        #         0,  # 0011
        #         0,  # 0012
        #         0,  # 0020
        #         0,  # 0021
        #         0,  # 0022
        #         0,  # 0100
        #         -Q22,  # 0101
        #         0,  # 0102
        #         Q22,  # 0110
        #         0,  # 0111
        #         0,  # 0112
        #         0,  # 0120
        #         0,  # 0121
        #         0,  # 0122
        #         0,  # 0200
        #         0,  # 0201
        #         -Q11,  # 0202
        #         0,  # 0210
        #         0,  # 0211
        #         0,  # 0212
        #         Q11,  # 0220
        #         0,  # 0221
        #         0,  # 0222
        #         0,  # 1000
        #         Q22,  # 1001
        #         0,  # 1002
        #         -Q22,  # 1010
        #         0,  # 1011
        #         0,  # 1012
        #         0,  # 1020
        #         0,  # 1021
        #         0,  # 1022
        #         0,  # 1100
        #         0,  # 1101
        #         0,  # 1102
        #         0,  # 1110
        #         0,  # 1111
        #         0,  # 1112
        #         0,  # 1120
        #         0,  # 1121
        #         0,  # 1122
        #         0,  # 1200
        #         0,  # 1201
        #         0,  # 1202
        #         0,  # 1210
        #         0,  # 1211
        #         -Q00,  # 1212
        #         0,  # 1220
        #         Q00,  # 1221
        #         0,  # 1222
        #         0,  # 2000
        #         0,  # 2001
        #         Q11,  # 2002
        #         0,  # 2010
        #         0,  # 2011
        #         0,  # 2012
        #         -Q11,  # 2020
        #         0,  # 2021
        #         0,  # 2022
        #         0,  # 2100
        #         0,  # 2101
        #         0,  # 2102
        #         0,  # 2110
        #         0,  # 2111
        #         Q00,  # 2112
        #         0,  # 2120
        #         -Q00,  # 2121
        #         0,  # 2122
        #         0,  # 2200
        #         0,  # 2201
        #         0,  # 2202
        #         0,  # 2210
        #         0,  # 2211
        #         0,  # 2212
        #         0,  # 2220
        #         0,  # 2221
        #         0,  # 2222
        #     ),
        #     dims=(3, 3, 3, 3),
        # )

        self.Ricci = 2 * self.metric
        self.scalar = ngsolve.CF(6)
        self.Einstein = -self.metric
        # curvature = g^{-1}
        self.curvature = ngsolve.CF(
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
        )

        self.Riemann = -ngsolve.Det(self.metric) * Einsum(
            "ija,klb,ab->ijkl",
            ngsolve.fem.LeviCivitaSymbol(3),
            ngsolve.fem.LeviCivitaSymbol(3),
            self.curvature,
        )
        return


class PoincareDisk:
    """
    Hyperbolic metric on the Poincare Disk B_1(0)= {(x,y) in R^2 : x^2+y^2 < 1}. Has constant negative curvature.
    """

    def __init__(self):
        self.metric = 4 / (1 - ngsolve.x**2 - ngsolve.y**2) ** 2 * ngsolve.Id(2)

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

        self.Riemann = (
            16
            / (1 - ngsolve.x**2 - ngsolve.y**2) ** 4
            * ngsolve.CF(
                (0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0), dims=(2, 2, 2, 2)
            )
        )
        self.Ricci = -4 / (1 - ngsolve.x**2 - ngsolve.y**2) ** 2 * ngsolve.Id(2)
        self.scalar = ngsolve.CF(-2)
        self.Einstein = ngsolve.CF((0, 0, 0, 0), dims=(2, 2))
        self.curvature = ngsolve.CF(-1)
        return


#
class HyperbolicH2:
    """
    Hyperbolic metric on H2={(x,y) in R^2 : y > 0}. Has constant negative curvature.
    """

    def __init__(self):
        self.metric = TensorField(1 / ngsolve.y**2 * ngsolve.Id(2), "11")
        self.chr1 = (
            -1 / ngsolve.y**3 * ngsolve.CF((0, -1, 1, 0, 1, 0, 0, 1), dims=(2, 2, 2))
        )
        self.chr2 = (
            -1 / ngsolve.y * ngsolve.CF((0, -1, 1, 0, 1, 0, 0, 1), dims=(2, 2, 2))
        )
        self.Riemann = (
            1
            / ngsolve.y**4
            * ngsolve.CF(
                (0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0), dims=(2, 2, 2, 2)
            )
        )
        self.Ricci = -1 / ngsolve.y**2 * ngsolve.Id(2)
        self.scalar = ngsolve.CF(-2)
        self.Einstein = ngsolve.CF((0, 0, 0, 0), dims=(2, 2))
        self.curvature = ngsolve.CF(-1)
        return


#
class HyperbolicH3:
    """
    Hyperbolic metric on H3={(x,y,z) in R^3 : z > 0}. Has constant negative curvature.
    """

    def __init__(self):
        self.metric = 1 / ngsolve.z**2 * ngsolve.Id(3)
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
        self.Riemann = None
        self.Ricci = -2 / ngsolve.z**2 * ngsolve.Id(3)
        self.scalar = -6
        # G_{ij}=R_{ij}-0.5*g_{ij}R
        self.Einstein = 1 / ngsolve.z**2 * ngsolve.Id(3)
        self.curvature = ngsolve.z**2 * ngsolve.Id(3)
        return


class Heisenberg:
    """
    Heisenberg metric on R^3. Has non-zero Ricci curvature.
    """

    def __init__(self):
        self.metric = ngsolve.CF(
            (1, 0, 0, 0, 1 + ngsolve.x**2, -ngsolve.x, 0, -ngsolve.x, 1), dims=(3, 3)
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
        self.Riemann = None
        self.Ricci = ngsolve.CF(
            (
                -0.5,
                0,
                0,
                0,
                0.5 * (ngsolve.x**2 - 1),
                -ngsolve.x / 2,
                0,
                -ngsolve.x / 2,
                0,
            ),
            dims=(3, 3),
        )
        self.scalar = -0.5
        self.Einstein = self.Ricci - 0.5 * self.scalar * self.metric
        self.curvature = None
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
        self.Riemann = (
            2
            * ngsolve.exp(4 * t)
            / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2) ** 3
            * ngsolve.CF(
                (0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0), dims=(2, 2, 2, 2)
            )
        )
        self.Ricci = (
            2
            * ngsolve.exp(4 * t)
            / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2) ** 2
            * ngsolve.Id(2)
        )
        self.scalar = (
            4 * ngsolve.exp(4 * t) / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2)
        )
        self.Einstein = ngsolve.CF((0, 0, 0, 0), dims=(2, 2))
        self.curvature = (
            2 * ngsolve.exp(4 * t) / (ngsolve.exp(4 * t) + ngsolve.x**2 + ngsolve.y**2)
        )
        return


class WarpedProduct:
    """
    Warped product metric on R^3.
    """

    def __init__(self):
        self.metric = ngsolve.CF(
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
        self.Riemann = None
        self.Ricci = None
        self.scalar = -6
        self.curvature = None
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
