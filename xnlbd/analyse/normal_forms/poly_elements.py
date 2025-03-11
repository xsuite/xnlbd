from __future__ import annotations

import warnings
from abc import ABC
from pathlib import Path
from typing import Protocol, Sequence, Union

import numpy as np
import xobjects as xo  # type: ignore
from scipy.special import binom  # type: ignore
from scipy.special import factorial  # type: ignore
from xtrack.particles.particles import Particles  # type: ignore

from .polynom_base import *

_pkg_root = Path(__file__).parent.absolute()


class PolyElement4D(ABC):
    ele_map: Map

    @staticmethod
    def MultiFringe(
        combined_kn: Sequence[float] | np.ndarray,
        combined_ks: Sequence[float] | np.ndarray,
        is_exit: int,
        order: int,
        part: Particles,
    ) -> None:
        pass

    @staticmethod
    def Drift(part: Particles, length: float) -> Map:

        rpp = part.rpp[0]

        ele_map = Map(
            x_poly=Polynom(
                terms=[Term(coeff=1.0, x_exp=1), Term(coeff=rpp * length, px_exp=1)]
            ),
            px_poly=Polynom(terms=[Term(coeff=1.0, px_exp=1)]),
            y_poly=Polynom(
                terms=[Term(coeff=1.0, y_exp=1), Term(coeff=rpp * length, py_exp=1)]
            ),
            py_poly=Polynom(terms=[Term(coeff=1.0, py_exp=1)]),
        )

        return ele_map

    @staticmethod
    def multipole_compute_dpx_dpy(
        part: Particles,
        knl: Sequence[float] | np.ndarray,
        ksl: Sequence[float] | np.ndarray,
        order: int,
        inv_factorial_order_0: float,
    ) -> tuple[Polynom, Polynom]:

        chi = part.chi[0]

        index = order
        inv_factorial = inv_factorial_order_0

        dpx_poly = Polynom(terms=[Term(coeff=(chi * knl[index] * inv_factorial))])
        dpy_poly = Polynom(terms=[Term(coeff=(chi * ksl[index] * inv_factorial))])
        x = Polynom(terms=[Term(coeff=1, x_exp=1)])
        y = Polynom(terms=[Term(coeff=1, y_exp=1)])
        while index > 0:
            zre = Polynom.sum_Polynoms(
                Polynom.product_Polynoms(dpx_poly, x, int(1e6)),
                Polynom.product_Polynoms(
                    Polynom.product_Coeff_Polynom(-1, dpy_poly), y, int(1e6)
                ),
            )
            zim = Polynom.sum_Polynoms(
                Polynom.product_Polynoms(dpx_poly, y, int(1e6)),
                Polynom.product_Polynoms(dpy_poly, x, int(1e6)),
            )

            inv_factorial *= index
            index -= 1

            this_knl = chi * knl[index]
            this_ksl = chi * ksl[index]

            dpx_poly = Polynom.sum_Polynoms(
                Polynom(terms=[Term(coeff=(this_knl * inv_factorial))]), zre
            )
            dpy_poly = Polynom.sum_Polynoms(
                Polynom(terms=[Term(coeff=(this_ksl * inv_factorial))]), zim
            )

        dpx_poly = Polynom.product_Coeff_Polynom(-1, dpx_poly)

        return dpx_poly, dpy_poly

    @staticmethod
    def Multipole(
        part: Particles,
        hxl: float,
        length: float,
        weight: float,
        knl: Sequence[float] | np.ndarray,
        ksl: Sequence[float] | np.ndarray,
        order: int,
        inv_factorial_order_0: float,
        knl_2: Sequence[float] | np.ndarray,
        ksl_2: Sequence[float] | np.ndarray,
        order_2: int,
        inv_factorial_order_2_0: float,
    ) -> Map:
        warnings.warn(
            "Synchrotron radiation in Multipole is not implemented for 4D normal forms."
        )
        ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            px_poly=Polynom(terms=[]),
            py_poly=Polynom(terms=[]),
        )
        px_poly = Polynom(terms=[Term(coeff=1, px_exp=1)])
        py_poly = Polynom(terms=[Term(coeff=1, py_exp=1)])

        dpx = Polynom(terms=[Term(coeff=0.0)])
        dpy = Polynom(terms=[Term(coeff=0.0)])

        if len(knl) != 0:
            dpx1, dpy1 = PolyElement4D.multipole_compute_dpx_dpy(
                part, knl, ksl, order, inv_factorial_order_0
            )
            dpx = Polynom.sum_Polynoms(dpx, Polynom.product_Coeff_Polynom(weight, dpx1))
            dpy = Polynom.sum_Polynoms(dpy, Polynom.product_Coeff_Polynom(weight, dpy1))

        if len(knl_2) != 0:
            dpx2, dpy2 = PolyElement4D.multipole_compute_dpx_dpy(
                part, knl_2, ksl_2, order_2, inv_factorial_order_2_0
            )
            dpx = Polynom.sum_Polynoms(dpx, Polynom.product_Coeff_Polynom(weight, dpx2))
            dpy = Polynom.sum_Polynoms(dpy, Polynom.product_Coeff_Polynom(weight, dpy2))

        if (hxl > 0) or (hxl < 0):
            delta = part.delta[0]
            chi = part.chi[0]

            hxlx = Polynom(terms=[Term(coeff=hxl, x_exp=1)])

            dpx = Polynom.sum_Polynoms(
                dpx, Polynom(terms=[Term(coeff=(hxl + hxl * delta))])
            )

            if length != 0:
                knl0 = 0.0

                if len(knl) != 0:
                    knl0 += knl[0]

                if len(knl_2) != 0:
                    knl0 += knl_2[0]

                b1l = chi * knl0 * weight

                dpx = Polynom.sum_Polynoms(
                    dpx,
                    Polynom.product_Coeff_Polynom(-b1l / length, hxlx),
                )

        ele_map.px_poly = Polynom.sum_Polynoms(px_poly, dpx)
        ele_map.py_poly = Polynom.sum_Polynoms(py_poly, dpy)

        ele_map.x_poly.remove_zero_terms()
        ele_map.x_poly.collect_terms()
        ele_map.px_poly.remove_zero_terms()
        ele_map.px_poly.collect_terms()
        ele_map.y_poly.remove_zero_terms()
        ele_map.y_poly.collect_terms()
        ele_map.py_poly.remove_zero_terms()
        ele_map.py_poly.collect_terms()

        return ele_map


class PolyIdentity4D(PolyElement4D):

    def __init__(self):
        self.ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            px_poly=Polynom(terms=[Term(coeff=1, px_exp=1)]),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            py_poly=Polynom(terms=[Term(coeff=1, py_exp=1)]),
        )


class PolyReferenceEnergyIncrease4D(PolyElement4D):

    def __init__(self):
        self.ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            px_poly=Polynom(terms=[Term(coeff=1, px_exp=1)]),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            py_poly=Polynom(terms=[Term(coeff=1, py_exp=1)]),
        )
        warnings.warn(
            "ReferenceEnergyIncrease affects longitudinal coordinates only, which is not supported for 4D normal forms, this element will be replaced with the identity transformation instead."
        )


class PolyMarker4D(PolyElement4D):

    def __init__(self):
        self.ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            px_poly=Polynom(terms=[Term(coeff=1, px_exp=1)]),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            py_poly=Polynom(terms=[Term(coeff=1, py_exp=1)]),
        )


class PolyDrift4D(PolyElement4D):

    def __init__(self, part: Particles, length: float) -> None:
        # rpp = part.rpp[0]

        # self.ele_map = Map(
        #     x_poly=Polynom(
        #         terms=[
        #             Term(coeff=1, x_exp=1),
        #             Term(coeff=(rpp * length), px_exp=1),
        #         ]
        #     ),
        #     px_poly=Polynom(terms=[Term(coeff=1, px_exp=1)]),
        #     y_poly=Polynom(
        #         terms=[
        #             Term(coeff=1, y_exp=1),
        #             Term(coeff=(rpp * length), py_exp=1),
        #         ]
        #     ),
        #     py_poly=Polynom(terms=[Term(coeff=1, py_exp=1)]),
        # )

        self.ele_map = PolyElement4D.Drift(part, length)


class PolyCavity4D(PolyElement4D):

    def __init__(self):
        self.ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            px_poly=Polynom(terms=[Term(coeff=1, px_exp=1)]),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            py_poly=Polynom(terms=[Term(coeff=1, py_exp=1)]),
        )
        warnings.warn(
            "Cavity affects longitudinal coordinates only, which is not supported for 4D normal forms, this element will be replaced with the identity transformation instead."
        )


class PolyXYShift4D(PolyElement4D):

    def __init__(self) -> None:

        raise NotImplementedError("XYShift is not supported for normal forms.")


### TODO: Elens


class PolyNonLinearLens4D(PolyElement4D):

    def __init__(self) -> None:

        raise NotImplementedError("NonLinearLens is not supported for normal forms.")


### TODO: Wire


class PolySRotation4D(PolyElement4D):

    def __init__(
        self,
        cos_z: float,
        sin_z: float,
    ) -> None:
        self.ele_map = Map(
            x_poly=Polynom(
                terms=[Term(coeff=cos_z, x_exp=1), Term(coeff=sin_z, y_exp=1)]
            ),
            px_poly=Polynom(
                terms=[
                    Term(coeff=cos_z, px_exp=1),
                    Term(coeff=sin_z, py_exp=1),
                ]
            ),
            y_poly=Polynom(
                terms=[
                    Term(coeff=-sin_z, x_exp=1),
                    Term(coeff=cos_z, y_exp=1),
                ]
            ),
            py_poly=Polynom(
                terms=[
                    Term(coeff=-sin_z, px_exp=1),
                    Term(coeff=cos_z, py_exp=1),
                ]
            ),
        )


### TODO: XRotation


### TODO; YRotation


class PolyZetaShift4D(PolyElement4D):

    def __init__(self):
        self.ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            px_poly=Polynom(terms=[Term(coeff=1, px_exp=1)]),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            py_poly=Polynom(terms=[Term(coeff=1, py_exp=1)]),
        )
        warnings.warn(
            "ZetaShift affects longitudinal coordinates only, which is not supported for 4D normal forms, this element will be replaced with the identity transformation instead."
        )


class PolyMultipole4D(PolyElement4D):

    def __init__(
        self,
        part: Particles,
        order: int,
        inv_factorial_order: float,
        knl: Sequence[float] | np.ndarray,
        ksl: Sequence[float] | np.ndarray,
        hxl: float,
        length: float,
        max_order: int,
    ):
        # warnings.warn(
        #     "Synchrotron radiation in Multipole is not implemented for 4D normal forms."
        # )
        # map = Map(
        #     x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
        #     y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
        #     px_poly=Polynom(terms=[]),
        #     py_poly=Polynom(terms=[]),
        # )
        # px_poly = Polynom(terms=[Term(coeff=1, px_exp=1)])
        # py_poly = Polynom(terms=[Term(coeff=1, py_exp=1)])

        # chi = part.chi[0]
        # delta = part.delta[0]

        # index = order
        # inv_factorial = inv_factorial_order

        # dpx_poly = Polynom(terms=[Term(coeff=(chi * knl[index] * inv_factorial))])
        # dpy_poly = Polynom(terms=[Term(coeff=(chi * ksl[index] * inv_factorial))])
        # x = Polynom(terms=[Term(coeff=1, x_exp=1)])
        # y = Polynom(terms=[Term(coeff=1, y_exp=1)])
        # while index > 0:
        #     zre = Polynom.sum_Polynoms(
        #         Polynom.product_Polynoms(dpx_poly, x, int(1e6)),
        #         Polynom.product_Polynoms(
        #             Polynom.product_Coeff_Polynom(-1, dpy_poly), y, int(1e6)
        #         ),
        #     )
        #     zim = Polynom.sum_Polynoms(
        #         Polynom.product_Polynoms(dpx_poly, y, int(1e6)),
        #         Polynom.product_Polynoms(dpy_poly, x, int(1e6)),
        #     )

        #     inv_factorial *= index
        #     index -= 1

        #     this_knl = chi * knl[index]
        #     this_ksl = chi * ksl[index]

        #     dpx_poly = Polynom.sum_Polynoms(
        #         Polynom(terms=[Term(coeff=(this_knl * inv_factorial))]), zre
        #     )
        #     dpy_poly = Polynom.sum_Polynoms(
        #         Polynom(terms=[Term(coeff=(this_ksl * inv_factorial))]), zim
        #     )

        # dpx_poly = Polynom.product_Coeff_Polynom(-1, dpx_poly)

        # if (hxl > 0) or (hxl < 0):
        #     dpx_poly = Polynom.sum_Polynoms(
        #         dpx_poly, Polynom(terms=[Term(coeff=(hxl + hxl * delta))])
        #     )

        #     if length != 0:
        #         knl0 = 0.0

        #         if len(knl) != 0:
        #             knl0 += knl[0]

        #         b1l = chi * knl0

        #         dpx_poly = Polynom.sum_Polynoms(
        #             dpx_poly,
        #             Polynom.product_Coeff_Polynom(
        #                 -b1l / length, Polynom(terms=[Term(coeff=hxl, x_exp=1)])
        #             ),
        #         )

        # map.px_poly = Polynom.sum_Polynoms(px_poly, dpx_poly)
        # map.py_poly = Polynom.sum_Polynoms(py_poly, dpy_poly)

        # map.x_poly.truncate_at_order(max_order)
        # map.x_poly.remove_zero_terms()
        # map.x_poly.collect_terms()

        # map.px_poly.truncate_at_order(max_order)
        # map.px_poly.remove_zero_terms()
        # map.px_poly.collect_terms()

        # map.y_poly.truncate_at_order(max_order)
        # map.y_poly.remove_zero_terms()
        # map.y_poly.collect_terms()

        # map.py_poly.truncate_at_order(max_order)
        # map.py_poly.remove_zero_terms()
        # map.py_poly.collect_terms()

        ele_map = PolyElement4D.Multipole(
            part, hxl, length, 1, knl, ksl, order, inv_factorial_order, [], [], -1, -1
        )

        ele_map.x_poly.truncate_at_order(max_order)
        ele_map.px_poly.truncate_at_order(max_order)
        ele_map.y_poly.truncate_at_order(max_order)
        ele_map.py_poly.truncate_at_order(max_order)

        self.ele_map = ele_map


class PolySimpleThinQuadrupole4D(PolyElement4D):

    def __init__(
        self,
        part: Particles,
        knl: Sequence[float],
    ) -> None:
        chi = part.chi[0]
        knl1 = knl[1]

        self.ele_map = Map(
            x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
            px_poly=Polynom(
                terms=[
                    Term(coeff=1, px_exp=1),
                    Term(coeff=-chi * knl1, x_exp=1),
                ]
            ),
            y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
            py_poly=Polynom(
                terms=[
                    Term(coeff=1, py_exp=1),
                    Term(coeff=chi * knl1, y_exp=1),
                ]
            ),
        )


### TODO: Bend


class PolySextupole4D(PolyElement4D):

    def __init__(
        self,
        part: Particles,
        k2: float,
        k2s: float,
        length: float,
        order: int,
        inv_factorial_order: float,
        knl: Sequence[float],
        ksl: Sequence[float],
        edge_entry_active: int,
        edge_exit_active: int,
        max_order: int,
    ) -> None:
        if edge_entry_active or edge_exit_active:
            raise NotImplementedError(
                "Multipole fringes are not supported for 4D normal forms."
            )

        knl_sext = [0.0, 0.0, k2 * length]
        ksl_sext = [0.0, 0.0, k2s * length]

        combined_kn = [0, 0, k2 / 2]
        combined_ks = [0, 0, k2s / 2]

        drift1_map = self.Drift(part, length / 2.0)

        multipole_map = self.Multipole(
            part,
            0.0,
            length,
            1,
            knl,
            ksl,
            order,
            inv_factorial_order,
            knl_sext,
            ksl_sext,
            2,
            0.5,
        )

        drift2_map = self.Drift(part, length / 2.0)

        ele_map = Map.composition_Map(drift1_map, multipole_map, max_order)
        ele_map = Map.composition_Map(ele_map, drift2_map, max_order)

        self.ele_map = ele_map


class PolyOctupole4D(PolyElement4D):

    def __init__(
        self,
        part: Particles,
        k3: float,
        k3s: float,
        length: float,
        order: int,
        inv_factorial_order: float,
        knl: Sequence[float],
        ksl: Sequence[float],
        edge_entry_active: int,
        edge_exit_active: int,
        max_order: int,
    ) -> None:
        if edge_entry_active or edge_exit_active:
            raise NotImplementedError(
                "Multipole fringes are not supported for 4D normal forms."
            )

        knl_oct = [0.0, 0.0, 0.0, k3 * length]
        ksl_oct = [0.0, 0.0, 0.0, k3s * length]

        combined_kn = [0, 0, 0, k3 / 6]
        combined_ks = [0, 0, 0, k3s / 6]

        drift1_map = self.Drift(part, length / 2.0)

        multipole_map = self.Multipole(
            part,
            0.0,
            length,
            1,
            knl,
            ksl,
            order,
            inv_factorial_order,
            knl_oct,
            ksl_oct,
            3,
            1.0 / 6.0,
        )

        drift2_map = self.Drift(part, length / 2.0)

        ele_map = Map.composition_Map(drift1_map, multipole_map, max_order)
        ele_map = Map.composition_Map(ele_map, drift2_map, max_order)

        self.ele_map = ele_map


### TODO: Quadrupole


### TODO: Solenoid


### TODO: CombineFunctionMagnet


### TODO: DipoleFringe


### TODO: Wedge


### TODO: SimpleThinBend4D


### TODO: RFMultipole


class PolyDipoleEdge4D(PolyElement4D):

    def __init__(self, part: Particles, r21: float, r43: float, model: int):
        chi = part.chi[0]

        if model == 0:
            self.ele_map = Map(
                x_poly=Polynom(terms=[Term(coeff=1, x_exp=1)]),
                px_poly=Polynom(
                    terms=[
                        Term(coeff=1, px_exp=1),
                        Term(coeff=(chi * r21), x_exp=1),
                    ]
                ),
                y_poly=Polynom(terms=[Term(coeff=1, y_exp=1)]),
                py_poly=Polynom(
                    terms=[
                        Term(coeff=1, py_exp=1),
                        Term(coeff=(chi * r43), y_exp=1),
                    ]
                ),
            )
        else:
            raise NotImplementedError(
                "Nonlinear DipoleEdge not yet implemented for 4D normal forms."
            )


### TODO: LineSegmentMap


### TODO: FirstOrderTaylorMap


### TODO: LinearTransferMatrix


### TODO: SecondOrderTaylorMap


class PolyHenonmap4D(PolyElement4D):

    def __init__(
        self,
        part: Particles,
        cos_omega_x: float,
        sin_omega_x: float,
        cos_omega_y: float,
        sin_omega_y: float,
        domegax: float,
        domegay: float,
        n_turns: int,
        twiss_params: Sequence[float],
        dx: float,
        ddx: float,
        fx_coeffs: Sequence[float],
        fx_x_exps: Sequence[int],
        fx_y_exps: Sequence[int],
        fy_coeffs: Sequence[float],
        fy_x_exps: Sequence[int],
        fy_y_exps: Sequence[int],
        norm: bool,
        max_order: int,
    ) -> None:
        if n_turns != 1:
            raise NotImplementedError(
                "Multi-turn Henon map is not supported for 4D normal forms."
            )

        delta = part.delta

        n_fx_coeffs = len(fx_coeffs)
        n_fy_coeffs = len(fy_coeffs)

        alpha_x = twiss_params[0]
        beta_x = twiss_params[1]
        alpha_y = twiss_params[2]
        beta_y = twiss_params[3]
        sqrt_beta_x = np.sqrt(beta_x)
        sqrt_beta_y = np.sqrt(beta_y)

        if norm:
            x_hat = Polynom(terms=[Term(coeff=1.0, x_exp=1)])
            px_hat = Polynom(terms=[Term(coeff=1.0, px_exp=1)])
            y_hat = Polynom(terms=[Term(coeff=1.0, y_exp=1)])
            py_hat = Polynom(terms=[Term(coeff=1.0, py_exp=1)])
        else:
            x_hat = Polynom(terms=[Term(coeff=1.0 / sqrt_beta_x, x_exp=1)])
            px_hat = Polynom(
                terms=[
                    Term(coeff=alpha_x / sqrt_beta_x, x_exp=1),
                    Term(coeff=sqrt_beta_x, px_exp=1),
                ]
            )
            y_hat = Polynom(terms=[Term(coeff=1.0 / sqrt_beta_y, y_exp=1)])
            py_hat = Polynom(
                terms=[
                    Term(coeff=alpha_y / sqrt_beta_y, y_exp=1),
                    Term(coeff=sqrt_beta_y, py_exp=1),
                ]
            )
        x_hat_f = Polynom(terms=[Term(coeff=dx * delta / sqrt_beta_x)])
        px_hat_f = Polynom(
            terms=[
                Term(
                    coeff=alpha_x * dx * delta / sqrt_beta_x + ddx * delta * sqrt_beta_x
                )
            ]
        )

        multipole_scale = 1.0 / (1.0 + delta)

        if domegax == 0:
            curr_cos_omega_x = cos_omega_x
            curr_sin_omega_x = sin_omega_x
        else:
            cos_domega_x = np.cos(domegax * delta)
            sin_domega_x = np.sin(domegax * delta)
            curr_cos_omega_x = cos_omega_x * cos_domega_x - sin_omega_x * sin_domega_x
            curr_sin_omega_x = sin_omega_x * cos_domega_x + cos_omega_x * sin_domega_x
        if domegay == 0:
            curr_cos_omega_y = cos_omega_y
            curr_sin_omega_y = sin_omega_y
        else:
            cos_domega_y = np.cos(domegay * delta)
            sin_domega_y = np.sin(domegay * delta)
            curr_cos_omega_y = cos_omega_y * cos_domega_y - sin_omega_y * sin_domega_y
            curr_sin_omega_y = sin_omega_y * cos_domega_y + cos_omega_y * sin_domega_y

        fx = Polynom(terms=[Term(coeff=0.0)])
        for i in range(n_fx_coeffs):
            prod = Polynom(terms=[Term(coeff=fx_coeffs[i] * multipole_scale)])
            x_power = fx_x_exps[i]
            y_power = fx_y_exps[i]
            for j in range(x_power):
                prod = Polynom.product_Polynoms(
                    prod, Polynom.product_Coeff_Polynom(sqrt_beta_x, x_hat), max_order
                )
            for j in range(y_power):
                prod = Polynom.product_Polynoms(
                    prod, Polynom.product_Coeff_Polynom(sqrt_beta_y, y_hat), max_order
                )
            fx = Polynom.sum_Polynoms(fx, prod)

        fy = Polynom(terms=[Term(coeff=0.0)])
        for i in range(n_fy_coeffs):
            prod = Polynom(terms=[Term(coeff=fy_coeffs[i] * multipole_scale)])
            x_power = fy_x_exps[i]
            y_power = fy_y_exps[i]
            for j in range(x_power):
                prod = Polynom.product_Polynoms(
                    prod, Polynom.product_Coeff_Polynom(sqrt_beta_x, x_hat), max_order
                )
            for j in range(y_power):
                prod = Polynom.product_Polynoms(
                    prod, Polynom.product_Coeff_Polynom(sqrt_beta_y, y_hat), max_order
                )
            fy = Polynom.sum_Polynoms(fy, prod)

        fx = Polynom.product_Coeff_Polynom(sqrt_beta_x, fx)
        fy = Polynom.product_Coeff_Polynom(sqrt_beta_y, fy)

        x_hat_new = Polynom.sum_Polynoms(
            Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    curr_cos_omega_x,
                    Polynom.sum_Polynoms(
                        x_hat, Polynom.product_Coeff_Polynom(-1, x_hat_f)
                    ),
                ),
                Polynom.product_Coeff_Polynom(
                    curr_sin_omega_x,
                    Polynom.sum_Polynoms(
                        px_hat,
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(-1, px_hat_f), fx
                        ),
                    ),
                ),
            ),
            x_hat_f,
        )
        px_hat_new = Polynom.sum_Polynoms(
            Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    -curr_sin_omega_x,
                    Polynom.sum_Polynoms(
                        x_hat, Polynom.product_Coeff_Polynom(-1, x_hat_f)
                    ),
                ),
                Polynom.product_Coeff_Polynom(
                    curr_cos_omega_x,
                    Polynom.sum_Polynoms(
                        px_hat,
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(-1, px_hat_f), fx
                        ),
                    ),
                ),
            ),
            px_hat_f,
        )
        y_hat_new = Polynom.sum_Polynoms(
            Polynom.product_Coeff_Polynom(curr_cos_omega_y, y_hat),
            Polynom.product_Coeff_Polynom(
                curr_sin_omega_y, Polynom.sum_Polynoms(py_hat, fy)
            ),
        )
        py_hat_new = Polynom.sum_Polynoms(
            Polynom.product_Coeff_Polynom(-curr_sin_omega_y, y_hat),
            Polynom.product_Coeff_Polynom(
                curr_cos_omega_y, Polynom.sum_Polynoms(py_hat, fy)
            ),
        )

        if norm:
            self.ele_map = Map(
                x_poly=x_hat_new,
                px_poly=px_hat_new,
                y_poly=y_hat_new,
                py_poly=py_hat_new,
            )
        else:
            self.ele_map = Map(
                x_poly=Polynom.product_Coeff_Polynom(sqrt_beta_x, x_hat_new),
                px_poly=Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(-alpha_x / sqrt_beta_x, x_hat_new),
                    Polynom.product_Coeff_Polynom(1.0 / sqrt_beta_x, px_hat_new),
                ),
                y_poly=Polynom.product_Coeff_Polynom(sqrt_beta_y, y_hat_new),
                py_poly=Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(-alpha_y / sqrt_beta_y, y_hat_new),
                    Polynom.product_Coeff_Polynom(1.0 / sqrt_beta_y, py_hat_new),
                ),
            )
        self.ele_map.x_poly.remove_zero_terms()
        self.ele_map.x_poly.truncate_at_order(max_order)
        self.ele_map.x_poly.collect_terms()
        self.ele_map.px_poly.remove_zero_terms()
        self.ele_map.px_poly.truncate_at_order(max_order)
        self.ele_map.px_poly.collect_terms()
        self.ele_map.y_poly.remove_zero_terms()
        self.ele_map.y_poly.truncate_at_order(max_order)
        self.ele_map.y_poly.collect_terms()
        self.ele_map.py_poly.remove_zero_terms()
        self.ele_map.py_poly.truncate_at_order(max_order)
        self.ele_map.py_poly.collect_terms()


class PolyModulatedHenonmap4D(PolyElement4D):

    def __init__(self) -> None:
        raise NotImplementedError(
            "Time dependent elements are not supported for 4D normal forms."
        )
