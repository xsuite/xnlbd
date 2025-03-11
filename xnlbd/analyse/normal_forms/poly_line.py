from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
import xtrack as xt  # type: ignore[import-untyped, import-not-found]
from xdeps.refs import BaseRef  # type: ignore[import-untyped, import-not-found]
from xtrack import Line  # type: ignore[import-untyped, import-not-found]
from xtrack.beam_elements.apertures import (  # type: ignore[import-untyped, import-not-found]
    LimitEllipse,
    LimitPolygon,
    LimitRacetrack,
    LimitRect,
    LimitRectEllipse,
    LongitudinalLimitRect,
)
from xtrack.beam_elements.elements import (  # type: ignore[import-untyped, import-not-found]
    Bend,
    Cavity,
    CombinedFunctionMagnet,
    DipoleEdge,
    DipoleFringe,
    Drift,
    Elens,
    FirstOrderTaylorMap,
    LinearTransferMatrix,
    LineSegmentMap,
    Marker,
    Multipole,
    NonLinearLens,
    Octupole,
    Quadrupole,
    ReferenceEnergyIncrease,
    RFMultipole,
    SecondOrderTaylorMap,
    Sextupole,
    SimpleThinBend,
    SimpleThinQuadrupole,
    Solenoid,
    SRotation,
    Wedge,
    Wire,
    XRotation,
    XYShift,
    YRotation,
    ZetaShift,
)
from xtrack.line import (  # type: ignore[import-untyped, import-not-found]
    LineVars,
    VarValues,
)
from xtrack.particles.particles import (  # type: ignore[import-untyped, import-not-found]
    Particles,
)

from xnlbd.tools import NormedParticles
from xnlbd.track.elements import Henonmap

from .numeric_normal_forms import NormalForm4D
from .poly_elements import *
from .polynom_base import Map, Polynom, Term


class PolyLine4D:
    """
    Class for 4D polynomial representation of a Line used for calculating
    one-turn maps and normal forms.
    """

    def __init__(
        self,
        line: Line,
        part: Particles,
        max_ele_order: int,
        max_map_order: int,
        nemitt_x: float = 1,
        nemitt_y: float = 1,
        nemitt_z: float = 1,
    ) -> None:
        """
        Initialiser for PolyLine4D class.

        Input:
            - line: Line object constructed with xtrack that represents the
              accelerator lattice
            - part: Particles object constructed by xtrack that represents the
              reference particle for the one-turn map and normal form
              calculation (attention: this may be different from the reference
              particle of the line)
            - max_ele_order: integer, maximum order at which individual
              element maps will be truncated
            - max_map_order: integer, maximum order at which the one-turn map
              will be truncated
            - nemitt_x: float, horizontal normalised emittance used for
              coordinate normalisation
            - nemitt_y: float, vertical normalised emittance used for
              coordinate normalisation
            - nemitt_z: float, longitudinal normalised emittance used for
              coordinate normalisation

        Output:
            -
        """

        self.poly_elements: list[PolyElement4D] = []
        self.one_turn_map: Map | None = None
        self.one_turn_map_real: Map | None = None

        self._complex_one_turn_map: Map | None = None

        self.normal_form: NormalForm4D | None = None

        self.max_map_order: int = max_map_order
        self.max_ele_order: int = max_ele_order

        tw = line.twiss(continue_on_closed_orbit_error=False)
        self.W_matrix: np.ndarray = tw.W_matrix[0].flatten()
        self.W_matrix_inv: np.ndarray = np.linalg.inv(tw.W_matrix[0]).flatten()
        self.twiss_data: np.ndarray = np.asarray(
            [
                nemitt_x,
                nemitt_y,
                tw.x[0],
                tw.px[0],
                tw.y[0],
                tw.py[0],
                tw.zeta[0],
                tw.ptau[0],
                nemitt_z,
            ]
        )
        self.part: Particles = part
        self.part_norm: NormedParticles = NormedParticles(
            tw, nemitt_x=nemitt_x, nemitt_y=nemitt_y, part=part
        )
        self.beta0: float = part.beta0
        self.gamma0: float = part.gamma0

        element_names = line.element_names
        element_refs = line.element_refs

        for name in element_names:
            ele = element_refs[name]._get_value()

            match ele:
                case LimitRect():
                    self.poly_elements.append(PolyIdentity4D())
                case LimitRacetrack():
                    self.poly_elements.append(PolyIdentity4D())
                case LimitEllipse():
                    self.poly_elements.append(PolyIdentity4D())
                case LimitPolygon():
                    self.poly_elements.append(PolyIdentity4D())
                case LimitRectEllipse():
                    self.poly_elements.append(PolyIdentity4D())
                case LongitudinalLimitRect():
                    self.poly_elements.append(PolyIdentity4D())
                case ReferenceEnergyIncrease():
                    self.poly_elements.append(PolyReferenceEnergyIncrease4D())
                case Marker():
                    self.poly_elements.append(PolyMarker4D())
                case Drift():
                    self.poly_elements.append(
                        PolyDrift4D(
                            part=part, length=element_refs[name].length._get_value()
                        )
                    )
                case Cavity():
                    self.poly_elements.append(PolyCavity4D())
                case XYShift():
                    self.poly_elements.append(PolyXYShift4D())
                case SRotation():
                    sin_z = element_refs[name].sin_z._get_value()
                    cos_z = element_refs[name].cos_z._get_value()

                    self.poly_elements.append(
                        PolySRotation4D(
                            sin_z=sin_z,
                            cos_z=cos_z,
                        )
                    )
                case ZetaShift():
                    self.poly_elements.append(PolyZetaShift4D())
                case Multipole():
                    knl = []
                    for i in range(len(element_refs[name].knl._get_value())):
                        knl.append(element_refs[name].knl[i]._get_value())
                    ksl = []
                    for i in range(len(element_refs[name].ksl._get_value())):
                        ksl.append(element_refs[name].ksl[i]._get_value())
                    hxl = element_refs[name].hxl._get_value()

                    self.poly_elements.append(
                        PolyMultipole4D(
                            part=part,
                            order=element_refs[name]._order._get_value(),
                            inv_factorial_order=element_refs[
                                name
                            ].inv_factorial_order._get_value(),
                            knl=knl,
                            ksl=ksl,
                            hxl=hxl,
                            length=element_refs[name].length._get_value(),
                            max_order=max_ele_order,
                        )
                    )
                case SimpleThinQuadrupole():
                    knl = []
                    for i in range(len(element_refs[name].knl._get_value())):
                        knl.append(element_refs[name].knl[i]._get_value())

                    self.poly_elements.append(
                        PolySimpleThinQuadrupole4D(part=part, knl=knl)
                    )
                case DipoleEdge():
                    r21 = element_refs[name]._r21._get_value()
                    r43 = element_refs[name]._r43._get_value()

                    self.poly_elements.append(
                        PolyDipoleEdge4D(
                            part=part,
                            r21=r21,
                            r43=r43,
                            model=element_refs[name]._model._get_value(),
                        )
                    )
                case Sextupole():
                    knl = []
                    for i in range(len(element_refs[name].knl._get_value())):
                        knl.append(element_refs[name].knl[i]._get_value())
                    ksl = []
                    for i in range(len(element_refs[name].ksl._get_value())):
                        ksl.append(element_refs[name].ksl[i]._get_value())

                    self.poly_elements.append(
                        PolySextupole4D(
                            part=part,
                            k2=element_refs[name].k2._get_value(),
                            k2s=element_refs[name].k2s._get_value(),
                            length=element_refs[name].length._get_value(),
                            order=element_refs[name]._order._get_value(),
                            inv_factorial_order=element_refs[
                                name
                            ].inv_factorial_order._get_value(),
                            knl=knl,
                            ksl=ksl,
                            edge_entry_active=element_refs[
                                name
                            ].edge_entry_active._get_value(),
                            edge_exit_active=element_refs[
                                name
                            ].edge_exit_active._get_value(),
                            max_order=max_ele_order,
                        )
                    )
                case Octupole():
                    knl = []
                    for i in range(len(element_refs[name].knl._get_value())):
                        knl.append(element_refs[name].knl[i]._get_value())
                    ksl = []
                    for i in range(len(element_refs[name].ksl._get_value())):
                        ksl.append(element_refs[name].ksl[i]._get_value())

                    self.poly_elements.append(
                        PolyOctupole4D(
                            part=part,
                            k3=element_refs[name].k3._get_value(),
                            k3s=element_refs[name].k3s._get_value(),
                            length=element_refs[name].length._get_value(),
                            order=element_refs[name]._order._get_value(),
                            inv_factorial_order=element_refs[
                                name
                            ].inv_factorial_order._get_value(),
                            knl=knl,
                            ksl=ksl,
                            edge_entry_active=element_refs[
                                name
                            ].edge_entry_active._get_value(),
                            edge_exit_active=element_refs[
                                name
                            ].edge_exit_active._get_value(),
                            max_order=max_ele_order,
                        )
                    )
                case Henonmap():
                    twiss_params = []
                    for i in range(len(element_refs[name].twiss_params._get_value())):
                        twiss_params.append(
                            element_refs[name].twiss_params[i]._get_value()
                        )
                    fx_coeffs = []
                    for i in range(len(element_refs[name].fx_coeffs._get_value())):
                        fx_coeffs.append(element_refs[name].fx_coeffs[i]._get_value())
                    fx_x_exps = []
                    for i in range(len(element_refs[name].fx_x_exps._get_value())):
                        fx_x_exps.append(element_refs[name].fx_x_exps[i]._get_value())
                    fx_y_exps = []
                    for i in range(len(element_refs[name].fx_y_exps._get_value())):
                        fx_y_exps.append(element_refs[name].fx_y_exps[i]._get_value())
                    fy_coeffs = []
                    for i in range(len(element_refs[name].fy_coeffs._get_value())):
                        fy_coeffs.append(element_refs[name].fy_coeffs[i]._get_value())
                    fy_x_exps = []
                    for i in range(len(element_refs[name].fy_x_exps._get_value())):
                        fy_x_exps.append(element_refs[name].fy_x_exps[i]._get_value())
                    fy_y_exps = []
                    for i in range(len(element_refs[name].fy_y_exps._get_value())):
                        fy_y_exps.append(element_refs[name].fy_y_exps[i]._get_value())

                    self.poly_elements.append(
                        # PolyHenonmap4D(
                        #     part=part,
                        #     cos_omega_x=element_refs[name].cos_omega_x._get_value(),
                        #     sin_omega_x=element_refs[name].sin_omega_x._get_value(),
                        #     cos_omega_y=element_refs[name].cos_omega_y._get_value(),
                        #     sin_omega_y=element_refs[name].sin_omega_y._get_value(),
                        #     domegax=element_refs[name].domegax._get_value(),
                        #     domegay=element_refs[name].domegay._get_value(),
                        #     n_turns=element_refs[name].n_turns._get_value(),
                        #     twiss_params=twiss_params,
                        #     dx=element_refs[name].dx._get_value(),
                        #     ddx=element_refs[name].ddx._get_value(),
                        #     fx_coeffs=fx_coeffs,
                        #     fx_x_exps=fx_x_exps,
                        #     fx_y_exps=fx_y_exps,
                        #     fy_coeffs=fy_coeffs,
                        #     fy_x_exps=fy_x_exps,
                        #     fy_y_exps=fy_y_exps,
                        #     norm=element_refs[name].norm._get_value(),
                        #     max_order=max_ele_order
                        # )
                        PolyHenonmap4D(
                            part=part,
                            cos_omega_x=line.element_refs[
                                name
                            ].cos_omega_x._get_value(),
                            sin_omega_x=line.element_refs[
                                name
                            ].sin_omega_x._get_value(),
                            cos_omega_y=line.element_refs[
                                name
                            ].cos_omega_y._get_value(),
                            sin_omega_y=line.element_refs[
                                name
                            ].sin_omega_y._get_value(),
                            domegax=line.element_refs[name].domegax._get_value(),
                            domegay=line.element_refs[name].domegay._get_value(),
                            n_turns=line.element_refs[name].n_turns._get_value(),
                            twiss_params=line.element_refs[
                                name
                            ].twiss_params._get_value(),
                            dx=line.element_refs[name].dx._get_value(),
                            ddx=line.element_refs[name].ddx._get_value(),
                            fx_coeffs=line.element_refs[name].fx_coeffs._get_value(),
                            fx_x_exps=line.element_refs[name].fx_x_exps._get_value(),
                            fx_y_exps=line.element_refs[name].fx_y_exps._get_value(),
                            fy_coeffs=line.element_refs[name].fy_coeffs._get_value(),
                            fy_x_exps=line.element_refs[name].fy_x_exps._get_value(),
                            fy_y_exps=line.element_refs[name].fy_y_exps._get_value(),
                            norm=line.element_refs[name].norm._get_value(),
                            max_order=max_ele_order,
                        )
                    )
                case _:
                    raise ValueError(f"{ele} is not implemented.")

    def set_max_map_order(self, max_map_order: int) -> None:
        """
        Function to set the maximum order at which the one-turn map will be
        truncated.

        Input:
            - max_map_order: integer, maximum order at which the one-turn map
              will be truncated

        Output:
            -
        """

        self.max_map_order = max_map_order

    def calculate_one_turn_map(self) -> None:
        """
        Function to calculate the one-turn map by composing the individual
        polynomial element maps and truncating at the given order.

        Input:
            -

        Output:
            -
        """

        one_turn_map = self.poly_elements[0].ele_map

        total = len(self.poly_elements)
        for i in range(1, len(self.poly_elements)):
            one_turn_map = Map.composition_Map(
                one_turn_map, self.poly_elements[i].ele_map, self.max_map_order
            )
            progress = (i / (total - 1)) * 100
            sys.stdout.write(f"\rCombining line elements: {progress:.2f}%")
            sys.stdout.flush()
        print("\nCombination of all line elements finished!")

        self.one_turn_map_real = one_turn_map

        gemitt_x = self.twiss_data[0] / self.beta0 / self.gamma0
        gemitt_y = self.twiss_data[1] / self.beta0 / self.gamma0
        gemitt_z = self.twiss_data[8] / self.beta0 / self.gamma0

        x = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[0] * np.sqrt(gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[1] * np.sqrt(gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[2] * np.sqrt(gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[3] * np.sqrt(gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[4] * self.part_norm.zeta_norm
                            + self.W_matrix[5] * self.part_norm.pzeta_norm
                        )
                        * np.sqrt(gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.twiss_data[2])]),
        )
        px = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[6] * np.sqrt(gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[7] * np.sqrt(gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[8] * np.sqrt(gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[9] * np.sqrt(gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[10] * self.part_norm.zeta_norm
                            + self.W_matrix[11] * self.part_norm.pzeta_norm
                        )
                        * np.sqrt(gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.twiss_data[3])]),
        )
        y = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[12] * np.sqrt(gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[13] * np.sqrt(gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[14] * np.sqrt(gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[15] * np.sqrt(gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[16] * self.part_norm.zeta_norm
                            + self.W_matrix[17] * self.part_norm.pzeta_norm
                        )
                        * np.sqrt(gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.twiss_data[4])]),
        )
        py = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[18] * np.sqrt(gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[19] * np.sqrt(gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[20] * np.sqrt(gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[21] * np.sqrt(gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[22] * self.part_norm.zeta_norm
                            + self.W_matrix[23] * self.part_norm.pzeta_norm
                        )
                        * np.sqrt(gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.twiss_data[5])]),
        )

        new_x_poly = Polynom(terms=[])
        for i in range(len(one_turn_map.x_poly.terms)):
            new_x_poly = Polynom.sum_Polynoms(
                new_x_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=one_turn_map.x_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, one_turn_map.x_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, one_turn_map.x_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, one_turn_map.x_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py, one_turn_map.x_poly.terms[i].py_exp, int(1e6)
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_x_poly.remove_zero_terms()
            # new_x_poly.collect_terms()
        new_x_poly = Polynom.sum_Polynoms(
            new_x_poly, Polynom(terms=[Term(coeff=-self.twiss_data[2])])
        )

        new_px_poly = Polynom(terms=[])
        for i in range(len(one_turn_map.px_poly.terms)):
            new_px_poly = Polynom.sum_Polynoms(
                new_px_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=one_turn_map.px_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, one_turn_map.px_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, one_turn_map.px_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, one_turn_map.px_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py,
                                    one_turn_map.px_poly.terms[i].py_exp,
                                    int(1e6),
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_px_poly.remove_zero_terms()
            # new_px_poly.collect_terms()
        new_px_poly = Polynom.sum_Polynoms(
            new_px_poly, Polynom(terms=[Term(coeff=-self.twiss_data[3])])
        )

        new_y_poly = Polynom(terms=[])
        for i in range(len(one_turn_map.y_poly.terms)):
            new_y_poly = Polynom.sum_Polynoms(
                new_y_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=one_turn_map.y_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, one_turn_map.y_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, one_turn_map.y_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, one_turn_map.y_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py, one_turn_map.y_poly.terms[i].py_exp, int(1e6)
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_y_poly.remove_zero_terms()
            # new_y_poly.collect_terms()
        new_y_poly = Polynom.sum_Polynoms(
            new_y_poly, Polynom(terms=[Term(coeff=-self.twiss_data[4])])
        )

        new_py_poly = Polynom(terms=[])
        for i in range(len(one_turn_map.py_poly.terms)):
            new_py_poly = Polynom.sum_Polynoms(
                new_py_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=one_turn_map.py_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, one_turn_map.py_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, one_turn_map.py_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, one_turn_map.py_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py,
                                    one_turn_map.py_poly.terms[i].py_exp,
                                    int(1e6),
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_py_poly.remove_zero_terms()
            # new_py_poly.collect_terms()
        new_py_poly = Polynom.sum_Polynoms(
            new_py_poly, Polynom(terms=[Term(coeff=-self.twiss_data[5])])
        )

        new_x_norm_poly = Polynom.product_Coeff_Polynom(
            coeff=1.0 / np.sqrt(gemitt_x),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[0], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[1], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[2], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[3], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[4] * self.part.zeta
                                            + self.W_matrix_inv[5] * self.part.pzeta
                                        )
                                    )
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        )
        new_px_norm_poly = Polynom.product_Coeff_Polynom(
            coeff=1.0 / np.sqrt(gemitt_x),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[6], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[7], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[8], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[9], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[10] * self.part.zeta
                                            + self.W_matrix_inv[11] * self.part.pzeta
                                        )
                                    )
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        )
        new_y_norm_poly = Polynom.product_Coeff_Polynom(
            coeff=1.0 / np.sqrt(gemitt_y),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[12], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[13], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[14], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[15], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[16] * self.part.zeta
                                            + self.W_matrix_inv[17] * self.part.pzeta
                                        )
                                    )
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        )
        new_py_norm_poly = Polynom.product_Coeff_Polynom(
            coeff=1.0 / np.sqrt(gemitt_y),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[18], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[19], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[20], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[21], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[22] * self.part.zeta
                                            + self.W_matrix_inv[23] * self.part.pzeta
                                        )
                                    )
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        )

        self.one_turn_map = Map(
            x_poly=new_x_norm_poly,
            px_poly=new_px_norm_poly,
            y_poly=new_y_norm_poly,
            py_poly=new_py_norm_poly,
        )

        for term in self.one_turn_map.x_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except TypeError:
                continue
        for term in self.one_turn_map.px_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except TypeError:
                continue
        for term in self.one_turn_map.y_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except TypeError:
                continue
        for term in self.one_turn_map.py_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except TypeError:
                continue

    def _calculate_complex_one_turn_map(self) -> None:
        """
        Function to calculate the one-turn map in complex normalised
        Courant-Snyder coordinates by composing the individual polynomial
        element maps and truncating at the given order.

        Input:
            -

        Output:
            -
        """

        if self.one_turn_map is None:
            self.calculate_one_turn_map()

        x_subs = Polynom(terms=[Term(coeff=0.5, x_exp=1), Term(coeff=0.5, px_exp=1)])
        px_subs = Polynom(
            terms=[Term(coeff=0.5j, x_exp=1), Term(coeff=-0.5j, px_exp=1)]
        )
        y_subs = Polynom(terms=[Term(coeff=0.5, y_exp=1), Term(coeff=0.5, py_exp=1)])
        py_subs = Polynom(
            terms=[Term(coeff=0.5j, y_exp=1), Term(coeff=-0.5j, py_exp=1)]
        )

        if self.one_turn_map is None:
            raise ValueError
        new_x_poly = Polynom(terms=[])
        for i in range(len(self.one_turn_map.x_poly.terms)):
            new_x_poly = Polynom.sum_Polynoms(
                new_x_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=self.one_turn_map.x_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x_subs, self.one_turn_map.x_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px_subs,
                                self.one_turn_map.x_poly.terms[i].px_exp,
                                int(1e6),
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y_subs,
                                    self.one_turn_map.x_poly.terms[i].y_exp,
                                    int(1e6),
                                ),
                                Polynom.power_Polynom(
                                    py_subs,
                                    self.one_turn_map.x_poly.terms[i].py_exp,
                                    int(1e6),
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_x_poly.remove_zero_terms()
            # new_x_poly.collect_terms()

        new_px_poly = Polynom(terms=[])
        for i in range(len(self.one_turn_map.px_poly.terms)):
            new_px_poly = Polynom.sum_Polynoms(
                new_px_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=self.one_turn_map.px_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x_subs, self.one_turn_map.px_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px_subs,
                                self.one_turn_map.px_poly.terms[i].px_exp,
                                int(1e6),
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y_subs,
                                    self.one_turn_map.px_poly.terms[i].y_exp,
                                    int(1e6),
                                ),
                                Polynom.power_Polynom(
                                    py_subs,
                                    self.one_turn_map.px_poly.terms[i].py_exp,
                                    int(1e6),
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_px_poly.remove_zero_terms()
            # new_px_poly.collect_terms()

        new_y_poly = Polynom(terms=[])
        for i in range(len(self.one_turn_map.y_poly.terms)):
            new_y_poly = Polynom.sum_Polynoms(
                new_y_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=self.one_turn_map.y_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x_subs, self.one_turn_map.y_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px_subs,
                                self.one_turn_map.y_poly.terms[i].px_exp,
                                int(1e6),
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y_subs,
                                    self.one_turn_map.y_poly.terms[i].y_exp,
                                    int(1e6),
                                ),
                                Polynom.power_Polynom(
                                    py_subs,
                                    self.one_turn_map.y_poly.terms[i].py_exp,
                                    int(1e6),
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_y_poly.remove_zero_terms()
            # new_y_poly.collect_terms()

        new_py_poly = Polynom(terms=[])
        for i in range(len(self.one_turn_map.py_poly.terms)):
            new_py_poly = Polynom.sum_Polynoms(
                new_py_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=self.one_turn_map.py_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x_subs, self.one_turn_map.py_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px_subs,
                                self.one_turn_map.py_poly.terms[i].px_exp,
                                int(1e6),
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y_subs,
                                    self.one_turn_map.py_poly.terms[i].y_exp,
                                    int(1e6),
                                ),
                                Polynom.power_Polynom(
                                    py_subs,
                                    self.one_turn_map.py_poly.terms[i].py_exp,
                                    int(1e6),
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
            # new_py_poly.remove_zero_terms()
            # new_py_poly.collect_terms()

        self._complex_one_turn_map = Map(
            x_poly=Polynom.sum_Polynoms(
                new_x_poly, Polynom.product_Coeff_Polynom(coeff=-1j, poly=new_px_poly)
            ),
            px_poly=Polynom.sum_Polynoms(
                new_x_poly, Polynom.product_Coeff_Polynom(coeff=1j, poly=new_px_poly)
            ),
            y_poly=Polynom.sum_Polynoms(
                new_y_poly, Polynom.product_Coeff_Polynom(coeff=-1j, poly=new_py_poly)
            ),
            py_poly=Polynom.sum_Polynoms(
                new_y_poly, Polynom.product_Coeff_Polynom(coeff=1j, poly=new_py_poly)
            ),
        )
        self._complex_one_turn_map.x_poly.collect_terms()
        self._complex_one_turn_map.px_poly.collect_terms()
        self._complex_one_turn_map.y_poly.collect_terms()
        self._complex_one_turn_map.py_poly.collect_terms()

    def calculate_normal_form(
        self,
        max_nf_order: int,
        res_space_dim: int,
        res_case: int,
        res_eig: list[complex] | None = None,
        res_basis1: list[int] | None = None,
        res_basis2: list[int] | None = None,
    ) -> None:
        """
        Function to calculate the normal forms of the one-turn map.

        Input:
            - max_nf_order: integer, maximum order of the normal form
            - res_case: integer, 0 for nonresonant normal forms, 1 for exactly
              resonant normal forms, 2 for quasiresonant normal forms
            - res_eig: list of complex resonant eigenvalues,
              i.e. [$e^{2i\pi Q_{x,res}}$, $e^{-2i\pi Q_{x,res}}$,
              $e^{2i\pi Q_{y,res}}$, $e^{-2i\pi Q_{y,res}}$], only needed if
              res_case is 1 or 2
            - res_basis1: list or integers, [n, m] which satisfy the resonance
              condition n*Q_x+m*Q_y=p, only needed if res_space_dim is 1 or 1
            - res_basis2: list or integers, [n, m] which satisfy the resonance
              condition n*Q_x+m*Q_y=p for a second resonance, only needed if
              res_space_dim is 2

        Output:
            -
        """

        self._calculate_complex_one_turn_map()

        if self._complex_one_turn_map is None:
            raise ValueError

        self.normal_form = NormalForm4D(
            complex_map=self._complex_one_turn_map,
            max_map_order=self.max_map_order,
            max_nf_order=max_nf_order,
            res_space_dim=res_space_dim,
            res_case=res_case,
            res_eig=res_eig,
            res_basis1=res_basis1,
            res_basis2=res_basis2,
        )

        self.normal_form.compute_normal_form()
