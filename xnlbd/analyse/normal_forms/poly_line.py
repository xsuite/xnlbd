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

    linear_poly_elements = (
        PolyIdentity4D,
        PolyReferenceEnergyIncrease4D,
        PolyMarker4D,
        PolyDrift4D, 
        PolyCavity4D,
        PolyZetaShift4D,
        # PolySimpleThinBend4D,
        PolySimpleThinQuadrupole4D,
        PolyDipoleEdge4D,
    )

    def __init__(
        self,
        line_in: Line,
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
            - line_in: Line object constructed with xtrack that represents the
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
        line = line_in.copy()

        try:
            self.exact_drifts = int(line.config.XTRACK_USE_EXACT_DRIFTS)
        except AttributeError:
            self.exact_drifts = 0

        unique_thick_ele_types = tuple(set(type(ele) for ele in line.elements if ele.isthick))
        if len(unique_thick_ele_types) != 1: # line contains thick elements other than drifts
            print("Thick elements are not supported for normal forms, \\ "
                "they will be sliced and replaced with thin equivalents! \\"
                "Tunes, chromaticities and other features of the machine \\"
                "(e.g. closed orbit bumps) may change. If these are of \\"
                "interest, please provide a line containing thin elements \\"
                "only.")
            line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Uniform(1))])

        self.poly_elements: list[PolyElement4D] = []
        self.one_turn_map: Map | None = None
        self.one_turn_map_real: Map | None = None

        self._complex_one_turn_map: Map | None = None

        self.normal_form: NormalForm4D | None = None

        self.max_map_order: int = max_map_order
        self.max_ele_order: int = max_ele_order

        self.tw = line.twiss(continue_on_closed_orbit_error=False, delta0=part.delta[0])
        self.W_matrix: list[np.ndarray] = []
        self.W_matrix_inv: list[np.ndarray] = []
        self.parts: list[Particles] = []
        self.parts_norm: list[NormedParticles] = []
        for i in range(len(line.element_names)):
            self.W_matrix.append(self.tw.W_matrix[i].flatten())
            self.W_matrix_inv.append(np.linalg.inv(self.tw.W_matrix[i]).flatten())
            part0 = xt.Particles(
                x=self.tw.x[i], 
                px=self.tw.px[i], 
                y=self.tw.y[i], 
                py=self.tw.py[i], 
                zeta=self.tw.zeta[i], 
                delta=self.tw.delta[i], 
                p0c=self.tw.particle_on_co.p0c, 
                mass0=self.tw.particle_on_co.mass0, 
                q0=self.tw.particle_on_co.q0
            )
            self.parts.append(part0)
            part_norm0 = NormedParticles(
                self.tw, nemitt_x=nemitt_x, nemitt_y=nemitt_y, part=part0
            )
            part_norm0.phys_to_norm(part0)
            self.parts_norm.append(part_norm0)

        self.beta0: float = part.beta0
        self.gamma0: float = part.gamma0
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.nemitt_z = nemitt_z
        self.gemitt_x = self.nemitt_x / self.beta0 / self.gamma0
        self.gemitt_y = self.nemitt_y / self.beta0 / self.gamma0
        self.gemitt_z = self.nemitt_z / self.beta0 / self.gamma0

        element_names = line.element_names
        elements = line.elements

        for i, name in enumerate(element_names):
            try:
                ele = elements[i].get_equivalent_element()
            except AttributeError:
                ele = elements[i]

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
                            part=part, length=ele.length, exact=self.exact_drifts, max_order=max_ele_order
                        )
                    )
                case Cavity():
                    self.poly_elements.append(PolyCavity4D())
                case XYShift():
                    self.poly_elements.append(PolyXYShift4D())
                case SRotation():
                    sin_z = ele.sin_z
                    cos_z = ele.cos_z

                    self.poly_elements.append(
                        PolySRotation4D(
                            sin_z=sin_z,
                            cos_z=cos_z,
                        )
                    )
                case ZetaShift():
                    self.poly_elements.append(PolyZetaShift4D())
                case Multipole():
                    self.poly_elements.append(
                        PolyMultipole4D(
                            part=part,
                            order=ele._order,
                            inv_factorial_order=ele.inv_factorial_order,
                            knl=ele.knl,
                            ksl=ele.ksl,
                            hxl=ele.hxl,
                            length=ele.length,
                            max_order=max_ele_order,
                        )
                    )
                case SimpleThinQuadrupole():
                    self.poly_elements.append(
                        PolySimpleThinQuadrupole4D(part=part, knl=ele.knl)
                    )
                case DipoleEdge():
                    r21 = ele._r21
                    r43 = ele._r43

                    self.poly_elements.append(
                        PolyDipoleEdge4D(
                            part=part,
                            r21=r21,
                            r43=r43,
                            model=ele._model,
                        )
                    )
                case Sextupole():
                    self.poly_elements.append(
                        PolySextupole4D(
                            part=part,
                            k2=ele.k2,
                            k2s=ele.k2s,
                            length=ele.length,
                            order=ele._order,
                            inv_factorial_order=ele.inv_factorial_order,
                            knl=ele.knl,
                            ksl=ele.ksl,
                            edge_entry_active=ele.edge_entry_active,
                            edge_exit_active=ele.edge_exit_active,
                            max_order=max_ele_order,
                            exact=self.exact_drifts
                        )
                    )
                case Octupole():
                    self.poly_elements.append(
                        PolyOctupole4D(
                            part=part,
                            k3=ele.k3,
                            k3s=ele.k3s,
                            length=ele.length,
                            order=ele._order,
                            inv_factorial_order=ele.inv_factorial_order,
                            knl=ele.knl,
                            ksl=ele.ksl,
                            edge_entry_active=ele.edge_entry_active,
                            edge_exit_active=ele.edge_exit_active,
                            max_order=max_ele_order,
                            exact=self.exact_drifts
                        )
                    )
                case Henonmap():
                    self.poly_elements.append(
                        PolyHenonmap4D(
                            part=part,
                            cos_omega_x=ele.cos_omega_x,
                            sin_omega_x=ele.sin_omega_x,
                            cos_omega_y=ele.cos_omega_y,
                            sin_omega_y=ele.sin_omega_y,
                            domegax=ele.domegax,
                            domegay=ele.domegay,
                            n_turns=ele.n_turns,
                            twiss_params=ele.twiss_params,
                            dx=ele.dx,
                            ddx=ele.ddx,
                            fx_coeffs=ele.fx_coeffs,
                            fx_x_exps=ele.fx_x_exps,
                            fx_y_exps=ele.fx_y_exps,
                            fy_coeffs=ele.fy_coeffs,
                            fy_x_exps=ele.fy_x_exps,
                            fy_y_exps=ele.fy_y_exps,
                            norm=ele.norm,
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

    def _calculate_one_turn_map_real(self) -> None:
        """
        Function to calculate the one-turn map by composing the individual
        polynomial element maps and truncating at the given order. The map 
        calculated will be in real physical space.

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

    def _get_coords_in_norm_at_ele(self, n: int) -> tuple[Polynom, Polynom, Polynom, Polynom]:
        """
        Function to compute the polynomial representations of physical space 
        coordinates as a function of normalised coordinates at a given element.

        Input:
            - n: integer, index of the element

        Output:
            - tuple of Polynom objects representing (x, px, y, py) in terms of 
              normalised equivalents
        """
        
        x = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[n][0] * np.sqrt(self.gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[n][1] * np.sqrt(self.gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[n][2] * np.sqrt(self.gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[n][3] * np.sqrt(self.gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[n][4] * self.parts_norm[n].zeta_norm[0]
                            + self.W_matrix[n][5] * self.parts_norm[n].pzeta_norm[0]
                        )
                        * np.sqrt(self.gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.tw.x[n])]),
        )
        px = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[n][6] * np.sqrt(self.gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[n][7] * np.sqrt(self.gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[n][8] * np.sqrt(self.gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[n][9] * np.sqrt(self.gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[n][10] * self.parts_norm[n].zeta_norm[0]
                            + self.W_matrix[n][11] * self.parts_norm[n].pzeta_norm[0]
                        )
                        * np.sqrt(self.gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.tw.px[n])]),
        )
        y = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[n][12] * np.sqrt(self.gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[n][13] * np.sqrt(self.gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[n][14] * np.sqrt(self.gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[n][15] * np.sqrt(self.gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[n][16] * self.parts_norm[n].zeta_norm[0]
                            + self.W_matrix[n][17] * self.parts_norm[n].pzeta_norm[0]
                        )
                        * np.sqrt(self.gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.tw.y[n])]),
        )
        py = Polynom.sum_Polynoms(
            Polynom(
                terms=[
                    Term(coeff=self.W_matrix[n][18] * np.sqrt(self.gemitt_x), x_exp=1),
                    Term(coeff=self.W_matrix[n][19] * np.sqrt(self.gemitt_x), px_exp=1),
                    Term(coeff=self.W_matrix[n][20] * np.sqrt(self.gemitt_y), y_exp=1),
                    Term(coeff=self.W_matrix[n][21] * np.sqrt(self.gemitt_y), py_exp=1),
                    Term(
                        coeff=(
                            self.W_matrix[n][22] * self.parts_norm[n].zeta_norm[0]
                            + self.W_matrix[n][23] * self.parts_norm[n].pzeta_norm[0]
                        )
                        * np.sqrt(self.gemitt_z)
                    ),
                ]
            ),
            Polynom(terms=[Term(self.tw.py[n])]),
        )

        return (x, px, y, py)
    
    def _get_norm_map_at_ele(self, n: int) -> Map:
        """
        Function to compute the map representation of a given element in 
        normalised space.

        Input:
            - n: integer, index of the element

        Output:
            - Map objects in normalised space
        """

        curr_ele_map = self.poly_elements[n].ele_map
        
        x, px, y, py = self._get_coords_in_norm_at_ele(n)
        
        new_x_poly = Polynom(terms=[])
        for i in range(len(curr_ele_map.x_poly.terms)):
            new_x_poly = Polynom.sum_Polynoms(
                new_x_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=curr_ele_map.x_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, curr_ele_map.x_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, curr_ele_map.x_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, curr_ele_map.x_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py, curr_ele_map.x_poly.terms[i].py_exp, int(1e6)
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
        new_x_poly = Polynom.sum_Polynoms(
            new_x_poly, Polynom(terms=[Term(coeff=-self.tw.x[n])])
        )

        new_px_poly = Polynom(terms=[])
        for i in range(len(curr_ele_map.px_poly.terms)):
            new_px_poly = Polynom.sum_Polynoms(
                new_px_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=curr_ele_map.px_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, curr_ele_map.px_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, curr_ele_map.px_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, curr_ele_map.px_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py,
                                    curr_ele_map.px_poly.terms[i].py_exp,
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
        new_px_poly = Polynom.sum_Polynoms(
            new_px_poly, Polynom(terms=[Term(coeff=-self.tw.px[n])])
        )

        new_y_poly = Polynom(terms=[])
        for i in range(len(curr_ele_map.y_poly.terms)):
            new_y_poly = Polynom.sum_Polynoms(
                new_y_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=curr_ele_map.y_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, curr_ele_map.y_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, curr_ele_map.y_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, curr_ele_map.y_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py, curr_ele_map.y_poly.terms[i].py_exp, int(1e6)
                                ),
                                int(1e6),
                            ),
                            int(1e6),
                        ),
                        int(1e6),
                    ),
                ),
            )
        new_y_poly = Polynom.sum_Polynoms(
            new_y_poly, Polynom(terms=[Term(coeff=-self.tw.y[n])])
        )

        new_py_poly = Polynom(terms=[])
        for i in range(len(curr_ele_map.py_poly.terms)):
            new_py_poly = Polynom.sum_Polynoms(
                new_py_poly,
                Polynom.product_Coeff_Polynom(
                    coeff=curr_ele_map.py_poly.terms[i].coeff,
                    poly=Polynom.product_Polynoms(
                        Polynom.power_Polynom(
                            x, curr_ele_map.py_poly.terms[i].x_exp, int(1e6)
                        ),
                        Polynom.product_Polynoms(
                            Polynom.power_Polynom(
                                px, curr_ele_map.py_poly.terms[i].px_exp, int(1e6)
                            ),
                            Polynom.product_Polynoms(
                                Polynom.power_Polynom(
                                    y, curr_ele_map.py_poly.terms[i].y_exp, int(1e6)
                                ),
                                Polynom.power_Polynom(
                                    py,
                                    curr_ele_map.py_poly.terms[i].py_exp,
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
        new_py_poly = Polynom.sum_Polynoms(
            new_py_poly, Polynom(terms=[Term(coeff=-self.tw.py[n])])
        )

        new_x_norm_poly = Polynom.product_Coeff_Polynom(
            coeff=1.0 / np.sqrt(self.gemitt_x),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[n][0], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[n][1], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[n][2], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[n][3], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[n][4] * self.parts[n].zeta[0]
                                            + self.W_matrix_inv[n][5] * self.parts[n].pzeta[0]
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
            coeff=1.0 / np.sqrt(self.gemitt_x),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[n][6], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[n][7], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[n][8], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[n][9], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[n][10] * self.parts[n].zeta[0]
                                            + self.W_matrix_inv[n][11] * self.parts[n].pzeta[0]
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
            coeff=1.0 / np.sqrt(self.gemitt_y),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[n][12], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[n][13], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[n][14], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[n][15], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[n][16] * self.parts[n].zeta[0]
                                            + self.W_matrix_inv[n][17] * self.parts[n].pzeta[0]
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
            coeff=1.0 / np.sqrt(self.gemitt_y),
            poly=Polynom.sum_Polynoms(
                Polynom.product_Coeff_Polynom(
                    coeff=self.W_matrix_inv[n][18], poly=new_x_poly
                ),
                Polynom.sum_Polynoms(
                    Polynom.product_Coeff_Polynom(
                        coeff=self.W_matrix_inv[n][19], poly=new_px_poly
                    ),
                    Polynom.sum_Polynoms(
                        Polynom.product_Coeff_Polynom(
                            coeff=self.W_matrix_inv[n][20], poly=new_y_poly
                        ),
                        Polynom.sum_Polynoms(
                            Polynom.product_Coeff_Polynom(
                                coeff=self.W_matrix_inv[n][21], poly=new_py_poly
                            ),
                            Polynom(
                                terms=[
                                    Term(
                                        coeff=(
                                            self.W_matrix_inv[n][22] * self.parts[n].zeta[0]
                                            + self.W_matrix_inv[n][23] * self.parts[n].pzeta[0]
                                        )
                                    )
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        )

        curr_ele_map_norm = Map(
            x_poly=new_x_norm_poly,
            px_poly=new_px_norm_poly,
            y_poly=new_y_norm_poly,
            py_poly=new_py_norm_poly,
        )

        for term in curr_ele_map_norm.x_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except (TypeError, IndexError) as e:
                continue
        for term in curr_ele_map_norm.px_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except (TypeError, IndexError) as e:
                continue
        for term in curr_ele_map_norm.y_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except (TypeError, IndexError) as e:
                continue
        for term in curr_ele_map_norm.py_poly.terms:
            try:
                term.coeff = term.coeff[0]  # type: ignore[index]
            except (TypeError, IndexError) as e:
                continue

        return curr_ele_map_norm

    def calculate_one_turn_map(self) -> None:
        """
        Function to calculate the one-turn map by composing the individual
        polynomial element maps directly in normalised space and truncating 
        at the given order. The map calculated will be in normalised space.

        Input:
            -

        Output:
            -
        """
        
        one_turn_map = PolyIdentity4D().ele_map

        nonlin_poly_ele_idx = []
        nonlin_poly_ele_norm = []

        total = len(self.poly_elements)
        for i in range(0, len(self.poly_elements)):
            if isinstance(self.poly_elements[i], self.linear_poly_elements):
                continue
            elif isinstance(self.poly_elements[i], PolyMultipole4D):
                if self.poly_elements[i].ele_map.get_max_order() < 2:
                    continue
                else:
                    nonlin_poly_ele_idx.append(i)
                    nonlin_poly_ele_norm.append(self._get_norm_map_at_ele(i))
            else:
                nonlin_poly_ele_idx.append(i)
                nonlin_poly_ele_norm.append(self._get_norm_map_at_ele(i))
        
        idx = 0
        for i in range(len(nonlin_poly_ele_idx)):
            d_mux = (self.tw.mux[nonlin_poly_ele_idx[i]] - self.tw.mux[idx]) * 2 * np.pi
            d_muy = (self.tw.muy[nonlin_poly_ele_idx[i]] - self.tw.muy[idx]) * 2 * np.pi
            curr_rot = Map(
                x_poly=Polynom(terms=[
                    Term(coeff=np.cos(d_mux), x_exp=1),
                    Term(coeff=np.sin(d_mux), px_exp=1)
                ]),
                px_poly=Polynom(terms=[
                    Term(coeff=-np.sin(d_mux), x_exp=1),
                    Term(coeff=np.cos(d_mux), px_exp=1)
                ]),
                y_poly=Polynom(terms=[
                    Term(coeff=np.cos(d_muy), y_exp=1),
                    Term(coeff=np.sin(d_muy), py_exp=1)
                ]),
                py_poly=Polynom(terms=[
                    Term(coeff=-np.sin(d_muy), y_exp=1),
                    Term(coeff=np.cos(d_muy), py_exp=1)
                ])
            )
            one_turn_map = Map.composition_Map(
                one_turn_map, curr_rot, self.max_map_order
            )
            one_turn_map = Map.composition_Map(
                one_turn_map, nonlin_poly_ele_norm[i], self.max_map_order
            )
            idx = nonlin_poly_ele_idx[i]
            progress = (nonlin_poly_ele_idx[i] / (total - 1)) * 100
            sys.stdout.write(f"\rCombining line elements: {progress:.2f}%")
            sys.stdout.flush()

        if nonlin_poly_ele_idx[-1] != (total - 1):
            d_mux = (self.tw.mux[total] - self.tw.mux[idx]) * 2 * np.pi
            d_muy = (self.tw.muy[total] - self.tw.muy[idx]) * 2 * np.pi
            curr_rot = Map(
                x_poly=Polynom(terms=[
                    Term(coeff=np.cos(d_mux), x_exp=1),
                    Term(coeff=np.sin(d_mux), px_exp=1)
                ]),
                px_poly=Polynom(terms=[
                    Term(coeff=-np.sin(d_mux), x_exp=1),
                    Term(coeff=np.cos(d_mux), px_exp=1)
                ]),
                y_poly=Polynom(terms=[
                    Term(coeff=np.cos(d_muy), y_exp=1),
                    Term(coeff=np.sin(d_muy), py_exp=1)
                ]),
                py_poly=Polynom(terms=[
                    Term(coeff=-np.sin(d_muy), y_exp=1),
                    Term(coeff=np.cos(d_muy), py_exp=1)
                ])
            )
            one_turn_map = Map.composition_Map(
                one_turn_map, curr_rot, self.max_map_order
            )
            progress = 100
            sys.stdout.write(f"\rCombining line elements: {progress:.2f}%")
            sys.stdout.flush()

        print("\nCombination of all line elements finished!")

        self.one_turn_map = one_turn_map

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
