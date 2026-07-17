from __future__ import annotations

import copy
import sys
import numpy as np
import contourpy as cpy
from typing import Union
from math import comb
from itertools import product
from scipy.optimize import least_squares
from math import factorial

import xtrack as xt  # type: ignore[import-untyped, import-not-found]
from xtrack import Line  # type: ignore[import-untyped, import-not-found]

from xnlbd.tools import NormedParticles

from .numeric_normal_forms import NormalForm4D
from .poly_elements import *
from .polynom_base import (
    Map, Polynom, Term,
    ParametricMap, ParametricPolynom, ParametricTerm,
    _ParamPoly, _make_fast_coeff,
)


class OneTurnMap4D:

    def __init__(
        self, 
        line: Line,
        max_gen_degree: int,
        max_map_order: int,
        nemitt_x: float = 1,
        nemitt_y: float = 1,
        nemitt_z: float = 1,
        sigma_limit: float = 10,
    ) -> None:
        self.one_turn_map = None
        self._complex_one_turn_map = None
        self.normal_form = None
        
        tw = line.twiss()

        W = tw.W_matrix[0]
        W_inv = np.linalg.inv(W)
        R = W_inv @ tw.R_matrix @ W
        self.R = R[:4, :4]

        if max_gen_degree < 3:
            raise ValueError(
                "Maximum generator order cannot be lower than 3, but " \
                f"{max_gen_degree} was provided."
            )
        self.max_gen_degree = max_gen_degree
        self.generator_degrees = np.arange(3, max_gen_degree+1)
        self.generator_exponents = {
            d: self._homogeneous_exponents(d)
            for d in self.generator_degrees
        }
        self.generator_sizes = {
            d: len(self.generator_exponents[d])
            for d in self.generator_degrees
        }
        
        self.max_map_order = max_map_order

        # Number of particles to track 100 times more than number of 
        # coefficients to fit
        N = 0
        for d in range(3, max_gen_degree+1):
            N += comb(4+d-1, d)
        N *= 100

        Jx = np.random.uniform(0, sigma_limit**2/2, N)
        Jy = np.random.uniform(0, sigma_limit**2/2, N)
        phix = np.random.uniform(0, 2*np.pi, N)
        phiy = np.random.uniform(0, 2*np.pi, N)

        x_norm  = np.sqrt(2*Jx) * np.cos(phix)
        px_norm = np.sqrt(2*Jx) * np.sin(phix)
        y_norm  = np.sqrt(2*Jy) * np.cos(phiy)
        py_norm = np.sqrt(2*Jy) * np.sin(phiy)

        part = xt.Particles(
            p0c=tw.particle_on_co.p0c,
            mass0=tw.particle_on_co.mass0,
            x=np.ones(N) * tw.x[0],
            px=np.ones(N) * tw.px[0],
            y=np.ones(N) * tw.y[0],
            py=np.ones(N) * tw.py[0],
            zeta=np.ones(N) * tw.zeta[0],
            ptau=np.ones(N) * tw.ptau[0],
        )
        part_norm = NormedParticles(
            tw, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=part
        )
        part_norm.phys_to_norm(part)
        part_norm.x_norm = x_norm
        part_norm.px_norm = px_norm
        part_norm.y_norm = y_norm
        part_norm.py_norm = py_norm
        part = part_norm.norm_to_phys(part)

        part_init = part.copy()
        line.track(part)
        part_fin = part.copy()

        part_norm_init = NormedParticles(
            tw, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=part_init
        )
        part_norm_init.phys_to_norm(part_init)
        part_norm_fin = NormedParticles(
            tw, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=part_fin
        )
        part_norm_fin.phys_to_norm(part_fin)

        self.z0 = np.vstack((
            part_norm_init.x_norm, 
            part_norm_init.px_norm, 
            part_norm_init.y_norm, 
            part_norm_init.py_norm
        ))
        self.z1 = np.vstack((
            part_norm_fin.x_norm, 
            part_norm_fin.px_norm, 
            part_norm_fin.y_norm, 
            part_norm_fin.py_norm
        ))


    @staticmethod
    def _homogeneous_exponents(degree) -> list[tuple]:
        """
        Function to compute all possible combinations of exponents of a 
        monomial in 4 variables of the given degree.

        Parameters:
        -----------
        degree : int
            Degree of the monominal.
        
        Returns:
        --------
        list[tuple]
            List of comb(4+degree-1, degree) tuples of 4 elements each.
        """
        all_exponents = []

        for exps in product(range(degree + 1), repeat=4):
            if sum(exps) == degree:
                all_exponents.append(tuple(exps))

        return all_exponents
    

    def _unpack_generators(self, params: Union[np.ndarray, list]) -> list[Polynom]:
        """
        Function to return the polynomials representing each generator order 
        individually with the provided coefficients.

        Parameters:
        -----------
        params : iterable (np.ndarray, list)
            Iterable containing the numerical values of the generator 
            coefficients of all orders.

        Returns:
        --------
        list[Polynom]
            List of Polynom objects representing the different orders of 
            generators.
        """

        generators = []

        idx = 0
        for degree in self.generator_degrees:
            exponents = self.generator_exponents[degree]
            n = len(exponents)
            coeffs = params[idx:idx+n]
            idx += n


            terms = []
            for c, e in zip(coeffs, exponents):
                if abs(c) > 1e-15:
                    terms.append(Term(
                        coeff=c,
                        x_exp=e[0],
                        px_exp=e[1],
                        y_exp=e[2],
                        py_exp=e[3]
                    ))

            generators.append(Polynom(terms=terms))

        return generators
    

    def _linear_polynomial(self, row: np.ndarray) -> Polynom:
        """
        Function to evaluate a polynomial obtained from matrix multiplication of 
        a row vector of coefficients and the column vector of coordinates.

        Parameters:
        -----------
        row : np.ndarray
            Row vector of coefficients.

        Returns:
        --------
        Polynom
            Polynomial representing the result of the multiplication.
        """

        basis = [
            (1,0,0,0),
            (0,1,0,0),
            (0,0,1,0),
            (0,0,0,1)
        ]

        terms = []

        for c, e in zip(row, basis):
            if abs(c) > 1e-15:
                terms.append(Term(
                    coeff=c,
                    x_exp=e[0],
                    px_exp=e[1],
                    y_exp=e[2],
                    py_exp=e[3]
                ))

        return Polynom(terms=terms)
    

    @staticmethod
    def _poisson_bracket(f: Polynom, g: Polynom, max_order: int) -> Polynom:
        """
        Function that returns the Poisson bracket of two polynomials f and g 
        truncated at a maximum order.

        Parameters:
        -----------
        f : Polynom
            Polynomial on the left side of the Poisson bracket.
        g : Polynom
            Polynomial on the right side of the Poisson bracket.
        max_order : int
            Maximum order above which result is truncated.

        Returns:
        --------
        Polynom
            Polynomial of degree `max_order`.
        """

        dfx  = Polynom.diff_Polynom(f, "x")
        dfpx = Polynom.diff_Polynom(f, "px")
        dfy  = Polynom.diff_Polynom(f, "y")
        dfpy = Polynom.diff_Polynom(f, "py")

        dgx  = Polynom.diff_Polynom(g, "x")
        dgpx = Polynom.diff_Polynom(g, "px")
        dgy  = Polynom.diff_Polynom(g, "y")
        dgpy = Polynom.diff_Polynom(g, "py")

        result = Polynom.product_Polynoms(dfx, dgpx, max_order)
        result = Polynom.sum_Polynoms(
            result,
            Polynom.product_Coeff_Polynom(-1, Polynom.product_Polynoms(
                dfpx, dgx, max_order
            ))
        )
        result = Polynom.sum_Polynoms(
            result,
            Polynom.product_Polynoms(dfy, dgpy, max_order)
        )
        result = Polynom.sum_Polynoms(
            result,
            Polynom.product_Coeff_Polynom(-1, Polynom.product_Polynoms(
                dfpy, dgy, max_order
            ))
        )

        return result
    

    def _lie_exp(
        self,
        f: Polynom,
        g: Polynom,
        order: int
    ) -> Polynom:
        """
        Function to evaluate the polynomial resulting from $e^{:f:}g$.

        Parameters:
        -----------
        f : Polynom
            Polynomial in the exponential.
        g : Polynom
            Polynomial outside the exponential.
        order : int
            Order up to which to retain terms in the resulting polynomial.

        Returns:
        --------
        Polynom
            Truncated polynomial representation of $e^{:f:}g$.
        """

        result = copy.deepcopy(g)
        current = copy.deepcopy(g)

        for n in range(1, order + 1):
            current = self._poisson_bracket(
                f,
                current,
                max_order=order
            )

            result = Polynom.sum_Polynoms(
                result,
                Polynom.product_Coeff_Polynom(1.0/factorial(n), current)
            )

        return result
    

    def _apply_generators(
        self,
        p: Polynom,
        generators: list[Polynom],
        order: int
    ) -> Polynom:
        """
        Function to apply Lie exponential of several generator orders.

        Parameters:
        -----------
        p : Polynom
            Polynomial to apply Lie exponentials to.
        generators : list[Polynom]
            List of polynomials representing generator of different orders.
        order : int
            Order up to which to retain terms in the resulting polynomial.

        Returns:
        --------
        Polynom
            Polynomial truncated at given order.
        """

        poly = copy.deepcopy(p)
        for f in generators:
            poly = self._lie_exp(f, poly, order)

        return poly
    

    def _build_map(self, params: Union[np.ndarray, list], order: int) -> Map:
        """
        Function to contruct map with given parameters by applying Lie 
        exponential of several orders of generators to linearly transformed 
        coordinates.

        Parameters:
        -----------
        params : np.ndarray | list
            Iterable of numerical values of map parametrs.
        order : int
            Order up to which to retain terms in the resulting map.

        Returns:
        --------
        Map
            Map of 4 polynomials.
        """

        generators = self._unpack_generators(params)

        Rx  = self._linear_polynomial(self.R[0])
        Rpx = self._linear_polynomial(self.R[1])
        Ry  = self._linear_polynomial(self.R[2])
        Rpy = self._linear_polynomial(self.R[3])

        
        Mx  = self._apply_generators(Rx,  generators, order)
        Mpx = self._apply_generators(Rpx, generators, order)
        My  = self._apply_generators(Ry,  generators, order)
        Mpy = self._apply_generators(Rpy, generators, order)

        map = Map(
            x_poly=Mx,
            px_poly=Mpx,
            y_poly=My,
            py_poly=Mpy,
        )

        return map
    

    @staticmethod
    def _evaluate_map(
        map: Map,
        coords: np.ndarray
    ) -> np.ndarray:
        result = map.substitute(coords[0], coords[1], coords[2], coords[3])

        return np.vstack(result).real
    

    def residuals(
        self,
        params: Union[np.ndarray, list],
        z0: np.ndarray,
        z1: np.ndarray,
        order: int
    ) -> np.ndarray:
        """
        Function to compute the residuals between coordinates tracked with full 
        lattice and those obtained by application of the one-turn map.

        Parameters:
        -----------
        params : np.ndarray | list
            Numerical parameters of the one-turn map.
        z0 : np.ndarray
            Initial coordinates.
        z1 : np.ndarray
            Final coordinates obtained from tracking with full lattice.
        order : int
            Order up to which to retain terms in the one-turn map.

        Returns:
        --------
        np.ndarray
            Array of predicted - tracked coordinates.
        """

        maps = self._build_map(params, order)

        pred = self._evaluate_map(maps, z0)

        return (pred - z1).ravel()


    def _fit_full_lie_map(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        order: int
    ) -> Map:
        """
        Function to perform least-squares fit of one-turn map with Lie map 
        ansatz.

        Parameters:
        -----------
        z0 : np.ndarray
            Initial coordinates.
        z1 : np.ndarray
            Final coordinates to match with fitted one-turn map application.
        order : int
            Order up to which to retain terms in the one-turn map.
        
        Returns:
        --------
        Map
            Fitted one-turn map.
        """
        
        nparams = sum(self.generator_sizes.values())

        x0 = np.zeros(nparams)

        result = least_squares(
            self.residuals,
            x0,
            args=(z0, z1, order),
            verbose=2,
            method='trf'
        )

        params = result.x

        generators = self._unpack_generators(params)

        map = self._build_map(params, order)

        return map


    def compute(self):
        """
        Function to compute one-turn map by least-squares fit.

        Paramaters:
        -----------

        Returns:
        --------
        """

        map = self._fit_full_lie_map(
            self.z0,
            self.z1,
            order=self.max_map_order
        )

        self.one_turn_map = map
    

    def _calculate_complex_one_turn_map(self) -> None:
        """
        Function to calculate the one-turn map in complex normalised
        Courant-Snyder coordinates by composing the individual polynomial
        element maps and truncating at the given order.

        Parameters:
        -----------

        Returns:
        --------
        """

        if self.one_turn_map is None:
            self.compute()

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

        Parameters:
        -----------
        max_nf_order : int
            Maximum order of the normal form.
        res_space_dim : int
            Dimension of the resonant space, 0 if nonresonant, 1 if single 
            resonance, 2 if double resonance.
        res_case : int
            0 for nonresonant normal forms, 1 for exactly resonant normal forms, 
            2 for quasiresonant normal forms.
        res_eig : list[complex]
            List of complex resonant eigenvalues, i.e. [$e^{2i\\pi Q_{x,res}}$, 
            $e^{-2i\\pi Q_{x,res}}$, $e^{2i\\pi Q_{y,res}}$, 
            $e^{-2i\\pi Q_{y,res}}$], only needed if res_case is 1 or 2.
        res_basis1: list[int]
            List of integers, [n, m] which satisfy the resonance condition 
            $n*Q_x+m*Q_y=p$, only needed if res_space_dim is 1 or 2.
        res_basis2: list[int]
            List of integers, [n, m] which satisfy the resonance condition 
            $n*Q_x+m*Q_y=p$ for a second resonance, only needed if 
            res_space_dim is 2.

        Returns:
        --------
        """

        self._calculate_complex_one_turn_map()

        if self._complex_one_turn_map is None:
            raise ValueError(
                "Complex one-turn map was not calculated."
            )

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

    def get_orbit_from_normal_form(
        self,
        coord_norm: np.ndarray,
        plane: str,
        res_order: Union[int, None] = None
    ) -> list[np.ndarray]:
        coords_nf = self.normal_form.norm_to_nf(
            coord_norm[0],
            coord_norm[1],
            coord_norm[2],
            coord_norm[3],
        )
        H_values = self.normal_form.H.substitute(
            coords_nf[0],
            coords_nf[1],
            coords_nf[2],
            coords_nf[3],
        )
        H_values = np.imag(H_values)

        if plane == "H":
            sigma_limit = np.nanmax(np.sqrt(coord_norm[0]**2 + coord_norm[1]**2))
            zeta1_re = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            zeta1_im = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            zeta2_re = np.zeros(1000)
            zeta2_im = np.zeros(1000)
        else:
            sigma_limit = np.nanmax(np.sqrt(coord_norm[2]**2 + coord_norm[3]**2))
            zeta1_re = np.zeros(1000)
            zeta1_im = np.zeros(1000)
            zeta2_re = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            zeta2_im = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)

        zeta1_re, zeta1_im = np.meshgrid(zeta1_re, zeta1_im)
        zeta2_re, zeta2_im = np.meshgrid(zeta2_re, zeta2_im)

        H_values_grid = self.normal_form.H.substitute(
            zeta1_re + 1j * zeta1_im,
            zeta1_re - 1j * zeta1_im,
            zeta2_re + 1j * zeta2_im,
            zeta2_re - 1j * zeta2_im,
        )
        H_values_grid = np.imag(H_values_grid)

        all_orbits = []
        for i in range(coord_norm.shape[1]):
            curr_zeta1_re = coords_nf[0][i].real
            curr_zeta1_im = coords_nf[0][i].imag
            curr_zeta2_re = coords_nf[2][i].real
            curr_zeta2_im = coords_nf[2][i].imag
            curr_H_value = H_values[i]
            if plane == "H":
                curr_inv = (curr_zeta1_re + 1j * curr_zeta1_im) * (curr_zeta1_re - 1j * curr_zeta1_im)
            else:
                curr_inv = (curr_zeta2_re + 1j * curr_zeta2_im) * (curr_zeta2_re - 1j * curr_zeta2_im)
            curr_inv = curr_inv.real
            
            # Extract contour
            if plane == "H":
                cont_gen = cpy.contour_generator(
                    zeta1_re,
                    -zeta1_im,
                    H_values_grid
                )
            else:
                cont_gen = cpy.contour_generator(
                    zeta2_re,
                    -zeta2_im,
                    H_values_grid
                )

            orbit = cont_gen.lines(curr_H_value)

            orbit_zeta_re = []
            orbit_zeta_im = []
            for sub_orbit in orbit:
                avg_inv = np.mean(
                   (sub_orbit[:, 0] - 1j*sub_orbit[:, 1])*(sub_orbit[:, 0] + 1j*sub_orbit[:, 1])
                )
                avg_inv = avg_inv.real
                if np.isclose(curr_inv, avg_inv, rtol=0.25):
                    orbit_zeta_re += list(sub_orbit[:, 0])
                    orbit_zeta_im += list(-sub_orbit[:, 1])
                else:
                    continue

            if plane == "H":
                orbit_zeta1 = np.asarray(orbit_zeta_re) + 1j * np.asarray(orbit_zeta_im)
                orbit_zeta1_c = np.asarray(orbit_zeta_re) - 1j * np.asarray(orbit_zeta_im)
                orbit_zeta2 = np.ones(len(orbit_zeta1)) * (curr_zeta2_re + 1j * curr_zeta2_im)
                orbit_zeta2_c = np.ones(len(orbit_zeta1)) * (curr_zeta2_re - 1j * curr_zeta2_im)
            else:
                orbit_zeta2 = np.asarray(orbit_zeta_re) + 1j * np.asarray(orbit_zeta_im)
                orbit_zeta2_c = np.asarray(orbit_zeta_re) - 1j * np.asarray(orbit_zeta_im)
                orbit_zeta1 = np.ones(len(orbit_zeta2)) * (curr_zeta1_re + 1j * curr_zeta1_im)
                orbit_zeta1_c = np.ones(len(orbit_zeta2)) * (curr_zeta1_re - 1j * curr_zeta1_im)

            # Transform back to normalised coordinates
            orbit_norm = self.normal_form.nf_to_norm(
                orbit_zeta1,
                orbit_zeta1_c,
                orbit_zeta2,
                orbit_zeta2_c,
            )
            all_orbits.append(orbit_norm)

        return all_orbits
    
    def get_orbit_from_normal_form_2(
        self,
        coord_norm: np.ndarray,
        plane: str
    ) -> list[np.ndarray]:
        if plane == "H":
            sigma_limit = np.max(np.sqrt(coord_norm[0]**2 + coord_norm[1]**2))
        else:
            sigma_limit = np.max(np.sqrt(coord_norm[2]**2 + coord_norm[3]**2))
        x_norm = coord_norm[0]
        px_norm = coord_norm[1]
        y_norm = coord_norm[2]
        py_norm = coord_norm[3]

        coords_nf = self.normal_form.norm_to_nf(
            x_norm,
            px_norm,
            y_norm,
            py_norm,
        )
        H_values = self.normal_form.H.substitute(
            coords_nf[0],
            coords_nf[1],
            coords_nf[2],
            coords_nf[3],
        )
        H_values = np.imag(H_values)
        
        Psi = copy.deepcopy(self.normal_form.Psi)
        H = copy.deepcopy(self.normal_form.H)

        max_order = 2 * Psi.get_max_order() * H.get_max_order()
        if plane == "H":
            z1 = Polynom(terms=[Term(1, x_exp=1), Term(-1j, px_exp=1)])
            z1c = Polynom(terms=[Term(1, x_exp=1), Term(1j, px_exp=1)])
            z2 = Polynom(terms=[Term(coeff=0)])
            z2c = Polynom(terms=[Term(coeff=0)])
        else:
            z1 = Polynom(terms=[Term(coeff=0)])
            z1c = Polynom(terms=[Term(coeff=0)])
            z2 = Polynom(terms=[Term(1, y_exp=1), Term(-1j, py_exp=1)])
            z2c = Polynom(terms=[Term(1, y_exp=1), Term(1j, py_exp=1)])
        Psi_4D_polys = [Psi.x_poly, Psi.px_poly, Psi.y_poly, Psi.py_poly]
        Psi_2D_polys = []
        for poly_4D in Psi_4D_polys:
            poly_2D = Polynom(terms=[Term(coeff=0)])
            for term_4D in poly_4D.terms:
                z2_z2c = Polynom.product_Polynoms(
                    Polynom.power_Polynom(z2, term_4D.y_exp, max_order),
                    Polynom.power_Polynom(z2c, term_4D.py_exp, max_order),
                    max_order
                )
                # z2_z2c.remove_zero_terms()
                z2_z2c.collect_terms()
                z1c_z2_z2c = Polynom.product_Polynoms(
                    Polynom.power_Polynom(z1c, term_4D.px_exp, max_order),
                    z2_z2c,
                    max_order
                )
                # z1c_z2_z2c.remove_zero_terms()
                z1c_z2_z2c.collect_terms()
                z1_z1c_z2_z2c = Polynom.product_Polynoms(
                    Polynom.power_Polynom(z1, term_4D.x_exp, max_order),
                    z1c_z2_z2c,
                    max_order
                )
                # z1_z1c_z2_z2c.remove_zero_terms()
                z1_z1c_z2_z2c.collect_terms()
                c_z1_z1c_z2_z2c = Polynom.product_Coeff_Polynom(term_4D.coeff, z1_z1c_z2_z2c)
                poly_2D = Polynom.sum_Polynoms(poly_2D, c_z1_z1c_z2_z2c)
            # poly_2D.remove_zero_terms()
            poly_2D.collect_terms()
            Psi_2D_polys.append(poly_2D)
        zeta1 = Psi_2D_polys[0]
        zeta1c = Psi_2D_polys[1]
        zeta2 = Psi_2D_polys[2]
        zeta2c = Psi_2D_polys[3]
        
        H_2D = Polynom(terms=[Term(coeff=0)])
        for term_4D in H.terms:
            z2_z2c = Polynom.product_Polynoms(
                Polynom.power_Polynom(zeta2, term_4D.y_exp, max_order),
                Polynom.power_Polynom(zeta2c, term_4D.py_exp, max_order),
                max_order
            )
            # z2_z2c.remove_zero_terms()
            z2_z2c.collect_terms()
            z1c_z2_z2c = Polynom.product_Polynoms(
                Polynom.power_Polynom(zeta1c, term_4D.px_exp, max_order),
                z2_z2c,
                max_order
            )
            # z1c_z2_z2c.remove_zero_terms()
            z1c_z2_z2c.collect_terms()
            z1_z1c_z2_z2c = Polynom.product_Polynoms(
                Polynom.power_Polynom(zeta1, term_4D.x_exp, max_order),
                z1c_z2_z2c,
                max_order
            )
            # z1_z1c_z2_z2c.remove_zero_terms()
            z1_z1c_z2_z2c.collect_terms()
            c_z1_z1c_z2_z2c = Polynom.product_Coeff_Polynom(term_4D.coeff, z1_z1c_z2_z2c)
            H_2D = Polynom.sum_Polynoms(H_2D, c_z1_z1c_z2_z2c)

        # Create dense grid on which to look for contours of H
        if plane == "H":
            _x_norm  = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            _px_norm = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            _x_norm, _px_norm = np.meshgrid(_x_norm, _px_norm)
            _y_norm = np.zeros(_x_norm.shape)
            _py_norm = np.zeros(_x_norm.shape)
        else:
            _y_norm  = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            _py_norm = np.linspace(-2*sigma_limit, 2*sigma_limit, 1000)
            _y_norm, _py_norm = np.meshgrid(_y_norm, _py_norm)
            _x_norm = np.zeros(_y_norm.shape)
            _px_norm = np.zeros(_y_norm.shape)
        z1 = _x_norm - 1j * _px_norm
        z1c = _x_norm + 1j * _px_norm
        z2 = _y_norm - 1j * _py_norm
        z2c = _y_norm + 1j * _py_norm

        H_values_grid = H_2D.substitute(
            z1,
            z1c,
            z2,
            z2c,
        )
        H_values_grid = np.imag(H_values_grid)

        # Create contour generator
        if plane == "H":
            cont_gen = cpy.contour_generator(
                _x_norm,
                _px_norm,
                H_values_grid
            )
        else:
            cont_gen = cpy.contour_generator(
                _y_norm,
                _py_norm,
                H_values_grid
            )

        # For each initial condition
        all_orbits = []
        for i in range(len(x_norm)):
            curr_H_value = H_values[i]

            # Extract contour
            orbit = cont_gen.lines(curr_H_value)
            
            for sub_orbit in orbit:
                all_orbits.append(np.vstack((sub_orbit[:, 0], sub_orbit[:, 1])))

        return all_orbits, H_2D, Psi_2D_polys


class ParametricOneTurnMap4D(OneTurnMap4D):
    """
    Variant of OneTurnMap4D that builds the Lie map structure once —
    symbolically, with callable coefficients — so that each call to
    `residuals` only evaluates those callables rather than re-running
    the full polynomial algebra.

    Usage is identical to OneTurnMap4D.  The one-time build cost is paid
    in __init__; every subsequent residual evaluation is cheap.
    """

    def __init__(
        self,
        line: Line,
        max_gen_degree: int,
        max_map_order: int,
        nemitt_x: float = 1,
        nemitt_y: float = 1,
        nemitt_z: float = 1,
        sigma_limit: float = 10,
    ) -> None:
        super().__init__(
            line, max_gen_degree, max_map_order,
            nemitt_x, nemitt_y, nemitt_z, sigma_limit,
        )
        self._parametric_map: ParametricMap = self._build_parametric_map(
            self.max_map_order
        )
        self._flatten_parametric_map()

    # ------------------------------------------------------------------
    # Parametric map construction (called once in __init__)
    # ------------------------------------------------------------------

    def _build_parametric_generators(self) -> list[ParametricPolynom]:
        """
        Return generator polynomials whose coefficients are callables
        ``lambda params: params[i]``.  Unlike _unpack_generators, all
        monomials are included because zero-pruning cannot be done before
        numerical params are available.
        """
        generators = []
        idx = 0
        for degree in self.generator_degrees:
            exponents = self.generator_exponents[degree]
            terms = []
            for j, e in enumerate(exponents):
                i = idx + j
                terms.append(ParametricTerm(
                    coeff=lambda params, i=i: params[i],
                    x_exp=e[0],
                    px_exp=e[1],
                    y_exp=e[2],
                    py_exp=e[3],
                ))
            generators.append(ParametricPolynom(terms=terms))
            idx += len(exponents)
        return generators

    @staticmethod
    def _parametric_poisson_bracket(
        f: ParametricPolynom,
        g: ParametricPolynom,
        max_order: int,
    ) -> ParametricPolynom:
        dfx  = ParametricPolynom.diff_Polynom(f, "x")
        dfpx = ParametricPolynom.diff_Polynom(f, "px")
        dfy  = ParametricPolynom.diff_Polynom(f, "y")
        dfpy = ParametricPolynom.diff_Polynom(f, "py")

        dgx  = ParametricPolynom.diff_Polynom(g, "x")
        dgpx = ParametricPolynom.diff_Polynom(g, "px")
        dgy  = ParametricPolynom.diff_Polynom(g, "y")
        dgpy = ParametricPolynom.diff_Polynom(g, "py")

        result = ParametricPolynom.product_Polynoms(dfx, dgpx, max_order)
        result = ParametricPolynom.sum_Polynoms(
            result,
            ParametricPolynom.product_Coeff_Polynom(
                -1, ParametricPolynom.product_Polynoms(dfpx, dgx, max_order)
            ),
        )
        result = ParametricPolynom.sum_Polynoms(
            result,
            ParametricPolynom.product_Polynoms(dfy, dgpy, max_order),
        )
        result = ParametricPolynom.sum_Polynoms(
            result,
            ParametricPolynom.product_Coeff_Polynom(
                -1, ParametricPolynom.product_Polynoms(dfpy, dgy, max_order)
            ),
        )
        return result

    def _parametric_lie_exp(
        self,
        f: ParametricPolynom,
        g: ParametricPolynom,
        order: int,
    ) -> ParametricPolynom:
        result = copy.deepcopy(g)
        current = copy.deepcopy(g)

        for n in range(1, order + 1):
            current = self._parametric_poisson_bracket(f, current, max_order=order)
            result = ParametricPolynom.sum_Polynoms(
                result,
                ParametricPolynom.product_Coeff_Polynom(1.0 / factorial(n), current),
            )
        return result

    def _apply_parametric_generators(
        self,
        p: ParametricPolynom,
        generators: list[ParametricPolynom],
        order: int,
    ) -> ParametricPolynom:
        poly = copy.deepcopy(p)
        for f in generators:
            poly = self._parametric_lie_exp(f, poly, order)
        return poly

    def _build_parametric_map(self, order: int) -> ParametricMap:
        generators = self._build_parametric_generators()

        Rx  = ParametricPolynom.from_polynom(self._linear_polynomial(self.R[0]))
        Rpx = ParametricPolynom.from_polynom(self._linear_polynomial(self.R[1]))
        Ry  = ParametricPolynom.from_polynom(self._linear_polynomial(self.R[2]))
        Rpy = ParametricPolynom.from_polynom(self._linear_polynomial(self.R[3]))

        Mx  = self._apply_parametric_generators(Rx,  generators, order)
        Mpx = self._apply_parametric_generators(Rpx, generators, order)
        My  = self._apply_parametric_generators(Ry,  generators, order)
        Mpy = self._apply_parametric_generators(Rpy, generators, order)

        return ParametricMap(x_poly=Mx, px_poly=Mpx, y_poly=My, py_poly=Mpy)

    def _flatten_parametric_map(
        self,
        _PP=_ParamPoly,
        _mfc=_make_fast_coeff,
    ) -> None:
        """
        Replace every term's deep callable chain with a flat numpy evaluator.

        Each coefficient c(params) is a polynomial in params.  We extract
        this polynomial by probing with _ParamPoly basis elements — one probe
        pass per term — then compile each coefficient to a fast function that
        evaluates the linear part via np.dot and handles any higher-degree
        monomials with a short Python loop.
        """
        nparams = sum(self.generator_sizes.values())
        probe = [_PP({(i,): 1.0}) for i in range(nparams)]

        # The lambda chain depth grows roughly as 2^(max_map_order).
        # We set a generous limit and restore it afterwards.
        old_limit = sys.getrecursionlimit()
        needed = max(old_limit, 2000 * 2 ** max(0, self.max_map_order - 4))
        sys.setrecursionlimit(needed)

        # Measure base stack depth so we can report a useful number on failure.
        base_depth = 0
        _f = sys._getframe()
        while _f is not None:
            base_depth += 1
            _f = _f.f_back

        try:
            for poly in (
                self._parametric_map.x_poly,
                self._parametric_map.px_poly,
                self._parametric_map.y_poly,
                self._parametric_map.py_poly,
            ):
                for term in poly.terms:
                    try:
                        result = term.coeff(probe)
                    except RecursionError:
                        raise RecursionError(
                            f"Recursion limit hit during map flattening "
                            f"(limit was {needed}, base stack depth was "
                            f"{base_depth}, so lambda chain depth exceeded "
                            f"~{needed - base_depth}). "
                            f"Increase the limit by passing a larger "
                            f"recursion_limit to _flatten_parametric_map, "
                            f"or call sys.setrecursionlimit() before "
                            f"instantiating ParametricOneTurnMap4D."
                        ) from None
                    pp = result if isinstance(result, _PP) else _PP({(): complex(result)})
                    term.coeff = _mfc(pp, nparams)
        finally:
            sys.setrecursionlimit(old_limit)

    # ------------------------------------------------------------------
    # Residuals — only numerical evaluation, no map construction
    # ------------------------------------------------------------------

    def residuals(
        self,
        params: Union[np.ndarray, list],
        z0: np.ndarray,
        z1: np.ndarray,
        _: int,
    ) -> np.ndarray:
        """
        Compute residuals using the prebuilt parametric map.  The trailing
        `order` argument (passed as `_`) is kept for API compatibility with
        the parent class but is unused: the map was already built at
        `max_map_order`.
        """
        pred = np.vstack(
            self._parametric_map.substitute(z0[0], z0[1], z0[2], z0[3], params)
        ).real
        return (pred - z1).ravel()