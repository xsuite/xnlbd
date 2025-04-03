from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Tuple, Union, cast

import numpy as np
import xtrack as xt  # type: ignore[import-untyped, import-not-found]


@dataclass
class Term:
    """
    Dataclass representing a term in a polynomial of 4 variables.

    Attributes:
        - coeff: float or complex, numerical value of the coefficient
        - x_exp: integer, exponent of the first variable, default `0`
        - px_exp: integer, exponent of the second variable, default `0`
        - y_exp: integer, exponent of the third variable, default `0`
        - py_exp: integer, exponent of the fourth variable, default `0`
    """

    coeff: float | complex
    x_exp: int = 0
    px_exp: int = 0
    y_exp: int = 0
    py_exp: int = 0

    def __str__(self) -> str:
        return (
            f"{self.coeff}\t( {self.x_exp} {self.px_exp} {self.y_exp} {self.py_exp} )"
        )

    def __eq__(self, other):
        if not isinstance(other, Term):
            return ValueError(
                "Term object can only be compared to another Term object!"
            )

        self_coeff_re_o = (
            0
            if np.isclose(self.coeff.real, 0, rtol=1e-14, atol=1e-16)
            else np.floor(np.log10(np.abs(self.coeff.real)))
        )
        self_coeff_im_o = (
            0
            if np.isclose(self.coeff.imag, 0, rtol=1e-14, atol=1e-16)
            else np.floor(np.log10(np.abs(self.coeff.imag)))
        )
        other_coeff_re_o = (
            0
            if np.isclose(other.coeff.real, 0, rtol=1e-14, atol=1e-16)
            else np.floor(np.log10(np.abs(other.coeff.real)))
        )
        other_coeff_im_o = (
            0
            if np.isclose(other.coeff.imag, 0, rtol=1e-14, atol=1e-16)
            else np.floor(np.log10(np.abs(other.coeff.imag)))
        )

        coeff_re_atol = float(f"1e{np.min([self_coeff_re_o, other_coeff_re_o])-10:.0f}")
        coeff_im_atol = float(f"1e{np.min([self_coeff_im_o, other_coeff_im_o])-10:.0f}")

        coeff_re_eq = np.isclose(self.coeff.real, other.coeff.real, rtol=1e-14, atol=coeff_re_atol)
        coeff_im_eq = np.isclose(self.coeff.imag, other.coeff.imag, rtol=1e-14, atol=coeff_im_atol)
        x_exp_eq = np.isclose(self.x_exp, other.x_exp, rtol=1e-14, atol=1e-16)
        px_exp_eq = np.isclose(self.px_exp, other.px_exp, rtol=1e-14, atol=1e-16)
        y_exp_eq = np.isclose(self.y_exp, other.y_exp, rtol=1e-14, atol=1e-16)
        py_exp_eq = np.isclose(self.py_exp, other.py_exp, rtol=1e-14, atol=1e-16)

        return np.all(
            [coeff_re_eq, coeff_im_eq, x_exp_eq, px_exp_eq, y_exp_eq, py_exp_eq]
        )

    def __hash__(self):
        rounded_coeff = complex(round(self.coeff.real, 14), round(self.coeff.imag, 14))

        return hash((rounded_coeff, self.x_exp, self.px_exp, self.y_exp, self.py_exp))

    @staticmethod
    def product_Terms(term1: Term, term2: Term, max_order: int) -> Term:
        """
        Function that evaluates the product of two terms, truncating at a
        given order.

        Input:
            - term1: Term, first term in the product
            - term2: Term, second term in the product
            - max_order: integer, order above which the product is truncated

        Output:
            - Term object representing the truncated product of the input terms
        """

        final_order = (
            term1.x_exp
            + term2.x_exp
            + term1.px_exp
            + term2.px_exp
            + term1.y_exp
            + term2.y_exp
            + term1.py_exp
            + term2.py_exp
        )

        if final_order > max_order:
            return Term(coeff=0)
        else:
            if np.isclose(term1.coeff, 0, rtol=1e-14, atol=1e-16):
                return Term(coeff=0)
            if np.isclose(term2.coeff, 0, rtol=1e-14, atol=1e-16):
                return Term(coeff=0)
            return Term(
                coeff=(term1.coeff * term2.coeff),
                x_exp=(term1.x_exp + term2.x_exp),
                px_exp=(term1.px_exp + term2.px_exp),
                y_exp=(term1.y_exp + term2.y_exp),
                py_exp=(term1.py_exp + term2.py_exp),
            )

    @staticmethod
    def power_Term(
        term: Term,
        pow: int,
        max_order: int,
    ) -> Term:
        """
        Function that evaluates a term raised to the given power, truncated
        at a given order.

        Input:
            - term: Term, term to be raised to a given power
            - pow: integer, power
            - max_order: integer, order above which the power is truncated

        Output:
            - Term object representing the input term to the power or 'pow',
              truncated at mx_order
        """

        if np.isclose(term.coeff, 0, rtol=1e-14, atol=1e-16):
            return Term(coeff=0)

        final_order = (
            term.x_exp * pow + term.px_exp * pow + term.y_exp * pow + term.py_exp * pow
        )

        if final_order > max_order:
            return Term(coeff=0)
        else:
            return Term(
                coeff=term.coeff**pow,
                x_exp=term.x_exp * pow,
                px_exp=term.px_exp * pow,
                y_exp=term.y_exp * pow,
                py_exp=term.py_exp * pow,
            )


@dataclass
class Polynom:
    """
    Dataclass representing a polynomial of 4 variables.

    Attributes:
        - terms: list of Term objects
    """

    terms: list[Term]

    def __str__(self) -> str:
        return "\n".join(map(str, self.terms))

    def __eq__(self, other):
        if not isinstance(other, Polynom):
            return ValueError(
                "Polynom object can only be compared to another Polynom object!"
            )

        self_terms = self.terms[:]
        other_terms = other.terms[:]

        if len(self_terms) != len(other_terms):
            return False

        for self_term in self_terms:
            for other_term in other_terms:
                if self_term == other_term:
                    other_terms.remove(other_term)
                    break

        eq = False
        if len(other_terms) == 0:
            eq = True

        return eq

    def truncate_at_order(self, max_order: int) -> None:
        """
        Function that truncates the polynomial at a given order.

        Input:
            - max_order: integer, order above which the polynomial is truncated

        Output:
            -
        """

        low_order_terms = []

        for term in self.terms:
            if (term.x_exp + term.px_exp + term.y_exp + term.py_exp) <= max_order:
                low_order_terms.append(term)

        self.terms = low_order_terms

    def remove_zero_terms(self) -> None:
        """
        Function that removes terms from the polynomial that can be considered
        zero (i.e. within machine precision) to speed up calculations.

        Input:
            -

        Output:
            -
        """

        nonzero_terms = []

        for term in self.terms:
            if np.isclose(term.coeff, 0, rtol=1e-14, atol=1e-16):
                continue
            nonzero_terms.append(term)

        self.terms = nonzero_terms

    def collect_terms(self) -> None:
        """
        Function that combines coefficients of terms that have the same
        exponents.

        Input:
            -

        Output:
            -
        """

        term_dict: dict[Tuple[int, int, int, int], Term] = {}

        for term in self.terms:
            key = (term.x_exp, term.px_exp, term.y_exp, term.py_exp)

            if key in term_dict:
                term_dict[key].coeff = term_dict[key].coeff + term.coeff
            else:
                term_dict[key] = term

        self.terms = list(term_dict.values())

    def substitute(
        self,
        x_val: np.ndarray,
        px_val: np.ndarray,
        y_val: np.ndarray,
        py_val: np.ndarray,
    ) -> np.ndarray:
        """
        Function that substitutes the input for the four variables x, px, y, py.

        Input:
            - x_val: float, complex or array of initial values of the first
              coordinate
            - px_val: float, complex or array of initial values of the second
              coordinate
            - y_val: float, complex or array of initial values of the third
              coordinate
            - py_val: float, complex or array of initial values of the fourth
              coordinate

        Output:
            - float, complex or array of the numerical value of the polynomial
              after substitution
        """

        sum = 0.0 + 0j
        for term in self.terms:
            sum += (
                term.coeff
                * x_val**term.x_exp
                * px_val**term.px_exp
                * y_val**term.y_exp
                * py_val**term.py_exp
            )
        return cast(np.ndarray, sum)

    @staticmethod
    def product_Coeff_Polynom(
        coeff: float | complex,
        poly: Polynom,
    ) -> Polynom:
        """
        Function that evaluates the product of a numerical coefficient and a
        polynomial.

        Input:
            - coeff: float or complex, the numerical coefficient to multiply
              with
            - poly: Polynom, the polynomial to be multiplied

        Output:
            - Polynom representing the new polynomial
        """

        new_terms = []

        for term in poly.terms:
            new_terms.append(
                Term(
                    coeff=(coeff * term.coeff),
                    x_exp=term.x_exp,
                    px_exp=term.px_exp,
                    y_exp=term.y_exp,
                    py_exp=term.py_exp,
                )
            )

        return Polynom(terms=new_terms)

    @staticmethod
    def sum_Polynoms(poly1: Polynom, poly2: Polynom) -> Polynom:
        """
        Function that evaluates the sum of two polynomials.

        Input:
            - poly1: Polynom, first polynomial in the sum
            - poly2: Polynom, second polynomial in the sum

        Output:
            - Polynom representing the new polynomial
        """

        new_terms = copy.deepcopy(poly1.terms) + copy.deepcopy(poly2.terms)

        new_poly = Polynom(terms=new_terms)
        new_poly.remove_zero_terms()
        new_poly.collect_terms()

        return new_poly

    @staticmethod
    def product_Polynoms(poly1: Polynom, poly2: Polynom, max_order: int) -> Polynom:
        """
        Function that evaluates the product of two polynomials truncated at a
        given order.

        Input:
            - poly1: Polynom, first polynomial in the product
            - poly2: Polynom, second polynomial in the product
            - max_order: integer, order above which the product is truncated

        Output:
            - Polynom representing the new polynomial
        """

        new_terms = []

        for term1 in poly1.terms:
            for term2 in poly2.terms:
                new_terms.append(Term.product_Terms(term1, term2, max_order))

        new_poly = Polynom(terms=new_terms)
        new_poly.remove_zero_terms()
        new_poly.collect_terms()

        return new_poly

    @staticmethod
    def power_Polynom(poly: Polynom, pow: int, max_order: int) -> Polynom:
        """
        Function that evaluates the power of a polynomial truncated at a given
        order.

        Input:
            - poly: Polynom, polynomial to be raised to the given power
            - pow: integer, the power to which the polynomial should be raised
            - max_order: integer, order above which the polynomial will be
              truncated

        Output:
            - Polynom representing the new polynomial
        """

        if pow == 0:
            return Polynom(
                terms=[
                    Term(
                        coeff=1,
                        x_exp=0,
                        px_exp=0,
                        y_exp=0,
                        py_exp=0,
                    )
                ]
            )
        elif pow == 1:
            return poly
        else:
            new_poly = Polynom.product_Polynoms(poly, poly, max_order)

            for _ in range(2, pow):
                new_poly = Polynom.product_Polynoms(new_poly, poly, max_order)

            return new_poly
        
    def get_max_order(self) -> int:
        """
        Function that returns the maximum order of the polynomial.

        Input:
            - 

        Output:
            - integer representing the maximum order of the polynomial
        """
        
        max_x_order = 0
        for term in self.terms:
            curr_order = (term.x_exp + term.px_exp + term.y_exp + term.py_exp)
            if curr_order > max_x_order:
                max_x_order = curr_order

        return max_x_order


@dataclass
class Map:
    """
    Dataclass representing a polynomial map of 4 variables.

    Attributes:
        - x_poly: Polynom, polynomial map of first coordinate
        - px_poly: Polynom, polynomial map of second coordinate
        - y_poly: Polynom, polynomial map of third coordinate
        - py_poly: Polynom, polynomial map of fourth coordinate
    """

    x_poly: Polynom
    px_poly: Polynom
    y_poly: Polynom
    py_poly: Polynom

    def substitute(
        self,
        x_val: np.ndarray,
        px_val: np.ndarray,
        y_val: np.ndarray,
        py_val: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Function that substitutes the input for the four variables x, px, y, py.

        Input:
            - x_val: array-like of initial values of the first
              coordinate
            - px_val: array-like of initial values of the second
              coordinate
            - y_val: array-like of initial values of the third
              coordinate
            - py_val: array-like of initial values of the fourth
              coordinate

        Output:
            - array-like of the numerical value of the polynomial
              after substitution
        """

        return (
            self.x_poly.substitute(x_val, px_val, y_val, py_val),
            self.px_poly.substitute(x_val, px_val, y_val, py_val),
            self.y_poly.substitute(x_val, px_val, y_val, py_val),
            self.py_poly.substitute(x_val, px_val, y_val, py_val),
        )
    
    def get_max_order(self) -> int:
        """
        Function that returns the maximum map order.

        Input:
            - x_val: array-like of initial values of the first
              coordinate
            - px_val: array-like of initial values of the second
              coordinate
            - y_val: array-like of initial values of the third
              coordinate
            - py_val: array-like of initial values of the fourth
              coordinate

        Output:
            - array-like of the numerical value of the polynomial
              after substitution
        """
        
        max_x_order = self.x_poly.get_max_order()
        max_px_order = self.px_poly.get_max_order()
        max_y_order = self.y_poly.get_max_order()
        max_py_order = self.py_poly.get_max_order()

        return np.max([max_x_order, max_px_order, max_y_order, max_py_order])

    def __str__(self) -> str:
        string = (
            "x:\n"
            + str(self.x_poly)
            + "\n"
            + "px:\n"
            + str(self.px_poly)
            + "\n"
            + "y:\n"
            + str(self.y_poly)
            + "\n"
            + "py:\n"
            + str(self.py_poly)
        )

        return string

    def __eq__(self, other):
        if not isinstance(other, Map):
            return ValueError("Map object can only be compared to another Map object!")

        x_poly_eq = self.x_poly == other.x_poly
        px_poly_eq = self.px_poly == other.px_poly
        y_poly_eq = self.y_poly == other.y_poly
        py_poly_eq = self.py_poly == other.py_poly

        return np.all([x_poly_eq, px_poly_eq, y_poly_eq, py_poly_eq])

    @staticmethod
    def composition_Map(map1: Map, map2: Map, max_order: int) -> Map:
        """
        Function that evaluates the composition of two maps truncated at a
        given order.

        Input:
            - map1: Map, inner map in the composition
            - map2: outer map in the composition
            - max_order: integer, order above which the polynomial components
              of the map are truncated

        Output:
            - Map representing the composition
        """

        x_poly = Polynom(terms=[])
        for term in map2.x_poly.terms:
            x_substituted = Polynom.power_Polynom(map1.x_poly, term.x_exp, max_order)
            px_substituted = Polynom.power_Polynom(map1.px_poly, term.px_exp, max_order)
            y_substituted = Polynom.power_Polynom(map1.y_poly, term.y_exp, max_order)
            py_substituted = Polynom.power_Polynom(map1.py_poly, term.py_exp, max_order)

            x_px_prod = Polynom.product_Polynoms(
                x_substituted, px_substituted, max_order
            )

            y_py_prod = Polynom.product_Polynoms(
                y_substituted, py_substituted, max_order
            )

            full_prod = Polynom.product_Polynoms(x_px_prod, y_py_prod, max_order)

            x_poly = Polynom.sum_Polynoms(
                x_poly, Polynom.product_Coeff_Polynom(term.coeff, full_prod)
            )
        x_poly.remove_zero_terms()
        x_poly.collect_terms()

        px_poly = Polynom(terms=[])
        for term in map2.px_poly.terms:
            x_substituted = Polynom.power_Polynom(map1.x_poly, term.x_exp, max_order)
            px_substituted = Polynom.power_Polynom(map1.px_poly, term.px_exp, max_order)
            y_substituted = Polynom.power_Polynom(map1.y_poly, term.y_exp, max_order)
            py_substituted = Polynom.power_Polynom(map1.py_poly, term.py_exp, max_order)

            x_px_prod = Polynom.product_Polynoms(
                x_substituted, px_substituted, max_order
            )
            y_py_prod = Polynom.product_Polynoms(
                y_substituted, py_substituted, max_order
            )

            full_prod = Polynom.product_Polynoms(x_px_prod, y_py_prod, max_order)

            px_poly = Polynom.sum_Polynoms(
                px_poly, Polynom.product_Coeff_Polynom(term.coeff, full_prod)
            )
        px_poly.remove_zero_terms()
        px_poly.collect_terms()

        y_poly = Polynom(terms=[])
        for term in map2.y_poly.terms:
            x_substituted = Polynom.power_Polynom(map1.x_poly, term.x_exp, max_order)
            px_substituted = Polynom.power_Polynom(map1.px_poly, term.px_exp, max_order)
            y_substituted = Polynom.power_Polynom(map1.y_poly, term.y_exp, max_order)
            py_substituted = Polynom.power_Polynom(map1.py_poly, term.py_exp, max_order)

            x_px_prod = Polynom.product_Polynoms(
                x_substituted, px_substituted, max_order
            )
            y_py_prod = Polynom.product_Polynoms(
                y_substituted, py_substituted, max_order
            )

            full_prod = Polynom.product_Polynoms(x_px_prod, y_py_prod, max_order)

            y_poly = Polynom.sum_Polynoms(
                y_poly, Polynom.product_Coeff_Polynom(term.coeff, full_prod)
            )
        y_poly.remove_zero_terms()
        y_poly.collect_terms()

        py_poly = Polynom(terms=[])
        for term in map2.py_poly.terms:
            x_substituted = Polynom.power_Polynom(map1.x_poly, term.x_exp, max_order)
            px_substituted = Polynom.power_Polynom(map1.px_poly, term.px_exp, max_order)
            y_substituted = Polynom.power_Polynom(map1.y_poly, term.y_exp, max_order)
            py_substituted = Polynom.power_Polynom(map1.py_poly, term.py_exp, max_order)

            x_px_prod = Polynom.product_Polynoms(
                x_substituted, px_substituted, max_order
            )
            y_py_prod = Polynom.product_Polynoms(
                y_substituted, py_substituted, max_order
            )

            full_prod = Polynom.product_Polynoms(x_px_prod, y_py_prod, max_order)

            py_poly = Polynom.sum_Polynoms(
                py_poly, Polynom.product_Coeff_Polynom(term.coeff, full_prod)
            )
        py_poly.remove_zero_terms()
        py_poly.collect_terms()

        return Map(x_poly=x_poly, px_poly=px_poly, y_poly=y_poly, py_poly=py_poly)
