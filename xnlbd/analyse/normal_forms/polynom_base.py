from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Tuple, Union, cast

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

        coeff_re_eq = np.isclose(
            self.coeff.real, other.coeff.real, rtol=1e-14, atol=coeff_re_atol
        )
        coeff_im_eq = np.isclose(
            self.coeff.imag, other.coeff.imag, rtol=1e-14, atol=coeff_im_atol
        )
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
        
    @staticmethod
    def diff_Polynom(poly: Polynom, var: str) -> Polynom:
        """
        Function that evaluates the power of a polynomial truncated at a given
        order.

        Input:
            - poly: Polynom, polynomial to be raised to the given power
            - var: name of variable w.r.t. which to differentiate

        Output:
            - Polynom representing the new polynomial
        """

        terms = []
        if var == "x":
            for term in poly.terms:
                if term.x_exp > 0:
                    terms.append(Term(
                        coeff=term.coeff * term.x_exp,
                        x_exp=term.x_exp - 1,
                        px_exp=term.px_exp,
                        y_exp=term.y_exp,
                        py_exp=term.py_exp,
                    ))
        elif var == "px":
            for term in poly.terms:
                if term.px_exp > 0:
                    terms.append(Term(
                        coeff=term.coeff * term.px_exp,
                        x_exp=term.x_exp,
                        px_exp=term.px_exp-1,
                        y_exp=term.y_exp,
                        py_exp=term.py_exp,
                    ))
        elif var == "y":
            for term in poly.terms:
                if term.y_exp > 0:
                    terms.append(Term(
                        coeff=term.coeff * term.y_exp,
                        x_exp=term.x_exp,
                        px_exp=term.px_exp,
                        y_exp=term.y_exp-1,
                        py_exp=term.py_exp,
                    ))
        elif var == "py":
            for term in poly.terms:
                if term.py_exp > 0:
                    terms.append(Term(
                        coeff=term.coeff * term.py_exp,
                        x_exp=term.x_exp,
                        px_exp=term.px_exp,
                        y_exp=term.y_exp,
                        py_exp=term.py_exp-1,
                    ))
        else:
            raise ValueError(
                f"Incorrect variable {var}."
            )
        
        return Polynom(terms=terms)

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
            curr_order = term.x_exp + term.px_exp + term.y_exp + term.py_exp
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
            -

        Output:
            - integer representing the maximum map order
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


@dataclass
class ParametricTerm:
    """
    Dataclass representing a term in a polynomial of 4 variables whose
    coefficient is a callable ``coeff(params) -> complex`` rather than a
    fixed number.  All algebraic operations mirror those of :class:`Term` but
    compose callables instead of multiplying scalars, so the symbolic
    structure is built once and evaluated cheaply at any ``params``.

    Attributes:
        - coeff: Callable[[params], complex], parametric coefficient
        - x_exp: integer, exponent of the first variable, default ``0``
        - px_exp: integer, exponent of the second variable, default ``0``
        - y_exp: integer, exponent of the third variable, default ``0``
        - py_exp: integer, exponent of the fourth variable, default ``0``
    """

    coeff: Callable
    x_exp: int = 0
    px_exp: int = 0
    y_exp: int = 0
    py_exp: int = 0

    def __str__(self) -> str:
        return (
            f"f(params)\t( {self.x_exp} {self.px_exp} {self.y_exp} {self.py_exp} )"
        )

    def __eq__(self, other):
        if not isinstance(other, ParametricTerm):
            return ValueError(
                "ParametricTerm object can only be compared to another ParametricTerm object!"
            )
        return (
            self.x_exp == other.x_exp
            and self.px_exp == other.px_exp
            and self.y_exp == other.y_exp
            and self.py_exp == other.py_exp
        )

    def __hash__(self):
        return hash((self.x_exp, self.px_exp, self.y_exp, self.py_exp))

    @classmethod
    def from_term(cls, term: Term) -> ParametricTerm:
        """Wrap a numeric :class:`Term` as a constant-callable ParametricTerm."""
        v = term.coeff
        return cls(
            coeff=lambda _, v=v: v,
            x_exp=term.x_exp,
            px_exp=term.px_exp,
            y_exp=term.y_exp,
            py_exp=term.py_exp,
        )

    @staticmethod
    def product_Terms(
        term1: "ParametricTerm",
        term2: "ParametricTerm",
        max_order: int,
    ) -> "ParametricTerm":
        """
        Function that evaluates the product of two parametric terms, truncating
        at a given order.

        Input:
            - term1: ParametricTerm, first term in the product
            - term2: ParametricTerm, second term in the product
            - max_order: integer, order above which the product is truncated

        Output:
            - ParametricTerm representing the truncated product
        """
        final_order = (
            term1.x_exp + term2.x_exp
            + term1.px_exp + term2.px_exp
            + term1.y_exp + term2.y_exp
            + term1.py_exp + term2.py_exp
        )
        if final_order > max_order:
            return type(term1)(coeff=lambda _: 0.0)
        c1, c2 = term1.coeff, term2.coeff
        return type(term1)(
            coeff=lambda params, c1=c1, c2=c2: c1(params) * c2(params),
            x_exp=term1.x_exp + term2.x_exp,
            px_exp=term1.px_exp + term2.px_exp,
            y_exp=term1.y_exp + term2.y_exp,
            py_exp=term1.py_exp + term2.py_exp,
        )

    @staticmethod
    def power_Term(
        term: "ParametricTerm",
        pow: int,
        max_order: int,
    ) -> "ParametricTerm":
        """
        Function that evaluates a parametric term raised to the given power,
        truncated at a given order.

        Input:
            - term: ParametricTerm, term to be raised to a given power
            - pow: integer, power
            - max_order: integer, order above which the power is truncated

        Output:
            - ParametricTerm representing the input term to the power 'pow',
              truncated at max_order
        """
        if pow == 0:
            return type(term)(coeff=lambda _: 1.0)

        final_order = (
            term.x_exp * pow
            + term.px_exp * pow
            + term.y_exp * pow
            + term.py_exp * pow
        )
        if final_order > max_order:
            return type(term)(coeff=lambda _: 0.0)

        c, p = term.coeff, pow
        return type(term)(
            coeff=lambda params, c=c, p=p: c(params) ** p,
            x_exp=term.x_exp * pow,
            px_exp=term.px_exp * pow,
            y_exp=term.y_exp * pow,
            py_exp=term.py_exp * pow,
        )


@dataclass
class ParametricPolynom:
    """
    Dataclass representing a polynomial of 4 variables with parametric
    coefficients.  Each term's coefficient is a callable so the symbolic
    structure (which monomials appear and how their coefficients depend on
    the parameter vector) is built once and evaluated cheaply.

    Note: :meth:`__eq__` compares only the monomial structure (exponents),
    not the coefficient callables, since callable equality is undecidable.

    Attributes:
        - terms: list of ParametricTerm objects
    """

    terms: list[ParametricTerm]

    def __str__(self) -> str:
        return "\n".join(map(str, self.terms))

    def __eq__(self, other):
        if not isinstance(other, ParametricPolynom):
            return ValueError(
                "ParametricPolynom object can only be compared to another ParametricPolynom object!"
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

        return len(other_terms) == 0

    @classmethod
    def from_polynom(cls, poly: Polynom) -> ParametricPolynom:
        """Wrap a numeric :class:`Polynom` as a ParametricPolynom."""
        return cls(terms=[ParametricTerm.from_term(t) for t in poly.terms])

    def truncate_at_order(self, max_order: int) -> None:
        """
        Function that truncates the polynomial at a given order.

        Input:
            - max_order: integer, order above which the polynomial is truncated
        """
        self.terms = [
            t for t in self.terms
            if (t.x_exp + t.px_exp + t.y_exp + t.py_exp) <= max_order
        ]

    def remove_zero_terms(self) -> None:
        """
        No-op: whether a callable coefficient evaluates to zero cannot be
        determined at build time.  Zero pruning happens implicitly during
        :meth:`substitute` when numerical values are available.
        """
        pass

    def collect_terms(self) -> None:
        """
        Function that combines coefficients of terms that have the same
        exponents by composing a new callable that sums the originals.
        """
        term_dict: dict[Tuple[int, int, int, int], ParametricTerm] = {}
        for term in self.terms:
            key = (term.x_exp, term.px_exp, term.y_exp, term.py_exp)
            if key in term_dict:
                c1 = term_dict[key].coeff
                c2 = term.coeff
                term_dict[key] = type(term)(
                    coeff=lambda params, c1=c1, c2=c2: c1(params) + c2(params),
                    x_exp=term.x_exp,
                    px_exp=term.px_exp,
                    y_exp=term.y_exp,
                    py_exp=term.py_exp,
                )
            else:
                term_dict[key] = term
        self.terms = list(term_dict.values())

    def substitute(
        self,
        x_val: np.ndarray,
        px_val: np.ndarray,
        y_val: np.ndarray,
        py_val: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Function that substitutes the input for the four variables x, px, y, py
        given numerical params.

        Input:
            - x_val: float, complex or array of initial values of x
            - px_val: float, complex or array of initial values of px
            - y_val: float, complex or array of initial values of y
            - py_val: float, complex or array of initial values of py
            - params: array of numerical parameter values

        Output:
            - float, complex or array of the numerical value of the polynomial
        """
        total = 0.0 + 0j
        for term in self.terms:
            total += (
                term.coeff(params)
                * x_val ** term.x_exp
                * px_val ** term.px_exp
                * y_val ** term.y_exp
                * py_val ** term.py_exp
            )
        return cast(np.ndarray, total)

    def get_max_order(self) -> int:
        """
        Function that returns the maximum order of the polynomial.
        """
        max_order = 0
        for term in self.terms:
            curr_order = term.x_exp + term.px_exp + term.y_exp + term.py_exp
            if curr_order > max_order:
                max_order = curr_order
        return max_order

    @staticmethod
    def product_Coeff_Polynom(
        coeff: Union[float, complex, Callable],
        poly: "ParametricPolynom",
    ) -> "ParametricPolynom":
        """
        Function that evaluates the product of a coefficient (numeric or
        callable) and a parametric polynomial.

        Input:
            - coeff: float, complex or Callable[[params], complex]
            - poly: ParametricPolynom, the polynomial to be multiplied

        Output:
            - ParametricPolynom representing the new polynomial
        """
        if callable(coeff):
            cf = coeff
        else:
            v = coeff
            cf = lambda _, v=v: v

        new_terms = []
        for term in poly.terms:
            tc = term.coeff
            new_terms.append(type(term)(
                coeff=lambda params, cf=cf, tc=tc: cf(params) * tc(params),
                x_exp=term.x_exp,
                px_exp=term.px_exp,
                y_exp=term.y_exp,
                py_exp=term.py_exp,
            ))
        return type(poly)(terms=new_terms)

    @staticmethod
    def sum_Polynoms(
        poly1: "ParametricPolynom",
        poly2: "ParametricPolynom",
    ) -> "ParametricPolynom":
        """
        Function that evaluates the sum of two parametric polynomials.

        Input:
            - poly1: ParametricPolynom, first polynomial in the sum
            - poly2: ParametricPolynom, second polynomial in the sum

        Output:
            - ParametricPolynom representing the sum
        """
        new_terms = copy.deepcopy(poly1.terms) + copy.deepcopy(poly2.terms)
        new_poly = type(poly1)(terms=new_terms)
        new_poly.remove_zero_terms()
        new_poly.collect_terms()
        return new_poly

    @staticmethod
    def product_Polynoms(
        poly1: "ParametricPolynom",
        poly2: "ParametricPolynom",
        max_order: int,
    ) -> "ParametricPolynom":
        """
        Function that evaluates the product of two parametric polynomials
        truncated at a given order.

        Input:
            - poly1: ParametricPolynom, first polynomial in the product
            - poly2: ParametricPolynom, second polynomial in the product
            - max_order: integer, order above which the product is truncated

        Output:
            - ParametricPolynom representing the product
        """
        new_terms = []
        for term1 in poly1.terms:
            for term2 in poly2.terms:
                new_terms.append(ParametricTerm.product_Terms(term1, term2, max_order))
        new_poly = type(poly1)(terms=new_terms)
        new_poly.remove_zero_terms()
        new_poly.collect_terms()
        return new_poly

    @staticmethod
    def power_Polynom(
        poly: "ParametricPolynom",
        pow: int,
        max_order: int,
    ) -> "ParametricPolynom":
        """
        Function that evaluates the power of a parametric polynomial truncated
        at a given order.

        Input:
            - poly: ParametricPolynom, polynomial to be raised to the given power
            - pow: integer, the power to which the polynomial should be raised
            - max_order: integer, order above which the polynomial will be truncated

        Output:
            - ParametricPolynom representing the power
        """
        if pow == 0:
            term_cls = type(poly.terms[0]) if poly.terms else None
            if term_cls is None:
                raise ValueError("Cannot raise an empty polynomial to power 0.")
            return type(poly)(terms=[term_cls(coeff=lambda _: 1.0)])
        elif pow == 1:
            return poly
        else:
            new_poly = ParametricPolynom.product_Polynoms(poly, poly, max_order)
            for _ in range(2, pow):
                new_poly = ParametricPolynom.product_Polynoms(new_poly, poly, max_order)
            return new_poly

    @staticmethod
    def diff_Polynom(poly: "ParametricPolynom", var: str) -> "ParametricPolynom":
        """
        Function that differentiates a parametric polynomial w.r.t. one variable.

        Input:
            - poly: ParametricPolynom, polynomial to differentiate
            - var: name of variable w.r.t. which to differentiate

        Output:
            - ParametricPolynom representing the derivative
        """
        terms = []
        if var == "x":
            for term in poly.terms:
                if term.x_exp > 0:
                    n, c = term.x_exp, term.coeff
                    terms.append(type(term)(
                        coeff=lambda params, c=c, n=n: n * c(params),
                        x_exp=term.x_exp - 1,
                        px_exp=term.px_exp,
                        y_exp=term.y_exp,
                        py_exp=term.py_exp,
                    ))
        elif var == "px":
            for term in poly.terms:
                if term.px_exp > 0:
                    n, c = term.px_exp, term.coeff
                    terms.append(type(term)(
                        coeff=lambda params, c=c, n=n: n * c(params),
                        x_exp=term.x_exp,
                        px_exp=term.px_exp - 1,
                        y_exp=term.y_exp,
                        py_exp=term.py_exp,
                    ))
        elif var == "y":
            for term in poly.terms:
                if term.y_exp > 0:
                    n, c = term.y_exp, term.coeff
                    terms.append(type(term)(
                        coeff=lambda params, c=c, n=n: n * c(params),
                        x_exp=term.x_exp,
                        px_exp=term.px_exp,
                        y_exp=term.y_exp - 1,
                        py_exp=term.py_exp,
                    ))
        elif var == "py":
            for term in poly.terms:
                if term.py_exp > 0:
                    n, c = term.py_exp, term.coeff
                    terms.append(type(term)(
                        coeff=lambda params, c=c, n=n: n * c(params),
                        x_exp=term.x_exp,
                        px_exp=term.px_exp,
                        y_exp=term.y_exp,
                        py_exp=term.py_exp - 1,
                    ))
        else:
            raise ValueError(f"Incorrect variable {var}.")
        return type(poly)(terms=terms)


@dataclass
class ParametricMap:
    """
    Dataclass representing a polynomial map of 4 variables with parametric
    coefficients.

    Attributes:
        - x_poly: ParametricPolynom, polynomial map of first coordinate
        - px_poly: ParametricPolynom, polynomial map of second coordinate
        - y_poly: ParametricPolynom, polynomial map of third coordinate
        - py_poly: ParametricPolynom, polynomial map of fourth coordinate
    """

    x_poly: ParametricPolynom
    px_poly: ParametricPolynom
    y_poly: ParametricPolynom
    py_poly: ParametricPolynom

    def __str__(self) -> str:
        return (
            "x:\n" + str(self.x_poly) + "\n"
            + "px:\n" + str(self.px_poly) + "\n"
            + "y:\n" + str(self.y_poly) + "\n"
            + "py:\n" + str(self.py_poly)
        )

    def __eq__(self, other):
        if not isinstance(other, ParametricMap):
            return ValueError(
                "ParametricMap object can only be compared to another ParametricMap object!"
            )
        return (
            self.x_poly == other.x_poly
            and self.px_poly == other.px_poly
            and self.y_poly == other.y_poly
            and self.py_poly == other.py_poly
        )

    @classmethod
    def from_map(cls, map: Map) -> ParametricMap:
        """Wrap a numeric :class:`Map` as a ParametricMap."""
        return cls(
            x_poly=ParametricPolynom.from_polynom(map.x_poly),
            px_poly=ParametricPolynom.from_polynom(map.px_poly),
            y_poly=ParametricPolynom.from_polynom(map.y_poly),
            py_poly=ParametricPolynom.from_polynom(map.py_poly),
        )

    def substitute(
        self,
        x_val: np.ndarray,
        px_val: np.ndarray,
        y_val: np.ndarray,
        py_val: np.ndarray,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function that substitutes the input for the four variables x, px, y, py
        given numerical params.

        Input:
            - x_val: array-like of initial values of the first coordinate
            - px_val: array-like of initial values of the second coordinate
            - y_val: array-like of initial values of the third coordinate
            - py_val: array-like of initial values of the fourth coordinate
            - params: array of numerical parameter values

        Output:
            - 4-tuple of arrays: mapped coordinates
        """
        return (
            self.x_poly.substitute(x_val, px_val, y_val, py_val, params),
            self.px_poly.substitute(x_val, px_val, y_val, py_val, params),
            self.y_poly.substitute(x_val, px_val, y_val, py_val, params),
            self.py_poly.substitute(x_val, px_val, y_val, py_val, params),
        )

    def get_max_order(self) -> int:
        """
        Function that returns the maximum map order.
        """
        return int(np.max([
            self.x_poly.get_max_order(),
            self.px_poly.get_max_order(),
            self.y_poly.get_max_order(),
            self.py_poly.get_max_order(),
        ]))

    @staticmethod
    def composition_Map(
        map1: ParametricMap,
        map2: ParametricMap,
        max_order: int,
    ) -> ParametricMap:
        """
        Function that evaluates the composition of two parametric maps truncated
        at a given order.

        Input:
            - map1: ParametricMap, inner map in the composition
            - map2: ParametricMap, outer map in the composition
            - max_order: integer, order above which polynomial components are truncated

        Output:
            - ParametricMap representing the composition
        """

        def _compose_poly(out_poly: ParametricPolynom) -> ParametricPolynom:
            result = ParametricPolynom(terms=[])
            for term in out_poly.terms:
                x_sub  = ParametricPolynom.power_Polynom(map1.x_poly,  term.x_exp,  max_order)
                px_sub = ParametricPolynom.power_Polynom(map1.px_poly, term.px_exp, max_order)
                y_sub  = ParametricPolynom.power_Polynom(map1.y_poly,  term.y_exp,  max_order)
                py_sub = ParametricPolynom.power_Polynom(map1.py_poly, term.py_exp, max_order)

                x_px_prod = ParametricPolynom.product_Polynoms(x_sub,     px_sub,   max_order)
                y_py_prod = ParametricPolynom.product_Polynoms(y_sub,     py_sub,   max_order)
                full_prod = ParametricPolynom.product_Polynoms(x_px_prod, y_py_prod, max_order)

                result = ParametricPolynom.sum_Polynoms(
                    result,
                    ParametricPolynom.product_Coeff_Polynom(term.coeff, full_prod),
                )
            result.remove_zero_terms()
            result.collect_terms()
            return result

        return ParametricMap(
            x_poly=_compose_poly(map2.x_poly),
            px_poly=_compose_poly(map2.px_poly),
            y_poly=_compose_poly(map2.y_poly),
            py_poly=_compose_poly(map2.py_poly),
        )


class _ParamPoly:
    """
    Lightweight polynomial-in-params representation built by operator
    overloading during a single probe evaluation of a deep callable chain.
    Only used internally during the map-flattening step.

    _terms maps sorted tuples of param indices to complex scalars, e.g.
    {(): 1.0, (2,): 3.0, (1, 4): -0.5} represents 1 + 3*p[2] - 0.5*p[1]*p[4].
    """

    __slots__ = ('_terms',)

    def __init__(self, terms: dict):
        self._terms = terms

    def _add_dicts(self, a: dict, b: dict) -> dict:
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) + v
        return result

    def __add__(self, other):
        cls = type(self)
        if isinstance(other, cls):
            return cls(self._add_dicts(self._terms, other._terms))
        else:
            d = dict(self._terms)
            d[()] = d.get((), 0) + other
            return cls(d)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        cls = type(self)
        if isinstance(other, cls):
            result: dict = {}
            for k1, v1 in self._terms.items():
                for k2, v2 in other._terms.items():
                    k = tuple(sorted(k1 + k2))
                    result[k] = result.get(k, 0) + v1 * v2
            return cls(result)
        else:
            return cls({k: other * v for k, v in self._terms.items()})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, n: int):
        cls = type(self)
        result = cls({(): 1.0})
        for _ in range(n):
            result = result * self
        return result


def _make_fast_coeff(pp: _ParamPoly, nparams: int) -> Callable:
    """
    Convert a _ParamPoly into a fast numpy-based callable ``f(params) -> scalar``.
    The linear part is evaluated as a dot product; higher-degree monomials
    fall back to a short Python loop (there are typically very few of them).
    """
    c1 = np.zeros(nparams, dtype=complex)
    const = 0.0 + 0j
    higher: list = []

    for k, v in pp._terms.items():
        if len(k) == 0:
            const += v
        elif len(k) == 1:
            c1[k[0]] += v
        else:
            higher.append((k, complex(v)))

    c1_frozen = c1.copy()
    if not higher:
        c0, c = const, c1_frozen
        return lambda params, c0=c0, c=c: c0 + np.dot(c, params)
    else:
        c0, c, ht = const, c1_frozen, higher
        def _eval(params, c0=c0, c=c, ht=ht):
            total = c0 + np.dot(c, params)
            for k, v in ht:
                m = v
                for i in k:
                    m = m * params[i]
                total += m
            return total
        return _eval
