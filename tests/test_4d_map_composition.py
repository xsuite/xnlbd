import numpy as np
import sympy as sy  # type: ignore[import-untyped]
import xpart as xp  # type: ignore[import-untyped, import-not-found]
import xtrack as xt  # type: ignore[import-untyped, import-not-found]

from xnlbd.analyse.normal_forms import *
from xnlbd.track import Henonmap


def truncate_polynomial(expr, max_order):
    expr = expr.expand()

    terms = expr.as_coefficients_dict()
    truncated_terms = []

    for term, coeff in terms.items():
        exponents = term.as_powers_dict()
        total_degree = sum(exponents.values())

        if total_degree <= max_order:
            truncated_terms.append(coeff * term)

    result = 0
    for term in truncated_terms:
        result += term

    return result


def sympy_combination(x, px, y, py, map4d, max_order):
    x1 = (
        map4d.ele_map.x_poly.terms[0].coeff
        * x ** (map4d.ele_map.x_poly.terms[0].x_exp)
        * px ** (map4d.ele_map.x_poly.terms[0].px_exp)
        * y ** (map4d.ele_map.x_poly.terms[0].y_exp)
        * py ** (map4d.ele_map.x_poly.terms[0].py_exp)
    )
    for term in map4d.ele_map.x_poly.terms[1:]:
        x1 += (
            term.coeff
            * x**term.x_exp
            * px**term.px_exp
            * y**term.y_exp
            * py**term.py_exp
        )
    x1 = truncate_polynomial(x1, max_order)
    px1 = (
        map4d.ele_map.px_poly.terms[0].coeff
        * x ** (map4d.ele_map.px_poly.terms[0].x_exp)
        * px ** (map4d.ele_map.px_poly.terms[0].px_exp)
        * y ** (map4d.ele_map.px_poly.terms[0].y_exp)
        * py ** (map4d.ele_map.px_poly.terms[0].py_exp)
    )
    for term in map4d.ele_map.px_poly.terms[1:]:
        px1 += (
            term.coeff
            * x**term.x_exp
            * px**term.px_exp
            * y**term.y_exp
            * py**term.py_exp
        )
    px1 = truncate_polynomial(px1, max_order)
    y1 = (
        map4d.ele_map.y_poly.terms[0].coeff
        * x ** (map4d.ele_map.y_poly.terms[0].x_exp)
        * px ** (map4d.ele_map.y_poly.terms[0].px_exp)
        * y ** (map4d.ele_map.y_poly.terms[0].y_exp)
        * py ** (map4d.ele_map.y_poly.terms[0].py_exp)
    )
    for term in map4d.ele_map.y_poly.terms[1:]:
        y1 += (
            term.coeff
            * x**term.x_exp
            * px**term.px_exp
            * y**term.y_exp
            * py**term.py_exp
        )
    y1 = truncate_polynomial(y1, max_order)
    py1 = (
        map4d.ele_map.py_poly.terms[0].coeff
        * x ** (map4d.ele_map.py_poly.terms[0].x_exp)
        * px ** (map4d.ele_map.py_poly.terms[0].px_exp)
        * y ** (map4d.ele_map.py_poly.terms[0].y_exp)
        * py ** (map4d.ele_map.py_poly.terms[0].py_exp)
    )
    for term in map4d.ele_map.py_poly.terms[1:]:
        py1 += (
            term.coeff
            * x**term.x_exp
            * px**term.px_exp
            * y**term.y_exp
            * py**term.py_exp
        )
    py1 = truncate_polynomial(py1, max_order)

    return x1, px1, y1, py1


def test_4d_map_composition():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=0.0, p0c=400e9, mass0=xp.PROTON_MASS_EV
    )

    alpha_x1 = 1.0
    beta_x1 = 100.0
    alpha_y1 = -2.0
    beta_y1 = 25.0
    alpha_x2 = 1.0
    beta_x2 = 100.0
    alpha_y2 = -2.0
    beta_y2 = 25.0
    alpha_x3 = 1.0
    beta_x3 = 100.0
    alpha_y3 = -2.0
    beta_y3 = 25.0
    Qx_res = 1.0 / 3.0
    Qy_res = 0.13
    Qx1 = 0.11
    Qy1 = 0.04
    Qx2 = 0.13
    Qy2 = 0.03
    Qx3 = 0.10
    Qy3 = 0.06
    nemitt_x = 1e-6
    nemitt_y = 1e-6

    rotation1 = Henonmap(
        omega_x=2 * np.pi * Qx1,
        omega_y=2 * np.pi * Qy1,
        twiss_params=[alpha_x1, beta_x1, alpha_y1, beta_y1],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[0.0, 0.0],
        norm=False,
    )
    sextupole = xt.Multipole(
        knl=[0.0, 0.0, 0.3, 0.0],
        ksl=[0.0, 0.0, 0.0, 0.0],
        hxl=0,
        length=0.74,
    )
    rotation2 = Henonmap(
        omega_x=2 * np.pi * Qx2,
        omega_y=2 * np.pi * Qy2,
        twiss_params=[alpha_x2, beta_x2, alpha_y2, beta_y2],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[0.0, 0.0],
        norm=False,
    )
    octupole1 = xt.Multipole(
        knl=[0.0, 0.0, 0.0, -50.0],
        ksl=[0.0, 0.0, 0.0, 0.0],
        hxl=0,
        length=0.705,
    )
    rotation3 = Henonmap(
        omega_x=2 * np.pi * Qx3,
        omega_y=2 * np.pi * Qy3,
        twiss_params=[alpha_x3, beta_x3, alpha_y3, beta_y3],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[0.0, 0.0],
        norm=False,
    )
    octupole2 = xt.Multipole(
        knl=[0.0, 0.0, 0.0, -50.0],
        ksl=[0.0, 0.0, 0.0, 0.0],
        hxl=0,
        length=0.705,
    )

    all_elements = [rotation1, sextupole, rotation2, octupole1, rotation3, octupole2]
    line = xt.Line(
        elements=[rotation1, sextupole, rotation2, octupole1, rotation3, octupole2],
        element_names=["rot1", "sext", "rot2", "oct1", "rot3", "oct2"],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=4e11)
    line.twiss_default["method"] = "4d"
    line.config.XTRACK_GLOBAL_XY_LIMIT = 0.1
    line.build_tracker()

    poly_line = PolyLine4D(line, line.particle_ref, 3, 4, nemitt_x, nemitt_y)
    poly_line.calculate_one_turn_map()
    final_map = poly_line.one_turn_map_real

    x = sy.Symbol("x")
    px = sy.Symbol("p_x")
    y = sy.Symbol("y")
    py = sy.Symbol("p_y")

    curr_x, curr_px, curr_y, curr_py = sympy_combination(
        x, px, y, py, poly_line.poly_elements[0], 4
    )
    for i in range(1, len(poly_line.poly_elements)):
        curr_x, curr_px, curr_y, curr_py = sympy_combination(
            curr_x, curr_px, curr_y, curr_py, poly_line.poly_elements[i], 4
        )
    curr_x = curr_x.expand().simplify().as_coefficients_dict()
    curr_px = curr_px.expand().simplify().as_coefficients_dict()
    curr_y = curr_y.expand().simplify().as_coefficients_dict()
    curr_py = curr_py.expand().simplify().as_coefficients_dict()

    test_x_poly = Polynom(terms=[])
    for term, coeff in curr_x.items():
        exponents = term.as_powers_dict()
        test_x_poly.terms.append(
            Term(
                coeff=coeff,
                x_exp=int(exponents.get(x, 0)),
                px_exp=int(exponents.get(px, 0)),
                y_exp=int(exponents.get(y, 0)),
                py_exp=int(exponents.get(py, 0)),
            )
        )
    test_px_poly = Polynom(terms=[])
    for term, coeff in curr_px.items():
        exponents = term.as_powers_dict()
        test_px_poly.terms.append(
            Term(
                coeff=coeff,
                x_exp=int(exponents.get(x, 0)),
                px_exp=int(exponents.get(px, 0)),
                y_exp=int(exponents.get(y, 0)),
                py_exp=int(exponents.get(py, 0)),
            )
        )
    test_y_poly = Polynom(terms=[])
    for term, coeff in curr_y.items():
        exponents = term.as_powers_dict()
        test_y_poly.terms.append(
            Term(
                coeff=coeff,
                x_exp=int(exponents.get(x, 0)),
                px_exp=int(exponents.get(px, 0)),
                y_exp=int(exponents.get(y, 0)),
                py_exp=int(exponents.get(py, 0)),
            )
        )
    test_py_poly = Polynom(terms=[])
    for term, coeff in curr_py.items():
        exponents = term.as_powers_dict()
        test_py_poly.terms.append(
            Term(
                coeff=coeff,
                x_exp=int(exponents.get(x, 0)),
                px_exp=int(exponents.get(px, 0)),
                y_exp=int(exponents.get(y, 0)),
                py_exp=int(exponents.get(py, 0)),
            )
        )

    test_map = Map(
        x_poly=test_x_poly,
        px_poly=test_px_poly,
        y_poly=test_y_poly,
        py_poly=test_py_poly,
    )
    for term in test_map.x_poly.terms:
        term.coeff = complex(term.coeff, 0)
    for term in test_map.px_poly.terms:
        term.coeff = complex(term.coeff, 0)
    for term in test_map.y_poly.terms:
        term.coeff = complex(term.coeff, 0)
    for term in test_map.py_poly.terms:
        term.coeff = complex(term.coeff, 0)
    test_map.x_poly.remove_zero_terms()
    test_map.x_poly.collect_terms()
    test_map.px_poly.remove_zero_terms()
    test_map.px_poly.collect_terms()
    test_map.y_poly.remove_zero_terms()
    test_map.y_poly.collect_terms()
    test_map.py_poly.remove_zero_terms()
    test_map.py_poly.collect_terms()

    assert final_map == test_map
