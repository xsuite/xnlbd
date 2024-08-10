from pathlib import Path

import numpy as np
import xobjects as xo  # type: ignore[import-untyped]
from scipy.special import factorial  # type: ignore[import-untyped]
from xtrack.base_element import BeamElement  # type: ignore[import-untyped]

_pkg_root = Path(__file__).parent.absolute()


class Henonmap(BeamElement):
    """Beam element representing a Henon-like map with an arbitrary polynomial kick.

    Parameters
    ----------
    omega_x : float
        Linear angular frequency in the horizontal plane. Default is ``0``.
    omega_y : float
        Linear angular frequency in the horizontal plane. Default is ``0``.
    n_turns : int
        Number of turns to track for. Default is ``1``. In general, tracking for
        multiple turns should be done in Python by wrapping the element in a line
        and providing ``num_turns`` for the ``track`` method, but the option to
        do it on the single element is included here via ``n_turns``, because the
        Henon map does represent an entire ring.
    twiss_params : array of floats
        An array of the form [alpha_x, beta_x, alpha_y, beta_y] used for coordinate
        normalisation and denormalisation. Default is ``None``, where a neutral
        array in the form of [0.0, 1.0, 0.0, 1.0] is passed.
    dqx : float
        A floating point number representing the value of the horizontal chromaticity
                in the ring. Default is ``0``.
    dqy : float
        A floating point number representing the value of the vertical chromaticity
                in the ring. Default is ``0``.
    dx : float
        A floating point number representing the value of horizontal dispersion at
                the location the multipole. Default is ``0``.
    ddx : float
        A floating point value representing the value of the derivative of horizontal
                dispersion at the location of the multipole. Default is ``0``.
    multipole_coeffs : array of floats
        An array of integrated normal multipole strengths in increasing multipole order.
        Integrated normal multipole strength of order n means the value of the nth
        derivative of the vertical magnetic field w.r.t x multiplied by the element length
        over the beam rigidity. The map only handles normal multipoles, not skew ones.
        Default is None, where a neutral array of [0.0] (i.e., no multipoles) is passed.
    norm : bool
        ``True`` if input coordinates are already normalised, ``False`` if not. Default
        is ``False``.

    Comments
    --------
    The properties of the object accessible after initialization are the following:
    sin_omega_x : float
        Sine of linear angular frequency in the horizontal plane.
    cos_omega_x : float
        Cosine of linear angular frequency in the horizontal plane.
    sin_omega_y: float
        Sine of linear angular frequency in the vertical plane.
    cos_omega_y: float
        Cosine of linear angular frequency in the vertical plane.
    twiss_params: array of floats
        An array of the form [alpha_x, beta_x, alpha_y, beta_y] used for coordinate
        normalisation and denormalisation.
    domegax: float
        A floating point number representing the value of the horizontal chromaticity
        in the ring multiplied by 2pi.
    domegay: float
        A floating point number representing the value of the vertical chromaticity
        in the ring multiplied by 2pi.
    dx: float
        A floating point number representing the value of horizontal dispersion at
        the location the multipole.
    ddx: float
        A floating point value representing the value of the derivative of horizontal
        dispersion at the location of the multipole.
    fx_coeffs: array of floats
        An array that contains the coefficients of monomials of the form x^n*y*m that
        represent the nonlinearities of the map in the horizontal plane. It is
        calculated at initialisation based on the multipole coefficients provided.
    fx_x_exps: array of floats
        An array containing the exponents, n, of x for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the horizontal plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    fx_y_exps: array of floats
        An array containing the exponents, m, of y for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the horizontal plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    fy_coeffs: array of floats
        An array that contains the coefficients of monomials of the form x^n*y*m that
        represent the nonlinearities of the map in the vertical plane. It is
        calculated at initialisation based on the multipole coefficients provided.
    fy_x_exps: array of floats
        An array containing the exponents, n, of x for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the vertical plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    fy_y_exps: array of floats
        An array containing the exponents, m, of y for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the vertical plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    n_fx_coeffs: int
        Length of the arrays fx_coeffs, fx_x_exps, and fx_y_exps.
    n_fy_coeffs: int
        Length of the arrays fy_coeffs, fy_x_exps, and fy_y_exps.
    n_turns: int
        Number of turns to track.
    norm: int
        1 if input coordinates are already normalised, 0 if not.

    Note
    ----
    - If the user wants to change the tune on a turn-by-turn basis in their
        simulation, they must do so by providing directly the sine and cosine of the
        new tune. This is done to speed up the code by avoiding unnecessary trig
        function evaluations when they are not absolutely needed and allow for fast
        long-term tracking.
    - The same applies for the multipole coefficients.

    """

    _xofields = {
        "sin_omega_x": xo.Float64,
        "cos_omega_x": xo.Float64,
        "sin_omega_y": xo.Float64,
        "cos_omega_y": xo.Float64,
        "n_turns": xo.Int64,
        "twiss_params": xo.Float64[:],
        "domegax": xo.Float64,
        "domegay": xo.Float64,
        "dx": xo.Float64,
        "ddx": xo.Float64,
        "fx_coeffs": xo.Float64[:],
        "fx_x_exps": xo.Int64[:],
        "fx_y_exps": xo.Int64[:],
        "fy_coeffs": xo.Float64[:],
        "fy_x_exps": xo.Int64[:],
        "fy_y_exps": xo.Int64[:],
        "n_fx_coeffs": xo.Int64,
        "n_fy_coeffs": xo.Int64,
        "norm": xo.Int64,
    }

    isthick = False
    behaves_like_drift = False

    _extra_c_sources = [_pkg_root.joinpath("elements_src/henonmap.h")]

    def __init__(
        self,
        omega_x=0.0,
        omega_y=0.0,
        n_turns=1,
        twiss_params=None,
        dqx=0,
        dqy=0,
        dx=0,
        ddx=0,
        multipole_coeffs=None,
        norm=False,
        **kwargs,
    ):
        if twiss_params is None:
            twiss_params = [0.0, 1.0, 0.0, 1.0]

        if multipole_coeffs is None:
            multipole_coeffs = [0.0]

        if "_xobject" not in kwargs:
            kwargs.setdefault("sin_omega_x", np.sin(omega_x))
            kwargs.setdefault("cos_omega_x", np.cos(omega_x))
            kwargs.setdefault("sin_omega_y", np.sin(omega_y))
            kwargs.setdefault("cos_omega_y", np.cos(omega_y))
            kwargs.setdefault("n_turns", n_turns)
            kwargs.setdefault("twiss_params", twiss_params)
            kwargs.setdefault("domegax", 2 * np.pi * dqx)
            kwargs.setdefault("domegay", 2 * np.pi * dqy)
            kwargs.setdefault("dx", dx)
            kwargs.setdefault("ddx", ddx)

            fx_coeffs = []
            fx_x_exps = []
            fx_y_exps = []
            fy_coeffs = []
            fy_x_exps = []
            fy_y_exps = []
            for n in range(2, len(multipole_coeffs) + 2):
                for k in range(0, n + 1):
                    if (k % 4) == 0:
                        fx_coeffs.append(
                            multipole_coeffs[n - 2] / factorial(k) / factorial(n - k)
                        )
                        fx_x_exps.append(n - k)
                        fx_y_exps.append(k)
                    elif (k % 4) == 2:
                        fx_coeffs.append(
                            -1
                            * multipole_coeffs[n - 2]
                            / factorial(k)
                            / factorial(n - k)
                        )
                        fx_x_exps.append(n - k)
                        fx_y_exps.append(k)
                    elif (k % 4) == 1:
                        fy_coeffs.append(
                            -1
                            * multipole_coeffs[n - 2]
                            / factorial(k)
                            / factorial(n - k)
                        )
                        fy_x_exps.append(n - k)
                        fy_y_exps.append(k)
                    else:
                        fy_coeffs.append(
                            multipole_coeffs[n - 2] / factorial(k) / factorial(n - k)
                        )
                        fy_x_exps.append(n - k)
                        fy_y_exps.append(k)
            kwargs.setdefault("fx_coeffs", fx_coeffs)
            kwargs.setdefault("fx_x_exps", fx_x_exps)
            kwargs.setdefault("fx_y_exps", fx_y_exps)
            kwargs.setdefault("fy_coeffs", fy_coeffs)
            kwargs.setdefault("fy_x_exps", fy_x_exps)
            kwargs.setdefault("fy_y_exps", fy_y_exps)
            kwargs.setdefault("n_fx_coeffs", len(fx_coeffs))
            kwargs.setdefault("n_fy_coeffs", len(fy_coeffs))

            if norm == True:
                kwargs.setdefault("norm", 1)
            else:
                kwargs.setdefault("norm", 0)

        super().__init__(**kwargs)

    has_backtrack = True


class ModulatedHenonmap(BeamElement):
    """Beam element representing a modulated Henon-like map with an arbitrary
    polynomial kick.

    Parameters
    ----------
    omega_x : array of floats
        List of linear angular frequencies in the horizontal plane. Default is a
        ``np.zeros(100)``.
    omega_y : array of floats
        List of linear angular frequencies in the horizontal plane. Default is a
        ``np.zeros(100)``.
    twiss_params : array of floats
        An array of the form [alpha_x, beta_x, alpha_y, beta_y] used for coordinate
        normalisation and denormalisation. Default is ``None``, where a neutral
        array in the form of [0.0, 1.0, 0.0, 1.0] is passed.
    dqx : float
        A floating point number representing the value of the horizontal chromaticity
                in the ring. Default is ``0``.
    dqy : float
        A floating point number representing the value of the vertical chromaticity
                in the ring. Default is ``0``.
    dx : float
        A floating point number representing the value of horizontal dispersion at
                the location the multipole. Default is ``0``.
    ddx : float
        A floating point value representing the value of the derivative of horizontal
                dispersion at the location of the multipole. Default is ``0``.
    multipole_coeffs : array of floats
        An array of integrated normal multipole strengths in increasing multipole order.
        Integrated normal multipole strength of order n means the value of the nth
        derivative of the vertical magnetic field w.r.t x multiplied by the element length
        over the beam rigidity. The map only handles normal multipoles, not skew ones.
        Default is None, where a neutral list of ``[[0.0]] * 100`` (i.e., no multipoles)
        is passed.
    norm : bool
        ``True`` if input coordinates are already normalised, ``False`` if not. Default
        is ``False``.

    Comments
    --------
    The properties of the object accessible after initialization are the following:
    sin_omega_x : array of floats
        Sine of linear angular frequencies in the horizontal plane.
    cos_omega_x : array of floats
        Cosine of linear angular frequencies in the horizontal plane.
    sin_omega_y: array of floats
        Sine of linear angular frequencies in the vertical plane.
    cos_omega_y: array of floats
        Cosine of linear angular frequencies in the vertical plane.
    twiss_params: array of floats
        An array of the form [alpha_x, beta_x, alpha_y, beta_y] used for coordinate
        normalisation and denormalisation.
    domegax: float
        A floating point number representing the value of the horizontal chromaticity
        in the ring multiplied by 2pi.
    domegay: float
        A floating point number representing the value of the vertical chromaticity
        in the ring multiplied by 2pi.
    dx: float
        A floating point number representing the value of horizontal dispersion at
        the location the multipole.
    ddx: float
        A floating point value representing the value of the derivative of horizontal
        dispersion at the location of the multipole.
    fx_coeffs: array of floats
        An array that contains the coefficients of monomials of the form x^n*y*m that
        represent the nonlinearities of the map in the horizontal plane. It is
        calculated at initialisation based on the multipole coefficients provided.
    fx_x_exps: array of floats
        An array containing the exponents, n, of x for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the horizontal plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    fx_y_exps: array of floats
        An array containing the exponents, m, of y for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the horizontal plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    fy_coeffs: array of floats
        An array that contains the coefficients of monomials of the form x^n*y*m that
        represent the nonlinearities of the map in the vertical plane. It is
        calculated at initialisation based on the multipole coefficients provided.
    fy_x_exps: array of floats
        An array containing the exponents, n, of x for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the vertical plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    fy_y_exps: array of floats
        An array containing the exponents, m, of y for all monomials of the form
        x^n*y^m that represent the nonlinearities of the map in the vertical plane.
        It is calculated at initialisation based on the multipole coefficients provided.
    n_fx_coeffs: int
        Length of the arrays fx_coeffs, fx_x_exps, and fx_y_exps.
    n_fy_coeffs: int
        Length of the arrays fy_coeffs, fy_x_exps, and fy_y_exps.
    n_turns: int
        Number of tunes available in the tune arrays, if a particle is tracked
        above that number, a mod operator is used to roll-over the array. If the
        particle has negative at_turn value, the array is rolled backwards.
    n_par_multipoles: int
        Number of different multipole strengths provided. It is evaluated
        automatically from the length of the multipole_coeffs array provided.
        It is used then at runtime to apply the same loop of strenghts to the
        according turns via modulo operator. If only a single multipole strength
        is provided, this value is 1, and the same strength is applied to all
        turns.
    norm: int
        1 if input coordinates are already normalised, 0 if not.

    Note
    ----
    - If the user wants to change the tune on a turn-by-turn basis in their
        simulation, they must do so by providing directly the sine and cosine of the
        new tune. This is done to speed up the code by avoiding unnecessary trig
        function evaluations when they are not absolutely needed and allow for fast
        long-term tracking.
    - The same applies for the multipole coefficients.

    """

    _xofields = {
        "sin_omega_x": xo.Float64[:],
        "cos_omega_x": xo.Float64[:],
        "sin_omega_y": xo.Float64[:],
        "cos_omega_y": xo.Float64[:],
        "n_turns": xo.Int64,
        "n_par_multipoles": xo.Int64,
        "twiss_params": xo.Float64[:],
        "domegax": xo.Float64,
        "domegay": xo.Float64,
        "dx": xo.Float64,
        "ddx": xo.Float64,
        "fx_coeffs": xo.Float64[:],
        "fx_x_exps": xo.Int64[:],
        "fx_y_exps": xo.Int64[:],
        "fy_coeffs": xo.Float64[:],
        "fy_x_exps": xo.Int64[:],
        "fy_y_exps": xo.Int64[:],
        "n_fx_coeffs": xo.Int64,
        "n_fy_coeffs": xo.Int64,
        "norm": xo.Int64,
    }

    isthick = False
    behaves_like_drift = False

    _extra_c_sources = [_pkg_root.joinpath("elements_src/modulatedhenonmap.h")]

    def __init__(
        self,
        omega_x=None,
        omega_y=None,
        twiss_params=None,
        dqx=0,
        dqy=0,
        dx=0,
        ddx=0,
        multipole_coeffs=None,
        norm=False,
        **kwargs,
    ):
        if omega_x is None:
            omega_x = np.zeros(100)
        if omega_y is None:
            omega_y = np.zeros(100)
        if len(omega_x) != len(omega_y):
            raise ValueError("omega_x and omega_y must have the same length.")

        if twiss_params is None:
            twiss_params = [0.0, 1.0, 0.0, 1.0]

        if multipole_coeffs is None:
            multipole_coeffs = [[0.0]] * 100

        if "_xobject" not in kwargs:
            kwargs.setdefault("sin_omega_x", np.sin(omega_x))
            kwargs.setdefault("cos_omega_x", np.cos(omega_x))
            kwargs.setdefault("sin_omega_y", np.sin(omega_y))
            kwargs.setdefault("cos_omega_y", np.cos(omega_y))
            kwargs.setdefault("n_turns", len(omega_x))
            kwargs.setdefault("n_par_multipoles", len(multipole_coeffs))
            kwargs.setdefault("twiss_params", twiss_params)
            kwargs.setdefault("domegax", 2 * np.pi * dqx)
            kwargs.setdefault("domegay", 2 * np.pi * dqy)
            kwargs.setdefault("dx", dx)
            kwargs.setdefault("ddx", ddx)

            fx_coeffs = []
            fx_x_exps = []
            fx_y_exps = []
            fy_coeffs = []
            fy_x_exps = []
            fy_y_exps = []
            for i, curr_multipole_coeffs in enumerate(multipole_coeffs):
                for n in range(2, len(curr_multipole_coeffs) + 2):
                    for k in range(0, n + 1):
                        if (k % 4) == 0:
                            fx_coeffs.append(
                                curr_multipole_coeffs[n - 2]
                                / factorial(k)
                                / factorial(n - k)
                            )
                            if i == 0:
                                fx_x_exps.append(n - k)
                                fx_y_exps.append(k)
                        elif (k % 4) == 2:
                            fx_coeffs.append(
                                -1
                                * curr_multipole_coeffs[n - 2]
                                / factorial(k)
                                / factorial(n - k)
                            )
                            if i == 0:
                                fx_x_exps.append(n - k)
                                fx_y_exps.append(k)
                        elif (k % 4) == 1:
                            fy_coeffs.append(
                                -1
                                * curr_multipole_coeffs[n - 2]
                                / factorial(k)
                                / factorial(n - k)
                            )
                            if i == 0:
                                fy_x_exps.append(n - k)
                                fy_y_exps.append(k)
                        else:
                            fy_coeffs.append(
                                curr_multipole_coeffs[n - 2]
                                / factorial(k)
                                / factorial(n - k)
                            )
                            if i == 0:
                                fy_x_exps.append(n - k)
                                fy_y_exps.append(k)
                if i == 0:
                    kwargs.setdefault("n_fx_coeffs", len(fx_coeffs))
                    kwargs.setdefault("n_fy_coeffs", len(fy_coeffs))
            kwargs.setdefault("fx_coeffs", fx_coeffs)
            kwargs.setdefault("fx_x_exps", fx_x_exps)
            kwargs.setdefault("fx_y_exps", fx_y_exps)
            kwargs.setdefault("fy_coeffs", fy_coeffs)
            kwargs.setdefault("fy_x_exps", fy_x_exps)
            kwargs.setdefault("fy_y_exps", fy_y_exps)

            if norm == True:
                kwargs.setdefault("norm", 1)
            else:
                kwargs.setdefault("norm", 0)

        super().__init__(**kwargs)

    has_backtrack = True
