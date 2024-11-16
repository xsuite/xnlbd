import copy
import pathlib

import numpy as np
import xpart as xp  # type: ignore
import xtrack as xt  # type: ignore

from xnlbd.track import Henonmap
from xnlbd.visualise.fixed_points import FPFinder

test_data_folder = pathlib.Path(__file__).parent.joinpath("test_data").absolute()


def test_2D_SPS_no_co_on_momentum():
    """
    Set lattice up
    """
    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tune and chromaticity
    line.vv["qh_setvalue"] = 26.332
    line.vv["qv_setvalue"] = 26.131061135802028
    line.vv["qph_setvalue"] = 0.4192872718150393
    line.vv["qpv_setvalue"] = 0.07550321565116734

    # Turn on strong sextupoles to excite 3rd order resonance
    line.vv["klse40602"] = 0.16642224665
    line.vv["klse52402"] = -0.134713395

    # Turn on octupoles to ensure stable islands are formed
    line.vv["klof"] = -6.0

    # Turn off RF
    line.vv["v200"] = 0.0

    # Turn a large orbit bump in one of the extraction regions
    line.vv["kmpsh41402"] = 0.00039440067863769437
    line.vv["kmplh41658"] = 0.00045922095845457876
    line.vv["kmplh41994"] = 0.00044190219049600475
    line.vv["kmpsh42198"] = -0.0002771794691164835

    # Cycle line so that the horizontal tune kicker is the first element
    line.cycle(name_first_element="mkqh.11653", inplace=True)

    """
    Find stable and unstable fixed points
    """
    ufp_limits = [[2.0, 12.0], [4.0, 12.0]]
    sfp_limits = [[-18.0, -6.0], [-16.0, -6.0]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, verbose=1)
    ufp, _ = fp.find_fp(ufp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)
    ufp_copy = copy.deepcopy(ufp)
    sfp, _ = fp.find_fp(sfp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)
    sfp_copy = copy.deepcopy(sfp)

    """
    Track and compare to itself
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=3,
        freeze_longitudinal=True,
    )
    assert np.isclose(ufp_copy["x"], particle_at_ufp.x, rtol=1e-13)
    assert np.isclose(ufp_copy["px"], particle_at_ufp.px, rtol=1e-13)
    assert np.isclose(ufp_copy["y"], particle_at_ufp.y, rtol=1e-13)
    assert np.isclose(ufp_copy["py"], particle_at_ufp.py, rtol=1e-13)
    assert np.isclose(ufp_copy["zeta"], particle_at_ufp.zeta, rtol=1e-13)
    assert np.isclose(
        ufp_copy["pzeta"], particle_at_ufp.ptau / line.particle_ref.beta0, rtol=1e-13
    )

    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=3,
        freeze_longitudinal=True,
    )
    assert np.isclose(sfp_copy["x"], particle_at_sfp.x, rtol=1e-13)
    assert np.isclose(sfp_copy["px"], particle_at_sfp.px, rtol=1e-13)
    assert np.isclose(sfp_copy["y"], particle_at_sfp.y, rtol=1e-13)
    assert np.isclose(sfp_copy["py"], particle_at_sfp.py, rtol=1e-13)
    assert np.isclose(sfp_copy["zeta"], particle_at_sfp.zeta, rtol=1e-13)
    assert np.isclose(
        sfp_copy["pzeta"], particle_at_sfp.ptau / line.particle_ref.beta0, rtol=1e-13
    )


def test_2D_SPS_no_co_off_momentum():
    """
    Set lattice up
    """
    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tune and chromaticity
    line.vv["qh_setvalue"] = 26.332
    line.vv["qv_setvalue"] = 26.131061135802028
    line.vv["qph_setvalue"] = 0.4192872718150393
    line.vv["qpv_setvalue"] = 0.07550321565116734

    # Turn on strong sextupoles to excite 3rd order resonance
    line.vv["klse40602"] = 0.3328444933
    line.vv["klse52402"] = -0.26942679

    # Turn on octupoles to ensure stable islands are formed
    line.vv["klof"] = -6.0

    # Turn off RF
    line.vv["v200"] = 0.0

    # Turn a large orbit bump in one of the extraction regions
    line.vv["kmpsh41402"] = 0.00039440067863769437
    line.vv["kmplh41658"] = 0.00045922095845457876
    line.vv["kmplh41994"] = 0.00044190219049600475
    line.vv["kmpsh42198"] = -0.0002771794691164835

    # Cycle line so that the horizontal tune kicker is the first element
    line.cycle(name_first_element="mkqh.11653", inplace=True)

    """
    Find stable and unstable fixed points
    """
    ufp_limits = [[-2.0, -0.5], [2.0, 4.5]]
    sfp_limits = [[-6.0, -2.5], [-5.5, -3.1]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, verbose=1)
    ufp, _ = fp.find_fp(
        ufp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, delta0=-0.00112
    )
    ufp_copy = copy.deepcopy(ufp)
    sfp, _ = fp.find_fp(
        sfp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, delta0=-0.00112
    )
    sfp_copy = copy.deepcopy(sfp)

    """
    Track and compare to itself
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=3,
        freeze_longitudinal=True,
    )
    assert np.isclose(ufp_copy["x"], particle_at_ufp.x, rtol=1e-13)
    assert np.isclose(ufp_copy["px"], particle_at_ufp.px, rtol=1e-13)
    assert np.isclose(ufp_copy["y"], particle_at_ufp.y, rtol=1e-13)
    assert np.isclose(ufp_copy["py"], particle_at_ufp.py, rtol=1e-13)
    assert np.isclose(ufp_copy["zeta"], particle_at_ufp.zeta, rtol=1e-13)
    assert np.isclose(
        ufp_copy["pzeta"], particle_at_ufp.ptau / line.particle_ref.beta0, rtol=1e-13
    )

    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=3,
        freeze_longitudinal=True,
    )
    assert np.isclose(sfp_copy["x"], particle_at_sfp.x, rtol=1e-13)
    assert np.isclose(sfp_copy["px"], particle_at_sfp.px, rtol=1e-13)
    assert np.isclose(sfp_copy["y"], particle_at_sfp.y, rtol=1e-13)
    assert np.isclose(sfp_copy["py"], particle_at_sfp.py, rtol=1e-13)
    assert np.isclose(sfp_copy["zeta"], particle_at_sfp.zeta, rtol=1e-13)
    assert np.isclose(
        sfp_copy["pzeta"], particle_at_sfp.ptau / line.particle_ref.beta0, rtol=1e-13
    )


def test_2D_SPS_co_on_momentum():
    """
    Set lattice up
    """

    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tune and chromaticity
    line.vv["qh_setvalue"] = 26.332
    line.vv["qv_setvalue"] = 26.131061135802028
    line.vv["qph_setvalue"] = 0.4192872718150393
    line.vv["qpv_setvalue"] = 0.07550321565116734

    # Turn on strong sextupoles to excite 3rd order resonance
    line.vv["klse40602"] = 0.16642224665 * 2
    line.vv["klse52402"] = -0.134713395 * 2

    # Turn on octupoles to ensure stable islands are formed
    line.vv["klof"] = -6.0

    # Turn off RF
    line.vv["v200"] = 0.0

    # Turn a large orbit bump in one of the extraction regions
    line.vv["kmpsh41402"] = 0.00039440067863769437
    line.vv["kmplh41658"] = 0.00045922095845457876
    line.vv["kmplh41994"] = 0.00044190219049600475
    line.vv["kmpsh42198"] = -0.0002771794691164835

    # Twiss and obtain closed orbit guess in bump
    twiss = line.twiss(continue_on_closed_orbit_error=False)
    co_guess = {
        "x": twiss.x[twiss.name == "teca.41777"],
        "px": twiss.px[twiss.name == "teca.41777"],
        "y": twiss.y[twiss.name == "teca.41777"],
        "py": twiss.py[twiss.name == "teca.41777"],
        "zeta": twiss.zeta[twiss.name == "teca.41777"],
        "ptau": twiss.ptau[twiss.name == "teca.41777"],
    }

    # Cycle line so that the horizontal tune kicker is the first element
    line.cycle(name_first_element="teca.41777", inplace=True)

    """
    Find stable and unstable fixed points
    """
    ufp_limits = [[4.0, 14.0], [2.0, 10.0]]
    sfp_limits = [[-18.0, -12.0], [-12.0, -8.0]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, co_guess=co_guess, verbose=1)
    ufp, _ = fp.find_fp(
        ufp_limits, co_guess=co_guess, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1
    )
    ufp_copy = copy.deepcopy(ufp)
    sfp, _ = fp.find_fp(
        sfp_limits, co_guess=co_guess, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1
    )
    sfp_copy = copy.deepcopy(sfp)

    """
    Track and compare to itself
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=3,
        freeze_longitudinal=True,
    )
    assert np.isclose(ufp_copy["x"], particle_at_ufp.x, rtol=1e-13)
    assert np.isclose(ufp_copy["px"], particle_at_ufp.px, rtol=1e-13)
    assert np.isclose(ufp_copy["y"], particle_at_ufp.y, rtol=1e-13)
    assert np.isclose(ufp_copy["py"], particle_at_ufp.py, rtol=1e-13)
    assert np.isclose(ufp_copy["zeta"], particle_at_ufp.zeta, rtol=1e-13)
    assert np.isclose(
        ufp_copy["pzeta"], particle_at_ufp.ptau / line.particle_ref.beta0, rtol=1e-13
    )

    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=3,
        freeze_longitudinal=True,
    )
    assert np.isclose(sfp_copy["x"], particle_at_sfp.x, rtol=1e-13)
    assert np.isclose(sfp_copy["px"], particle_at_sfp.px, rtol=1e-13)
    assert np.isclose(sfp_copy["y"], particle_at_sfp.y, rtol=1e-13)
    assert np.isclose(sfp_copy["py"], particle_at_sfp.py, rtol=1e-13)
    assert np.isclose(sfp_copy["zeta"], particle_at_sfp.zeta, rtol=1e-13)
    assert np.isclose(
        sfp_copy["pzeta"], particle_at_sfp.ptau / line.particle_ref.beta0, rtol=1e-13
    )


def test_2D_Henon():
    """
    Set up line with Henon map element
    """
    alpha_x = 0.0
    beta_x = 100.0
    alpha_y = 0.0
    beta_y = 1.0  # not very realistic, but reduces coupling and makes it more 2D-like
    Qx = 0.211
    Qy = 0.12
    K2 = -1.0

    henon = Henonmap(
        omega_x=2 * np.pi * Qx,
        omega_y=2 * np.pi * Qy,
        twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[K2],
        norm=False,
    )
    drift = xt.Drift(length=0.0)  # necessary for aperture check

    line = xt.Line(elements=[henon, drift], element_names=["henon", "zero_len_drift"])
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=4e11)
    line.twiss_default["method"] = "4d"
    line.config.XTRACK_GLOBAL_XY_LIMIT = 0.1
    line.build_tracker()

    """
    Find stable and unstable fixed points
    """
    ufp_limits = [[8.0, 22.0], [11.0, 21.0]]
    sfp_limits = [[-28.0, -14.0], [-12.0, 2.0]]

    fp = FPFinder(line, order=5, planes="H", tol=1e-13, verbose=1)
    ufp, _ = fp.find_fp(ufp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)
    ufp_copy = copy.deepcopy(ufp)
    sfp, _ = fp.find_fp(sfp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)
    sfp_copy = copy.deepcopy(sfp)

    """
    Track and compare to itself
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=5,
        freeze_longitudinal=True,
    )
    assert np.isclose(ufp_copy["x"], particle_at_ufp.x, rtol=1e-13)
    assert np.isclose(ufp_copy["px"], particle_at_ufp.px, rtol=1e-13)
    assert np.isclose(ufp_copy["y"], particle_at_ufp.y, rtol=1e-13)
    assert np.isclose(ufp_copy["py"], particle_at_ufp.py, rtol=1e-13)
    assert np.isclose(ufp_copy["zeta"], particle_at_ufp.zeta, rtol=1e-13)
    assert np.isclose(
        ufp_copy["pzeta"], particle_at_ufp.ptau / line.particle_ref.beta0, rtol=1e-13
    )

    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=5,
        freeze_longitudinal=True,
    )
    assert np.isclose(sfp_copy["x"], particle_at_sfp.x, rtol=1e-13)
    assert np.isclose(sfp_copy["px"], particle_at_sfp.px, rtol=1e-13)
    assert np.isclose(sfp_copy["y"], particle_at_sfp.y, rtol=1e-13)
    assert np.isclose(sfp_copy["py"], particle_at_sfp.py, rtol=1e-13)
    assert np.isclose(sfp_copy["zeta"], particle_at_sfp.zeta, rtol=1e-13)
    assert np.isclose(
        sfp_copy["pzeta"], particle_at_sfp.ptau / line.particle_ref.beta0, rtol=1e-13
    )


def test_4D_Henon():
    """
    Set up line with Henon map element
    """
    alpha_x = 0.0
    beta_x = 1.0
    alpha_y = 0.0
    beta_y = 1.0
    Qx = 0.1
    Qy = 0.1
    K2 = 2.0

    henon = Henonmap(
        omega_x=2 * np.pi * Qx,
        omega_y=2 * np.pi * Qy,
        twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[K2],
        norm=False,
    )
    drift = xt.Drift(length=0.0)  # necessary for aperture check

    line = xt.Line(elements=[henon, drift], element_names=["henon", "zero_len_drift"])
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=4e11)
    line.twiss_default["method"] = "4d"
    line.config.XTRACK_GLOBAL_XY_LIMIT = 100
    line.build_tracker()

    """
    Find fixed point
    """
    fp_limits = [[-0.4, -0.3], [0.05, 0.15], [0.5, 0.6], [-0.2, -0.1]]
    fp = FPFinder(line, order=1, planes="HV", tol=1e-13, verbose=1)
    fp_estimate, _ = fp.find_fp(fp_limits, nemitt_x=400, nemitt_y=400, nemitt_z=1)
    fp_estimate_copy = copy.deepcopy(fp_estimate)

    """
    Calculate fixed point analytically and compare
    """
    omega_x = 2 * np.pi * Qx
    omega_y = 2 * np.pi * Qy
    chi = beta_y / beta_x

    x_ana = -1 / chi * np.tan(omega_y / 2.0)
    px_ana = -x_ana * np.tan(omega_x / 2.0)
    y_ana = (
        1
        / chi
        * np.sqrt(
            np.tan(omega_y / 2.0) ** 2 / chi**2
            + 2 * np.tan(omega_x / 2.0) * np.tan(omega_y / 2.0) / chi**2
        )
    )
    py_ana = -y_ana * np.tan(omega_y / 2.0)

    assert np.isclose(fp_estimate_copy["x"], x_ana, rtol=1e-13)
    assert np.isclose(fp_estimate_copy["px"], px_ana, rtol=1e-13)
    assert np.isclose(fp_estimate_copy["y"], y_ana, rtol=1e-13)
    assert np.isclose(fp_estimate_copy["py"], py_ana, rtol=1e-13)

    """
    Track and compare to itself
    """
    particle_at_fp = xt.Particles(
        **fp_estimate,
    )
    line.track(
        particle_at_fp,
        num_turns=5,
        freeze_longitudinal=True,
    )

    assert np.isclose(fp_estimate_copy["x"], particle_at_fp.x, rtol=1e-13)
    assert np.isclose(fp_estimate_copy["px"], particle_at_fp.px, rtol=1e-13)
    assert np.isclose(fp_estimate_copy["y"], particle_at_fp.y, rtol=1e-13)
    assert np.isclose(fp_estimate_copy["py"], particle_at_fp.py, rtol=1e-13)
