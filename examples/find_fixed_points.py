import copy
import pathlib

import matplotlib
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.tools import NormedParticles
from xnlbd.track import Henonmap
from xnlbd.visualise.fixed_points import FPFinder
from xnlbd.visualise.orbits import get_orbit_points

example_data_folder = pathlib.Path(__file__).parent.joinpath("example_data").absolute()


def example_2D_SPS_no_co_on_momentum():
    """
    Set lattice up
    """
    # Load xsuite line
    line = xt.Line.from_json(example_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

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

    # Twiss
    twiss = line.twiss(continue_on_closed_orbit_error=False)

    """
    Find a stable and an unstable fixed point
    """
    ufp_limits = [[2.0, 12.0], [4.0, 12.0]]
    sfp_limits = [[-18.0, -6.0], [-16.0, -6.0]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, verbose=1)
    ufp, ufp_norm = fp.find_fp(ufp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)
    sfp, sfp_norm = fp.find_fp(sfp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)

    """
    Track fixed points to recover all fixed points
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=3,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_ufp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_ufp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_ufp
    )
    all_ufp_norm.phys_to_norm(all_ufp)
    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=3,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_sfp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_sfp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_sfp
    )
    all_sfp_norm.phys_to_norm(all_sfp)

    """
    Look at phase space and fixed points in horizontal plane
    """
    orbits_mkqh = get_orbit_points(
        line,
        element="mkqh.11653",
        planes="H",
        num_pts=60,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )

    fs = 16
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    subfigs = fig.subfigures(1, 1, hspace=0.1)
    ax = subfigs.subplots(1, 2)
    ax[0].plot(
        orbits_mkqh["H_orbit_points"]["x"] * 1e3,
        orbits_mkqh["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[0].plot(
        all_ufp.x * 1e3,
        all_ufp.px * 1e6,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[0].plot(
        all_sfp.x * 1e3,
        all_sfp.px * 1e6,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax[0].tick_params(axis="both", labelsize=fs - 2)
    ax[1].plot(
        orbits_mkqh["H_orbit_points_norm"]["x_norm"],
        orbits_mkqh["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[1].plot(
        all_ufp_norm.x_norm,
        all_ufp_norm.px_norm,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[1].plot(
        all_sfp_norm.x_norm,
        all_sfp_norm.px_norm,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax[1].tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("MKQH.11653", fontsize=fs)

    plt.show()


def example_2D_SPS_no_co_off_momentum():
    """
    Set lattice up
    """
    # Load xsuite line
    line = xt.Line.from_json(example_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

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

    # Twiss
    twiss = line.twiss(continue_on_closed_orbit_error=False, delta0=-0.00112)

    """
    Find a stable and an unstable fixed point
    """
    ufp_limits = [[-2.0, -0.5], [2.0, 4.5]]
    sfp_limits = [[-6.0, -2.5], [-5.5, -3.1]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, verbose=1)
    ufp, ufp_norm = fp.find_fp(
        ufp_limits, delta0=-0.00112, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1
    )
    sfp, sfp_norm = fp.find_fp(
        sfp_limits, delta0=-0.00112, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1
    )

    """
    Track fixed points to recover all fixed points
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=3,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_ufp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_ufp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_ufp
    )
    all_ufp_norm.phys_to_norm(all_ufp)
    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=3,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_sfp = copy.deepcopy(line.record_last_track).to_dict()["data"]
    all_sfp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_sfp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_sfp
    )
    all_sfp_norm.phys_to_norm(all_sfp)

    """
    Look at phase space and fixed points in horizontal plane
    """
    orbits_mkqh = get_orbit_points(
        line,
        element="mkqh.11653",
        planes="H",
        num_pts=60,
        delta0=-0.00112,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )

    fs = 16
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    subfigs = fig.subfigures(1, 1, hspace=0.1)
    ax = subfigs.subplots(1, 2)
    ax[0].plot(
        orbits_mkqh["H_orbit_points"]["x"] * 1e3,
        orbits_mkqh["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[0].plot(
        all_ufp.x * 1e3,
        all_ufp.px * 1e6,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[0].plot(
        all_sfp.x * 1e3,
        all_sfp.px * 1e6,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax[0].tick_params(axis="both", labelsize=fs - 2)
    ax[1].plot(
        orbits_mkqh["H_orbit_points_norm"]["x_norm"],
        orbits_mkqh["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[1].plot(
        all_ufp_norm.x_norm,
        all_ufp_norm.px_norm,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[1].plot(
        all_sfp_norm.x_norm,
        all_sfp_norm.px_norm,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax[1].tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("MKQH.11653", fontsize=fs)

    plt.show()


def example_2D_SPS_co_on_momentum():
    """
    Set lattice up
    """
    # Load xsuite line
    line = xt.Line.from_json(example_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

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
    twiss = line.twiss(continue_on_closed_orbit_error=False, co_guess=co_guess)

    """
    Find a stable and an unstable fixed point
    """
    ufp_limits = [[4.0, 14.0], [2.0, 10.0]]
    sfp_limits = [[-18.0, -12.0], [-12.0, -8.0]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, co_guess=co_guess, verbose=1)
    ufp, ufp_norm = fp.find_fp(
        ufp_limits, co_guess=co_guess, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1
    )
    sfp, sfp_norm = fp.find_fp(
        sfp_limits, co_guess=co_guess, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1
    )

    """
    Track fixed points to recover all fixed points
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=3,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_ufp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_ufp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_ufp
    )
    all_ufp_norm.phys_to_norm(all_ufp)
    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=3,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_sfp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_sfp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_sfp
    )
    all_sfp_norm.phys_to_norm(all_sfp)

    """
    Look at phase space and fixed points in horizontal plane
    """
    orbits_teca = get_orbit_points(
        line,
        element="teca.41777",
        planes="H",
        co_guess=co_guess,
        num_pts=60,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )

    fs = 16
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    subfigs = fig.subfigures(1, 1, hspace=0.1)
    ax = subfigs.subplots(1, 2)
    ax[0].plot(
        orbits_teca["H_orbit_points"]["x"] * 1e3,
        orbits_teca["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[0].plot(
        all_ufp.x * 1e3,
        all_ufp.px * 1e6,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[0].plot(
        all_sfp.x * 1e3,
        all_sfp.px * 1e6,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax[0].tick_params(axis="both", labelsize=fs - 2)
    ax[1].plot(
        orbits_teca["H_orbit_points_norm"]["x_norm"],
        orbits_teca["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[1].plot(
        all_ufp_norm.x_norm,
        all_ufp_norm.px_norm,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[1].plot(
        all_sfp_norm.x_norm,
        all_sfp_norm.px_norm,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax[1].tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("TECA.41777", fontsize=fs)

    plt.show()


def example_2D_Henon():
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

    twiss = line.twiss(continue_on_closed_orbit_error=False)

    """
    Find a stable and an unstable fixed point
    """
    ufp_limits = [[8.0, 22.0], [11.0, 21.0]]
    sfp_limits = [[-28.0, -14.0], [-12.0, 2.0]]

    fp = FPFinder(line, order=5, planes="H", tol=1e-13, verbose=1)
    ufp, ufp_norm = fp.find_fp(ufp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)
    sfp, sfp_norm = fp.find_fp(sfp_limits, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1)

    """
    Track fixed points to recover all fixed points
    """
    particle_at_ufp = xt.Particles(
        **ufp,
    )
    line.track(
        particle_at_ufp,
        num_turns=5,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_ufp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_ufp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_ufp
    )
    all_ufp_norm.phys_to_norm(all_ufp)
    particle_at_sfp = xt.Particles(
        **sfp,
    )
    line.track(
        particle_at_sfp,
        num_turns=5,
        freeze_longitudinal=True,
        turn_by_turn_monitor=True,
    )
    all_sfp = xt.Particles(**copy.deepcopy(line.record_last_track).to_dict()["data"])
    all_sfp_norm = NormedParticles(
        twiss, nemitt_x=1e-6, nemitt_y=1e-6, nemitt_z=1, part=all_sfp
    )
    all_sfp_norm.phys_to_norm(all_sfp)

    """
    Find signs to colour phase space to demonstrate the principle
    """
    X_grid_range = np.linspace(-40.0, 30.0, 100)
    Px_grid_range = np.linspace(-40.0, 40.0, 100)
    X_grid, Px_grid = np.meshgrid(X_grid_range, Px_grid_range)
    X = X_grid.flatten()
    Px = Px_grid.flatten()
    points = np.vstack((X, Px))
    signs_2d = fp._grid_signs(points, X_grid.shape)
    signs = (signs_2d[0, :, :].flatten() * 2 + signs_2d[1, :, :].flatten() + 3) * 0.5
    colours = ["blue", "yellow", "limegreen", "red"]
    custom_cmap = matplotlib.colors.ListedColormap(colours)

    """
    Look at phase space and fixed points in horizontal plane
    """
    orbits = get_orbit_points(line, element="henon", planes="H", num_pts=60)

    fs = 16
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    subfigs = fig.subfigures(1, 1, hspace=0.1)
    ax = subfigs.subplots(1, 2)
    ax[0].plot(
        orbits["H_orbit_points"]["x"] * 1e3,
        orbits["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[0].plot(
        all_ufp.x * 1e3,
        all_ufp.px * 1e6,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[0].plot(
        all_sfp.x * 1e3,
        all_sfp.px * 1e6,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax[0].tick_params(axis="both", labelsize=fs - 2)
    ax[1].plot(
        orbits["H_orbit_points_norm"]["x_norm"],
        orbits["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax[1].plot(
        all_ufp_norm.x_norm,
        all_ufp_norm.px_norm,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    ax[1].plot(
        all_sfp_norm.x_norm,
        all_sfp_norm.px_norm,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    ax[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax[1].tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("Henon map", fontsize=fs)

    plt.figure(figsize=(8, 8), layout="tight")
    X_half_space = (X_grid_range[1] - X_grid_range[0]) / 2.0
    Px_half_space = (Px_grid_range[1] - Px_grid_range[0]) / 2.0
    for i in range(len(X)):
        plt.fill(
            np.asarray(
                [
                    X[i] - X_half_space,
                    X[i] - X_half_space,
                    X[i] + X_half_space,
                    X[i] + X_half_space,
                    X[i] - X_half_space,
                ]
            ),
            np.asarray(
                [
                    Px[i] - Px_half_space,
                    Px[i] + Px_half_space,
                    Px[i] + Px_half_space,
                    Px[i] - Px_half_space,
                    Px[i] - Px_half_space,
                ]
            ),
            edgecolor=colours[int(signs[i])],
            facecolor=colours[int(signs[i])],
        )
    plt.plot(
        orbits["H_orbit_points_norm"]["x_norm"],
        orbits["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    plt.plot(
        all_ufp_norm.x_norm,
        all_ufp_norm.px_norm,
        color="magenta",
        marker="X",
        markersize=10,
        linestyle="None",
    )
    plt.plot(
        all_sfp_norm.x_norm,
        all_sfp_norm.px_norm,
        color="cyan",
        marker="*",
        markersize=10,
        linestyle="None",
    )
    plt.xlim((-40.0, 30.0))
    plt.ylim((-40.0, 40.0))
    plt.xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    plt.ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    plt.tick_params(axis="both", labelsize=fs - 2)

    plt.show()


def example_4D_Henon():
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
    fp_estimate, fp_estimate_norm = fp.find_fp(
        fp_limits, nemitt_x=400, nemitt_y=400, nemitt_z=1
    )
    fp_estimate_copy = copy.deepcopy(fp_estimate)

    """
    Calculate fixed point analytically
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

    """
    Compare fixed point found to analytical
    """
    diff_x_ana = fp_estimate_copy["x"][0] - x_ana
    diff_px_ana = fp_estimate_copy["px"][0] - px_ana
    diff_y_ana = fp_estimate_copy["y"][0] - y_ana
    diff_py_ana = fp_estimate_copy["py"][0] - py_ana

    """
    Track fixed point and check that it stays same
    """
    particle_at_fp = xt.Particles(
        p0c=line.particle_ref.p0c,
        **fp_estimate,
    )
    line.track(
        particle_at_fp,
        num_turns=5,
        freeze_longitudinal=True,
    )

    diff_x_track = fp_estimate_copy["x"][0] - particle_at_fp.x[0]
    diff_px_track = fp_estimate_copy["px"][0] - particle_at_fp.px[0]
    diff_y_track = fp_estimate_copy["y"][0] - particle_at_fp.y[0]
    diff_py_track = fp_estimate_copy["py"][0] - particle_at_fp.py[0]
    diff_zeta_track = fp_estimate_copy["zeta"][0] - particle_at_fp.zeta[0]
    diff_ptau_track = fp_estimate_copy["ptau"][0] - particle_at_fp.ptau[0]

    """
    Print differences
    """
    print("Differences to analytical:")
    print("x:", diff_x_ana)
    print("px:", diff_px_ana)
    print("y:", diff_y_ana)
    print("py:", diff_py_ana)

    print("\nDifferences to tracked:")
    print("x:", diff_x_track)
    print("px:", diff_px_track)
    print("y:", diff_y_track)
    print("py:", diff_py_track)
    print("zeta:", diff_zeta_track)
    print("ptau:", diff_ptau_track)


def main():
    """
    The following is an example of finding stable and unstable fixed points
    using the SPS lattice near the 3rd order resonance at a location where there
    is negligible closed orbit and we are looking at an on-momentum particle
    with the RF off.
    """
    example_2D_SPS_no_co_on_momentum()

    """
    The following is an example of finding stable and unstable fixed points 
    using the SPS lattice near the 3rd order resonance at a location where there 
    is negligible closed orbit and we are looking at an off-momentum particle 
    with the RF off. The topology of the phase space for the off-momentum 
    particle is different to that of the on-momentum one.
    """
    example_2D_SPS_no_co_off_momentum()

    """
    The following is an example of finding stable and unstable fixed points 
    using the SPS lattice near the 3rd order resonance at a location where there 
    is non-negligible closed orbit and we are looking at an on-momentum particle 
    with the RF off.
    """
    example_2D_SPS_co_on_momentum()

    """
    The following is an example of finding stable and unstable fixed points of 
    the 5th iteration of the Henon map with a single sextupole. This example 
    also includes an illustration of the principle of the method by plotting 
    also the normalised phase space coloured according to the topology, although 
    this should not normally be accessed by the user.
    """
    example_2D_Henon()

    """
    The following is an example of finding a fixed point of ine iteration of 
    the 4D Henon map, since this can also be calculated analytically.
    """
    example_4D_Henon()


if __name__ == "__main__":
    main()
