import pathlib

import matplotlib
import matplotlib.pyplot as plt
import xpart as xp
import xtrack as xt

from xnlbd.visualise.orbits import get_orbit_points

test_data_folder = pathlib.Path(__file__).parent.joinpath("example_data").absolute()


def main():
    """
    Example python script for inspecting phase spaces using the SPS lattice.
    In the horizontal plane the 3rd order resonance is strongly excited such
    that stable islands form, while in the vertical plane the phase space
    remains linear. The RF is turned on, therefore the RF bucket is visible in
    the logitudinal plane.
    """

    """
    Set lattice up
    """

    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tune and chromaticity
    line.vv["qh_setvalue"] = 26.334
    line.vv["qv_setvalue"] = 26.131061135802028
    line.vv["qph_setvalue"] = 0.4192872718150393
    line.vv["qpv_setvalue"] = 0.07550321565116734

    # Turn on strong sextupoles to excite 3rd order resonance
    line.vv["klse40602"] = 0.3328444933
    line.vv["klse52402"] = -0.26942679

    # Turn on octupoles to ensure stable islands are formed
    line.vv["klof"] = -6.0

    # Turn on RF
    line.vv["v200"] = 2e6

    # Turn a large orbit bump in one of the extraction regions
    line.vv["kmpsh41402"] = 0.00039440067863769437
    line.vv["kmplh41658"] = 0.00045922095845457876
    line.vv["kmplh41994"] = 0.00044190219049600475
    line.vv["kmpsh42198"] = -0.0002771794691164835

    """
    Look at phase space in all 3 planes at the horizontal tune kicker.
    At this location the closed orbit is 0 and dispersion is small.
    """
    orbits_mkqh = get_orbit_points(line, element="mkqh.11653", planes="HVL", num_pts=30)

    fs = 16
    fig = plt.figure(figsize=(8, 12), layout="constrained")
    subfigs = fig.subfigures(3, 1, hspace=0.1)
    ax_top = subfigs[0].subplots(1, 2)
    ax_mid = subfigs[1].subplots(1, 2)
    ax_bot = subfigs[2].subplots(1, 1)
    ax_top[0].plot(
        orbits_mkqh["H_orbit_points"]["x"] * 1e3,
        orbits_mkqh["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_top[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax_top[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax_top[0].tick_params(axis="both", labelsize=fs - 2)
    ax_top[1].plot(
        orbits_mkqh["H_orbit_points_norm"]["x_norm"] * 1e3,
        orbits_mkqh["H_orbit_points_norm"]["px_norm"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_top[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax_top[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax_top[1].tick_params(axis="both", labelsize=fs - 2)
    ax_mid[0].plot(
        orbits_mkqh["V_orbit_points"]["y"] * 1e3,
        orbits_mkqh["V_orbit_points"]["py"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_mid[0].set_xlabel(r"$y$ [mm]", fontsize=fs)
    ax_mid[0].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
    ax_mid[0].tick_params(axis="both", labelsize=fs - 2)
    ax_mid[1].plot(
        orbits_mkqh["V_orbit_points_norm"]["y_norm"] * 1e3,
        orbits_mkqh["V_orbit_points_norm"]["py_norm"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_mid[1].set_xlabel(r"$\hat{y}$ [a.u.]", fontsize=fs)
    ax_mid[1].set_ylabel(r"$\hat{p}_y$ [a.u.]", fontsize=fs)
    ax_mid[1].tick_params(axis="both", labelsize=fs - 2)
    ax_bot.plot(
        orbits_mkqh["L_orbit_points"]["zeta"],
        orbits_mkqh["L_orbit_points"]["delta"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_bot.set_xlabel(r"$\zeta$ [m]", fontsize=fs)
    ax_bot.set_ylabel(r"$\delta$ [$10^{-3}$]", fontsize=fs)
    ax_bot.tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("MKQH.11653", fontsize=fs)

    """
    Look at phase space in all 3 planes in the extraction region, where 
    we set up a large closed orbit bump.
    """
    orbits_teca = get_orbit_points(line, element="teca.41777", planes="HVL", num_pts=30)

    fig = plt.figure(figsize=(8, 12), layout="constrained")
    subfigs = fig.subfigures(3, 1, hspace=0.1)
    ax_top = subfigs[0].subplots(1, 2)
    ax_mid = subfigs[1].subplots(1, 2)
    ax_bot = subfigs[2].subplots(1, 1)
    ax_top[0].plot(
        orbits_teca["H_orbit_points"]["x"] * 1e3,
        orbits_teca["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_top[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax_top[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax_top[0].tick_params(axis="both", labelsize=fs - 2)
    ax_top[1].plot(
        orbits_teca["H_orbit_points_norm"]["x_norm"] * 1e3,
        orbits_teca["H_orbit_points_norm"]["px_norm"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_top[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax_top[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax_top[1].tick_params(axis="both", labelsize=fs - 2)
    ax_mid[0].plot(
        orbits_teca["V_orbit_points"]["y"] * 1e3,
        orbits_teca["V_orbit_points"]["py"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_mid[0].set_xlabel(r"$y$ [mm]", fontsize=fs)
    ax_mid[0].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
    ax_mid[0].tick_params(axis="both", labelsize=fs - 2)
    ax_mid[1].plot(
        orbits_teca["V_orbit_points_norm"]["y_norm"] * 1e3,
        orbits_teca["V_orbit_points_norm"]["py_norm"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_mid[1].set_xlabel(r"$\hat{y}$ [a.u.]", fontsize=fs)
    ax_mid[1].set_ylabel(r"$\hat{p}_y$ [a.u.]", fontsize=fs)
    ax_mid[1].tick_params(axis="both", labelsize=fs - 2)
    ax_bot.plot(
        orbits_teca["L_orbit_points"]["zeta"],
        orbits_teca["L_orbit_points"]["delta"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_bot.set_xlabel(r"$\zeta$ [m]", fontsize=fs)
    ax_bot.set_ylabel(r"$\delta$ [$10^{-3}$]", fontsize=fs)
    ax_bot.tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("TECA.41777", fontsize=fs)

    """
    Look at transverse phase space for an on-momentum and off-momentum 
    particle in the arc where dispersion is large. RF is turned off.
    """
    line.vv["v200"] = 0.0

    orbits_arc_on = get_orbit_points(line, element="bph.13008", planes="HV", num_pts=30)

    orbits_arc_off = get_orbit_points(
        line, element="bph.13008", planes="HV", num_pts=30, delta0=-0.0015
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout="tight")
    axes[0, 0].plot(
        orbits_arc_on["H_orbit_points"]["x"],
        orbits_arc_on["H_orbit_points"]["px"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    axes[0, 0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    axes[0, 0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    axes[0, 0].tick_params(axis="both", labelsize=fs - 2)
    axes[0, 0].set_title(r"$\delta=0$", fontsize=fs)
    axes[0, 1].plot(
        orbits_arc_off["H_orbit_points"]["x"],
        orbits_arc_off["H_orbit_points"]["px"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    axes[0, 1].set_xlabel(r"$x$ [mm]", fontsize=fs)
    axes[0, 1].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    axes[0, 1].tick_params(axis="both", labelsize=fs - 2)
    axes[0, 1].set_title(r"$\delta=-0.0015$", fontsize=fs)
    axes[1, 0].plot(
        orbits_arc_on["V_orbit_points"]["y"],
        orbits_arc_on["V_orbit_points"]["py"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    axes[1, 0].set_xlabel(r"$y$ [mm]", fontsize=fs)
    axes[1, 0].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
    axes[1, 0].tick_params(axis="both", labelsize=fs - 2)
    axes[1, 1].plot(
        orbits_arc_off["V_orbit_points"]["y"],
        orbits_arc_off["V_orbit_points"]["py"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    axes[1, 1].set_xlabel(r"$y$ [mm]", fontsize=fs)
    axes[1, 1].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
    axes[1, 1].tick_params(axis="both", labelsize=fs - 2)
    plt.show()


if __name__ == "__main__":
    main()
