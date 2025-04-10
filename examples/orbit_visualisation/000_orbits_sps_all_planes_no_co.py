import pathlib

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.visualise.orbits import get_orbit_points

example_data_folder = (
    pathlib.Path(__file__).parent.parent.joinpath("example_data").absolute()
)


def main():
    """
    Example python script for inspecting phase spaces in all 3 planes using the
    SPS lattice. In the horizontal plane the 3rd order resonance is strongly
    excited such that stable islands form, while in the vertical plane the phase
    space remains linear. The RF is turned on, so the RF bucket is visible in
    the longitudinal plane.
    """

    """
    Set lattice up
    """
    # Load xsuite line
    line = xt.Line.from_json(example_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

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
    Get orbits in 3 planes at the horizontal tune kicker.
    At this location the closed orbit is 0 and dispersion is small.
    """
    orbits = get_orbit_points(
        line,
        element="mkqh.11653",
        planes="HVL",
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
        num_pts=60,
    )

    """
    Plot
    """
    fs = 16
    fig = plt.figure(figsize=(8, 12), layout="constrained")
    subfigs = fig.subfigures(3, 1, hspace=0.1)
    ax_top = subfigs[0].subplots(1, 2)
    ax_mid = subfigs[1].subplots(1, 2)
    ax_bot = subfigs[2].subplots(1, 1)
    ax_top[0].plot(
        orbits["H_orbit_points"]["x"] * 1e3,
        orbits["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_top[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax_top[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax_top[0].tick_params(axis="both", labelsize=fs - 2)
    ax_top[1].plot(
        orbits["H_orbit_points_norm"]["x_norm"],
        orbits["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_top[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax_top[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax_top[1].tick_params(axis="both", labelsize=fs - 2)
    ax_mid[0].plot(
        orbits["V_orbit_points"]["y"] * 1e3,
        orbits["V_orbit_points"]["py"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_mid[0].set_xlabel(r"$y$ [mm]", fontsize=fs)
    ax_mid[0].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
    ax_mid[0].tick_params(axis="both", labelsize=fs - 2)
    ax_mid[1].plot(
        orbits["V_orbit_points_norm"]["y_norm"],
        orbits["V_orbit_points_norm"]["py_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_mid[1].set_xlabel(r"$\hat{y}$ [a.u.]", fontsize=fs)
    ax_mid[1].set_ylabel(r"$\hat{p}_y$ [a.u.]", fontsize=fs)
    ax_mid[1].tick_params(axis="both", labelsize=fs - 2)
    ax_bot.plot(
        orbits["L_orbit_points"]["zeta"],
        orbits["L_orbit_points"]["delta"] * 1e3,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    ax_bot.set_xlabel(r"$\zeta$ [m]", fontsize=fs)
    ax_bot.set_ylabel(r"$\delta$ [$10^{-3}$]", fontsize=fs)
    ax_bot.tick_params(axis="both", labelsize=fs - 2)
    fig.suptitle("MKQH.11653", fontsize=fs)

    plt.show()


if __name__ == "__main__":
    main()
