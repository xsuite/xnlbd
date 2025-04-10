import pathlib

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import xpart as xp  # type: ignore[import-untyped, import-not-found]
import xtrack as xt  # type: ignore[import-untyped, import-not-found]

from xnlbd.track.elements import Henonmap
from xnlbd.visualise.fixed_points import FPFinder
from xnlbd.visualise.orbits import get_orbit_points
from xnlbd.visualise.separatrix import approximate_separatrix_by_region_2D

example_data_folder = pathlib.Path(__file__).parent.joinpath("example_data").absolute()


def example_1():
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

    nemitt_x = 1e-6
    nemitt_y = 1e-6

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

    fp = FPFinder(line, order=5, planes="H", tol=1e-13, verbose=0)
    ufp, ufp_norm = fp.find_fp(ufp_limits, nemitt_x=nemitt_x, nemitt_y=nemitt_y)
    sfp, sfp_norm = fp.find_fp(sfp_limits, nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    """
    Get orbits for plotting
    """
    orbits = get_orbit_points(line, element="henon", planes="H", num_pts=60)

    """
    Find separatrix 
    """
    sep = approximate_separatrix_by_region_2D(
        line=line,
        twiss=twiss,
        plane="H",
        ufp=ufp,
        sfp=sfp,
        epsilon=0.05,
        order=5,
        num_turns=10000,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )

    """
    Plot
    """
    fs = 16
    isl_colours = [
        "mediumseagreen",
        "mediumturquoise",
        "deepskyblue",
        "royalblue",
        "slateblue",
    ]
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
    isl_num = 0
    for key in sep["separatrix"].keys():
        if "core" in key:
            ax[0].plot(
                sep["separatrix"][key]["x"] * 1e3,
                sep["separatrix"][key]["px"] * 1e6,
                color="orange",
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix'][key]['area']:.2e})",
            )
        if "island" in key:
            ax[0].plot(
                sep["separatrix"][key]["x"] * 1e3,
                sep["separatrix"][key]["px"] * 1e6,
                color=isl_colours[isl_num],
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix'][key]['area']:.2e})",
            )
            isl_num += 1
    ax[0].plot(
        ufp["x"] * 1e3,
        ufp["px"] * 1e6,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
        label="unstable fixed point",
    )
    ax[0].plot(
        sfp["x"] * 1e3,
        sfp["px"] * 1e6,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
        label="stable fixed point",
    )
    ax[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax[0].tick_params(axis="both", labelsize=fs - 2)
    ax[0].legend(loc=2, fontsize=fs - 4)
    ax[1].plot(
        orbits["H_orbit_points_norm"]["x_norm"],
        orbits["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    isl_num = 0
    for key in sep["separatrix_norm"].keys():
        if "core" in key:
            ax[1].plot(
                sep["separatrix_norm"][key]["x_norm"],
                sep["separatrix_norm"][key]["px_norm"],
                color="orange",
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix_norm'][key]['area_norm']:.2f})",
            )
        if "island" in key:
            ax[1].plot(
                sep["separatrix_norm"][key]["x_norm"],
                sep["separatrix_norm"][key]["px_norm"],
                color=isl_colours[isl_num],
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix_norm'][key]['area_norm']:.2f})",
            )
            isl_num += 1
    ax[1].plot(
        ufp_norm["x_norm"],
        ufp_norm["px_norm"],
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
        label="unstable fixed point",
    )
    ax[1].plot(
        sfp_norm["x_norm"],
        sfp_norm["px_norm"],
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
        label="stable fixed point",
    )
    ax[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax[1].tick_params(axis="both", labelsize=fs - 2)
    ax[1].legend(loc=2, fontsize=fs - 4)
    fig.suptitle("Henon map", fontsize=fs)

    plt.show()


def example_2():
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

    # Turn off RF
    line.vv["v200"] = 0.0

    # Define off-momentum
    delta_part = -0.0013

    # Define beam emittances
    nemitt_x = 1e-6
    nemitt_y = 1e-6

    # Cycle line
    line.cycle(name_first_element="bph.13008", inplace=True)

    # Twiss
    twiss = line.twiss(continue_on_closed_orbit_error=False, delta0=delta_part)

    """
    Find a stable and an unstable fixed point
    """
    ufp_limits = [[-1.8, 0.7], [1.1, 3.6]]
    sfp_limits = [[-6.8, -3.4], [-6.3, -3.2]]

    fp = FPFinder(line, order=3, planes="H", tol=1e-13, verbose=0)
    ufp, ufp_norm = fp.find_fp(
        ufp_limits, delta0=delta_part, nemitt_x=nemitt_x, nemitt_y=nemitt_y
    )
    sfp, sfp_norm = fp.find_fp(
        sfp_limits, delta0=delta_part, nemitt_x=nemitt_x, nemitt_y=nemitt_y
    )

    """
    Get orbits for plotting
    """
    orbits = get_orbit_points(
        line,
        element="bph.13008",
        planes="HV",
        num_pts=60,
        delta0=delta_part,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )

    """
    Find separatrix 
    """
    sep = approximate_separatrix_by_region_2D(
        line=line,
        twiss=twiss,
        plane="H",
        ufp=ufp,
        sfp=sfp,
        epsilon=0.01,
        order=3,
        num_turns=1000,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )

    """
    Plot
    """
    fs = 16
    isl_colours = [
        "mediumseagreen",
        "mediumturquoise",
        "deepskyblue",
        "royalblue",
        "slateblue",
    ]
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
    isl_num = 0
    for key in sep["separatrix"].keys():
        if "core" in key:
            ax[0].plot(
                sep["separatrix"][key]["x"] * 1e3,
                sep["separatrix"][key]["px"] * 1e6,
                color="orange",
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix'][key]['area']:.2e})",
            )
        if "island" in key:
            ax[0].plot(
                sep["separatrix"][key]["x"] * 1e3,
                sep["separatrix"][key]["px"] * 1e6,
                color=isl_colours[isl_num],
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix'][key]['area']:.2e})",
            )
            isl_num += 1
    ax[0].plot(
        ufp["x"] * 1e3,
        ufp["px"] * 1e6,
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
        label="unstable fixed point",
    )
    ax[0].plot(
        sfp["x"] * 1e3,
        sfp["px"] * 1e6,
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
        label="stable fixed point",
    )
    ax[0].set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax[0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax[0].tick_params(axis="both", labelsize=fs - 2)
    ax[0].legend(loc=2, fontsize=fs - 4)
    ax[1].plot(
        orbits["H_orbit_points_norm"]["x_norm"],
        orbits["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    isl_num = 0
    for key in sep["separatrix_norm"].keys():
        if "core" in key:
            ax[1].plot(
                sep["separatrix_norm"][key]["x_norm"],
                sep["separatrix_norm"][key]["px_norm"],
                color="orange",
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix_norm'][key]['area_norm']:.2f})",
            )
        if "island" in key:
            ax[1].plot(
                sep["separatrix_norm"][key]["x_norm"],
                sep["separatrix_norm"][key]["px_norm"],
                color=isl_colours[isl_num],
                marker="None",
                linestyle="-",
                linewidth=3,
                label=f"{key} (A={sep['separatrix_norm'][key]['area_norm']:.2f})",
            )
            isl_num += 1
    ax[1].plot(
        ufp_norm["x_norm"],
        ufp_norm["px_norm"],
        color="red",
        marker="X",
        markersize=10,
        linestyle="None",
        label="unstable fixed point",
    )
    ax[1].plot(
        sfp_norm["x_norm"],
        sfp_norm["px_norm"],
        color="green",
        marker="*",
        markersize=10,
        linestyle="None",
        label="stable fixed point",
    )
    ax[1].set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax[1].set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax[1].tick_params(axis="both", labelsize=fs - 2)
    ax[1].legend(loc=2, fontsize=fs - 4)
    fig.suptitle("SPS at BPH.13008", fontsize=fs)

    plt.show()


def main():
    example_1()
    example_2()


if __name__ == "__main__":
    main()
