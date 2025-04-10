import copy
import pathlib

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.tools import NormedParticles
from xnlbd.visualise.fixed_points import FPFinder
from xnlbd.visualise.orbits import get_orbit_points

example_data_folder = (
    pathlib.Path(__file__).parent.parent.joinpath("example_data").absolute()
)


def main():
    """
    Example python script for finding stable and unstable fixed points using the
    SPS lattice near the 3rd order resonance at a location where there is
    negligible closed orbit and we are looking at an off-momentum particle
    with the RF off.
    """

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
    orbits = get_orbit_points(
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
    fig.suptitle("MKQH.11653", fontsize=fs)

    plt.show()


if __name__ == "__main__":
    main()
