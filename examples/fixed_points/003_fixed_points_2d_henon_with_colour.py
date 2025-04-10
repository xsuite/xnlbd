import copy

import matplotlib
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.tools import NormedParticles
from xnlbd.track import Henonmap
from xnlbd.visualise.fixed_points import FPFinder
from xnlbd.visualise.orbits import get_orbit_points


def main():
    """
    Example script of finding stable and unstable fixed points of
    the 5th iteration of the Henon map with a single sextupole. This example
    also includes an illustration of the principle of the method by plotting
    also the normalised phase space coloured according to the topology.
    """

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

    """
    Plot
    """
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


if __name__ == "__main__":
    main()
