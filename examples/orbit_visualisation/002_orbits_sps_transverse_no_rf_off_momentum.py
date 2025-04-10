import pathlib

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.visualise.orbits import get_orbit_points

example_data_folder = (
    pathlib.Path(__file__).parent.parent.joinpath("example_data").absolute()
)


def main():
    """
    Example python script for inspecting phase spaces in the transverse planes
    using the SPS lattice. In the horizontal plane the 3rd order resonance is
    strongly excited such that stable islands form, while in the vertical plane
    the phase space remains linear. The RF is turned off, and we inspect two
    particles: one on-momentum, one off-momentum. The orbits of the two
    particles will be different.
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

    # Turn off RF
    line.vv["v200"] = 0.0

    # Turn a large orbit bump in one of the extraction regions
    line.vv["kmpsh41402"] = 0.00039440067863769437
    line.vv["kmplh41658"] = 0.00045922095845457876
    line.vv["kmplh41994"] = 0.00044190219049600475
    line.vv["kmpsh42198"] = -0.0002771794691164835

    """
    Get orbits in transverse phase space for an on-momentum and off-momentum 
    particle in the arc where dispersion is large. RF is turned off.
    """
    orbits_on = get_orbit_points(
        line, element="bph.13008", planes="HV", nemitt_x=1e-6, nemitt_y=1e-6, num_pts=60
    )

    orbits_off = get_orbit_points(
        line,
        element="bph.13008",
        planes="HV",
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        num_pts=60,
        delta0=-0.0015,
    )

    """
    Plot
    """
    fs = 16
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout="tight")
    axes[0, 0].plot(
        orbits_on["H_orbit_points"]["x"] * 1e3,
        orbits_on["H_orbit_points"]["px"] * 1e6,
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
        orbits_off["H_orbit_points"]["x"] * 1e3,
        orbits_off["H_orbit_points"]["px"] * 1e6,
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
        orbits_on["V_orbit_points"]["y"] * 1e3,
        orbits_on["V_orbit_points"]["py"] * 1e6,
        color="black",
        marker=".",
        markersize=0.1,
        linestyle="None",
    )
    axes[1, 0].set_xlabel(r"$y$ [mm]", fontsize=fs)
    axes[1, 0].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
    axes[1, 0].tick_params(axis="both", labelsize=fs - 2)
    axes[1, 1].plot(
        orbits_off["V_orbit_points"]["y"] * 1e3,
        orbits_off["V_orbit_points"]["py"] * 1e6,
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
