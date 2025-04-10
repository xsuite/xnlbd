import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xnlbd.analyse.normal_forms import *
from xnlbd.track import Henonmap
from xnlbd.visualise.orbits import get_orbit_points

example_data_folder = (
    pathlib.Path(__file__).parent.parent.joinpath("example_data").absolute()
)


def main():
    """
    Example script of computing the normalising transformation and interpolating
    Hamiltonian for the SPS in a quasiresonant (3rd order) case. The orbits are
    plotted in physical space, Courant-Snyder normalised coordinates and normal
    form coordinates. The contours of the interpolating Hamiltonian are also
    plotted in normal form space.
    """

    """
    Set up SPS line
    """

    line = xt.Line.from_json(example_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    line.vv["qh_setvalue"] = 26.336
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

    line.remove_markers()
    line.merge_consecutive_drifts()

    tw = line.twiss()

    nemitt_x = 1e-6
    nemitt_y = 1e-6

    Qx_res = 1.0 / 3.0
    Qy_res = tw.qy

    print(line.element_names[0])

    """
    Get at phase space portrait in horizontal plane
    """
    orbits = get_orbit_points(
        line,
        element="veqf.10010.a_aper",
        planes="H",
        num_pts=50,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )

    """
    Calculate normal forms
    """
    p_line = PolyLine4D(
        line,
        line.particle_ref,
        max_ele_order=3,
        max_map_order=4,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )

    p_line.calculate_one_turn_map()

    p_line.calculate_normal_form(
        max_nf_order=6,
        res_space_dim=1,
        res_case=2,
        res_eig=[
            np.exp(1j * 2 * np.pi * Qx_res),
            np.exp(-1j * 2 * np.pi * Qx_res),
            np.exp(1j * 2 * np.pi * Qy_res),
            np.exp(-1j * 2 * np.pi * Qy_res),
        ],
        res_basis1=[3, 0],
    )

    zeta1_re = np.linspace(-30.0, 30.0, 100)
    zeta1_im = np.linspace(-30.0, 30.0, 100)
    zeta2_re = np.zeros(100)
    zeta2_im = np.zeros(100)
    Zeta1_re, Zeta1_im = np.meshgrid(zeta1_re, zeta1_im)
    Zeta2_re, Zeta2_im = np.meshgrid(zeta2_re, zeta2_im)
    H_values = p_line.normal_form.H.substitute(
        Zeta1_re + 1j * Zeta1_im,
        Zeta1_re - 1j * Zeta1_im,
        Zeta2_re + 1j * Zeta2_im,
        Zeta2_re - 1j * Zeta2_im,
    )
    H_values = np.imag(H_values)
    zeta1_re_level = np.linspace(0, 30.0, 20)
    zeta1_im_level = np.zeros(20)
    zeta2_re_level = np.zeros(20)
    zeta2_im_level = np.zeros(20)
    H_levels = p_line.normal_form.H.substitute(
        zeta1_re_level + 1j * zeta1_im_level,
        zeta1_re_level - 1j * zeta1_im_level,
        zeta2_re_level + 1j * zeta2_im_level,
        zeta2_re_level - 1j * zeta2_im_level,
    )
    H_levels = np.imag(H_levels)

    nf_orbits = p_line.normal_form.norm_to_nf(
        orbits["H_orbit_points_norm"]["x_norm"].flatten(),
        orbits["H_orbit_points_norm"]["px_norm"].flatten(),
        orbits["H_orbit_points_norm"]["y_norm"].flatten(),
        orbits["H_orbit_points_norm"]["py_norm"].flatten(),
    )

    """
    Plot
    """
    fs = 24

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        width_ratios=[1, 1, 1],
        layout="tight",
    )
    ax_qr_re = axes[0]
    ax_qr_cs = axes[1]
    ax_qr_nf = axes[2]

    ax_qr_re.plot(
        orbits["H_orbit_points"]["x"] * 1e3,
        orbits["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    ax_qr_re.set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax_qr_re.set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax_qr_re.tick_params(axis="both", labelsize=fs - 2)

    ax_qr_cs.plot(
        orbits["H_orbit_points_norm"]["x_norm"],
        orbits["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    ax_qr_cs.set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax_qr_cs.set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax_qr_cs.tick_params(axis="both", labelsize=fs - 2)

    ax_qr_nf.plot(
        np.real(nf_orbits[0]),
        -np.imag(nf_orbits[0]),
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    contour = ax_qr_nf.contour(
        Zeta1_re,
        -Zeta1_im,
        H_values,
        levels=np.sort(H_levels),
        linestyles="--",
        cmap="winter",
    )

    ax_qr_nf.set_xlabel(r"Re[$\zeta$] [a.u.]", fontsize=fs)
    ax_qr_nf.set_ylabel(r"-Im[$\zeta$] [a.u.]", fontsize=fs)
    ax_qr_nf.tick_params(axis="both", labelsize=fs - 2)

    plt.show()


if __name__ == "__main__":
    main()
