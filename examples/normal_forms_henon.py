import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xnlbd.analyse.normal_forms import *
from xnlbd.track import Henonmap
from xnlbd.visualise.orbits import get_orbit_points


def main():
    """
    Set up line with Henon map element
    """
    alpha_x = 1.0
    beta_x = 100.0
    alpha_y = -2.0
    beta_y = 25.0
    Qx_res1 = 0.25
    Qy_res1 = 0.29
    Qx1 = 0.252
    Qy1 = 0.29
    Qx_res2 = 1.0 / 4.0
    Qy_res2 = 0.18
    Qx2 = 0.3
    Qy2 = 0.18
    K2 = 1.0
    K3 = 0.0
    nemitt_x = 1e-6
    nemitt_y = 1e-6

    henon1 = Henonmap(
        omega_x=2 * np.pi * Qx1,
        omega_y=2 * np.pi * Qy1,
        twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[K2, K3],
        norm=False,
    )
    henon2 = Henonmap(
        omega_x=2 * np.pi * Qx2,
        omega_y=2 * np.pi * Qy2,
        twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
        dqx=0.0,
        dqy=0.0,
        dx=0.0,
        ddx=0.0,
        multipole_coeffs=[K2, K3],
        norm=False,
    )
    drift1 = xt.Drift(length=0.0)  # necessary for aperture check
    drift2 = xt.Drift(length=0.0)

    line1 = xt.Line(
        elements=[drift1, henon1], element_names=["zero_len_drift", "henon"]
    )
    line1.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=4e11)
    line1.twiss_default["method"] = "4d"
    line1.config.XTRACK_GLOBAL_XY_LIMIT = 0.1
    line1.build_tracker()

    line2 = xt.Line(
        elements=[drift2, henon2], element_names=["zero_len_drift", "henon"]
    )
    line2.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=4e11)
    line2.twiss_default["method"] = "4d"
    line2.config.XTRACK_GLOBAL_XY_LIMIT = 0.1
    line2.build_tracker()

    """
    Get at phase space portrait in horizontal plane
    """
    orbits1 = get_orbit_points(line1, element="henon", planes="H", num_pts=30)

    orbits2 = get_orbit_points(line2, element="henon", planes="H", num_pts=30)

    """
    Calculate normal forms
    """
    poly_line1 = PolyLine4D(
        line1,
        line1.particle_ref,
        max_ele_order=2,
        max_map_order=2,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )
    poly_line1.calculate_one_turn_map()
    poly_line1.calculate_normal_form(
        max_nf_order=6,
        res_space_dim=1,
        res_case=2,
        res_eig=[
            np.exp(1j * 2 * np.pi * Qx_res1),
            np.exp(-1j * 2 * np.pi * Qx_res1),
            np.exp(1j * 2 * np.pi * Qy_res1),
            np.exp(-1j * 2 * np.pi * Qy_res1),
        ],
        res_basis1=[4, 0],
    )

    nf_orbits1 = poly_line1.normal_form.norm_to_nf(
        orbits1["H_orbit_points_norm"]["x_norm"].flatten(),
        orbits1["H_orbit_points_norm"]["px_norm"].flatten(),
        orbits1["H_orbit_points_norm"]["y_norm"].flatten(),
        orbits1["H_orbit_points_norm"]["py_norm"].flatten(),
    )

    zeta1_re1 = np.linspace(-36.0, 40.0, 100)
    zeta1_im1 = np.linspace(-40.0, 36.0, 100)
    zeta2_re1 = np.zeros(100)
    zeta2_im1 = np.zeros(100)
    Zeta1_re1, Zeta1_im1 = np.meshgrid(zeta1_re1, zeta1_im1)
    Zeta2_re1, Zeta2_im1 = np.meshgrid(zeta2_re1, zeta2_im1)
    H_values1 = poly_line1.normal_form.H.substitute(
        Zeta1_re1 + 1j * Zeta1_im1,
        Zeta1_re1 - 1j * Zeta1_im1,
        Zeta2_re1 + 1j * Zeta2_im1,
        Zeta2_re1 - 1j * Zeta2_im1,
    )
    H_values1 = np.imag(H_values1)
    zeta1_re_level1 = np.linspace(0, 40.0, 15)
    zeta1_im_level1 = np.zeros(15)
    zeta2_re_level1 = np.zeros(15)
    zeta2_im_level1 = np.zeros(15)
    H_levels1 = poly_line1.normal_form.H.substitute(
        zeta1_re_level1 + 1j * zeta1_im_level1,
        zeta1_re_level1 - 1j * zeta1_im_level1,
        zeta2_re_level1 + 1j * zeta2_im_level1,
        zeta2_re_level1 - 1j * zeta2_im_level1,
    )
    H_levels1 = np.imag(H_levels1)

    poly_line2 = PolyLine4D(
        line2,
        line2.particle_ref,
        max_ele_order=2,
        max_map_order=2,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )
    poly_line2.calculate_one_turn_map()
    poly_line2.calculate_normal_form(max_nf_order=6, res_space_dim=0, res_case=0)

    nf_orbits2 = poly_line2.normal_form.norm_to_nf(
        orbits2["H_orbit_points_norm"]["x_norm"].flatten(),
        orbits2["H_orbit_points_norm"]["px_norm"].flatten(),
        orbits2["H_orbit_points_norm"]["y_norm"].flatten(),
        orbits2["H_orbit_points_norm"]["py_norm"].flatten(),
    )

    zeta1_re2 = np.linspace(-36.0, 35.0, 100)
    zeta1_im2 = np.linspace(-37.0, 36.0, 100)
    zeta2_re2 = np.zeros(100)
    zeta2_im2 = np.zeros(100)
    Zeta1_re2, Zeta1_im2 = np.meshgrid(zeta1_re2, zeta1_im2)
    Zeta2_re2, Zeta2_im2 = np.meshgrid(zeta2_re2, zeta2_im2)
    H_values2 = poly_line2.normal_form.H.substitute(
        Zeta1_re2 + 1j * Zeta1_im2,
        Zeta1_re2 - 1j * Zeta1_im2,
        Zeta2_re2 + 1j * Zeta2_im2,
        Zeta2_re2 - 1j * Zeta2_im2,
    )
    H_values2 = np.imag(H_values2)
    zeta1_re_level2 = np.linspace(0, 35.0, 15)
    zeta1_im_level2 = np.zeros(15)
    zeta2_re_level2 = np.zeros(15)
    zeta2_im_level2 = np.zeros(15)
    H_levels2 = poly_line2.normal_form.H.substitute(
        zeta1_re_level2 + 1j * zeta1_im_level2,
        zeta1_re_level2 - 1j * zeta1_im_level2,
        zeta2_re_level2 + 1j * zeta2_im_level2,
        zeta2_re_level2 - 1j * zeta2_im_level2,
    )
    H_levels2 = np.imag(H_levels2)

    """
    Plot
    """

    fs = 24

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        height_ratios=[1, 1],
        width_ratios=[1, 1, 1],
        layout="tight",
    )
    ax_qr_re = axes[0, 0]
    ax_qr_cs = axes[0, 1]
    ax_qr_nf = axes[0, 2]
    ax_nr_re = axes[1, 0]
    ax_nr_cs = axes[1, 1]
    ax_nr_nf = axes[1, 2]

    ax_qr_re.plot(
        orbits1["H_orbit_points"]["x"] * 1e3,
        orbits1["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    ax_qr_re.set_xlim((-18.0, 20.0))
    ax_qr_re.set_ylim((-280.0, 150.0))
    ax_qr_re.set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax_qr_re.set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax_qr_re.tick_params(axis="both", labelsize=fs - 2)

    ax_nr_re.plot(
        orbits2["H_orbit_points"]["x"] * 1e3,
        orbits2["H_orbit_points"]["px"] * 1e6,
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    ax_nr_re.set_xlim((-16.0, 17.0))
    ax_nr_re.set_ylim((-380.0, 185.0))
    ax_nr_re.set_xlabel(r"$x$ [mm]", fontsize=fs)
    ax_nr_re.set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    ax_nr_re.tick_params(axis="both", labelsize=fs - 2)

    ax_qr_cs.plot(
        orbits1["H_orbit_points_norm"]["x_norm"],
        orbits1["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    ax_qr_cs.set_xlim((-36.0, 40.0))
    ax_qr_cs.set_ylim((-40.0, 36.0))
    ax_qr_cs.set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax_qr_cs.set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax_qr_cs.tick_params(axis="both", labelsize=fs - 2)

    ax_nr_cs.plot(
        orbits2["H_orbit_points_norm"]["x_norm"],
        orbits2["H_orbit_points_norm"]["px_norm"],
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    ax_nr_cs.set_xlim((-35.0, 35.0))
    ax_nr_cs.set_ylim((-45.0, 31.0))
    ax_nr_cs.set_xlabel(r"$\hat{x}$ [a.u.]", fontsize=fs)
    ax_nr_cs.set_ylabel(r"$\hat{p}_x$ [a.u.]", fontsize=fs)
    ax_nr_cs.tick_params(axis="both", labelsize=fs - 2)

    ax_qr_nf.plot(
        np.real(nf_orbits1[0]),
        -np.imag(nf_orbits1[0]),
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    contour02 = ax_qr_nf.contour(
        Zeta1_re1,
        -Zeta1_im1,
        H_values1,
        levels=np.sort(H_levels1),
        linestyles="--",
        cmap="winter",
    )
    ax_qr_nf.set_xlim((-36.0, 40.0))
    ax_qr_nf.set_ylim((-40.0, 36.0))
    ax_qr_nf.set_xlabel(r"Re[$\zeta$] [a.u.]", fontsize=fs)
    ax_qr_nf.set_ylabel(r"-Im[$\zeta$] [a.u.]", fontsize=fs)
    ax_qr_nf.tick_params(axis="both", labelsize=fs - 2)

    ax_nr_nf.plot(
        np.real(nf_orbits2[0]),
        -np.imag(nf_orbits2[0]),
        color="black",
        marker=".",
        markersize=0.5,
        linestyle="None",
    )
    contour12 = ax_nr_nf.contour(
        Zeta1_re2,
        -Zeta1_im2,
        H_values2,
        levels=np.sort(H_levels2),
        linestyles="--",
        cmap="winter",
    )
    ax_nr_nf.set_xlim((-36.0, 35.0))
    ax_nr_nf.set_ylim((-37.0, 36.0))
    ax_nr_nf.set_xlabel(r"Re[$\zeta$] [a.u.]", fontsize=fs)
    ax_nr_nf.set_ylabel(r"-Im[$\zeta$] [a.u.]", fontsize=fs)
    ax_nr_nf.tick_params(axis="both", labelsize=fs - 2)

    plt.show()


if __name__ == "__main__":
    main()
