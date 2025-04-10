import copy

import numpy as np
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.tools import NormedParticles
from xnlbd.track import Henonmap
from xnlbd.visualise.fixed_points import FPFinder
from xnlbd.visualise.orbits import get_orbit_points


def main():
    """
    Example script of finding a fixed point of one iteration of
    the 4D Henon map, since this can also be calculated analytically.
    """

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


if __name__ == "__main__":
    main()
