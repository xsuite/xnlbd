import copy
from typing import Tuple, Union, cast

import numpy as np
import xpart as xp  # type: ignore[import-untyped, import-not-found]
import xtrack as xt  # type: ignore[import-untyped, import-not-found]
from xtrack import Line  # type: ignore[import-untyped, import-not-found]
from xtrack.twiss import TwissTable  # type: ignore[import-untyped, import-not-found]

from xnlbd.tools import NormedParticles


def separatrix_points_2D(
    line: Line,
    twiss: TwissTable,
    plane: str,
    order: int,
    ufp: dict[str, float],
    epsilon: float,
    num_turns: int,
    nemitt_x: float = 1,
    nemitt_y: float = 1,
    nemitt_z: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function that returns points close to the separatrix in 2D by tracking from
    the unstable (hyperblic) fixed point.

    Input:
        - line: xsuite Line
        - twiss: xsuite TwissTable obtained with the correct delta0, which is
          the same as for the unstable fixed point passed
        - plane: 'H' for horizontal, 'V' for vertical, 'L' for longitudinal
        - order: integer, order of the resnance and unstable fixed point
        - ufp: dictionary containing the coordinates of the unstable fixed
          point obtained with xnlbd.visualise.FPFinder
        - epsilon: float, small displacement from the unstable fixed point used
          for computing numerical derivative and the initial condition for
          tracking
        - num_turns: integer, the number of turns to track an initial condition
          from close to the separatrix
        - nemitt_x: float, the normalised emittance in the horizontal plane
            used for converting to and from normalised coordinates, default
            `1e-6`
        - nemitt_y: float, normalised emittance in the vertical plane used
            for converting to and from normalised coordinates, default `1e-6`
        - nemitt_z: float, normalised emittance in the longitudinal plane
            used for converting to and from normalised coordinates, default
            `1.0`

    Output:
        - dictionary with coordinates of points close to the separatrix in real
          space and in normalised space
    """

    # Compute Jacobian and its eigenvectors and eigenvalues
    part_at_ufp = xt.Particles(
        **ufp,
    )
    part_eps_p_dir1 = part_at_ufp.copy()
    part_eps_n_dir1 = part_at_ufp.copy()
    part_eps_p_dir2 = part_at_ufp.copy()
    part_eps_n_dir2 = part_at_ufp.copy()

    if plane == "H":
        part_eps_p_dir1_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_p_dir1,
        )
        part_eps_p_dir1_norm.x_norm += epsilon
        part_eps_p_dir1 = part_eps_p_dir1_norm.norm_to_phys(part_eps_p_dir1)
        part_eps_n_dir1_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_n_dir1,
        )
        part_eps_n_dir1_norm.x_norm -= epsilon
        part_eps_n_dir1 = part_eps_n_dir1_norm.norm_to_phys(part_eps_n_dir1)
        part_eps_p_dir2_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_p_dir2,
        )
        part_eps_p_dir2_norm.px_norm += epsilon
        part_eps_p_dir2 = part_eps_p_dir2_norm.norm_to_phys(part_eps_p_dir2)
        part_eps_n_dir2_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_n_dir2,
        )
        part_eps_n_dir2_norm.px_norm -= epsilon
        part_eps_n_dir2 = part_eps_n_dir2_norm.norm_to_phys(part_eps_n_dir2)
    elif plane == "V":
        part_eps_p_dir1_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_p_dir1,
        )
        part_eps_p_dir1_norm.y_norm += epsilon
        part_eps_p_dir1 = part_eps_p_dir1_norm.norm_to_phys(part_eps_p_dir1)
        part_eps_n_dir1_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_n_dir1,
        )
        part_eps_n_dir1_norm.y_norm -= epsilon
        part_eps_n_dir1 = part_eps_n_dir1_norm.norm_to_phys(part_eps_n_dir1)
        part_eps_p_dir2_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_p_dir2,
        )
        part_eps_p_dir2_norm.py_norm += epsilon
        part_eps_p_dir2 = part_eps_p_dir2_norm.norm_to_phys(part_eps_p_dir2)
        part_eps_n_dir2_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_n_dir2,
        )
        part_eps_n_dir2_norm.py_norm -= epsilon
        part_eps_n_dir2 = part_eps_n_dir2_norm.norm_to_phys(part_eps_n_dir2)
    elif plane == "L":
        part_eps_p_dir1_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_p_dir1,
        )
        part_eps_p_dir1_norm.zeta_norm += epsilon
        part_eps_p_dir1 = part_eps_p_dir1_norm.norm_to_phys(part_eps_p_dir1)
        part_eps_n_dir1_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_n_dir1,
        )
        part_eps_n_dir1_norm.zeta_norm -= epsilon
        part_eps_n_dir1 = part_eps_n_dir1_norm.norm_to_phys(part_eps_n_dir1)
        part_eps_p_dir2_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_p_dir2,
        )
        part_eps_p_dir2_norm.pzta_norm += epsilon
        part_eps_p_dir2 = part_eps_p_dir2_norm.norm_to_phys(part_eps_p_dir2)
        part_eps_n_dir2_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_eps_n_dir2,
        )
        part_eps_n_dir2_norm.pzeta_norm -= epsilon
        part_eps_n_dir2 = part_eps_n_dir2_norm.norm_to_phys(part_eps_n_dir2)
    else:
        raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

    line.track(part_eps_p_dir1, num_turns=order)
    part_eps_p_dir1_norm.phys_to_norm(part_eps_p_dir1)

    line.track(part_eps_n_dir1, num_turns=order)
    part_eps_n_dir1_norm.phys_to_norm(part_eps_n_dir1)

    line.track(part_eps_p_dir2, num_turns=order)
    part_eps_p_dir2_norm.phys_to_norm(part_eps_p_dir2)

    line.track(part_eps_n_dir1, num_turns=order)
    part_eps_n_dir1_norm.phys_to_norm(part_eps_n_dir1)

    if plane == "H":
        J_matrix = np.asarray(
            [
                [
                    (part_eps_p_dir1_norm.x_norm - part_eps_n_dir1_norm.x_norm)[0]
                    / (2 * epsilon),
                    (part_eps_p_dir1_norm.px_norm - part_eps_n_dir1_norm.px_norm)[0]
                    / (2 * epsilon),
                ],
                [
                    (part_eps_p_dir2_norm.x_norm - part_eps_n_dir2_norm.x_norm)[0]
                    / (2 * epsilon),
                    (part_eps_p_dir2_norm.px_norm - part_eps_n_dir2_norm.px_norm)[0]
                    / (2 * epsilon),
                ],
            ]
        )
    elif plane == "V":
        J_matrix = np.asarray(
            [
                [
                    (part_eps_p_dir1_norm.y_norm - part_eps_n_dir1_norm.y_norm)[0]
                    / (2 * epsilon),
                    (part_eps_p_dir1_norm.py_norm - part_eps_n_dir1_norm.py_norm)[0]
                    / (2 * epsilon),
                ],
                [
                    (part_eps_p_dir2_norm.y_norm - part_eps_n_dir2_norm.y_norm)[0]
                    / (2 * epsilon),
                    (part_eps_p_dir2_norm.py_norm - part_eps_n_dir2_norm.py_norm)[0]
                    / (2 * epsilon),
                ],
            ]
        )
    elif plane == "L":
        J_matrix = np.asarray(
            [
                [
                    (part_eps_p_dir1_norm.zeta_norm - part_eps_n_dir1_norm.zeta_norm)[0]
                    / (2 * epsilon),
                    (part_eps_p_dir1_norm.pzeta_norm - part_eps_n_dir1_norm.pzeta_norm)[
                        0
                    ]
                    / (2 * epsilon),
                ],
                [
                    (part_eps_p_dir2_norm.zeta_norm - part_eps_n_dir2_norm.zeta_norm)[0]
                    / (2 * epsilon),
                    (part_eps_p_dir2_norm.pzeta_norm - part_eps_n_dir2_norm.pzeta_norm)[
                        0
                    ]
                    / (2 * epsilon),
                ],
            ]
        )
    else:
        raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")
    eigenvalues, eigenvectors = np.linalg.eig(J_matrix)
    stable_dir_idx = np.where(eigenvalues < 1)[0]
    unstable_dir_idx = np.where(eigenvalues >= 1)[0]
    stable_dir_vect = eigenvectors[:, stable_dir_idx].flatten()
    unstable_dir_vect = eigenvectors[:, unstable_dir_idx].flatten()

    # Track particle forwards along unstable direction
    part_unstable = part_at_ufp.copy()
    part_unstable_norm = NormedParticles(
        twiss,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        nemitt_z=nemitt_z,
        part=part_unstable,
    )
    if plane == "H":
        part_unstable_norm.x_norm += unstable_dir_vect[0] * epsilon
        part_unstable_norm.px_norm += unstable_dir_vect[1] * epsilon
    elif plane == "V":
        part_unstable_norm.y_norm += unstable_dir_vect[0] * epsilon
        part_unstable_norm.py_norm += unstable_dir_vect[1] * epsilon
    elif plane == "L":
        part_unstable_norm.x_norm += unstable_dir_vect[0] * epsilon
        part_unstable_norm.px_norm += unstable_dir_vect[1] * epsilon
    else:
        raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")
    part_unstable = part_unstable_norm.norm_to_phys(part_unstable)

    line.track(part_unstable, num_turns=num_turns, turn_by_turn_monitor=True)
    unstable_points = line.record_last_track.to_dict()["data"]
    part_unstable_all = xt.Particles(
        p0c=ufp["p0c"],
        mass0=ufp["mass0"],
        x=np.asarray(unstable_points["x"]),
        px=np.asarray(unstable_points["px"]),
        y=np.asarray(unstable_points["y"]),
        py=np.asarray(unstable_points["py"]),
        zeta=np.asarray(unstable_points["zeta"]),
        ptau=np.asarray(unstable_points["ptau"]),
    )
    part_unstable_norm_all = NormedParticles(
        twiss,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        nemitt_z=nemitt_z,
        part=part_unstable_all,
    )

    # TODO: track particle backwards along stable direction
    # This is currently not possible, as last turns monitor does not work
    # with backtracking

    # Construct data to return
    separatrix_points = {
        "x": np.asarray(part_unstable_all.x),
        "px": np.asarray(part_unstable_all.px),
        "y": np.asarray(part_unstable_all.y),
        "py": np.asarray(part_unstable_all.py),
        "zeta": np.asarray(part_unstable_all.zeta),
        "ptau": np.asarray(part_unstable_all.ptau),
        "pzeta": np.asarray(part_unstable_all.pzeta),
        "delta": np.asarray(part_unstable_all.delta),
        "p0c": np.asarray(part_unstable_all.p0c),
    }
    separatrix_points_norm = {
        "x_norm": np.asarray(part_unstable_norm_all.x_norm),
        "px_norm": np.asarray(part_unstable_norm_all.px_norm),
        "y_norm": np.asarray(part_unstable_norm_all.y_norm),
        "py_norm": np.asarray(part_unstable_norm_all.py_norm),
        "zeta_norm": np.asarray(part_unstable_norm_all.zeta_norm),
        "pzeta_norm": np.asarray(part_unstable_norm_all.pzeta_norm),
    }

    return {"separatrix": separatrix_points, "separatrix_norm": separatrix_points_norm}


def _area_by_triangulation_2D(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    fp: Tuple[float, float],
) -> float:
    """
    Function that calculates the area of a polygon by triangulation.

    Input:
        - x: array-like, coordinates along one dimension
        - y: array-like, coordinates along other dimension
        - fp: tuple of floats representing the fixed point which is to be
          taken as the centre for the triangulation

    Outpu:
        - float representing the area
    """

    orbit_area = 0.0
    for i in range(len(x) - 1, -1, -1):
        orbit_area += (
            abs(
                x[i] * (y[i - 1] - fp[1])
                + x[i - 1] * (fp[1] - y[i])
                + fp[0] * (y[i] - y[i - 1])
            )
            / 2.0
        )

    return orbit_area


def _core_separatrix_2D(
    line: Line,
    twiss: TwissTable,
    plane: str,
    ufp: dict[str, float],
    epsilon: float,
    num_turns: int,
    sampling: int = 1,
    nemitt_x: float = 1,
    nemitt_y: float = 1,
    nemitt_z: float = 1.0,
) -> dict[str, dict[str, Union[np.ndarray, float]]]:
    """
    Function that returns the approximate separatrix of the core region.

    Input:
        - line: xsuite Line
        - twiss: xsuite TwissTable obtained with the correct delta0, which is
          the same as for the unstable fixed point passed
        - plane: 'H' for horizontal, 'V' for vertical, 'L' for longitudinal
        - ufp: dictionary containing the coordinates of the unstable fixed
          point obtained with xnlbd.visualise.FPFinder
        - epsilon: float, small displacement from the unstable fixed point used
          for computing numerical derivative and the initial condition for
          tracking
        - num_turns: integer, the number of turns to track an initial condition
          from close to the separatrix
        - sampling: integer, frequency in turns of the sampling of separatrix
            points, should be set only if an exciter is
            used to generate the island(s)
        - nemitt_x: float, the normalised emittance in the horizontal plane
            used for converting to and from normalised coordinates, default
            `1e-6`
        - nemitt_y: float, normalised emittance in the vertical plane used
            for converting to and from normalised coordinates, default `1e-6`
        - nemitt_z: float, normalised emittance in the longitudinal plane
            used for converting to and from normalised coordinates, default
            `1.0`

    Output:
        - dictionary with coordinates of points close to the separatrix in real
          space and normalised space
    """

    # Track a particle from close to the unstable fixed point displaced
    # towards the closed orbit
    part_at_ufp = xt.Particles(
        **ufp,
    )
    part_at_ufp_norm = NormedParticles(
        twiss,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        nemitt_z=nemitt_z,
        part=part_at_ufp,
    )
    part_at_co = twiss.particle_on_co.copy()
    part_at_co_norm = NormedParticles(
        twiss,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        nemitt_z=nemitt_z,
        part=part_at_co,
    )

    if plane == "H":
        dir_vect = np.asarray(
            [
                (part_at_co_norm.x_norm - part_at_ufp_norm.x_norm)[0],
                (part_at_co_norm.px_norm - part_at_ufp_norm.px_norm)[0],
            ]
        )
        r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
        dir_vect *= 1 / r
        part_at_ufp_norm.x_norm += dir_vect[0] * epsilon * r
        part_at_ufp_norm.px_norm += dir_vect[1] * epsilon * r
    elif plane == "V":
        dir_vect = np.asarray(
            [
                (part_at_co_norm.y_norm - part_at_ufp_norm.y_norm)[0],
                (part_at_co_norm.py_norm - part_at_ufp_norm.py_norm)[0],
            ]
        )
        r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
        dir_vect *= 1 / r
        part_at_ufp_norm.y_norm += dir_vect[0] * epsilon * r
        part_at_ufp_norm.py_norm += dir_vect[1] * epsilon * r
    elif plane == "L":
        dir_vect = np.asarray(
            [
                (part_at_co_norm.zeta_norm - part_at_ufp_norm.zeta_norm)[0],
                (part_at_co_norm.pzeta_norm - part_at_ufp_norm.pzeta_norm)[0],
            ]
        )
        r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
        dir_vect *= 1 / r
        part_at_ufp_norm.zeta_norm += dir_vect[0] * epsilon * r
        part_at_ufp_norm.pzeta_norm += dir_vect[1] * epsilon * r
    else:
        raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

    part_at_ufp = part_at_ufp_norm.norm_to_phys(part_at_ufp)

    line.track(part_at_ufp, num_turns=num_turns, turn_by_turn_monitor=True)
    core_points_rec = line.record_last_track.to_dict()["data"]
    part_core_all = xt.Particles(
        p0c=ufp["p0c"],
        mass0=ufp["mass0"],
        x=np.asarray(core_points_rec["x"]),
        px=np.asarray(core_points_rec["px"]),
        y=np.asarray(core_points_rec["y"]),
        py=np.asarray(core_points_rec["py"]),
        zeta=np.asarray(core_points_rec["zeta"]),
        ptau=np.asarray(core_points_rec["ptau"]),
    )
    part_core_norm_all = NormedParticles(
        twiss,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        nemitt_z=nemitt_z,
        part=part_core_all,
    )
    core_points: dict[str, Union[np.ndarray, float]] = {
        "x": np.asarray(part_core_all.x)[::sampling],
        "px": np.asarray(part_core_all.px)[::sampling],
        "y": np.asarray(part_core_all.y)[::sampling],
        "py": np.asarray(part_core_all.py)[::sampling],
        "zeta": np.asarray(part_core_all.zeta)[::sampling],
        "ptau": np.asarray(part_core_all.ptau)[::sampling],
        "pzeta": np.asarray(part_core_all.pzeta)[::sampling],
        "delta": np.asarray(part_core_all.delta)[::sampling],
        "p0c": np.asarray(part_core_all.p0c)[::sampling],
    }
    core_points_norm: dict[str, Union[np.ndarray, float]] = {
        "x_norm": np.asarray(part_core_norm_all.x_norm)[::sampling],
        "px_norm": np.asarray(part_core_norm_all.px_norm)[::sampling],
        "y_norm": np.asarray(part_core_norm_all.y_norm)[::sampling],
        "py_norm": np.asarray(part_core_norm_all.py_norm)[::sampling],
        "zeta_norm": np.asarray(part_core_norm_all.zeta_norm)[::sampling],
        "pzeta_norm": np.asarray(part_core_norm_all.pzeta_norm)[::sampling],
    }

    # Order points by angle around closed orbit
    if plane == "H":
        theta = np.arctan2(
            core_points_norm["px_norm"] - part_at_co_norm.px_norm[0],
            core_points_norm["x_norm"] - part_at_co_norm.x_norm[0],
        )
        theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
        sort_idx = np.argsort(theta)
    elif plane == "V":
        theta = np.arctan2(
            core_points_norm["py_norm"] - part_at_co_norm.py_norm[0],
            core_points_norm["y_norm"] - part_at_co_norm.y_norm[0],
        )
        theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
        sort_idx = np.argsort(theta)
    elif plane == "L":
        theta = np.arctan2(
            core_points_norm["pzeta_norm"] - part_at_co_norm.pzeta_norm[0],
            core_points_norm["zeta_norm"] - part_at_co_norm.zeta_norm[0],
        )
        theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
        sort_idx = np.argsort(theta)
    else:
        raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")
    for key in core_points.keys():
        core_points[key] = core_points[key][sort_idx]  # type: ignore[index]
        core_points[key] = np.hstack((cast(np.ndarray, core_points[key]), core_points[key][0]))  # type: ignore[index]
    for key in core_points_norm.keys():
        core_points_norm[key] = core_points_norm[key][sort_idx]  # type: ignore[index]
        core_points_norm[key] = np.hstack(
            (cast(np.ndarray, core_points_norm[key]), core_points_norm[key][0])  # type: ignore[index]
        )

    # cast(dict[str, Union[np.ndarray, float]], core_points)
    # cast(dict[str, Union[np.ndarray, float]], core_points_norm)

    # Calculate area
    if plane == "H":
        core_points["area"] = _area_by_triangulation_2D(
            cast(np.ndarray, core_points["x"]),
            cast(np.ndarray, core_points["px"]),
            (part_at_co.x[0], part_at_co.px[0]),
        )
        core_points_norm["area_norm"] = _area_by_triangulation_2D(
            cast(np.ndarray, core_points_norm["x_norm"]),
            cast(np.ndarray, core_points_norm["px_norm"]),
            (part_at_co_norm.x_norm[0], part_at_co_norm.px_norm[0]),
        )
    elif plane == "V":
        core_points["area"] = _area_by_triangulation_2D(
            cast(np.ndarray, core_points["y"]),
            cast(np.ndarray, core_points["py"]),
            (part_at_co.y[0], part_at_co.py[0]),
        )
        core_points_norm["area_norm"] = _area_by_triangulation_2D(
            cast(np.ndarray, core_points_norm["y_norm"]),
            cast(np.ndarray, core_points_norm["py_norm"]),
            (part_at_co_norm.y_norm[0], part_at_co_norm.py_norm[0]),
        )
    elif plane == "L":
        core_points["area"] = _area_by_triangulation_2D(
            cast(np.ndarray, core_points["zeta"]),
            cast(np.ndarray, core_points["pzeta"]),
            (part_at_co.zeta[0], part_at_co.pzeta[0]),
        )
        core_points_norm["area_norm"] = _area_by_triangulation_2D(
            cast(np.ndarray, core_points_norm["zeta_norm"]),
            cast(np.ndarray, core_points_norm["pzeta_norm"]),
            (part_at_co_norm.zeta_norm[0], part_at_co_norm.pzeta_norm[0]),
        )
    else:
        raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

    return {
        "separatrix": core_points,
        "separatrix_norm": core_points_norm,
    }


def _island_separatrix_2D(
    line: Line,
    twiss: TwissTable,
    plane: str,
    ufp: dict[str, float],
    sfp: dict[str, float],
    epsilon: float,
    order: int,
    num_turns: int,
    sampling: int = 1,
    nemitt_x: float = 1,
    nemitt_y: float = 1,
    nemitt_z: float = 1.0,
) -> dict[str, dict[str, dict[str, Union[np.ndarray, float]]]]:
    """
    Function that returns the approximate separatrix of the island regions.

    Input:
        - line: xsuite Line
        - twiss: xsuite TwissTable obtained with the correct delta0, which is
          the same as for the unstable fixed point passed
        - plane: 'H' for horizontal, 'V' for vertical, 'L' for longitudinal
        - ufp: dictionary containing the coordinates of the unstable fixed
          point obtained with xnlbd.visualise.FPFinder
        - sfp: dictionary containing the coordinates of the stable fixed
          point obtained with xnlbd.visualise.FPFinder
        - epsilon: float, small displacement from the unstable fixed point used
          for computing numerical derivative and the initial condition for
          tracking
        - order: integer, order of the resonance
        - num_turns: integer, the number of turns to track an initial condition
          from close to the separatrix
        - sampling: integer, frequency in turns of the sampling of separatrix
            points, should be set only if an exciter is
            used to generate the island(s)
        - nemitt_x: float, the normalised emittance in the horizontal plane
            used for converting to and from normalised coordinates, default
            `1e-6`
        - nemitt_y: float, normalised emittance in the vertical plane used
            for converting to and from normalised coordinates, default `1e-6`
        - nemitt_z: float, normalised emittance in the longitudinal plane
            used for converting to and from normalised coordinates, default
            `1.0`

    Output:
        - dictionary with coordinates of points close to the separatrix in real
          space and in normalised space
    """

    if order == 1:
        part_at_ufp = xt.Particles(
            **ufp,
        )
        all_ufp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_ufp,
        )
        all_ufp_norm.phys_to_norm(part_at_ufp)
        part_at_sfp = xt.Particles(
            **sfp,
        )
        all_sfp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_sfp,
        )
        all_sfp_norm.phys_to_norm(part_at_sfp)

        # Track a particle from close to the unstable fixed point displaced
        # towards the stable fixed point
        part_at_ufp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_ufp,
        )

        if plane == "H":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.x_norm[0] - part_at_ufp_norm.x_norm)[0],
                    (all_sfp_norm.px_norm[0] - part_at_ufp_norm.px_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= 1 / r
            part_at_ufp_norm.x_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.px_norm += dir_vect[1] * epsilon * r
        elif plane == "V":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.y_norm[0] - part_at_ufp_norm.y_norm)[0],
                    (all_sfp_norm.py_norm[0] - part_at_ufp_norm.py_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= 1 / r
            part_at_ufp_norm.y_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.py_norm += dir_vect[1] * epsilon * r
        elif plane == "L":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.zeta_norm[0] - part_at_ufp_norm.zeta_norm)[0],
                    (all_sfp_norm.pzeta_norm[0] - part_at_ufp_norm.pzeta_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= 1 / r
            part_at_ufp_norm.zeta_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.pzeta_norm += dir_vect[1] * epsilon * r
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        part_at_ufp = part_at_ufp_norm.norm_to_phys(part_at_ufp)

        line.track(part_at_ufp, num_turns=num_turns, turn_by_turn_monitor=True)
        curr_island_points_rec = line.record_last_track.to_dict()["data"]
        good_idx = np.where(curr_island_points_rec["state"] == 1)[0]
        for key in curr_island_points_rec.keys():
            try:
                curr_island_points_rec[key] = curr_island_points_rec[key][good_idx]
            except IndexError:
                continue
        part_island_all = xt.Particles(
            p0c=ufp["p0c"],
            mass0=ufp["mass0"],
            x=np.asarray(curr_island_points_rec["x"]),
            px=np.asarray(curr_island_points_rec["px"]),
            y=np.asarray(curr_island_points_rec["y"]),
            py=np.asarray(curr_island_points_rec["py"]),
            zeta=np.asarray(curr_island_points_rec["zeta"]),
            ptau=np.asarray(curr_island_points_rec["ptau"]),
        )
        part_island_norm_all = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_island_all,
        )
        island_points: dict[str, dict[str, Union[np.ndarray, float]]] = {}
        island_points_norm: dict[str, dict[str, Union[np.ndarray, float]]] = {}
        curr_island_points: dict[str, Union[np.ndarray, float]] = {
            "x": np.asarray(part_island_all.x)[::sampling],
            "px": np.asarray(part_island_all.px)[::sampling],
            "y": np.asarray(part_island_all.y)[::sampling],
            "py": np.asarray(part_island_all.py)[::sampling],
            "zeta": np.asarray(part_island_all.zeta)[::sampling],
            "ptau": np.asarray(part_island_all.ptau)[::sampling],
            "pzeta": np.asarray(part_island_all.pzeta)[::sampling],
            "delta": np.asarray(part_island_all.delta)[::sampling],
            "p0c": np.asarray(part_island_all.p0c)[::sampling],
        }
        curr_island_points_norm: dict[str, Union[np.ndarray, float]] = {
            "x_norm": np.asarray(part_island_norm_all.x_norm)[::sampling],
            "px_norm": np.asarray(part_island_norm_all.px_norm)[::sampling],
            "y_norm": np.asarray(part_island_norm_all.y_norm)[::sampling],
            "py_norm": np.asarray(part_island_norm_all.py_norm)[::sampling],
            "zeta_norm": np.asarray(part_island_norm_all.zeta_norm)[::sampling],
            "pzeta_norm": np.asarray(part_island_norm_all.pzeta_norm)[::sampling],
        }
        island_points["core"] = curr_island_points
        island_points_norm["core"] = curr_island_points_norm

        # Order points by angle and calculate the area as well
        if plane == "H":
            theta = np.arctan2(
                (island_points_norm["core"]["px_norm"] - all_sfp_norm.px_norm[0]),
                (island_points_norm["core"]["x_norm"] - all_sfp_norm.x_norm[0]),
            )
            theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
            sort_idx = np.argsort(theta)
            for key in island_points["core"].keys():
                island_points["core"][key] = island_points["core"][key][sort_idx]  # type: ignore[index]
                island_points["core"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points["core"][key]),
                        island_points["core"][key][0],  # type: ignore[index]
                    )
                )
            for key in island_points_norm["core"].keys():
                island_points_norm["core"][key] = island_points_norm["core"][key][  # type: ignore[index]
                    sort_idx
                ]
                island_points_norm["core"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points_norm["core"][key]),
                        island_points_norm["core"][key][0],  # type: ignore[index]
                    )
                )
            island_points["core"]["area"] = _area_by_triangulation_2D(
                cast(np.ndarray, island_points["core"]["x"]),
                cast(np.ndarray, island_points["core"]["px"]),
                (part_at_sfp.x[0], part_at_sfp.px[0]),
            )
            island_points_norm["core"]["area_norm"] = _area_by_triangulation_2D(
                cast(np.ndarray, island_points_norm["core"]["x_norm"]),
                cast(np.ndarray, island_points_norm["core"]["px_norm"]),
                (all_sfp_norm.x_norm[0], all_sfp_norm.px_norm[0]),
            )
        elif plane == "V":
            theta = np.arctan2(
                (island_points_norm["core"]["py_norm"] - all_sfp_norm.py_norm[0]),
                (island_points_norm["core"]["y_norm"] - all_sfp_norm.y_norm[0]),
            )
            theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
            sort_idx = np.argsort(theta)
            for key in island_points["core"].keys():
                island_points["core"][key] = island_points["core"][key][sort_idx]  # type: ignore[index]
                island_points["core"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points["core"][key]),
                        island_points["core"][key][0],  # type: ignore[index]
                    )
                )
            for key in island_points_norm["core"].keys():
                island_points_norm["core"][key] = island_points_norm["core"][key][  # type: ignore[index]
                    sort_idx
                ]
                island_points_norm["core"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points_norm["core"][key]),
                        island_points_norm["core"][key][0],  # type: ignore[index]
                    )
                )
            island_points["core"]["area"] = _area_by_triangulation_2D(
                cast(np.ndarray, island_points["core"]["y"]),
                cast(np.ndarray, island_points["core"]["py"]),
                (part_at_sfp.y[0], part_at_sfp.py[0]),
            )
            island_points_norm["core"]["area_norm"] = _area_by_triangulation_2D(
                cast(np.ndarray, island_points_norm["core"]["y_norm"]),
                cast(np.ndarray, island_points_norm["core"]["py_norm"]),
                (all_sfp_norm.y_norm[0], all_sfp_norm.py_norm[0]),
            )
        elif plane == "L":
            theta = np.arctan2(
                (island_points_norm["core"]["pzeta_norm"] - all_sfp_norm.pzeta_norm[0]),
                (island_points_norm["core"]["zeta_norm"] - all_sfp_norm.zeta_norm[0]),
            )
            theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
            sort_idx = np.argsort(theta)
            for key in island_points["core"].keys():
                island_points["core"][key] = island_points["core"][key][sort_idx]  # type: ignore[index]
                island_points["core"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points["core"][key]),
                        island_points["core"][key][0],  # type: ignore[index]
                    )
                )
            for key in island_points_norm["core"].keys():
                island_points_norm["core"][key] = island_points_norm["core"][key][  # type: ignore[index]
                    sort_idx
                ]
                island_points_norm["core"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points_norm["core"][key]),
                        island_points_norm["core"][key][0],  # type: ignore[index]
                    )
                )
            island_points["core"]["area"] = _area_by_triangulation_2D(
                cast(np.ndarray, island_points["core"]["zeta"]),
                cast(np.ndarray, island_points["core"]["pzeta"]),
                (part_at_sfp.zeta[0], part_at_sfp.pzeta[0]),
            )
            island_points_norm["core"]["area_norm"] = _area_by_triangulation_2D(
                cast(np.ndarray, island_points_norm["core"]["zeta_norm"]),
                cast(np.ndarray, island_points_norm["core"]["pzeta_norm"]),
                (all_sfp_norm.zeta_norm[0], all_sfp_norm.pzeta_norm[0]),
            )
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        part_at_ufp = xt.Particles(
            **ufp,
        )
        all_ufp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_ufp,
        )
        all_ufp_norm.phys_to_norm(part_at_ufp)
        part_at_sfp = xt.Particles(
            **sfp,
        )
        all_sfp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_sfp,
        )
        all_sfp_norm.phys_to_norm(part_at_sfp)

        # Track a particle from close to the unstable fixed point displaced
        # away from the stable fixed point
        part_at_ufp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_ufp,
        )

        if plane == "H":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.x_norm[0] - part_at_ufp_norm.x_norm)[0],
                    (all_sfp_norm.px_norm[0] - part_at_ufp_norm.px_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= -1 / r
            part_at_ufp_norm.x_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.px_norm += dir_vect[1] * epsilon * r
        elif plane == "V":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.y_norm[0] - part_at_ufp_norm.y_norm)[0],
                    (all_sfp_norm.py_norm[0] - part_at_ufp_norm.py_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= -1 / r
            part_at_ufp_norm.y_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.py_norm += dir_vect[1] * epsilon * r
        elif plane == "L":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.zeta_norm[0] - part_at_ufp_norm.zeta_norm)[0],
                    (all_sfp_norm.pzeta_norm[0] - part_at_ufp_norm.pzeta_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= -1 / r
            part_at_ufp_norm.zeta_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.pzeta_norm += dir_vect[1] * epsilon * r
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        part_at_ufp = part_at_ufp_norm.norm_to_phys(part_at_ufp)

        line.track(part_at_ufp, num_turns=num_turns, turn_by_turn_monitor=True)
        curr_island_points = line.record_last_track.to_dict()["data"]
        good_idx = np.where(curr_island_points["state"] == 1)[0]
        for key in curr_island_points.keys():
            try:
                curr_island_points[key] = curr_island_points[key][good_idx]  # type: ignore[index]
            except IndexError:
                continue
        part_island_all = xt.Particles(
            p0c=ufp["p0c"],
            mass0=ufp["mass0"],
            x=np.asarray(curr_island_points["x"]),
            px=np.asarray(curr_island_points["px"]),
            y=np.asarray(curr_island_points["y"]),
            py=np.asarray(curr_island_points["py"]),
            zeta=np.asarray(curr_island_points["zeta"]),
            ptau=np.asarray(curr_island_points["ptau"]),
        )
        part_island_norm_all = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_island_all,
        )
        curr_island_points = {
            "x": np.asarray(part_island_all.x)[::sampling],
            "px": np.asarray(part_island_all.px)[::sampling],
            "y": np.asarray(part_island_all.y)[::sampling],
            "py": np.asarray(part_island_all.py)[::sampling],
            "zeta": np.asarray(part_island_all.zeta)[::sampling],
            "ptau": np.asarray(part_island_all.ptau)[::sampling],
            "pzeta": np.asarray(part_island_all.pzeta)[::sampling],
            "delta": np.asarray(part_island_all.delta)[::sampling],
            "p0c": np.asarray(part_island_all.p0c)[::sampling],
        }
        curr_island_points_norm = {
            "x_norm": np.asarray(part_island_norm_all.x_norm)[::sampling],
            "px_norm": np.asarray(part_island_norm_all.px_norm)[::sampling],
            "y_norm": np.asarray(part_island_norm_all.y_norm)[::sampling],
            "py_norm": np.asarray(part_island_norm_all.py_norm)[::sampling],
            "zeta_norm": np.asarray(part_island_norm_all.zeta_norm)[::sampling],
            "pzeta_norm": np.asarray(part_island_norm_all.pzeta_norm)[::sampling],
        }
        island_points["island 0"] = curr_island_points
        island_points_norm["island 0"] = curr_island_points_norm

        # Order points by angle and calculate the area as well
        if plane == "H":
            theta = np.arctan2(
                (island_points_norm["island 0"]["px_norm"] - all_sfp_norm.px_norm[0]),
                (island_points_norm["island 0"]["x_norm"] - all_sfp_norm.x_norm[0]),
            )
            theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
            sort_idx = np.argsort(theta)
            for key in island_points["island 0"].keys():
                island_points["island 0"][key] = island_points["island 0"][key][  # type: ignore[index]
                    sort_idx
                ]
                island_points["island 0"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points["island 0"][key]),
                        island_points["island 0"][key][0],  # type: ignore[index]
                    )
                )
            for key in island_points_norm["island 0"].keys():
                island_points_norm["island 0"][key] = island_points_norm["island 0"][  # type: ignore[index]
                    key
                ][
                    sort_idx
                ]
                island_points_norm["island 0"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points_norm["island 0"][key]),
                        island_points_norm["island 0"][key][0],  # type: ignore[index]
                    )
                )
            island_points["island 0"]["area"] = (
                _area_by_triangulation_2D(
                    cast(np.ndarray, island_points["island 0"]["x"]),
                    cast(np.ndarray, island_points["island 0"]["px"]),
                    (part_at_sfp.x[0], part_at_sfp.px[0]),
                )
                - island_points["core"]["area"]
            )
            island_points_norm["island 0"]["area_norm"] = (
                _area_by_triangulation_2D(
                    cast(np.ndarray, island_points_norm["island 0"]["x_norm"]),
                    cast(np.ndarray, island_points_norm["island 0"]["px_norm"]),
                    (all_sfp_norm.x_norm[0], all_sfp_norm.px_norm[0]),
                )
                - island_points_norm["core"]["area_norm"]
            )
        elif plane == "V":
            theta = np.arctan2(
                (island_points_norm["island 0"]["py_norm"] - all_sfp_norm.py_norm[0]),
                (island_points_norm["island 0"]["y_norm"] - all_sfp_norm.y_norm[0]),
            )
            theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
            sort_idx = np.argsort(theta)
            for key in island_points["island 0"].keys():
                island_points["island 0"][key] = island_points["island 0"][key][  # type: ignore[index]
                    sort_idx
                ]
                island_points["island 0"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points["island 0"][key]),
                        island_points["island 0"][key][0],  # type: ignore[index]
                    )
                )
            for key in island_points_norm["island 0"].keys():
                island_points_norm["island 0"][key] = island_points_norm["island 0"][  # type: ignore[index]
                    key
                ][
                    sort_idx
                ]
                island_points_norm["island 0"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points_norm["island 0"][key]),
                        island_points_norm["island 0"][key][0],  # type: ignore[index]
                    )
                )
            island_points["island 0"]["area"] = (
                _area_by_triangulation_2D(
                    cast(np.ndarray, island_points["island 0"]["y"]),
                    cast(np.ndarray, island_points["island 0"]["py"]),
                    (part_at_sfp.y[0], part_at_sfp.py[0]),
                )
                - island_points["core"]["area"]
            )
            island_points_norm["island 0"]["area_norm"] = (
                _area_by_triangulation_2D(
                    cast(np.ndarray, island_points_norm["island 0"]["y_norm"]),
                    cast(np.ndarray, island_points_norm["island 0"]["py_norm"]),
                    (all_sfp_norm.y_norm[0], all_sfp_norm.py_norm[0]),
                )
                - island_points_norm["core"]["area_norm"]
            )
        elif plane == "L":
            theta = np.arctan2(
                (
                    island_points_norm["island 0"]["pzeta_norm"]
                    - all_sfp_norm.pzeta_norm[0]
                ),
                (
                    island_points_norm["island 0"]["zeta_norm"]
                    - all_sfp_norm.zeta_norm[0]
                ),
            )
            theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
            sort_idx = np.argsort(theta)
            for key in island_points["island 0"].keys():
                island_points["island 0"][key] = island_points["island 0"][key][  # type: ignore[index]
                    sort_idx
                ]
                island_points["island 0"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points["island 0"][key]),
                        island_points["island 0"][key][0],  # type: ignore[index]
                    )
                )
            for key in island_points_norm["island 0"].keys():
                island_points_norm["island 0"][key] = island_points_norm["island 0"][  # type: ignore[index]
                    key
                ][
                    sort_idx
                ]
                island_points_norm["island 0"][key] = np.hstack(
                    (
                        cast(np.ndarray, island_points_norm["island 0"][key]),
                        island_points_norm["island 0"][key][0],  # type: ignore[index]
                    )
                )
            island_points["island 0"]["area"] = (
                _area_by_triangulation_2D(
                    cast(np.ndarray, island_points["island 0"]["zeta"]),
                    cast(np.ndarray, island_points["island 0"]["pzeta"]),
                    (part_at_sfp.zeta[0], part_at_sfp.pzeta[0]),
                )
                - island_points["core"]["area"]
            )
            island_points_norm["island 0"]["area_norm"] = (
                _area_by_triangulation_2D(
                    cast(np.ndarray, island_points_norm["island 0"]["zeta_norm"]),
                    cast(np.ndarray, island_points_norm["island 0"]["pzeta_norm"]),
                    (all_sfp_norm.zeta_norm[0], all_sfp_norm.pzeta_norm[0]),
                )
                - island_points_norm["core"]["area_norm"]
            )
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        return {"separatrix": island_points, "separatrix_norm": island_points_norm}
    else:
        # Find all stable and unstable fixed points
        part_at_ufp = xt.Particles(
            **ufp,
        )
        line.track(
            part_at_ufp,
            num_turns=order * sampling,
            turn_by_turn_monitor=True,
        )
        tmp = copy.deepcopy(line.record_last_track).to_dict()["data"]
        for key in tmp.keys():
            try:
                tmp[key] = tmp[key][::sampling]
            except IndexError:
                continue
        all_ufp = xt.Particles(**tmp)
        all_ufp_norm = NormedParticles(
            twiss, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=all_ufp
        )
        all_ufp_norm.phys_to_norm(all_ufp)
        part_at_sfp = xt.Particles(
            **sfp,
        )
        line.track(
            part_at_sfp,
            num_turns=order,
            turn_by_turn_monitor=True,
        )
        tmp = copy.deepcopy(line.record_last_track).to_dict()["data"]
        for key in tmp.keys():
            try:
                tmp[key] = tmp[key][::sampling]
            except IndexError:
                continue
        all_sfp = xt.Particles(**tmp)
        all_sfp_norm = NormedParticles(
            twiss, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=all_sfp
        )
        all_sfp_norm.phys_to_norm(all_sfp)

        # Find closest unstable fixed point to the first stable fixed point
        if plane == "H":
            dist = np.sqrt(
                (all_ufp_norm.x_norm - all_sfp_norm.x_norm[0]) ** 2
                + (all_ufp_norm.px_norm - all_sfp_norm.px_norm[0]) ** 2
            )
            ufp_idx = np.argmin(dist)
        elif plane == "V":
            dist = np.sqrt(
                (all_ufp_norm.y_norm - all_sfp_norm.y_norm[0]) ** 2
                + (all_ufp_norm.py_norm - all_sfp_norm.py_norm[0]) ** 2
            )
            ufp_idx = np.argmin(dist)
        elif plane == "L":
            dist = np.sqrt(
                (all_ufp_norm.zeta_norm - all_sfp_norm.zeta_norm[0]) ** 2
                + (all_ufp_norm.pzeta_norm - all_sfp_norm.pzeta_norm[0]) ** 2
            )
            ufp_idx = np.argmin(dist)
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        # Track a particle from close to the unstable fixed point displaced
        # towards the stable fixed point
        part_at_ufp_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_at_ufp,
        )
        part_at_ufp_norm.x_norm = all_ufp_norm.x_norm[ufp_idx]
        part_at_ufp_norm.px_norm = all_ufp_norm.px_norm[ufp_idx]
        part_at_ufp_norm.y_norm = all_ufp_norm.y_norm[ufp_idx]
        part_at_ufp_norm.py_norm = all_ufp_norm.py_norm[ufp_idx]
        part_at_ufp_norm.zeta_norm = all_ufp_norm.zeta_norm[ufp_idx]
        part_at_ufp_norm.pzeta_norm = all_ufp_norm.pzeta_norm[ufp_idx]
        part_at_ufp = part_at_ufp_norm.norm_to_phys(part_at_ufp)

        if plane == "H":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.x_norm[0] - part_at_ufp_norm.x_norm)[0],
                    (all_sfp_norm.px_norm[0] - part_at_ufp_norm.px_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= 1 / r
            part_at_ufp_norm.x_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.px_norm += dir_vect[1] * epsilon * r
        elif plane == "V":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.y_norm[0] - part_at_ufp_norm.y_norm)[0],
                    (all_sfp_norm.py_norm[0] - part_at_ufp_norm.py_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= 1 / r
            part_at_ufp_norm.y_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.py_norm += dir_vect[1] * epsilon * r
        elif plane == "L":
            dir_vect = np.asarray(
                [
                    (all_sfp_norm.zeta_norm[0] - part_at_ufp_norm.zeta_norm)[0],
                    (all_sfp_norm.pzeta_norm[0] - part_at_ufp_norm.pzeta_norm)[0],
                ]
            )
            r = np.sqrt(dir_vect[0] ** 2 + dir_vect[1] ** 2)
            dir_vect *= 1 / r
            part_at_ufp_norm.zeta_norm += dir_vect[0] * epsilon * r
            part_at_ufp_norm.pzeta_norm += dir_vect[1] * epsilon * r
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        part_at_ufp = part_at_ufp_norm.norm_to_phys(part_at_ufp)

        line.track(part_at_ufp, num_turns=num_turns, turn_by_turn_monitor=True)
        island_points_rec = line.record_last_track.to_dict()["data"]
        good_idx = np.where(island_points_rec["state"] == 1)[0]
        for key in island_points_rec.keys():
            try:
                island_points_rec[key] = island_points_rec[key][good_idx]
            except IndexError:
                continue
        part_island_all = xt.Particles(
            p0c=ufp["p0c"],
            mass0=ufp["mass0"],
            x=np.asarray(island_points_rec["x"]),
            px=np.asarray(island_points_rec["px"]),
            y=np.asarray(island_points_rec["y"]),
            py=np.asarray(island_points_rec["py"]),
            zeta=np.asarray(island_points_rec["zeta"]),
            ptau=np.asarray(island_points_rec["ptau"]),
        )
        part_island_norm_all = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=part_island_all,
        )
        island_points = {}
        island_points_norm = {}
        for i in range(order):
            curr_island_points = {
                "x": np.asarray(part_island_all.x)[::sampling][i::order],
                "px": np.asarray(part_island_all.px)[::sampling][i::order],
                "y": np.asarray(part_island_all.y)[::sampling][i::order],
                "py": np.asarray(part_island_all.py)[::sampling][i::order],
                "zeta": np.asarray(part_island_all.zeta)[::sampling][i::order],
                "ptau": np.asarray(part_island_all.ptau)[::sampling][i::order],
                "pzeta": np.asarray(part_island_all.pzeta)[::sampling][i::order],
                "delta": np.asarray(part_island_all.delta)[::sampling][i::order],
                "p0c": np.asarray(part_island_all.p0c)[::sampling][i::order],
            }
            curr_island_points_norm = {
                "x_norm": np.asarray(part_island_norm_all.x_norm)[::sampling][i::order],
                "px_norm": np.asarray(part_island_norm_all.px_norm)[::sampling][
                    i::order
                ],
                "y_norm": np.asarray(part_island_norm_all.y_norm)[::sampling][i::order],
                "py_norm": np.asarray(part_island_norm_all.py_norm)[::sampling][
                    i::order
                ],
                "zeta_norm": np.asarray(part_island_norm_all.zeta_norm)[::sampling][
                    i::order
                ],
                "pzeta_norm": np.asarray(part_island_norm_all.pzeta_norm)[::sampling][
                    i::order
                ],
            }
            island_points[f"island {i}"] = curr_island_points
            island_points_norm[f"island {i}"] = curr_island_points_norm

        # For each island order points by angle and calculate
        # the area as well (it should be the same for all islands)
        if plane == "H":
            for i in range(order):
                theta = np.arctan2(
                    (
                        island_points_norm[f"island {i}"]["px_norm"]
                        - all_sfp_norm.px_norm[i]
                    ),
                    (
                        island_points_norm[f"island {i}"]["x_norm"]
                        - all_sfp_norm.x_norm[i]
                    ),
                )
                theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
                sort_idx = np.argsort(theta)
                for key in island_points[f"island {i}"].keys():
                    island_points[f"island {i}"][key] = island_points[f"island {i}"][  # type: ignore[index]
                        key
                    ][
                        sort_idx
                    ]
                    island_points[f"island {i}"][key] = np.hstack(
                        (
                            cast(np.ndarray, island_points[f"island {i}"][key]),
                            island_points[f"island {i}"][key][0],  # type: ignore[index]
                        )
                    )
                for key in island_points_norm[f"island {i}"].keys():
                    island_points_norm[f"island {i}"][key] = island_points_norm[  # type: ignore[index]
                        f"island {i}"
                    ][
                        key
                    ][
                        sort_idx
                    ]
                    island_points_norm[f"island {i}"][key] = np.hstack(
                        (
                            cast(np.ndarray, island_points_norm[f"island {i}"][key]),
                            island_points_norm[f"island {i}"][key][0],  # type: ignore[index]
                        )
                    )
                island_points[f"island {i}"]["area"] = _area_by_triangulation_2D(
                    cast(np.ndarray, island_points[f"island {i}"]["x"]),
                    cast(np.ndarray, island_points[f"island {i}"]["px"]),
                    (all_sfp.x[i], all_sfp.px[i]),
                )
                island_points["stable fps"] = all_sfp
                island_points["unstable fps"] = all_ufp
                island_points_norm[f"island {i}"]["area_norm"] = (
                    _area_by_triangulation_2D(
                        cast(np.ndarray, island_points_norm[f"island {i}"]["x_norm"]),
                        cast(np.ndarray, island_points_norm[f"island {i}"]["px_norm"]),
                        (all_sfp_norm.x_norm[i], all_sfp_norm.px_norm[i]),
                    )
                )
                island_points_norm["stable fps"] = all_sfp_norm
                island_points_norm["unstable fps"] = all_ufp_norm
        elif plane == "V":
            for i in range(order):
                theta = np.arctan2(
                    (
                        island_points_norm[f"island {i}"]["py_norm"]
                        - all_sfp_norm.py_norm[i]
                    ),
                    (
                        island_points_norm[f"island {i}"]["y_norm"]
                        - all_sfp_norm.y_norm[i]
                    ),
                )
                theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
                sort_idx = np.argsort(theta)
                for key in island_points[f"island {i}"].keys():
                    island_points[f"island {i}"][key] = island_points[f"island {i}"][  # type: ignore[index]
                        key
                    ][
                        sort_idx
                    ]
                    island_points[f"island {i}"][key] = np.hstack(
                        (
                            cast(np.ndarray, island_points[f"island {i}"][key]),
                            island_points[f"island {i}"][key][0],  # type: ignore[index]
                        )
                    )
                for key in island_points_norm[f"island {i}"].keys():
                    island_points_norm[f"island {i}"][key] = island_points_norm[  # type: ignore[index]
                        f"island {i}"
                    ][
                        key
                    ][
                        sort_idx
                    ]
                    island_points_norm[f"island {i}"][key] = np.hstack(
                        (
                            cast(np.ndarray, island_points_norm[f"island {i}"][key]),
                            island_points_norm[f"island {i}"][key][0],  # type: ignore[index]
                        )
                    )
                island_points[f"island {i}"]["area"] = _area_by_triangulation_2D(
                    cast(np.ndarray, island_points[f"island {i}"]["y"]),
                    cast(np.ndarray, island_points[f"island {i}"]["py"]),
                    (all_sfp.y[i], all_sfp.py[i]),
                )
                island_points["stable fps"] = all_sfp
                island_points["unstable fps"] = all_ufp
                island_points_norm[f"island {i}"]["area_norm"] = (
                    _area_by_triangulation_2D(
                        cast(np.ndarray, island_points_norm[f"island {i}"]["y_norm"]),
                        cast(np.ndarray, island_points_norm[f"island {i}"]["py_norm"]),
                        (all_sfp_norm.y_norm[i], all_sfp_norm.py_norm[i]),
                    )
                )
                island_points_norm["stable fps"] = all_sfp_norm
                island_points_norm["unstable fps"] = all_ufp_norm
        elif plane == "L":
            for i in range(order):
                theta = np.arctan2(
                    (
                        island_points_norm[f"island {i}"]["pzeta_norm"]
                        - all_sfp_norm.pzeta_norm[i]
                    ),
                    (
                        island_points_norm[f"island {i}"]["zeta_norm"]
                        - all_sfp_norm.zeta_norm[i]
                    ),
                )
                theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
                sort_idx = np.argsort(theta)
                for key in island_points[f"island {i}"].keys():
                    island_points[f"island {i}"][key] = island_points[f"island {i}"][  # type: ignore[index]
                        key
                    ][
                        sort_idx
                    ]
                    island_points[f"island {i}"][key] = np.hstack(
                        (
                            cast(np.ndarray, island_points[f"island {i}"][key]),
                            island_points[f"island {i}"][key][0],  # type: ignore[index]
                        )
                    )
                for key in island_points_norm[f"island {i}"].keys():
                    island_points_norm[f"island {i}"][key] = island_points_norm[  # type: ignore[index]
                        f"island {i}"
                    ][
                        key
                    ][
                        sort_idx
                    ]
                    island_points_norm[f"island {i}"][key] = np.hstack(
                        (
                            cast(np.ndarray, island_points_norm[f"island {i}"][key]),
                            island_points_norm[f"island {i}"][key][0],  # type: ignore[index]
                        )
                    )
                island_points[f"island {i}"]["area"] = _area_by_triangulation_2D(
                    cast(np.ndarray, island_points[f"island {i}"]["zeta"]),
                    cast(np.ndarray, island_points[f"island {i}"]["pzeta"]),
                    (all_sfp.zeta[i], all_sfp.pzeta[i]),
                )
                island_points["stable fps"] = all_sfp
                island_points["unstable fps"] = all_ufp
                island_points_norm[f"island {i}"]["area_norm"] = (
                    _area_by_triangulation_2D(
                        cast(
                            np.ndarray, island_points_norm[f"island {i}"]["zeta_norm"]
                        ),
                        cast(
                            np.ndarray, island_points_norm[f"island {i}"]["pzeta_norm"]
                        ),
                        (all_sfp_norm.zeta_norm[i], all_sfp_norm.pzeta_norm[i]),
                    )
                )
                island_points_norm["stable fps"] = all_sfp_norm
                island_points_norm["unstable fps"] = all_ufp_norm
        else:
            raise ValueError("Incorrect plane requested, it must be 'H', 'V' or 'L'.")

        return {"separatrix": island_points, "separatrix_norm": island_points_norm}


def approximate_separatrix_by_region_2D(
    line: Line,
    twiss: TwissTable,
    plane: str,
    ufp: dict[str, float],
    sfp: Union[dict[str, float], None],
    epsilon: float,
    order: int,
    num_turns: int,
    sampling: int = 1,
    nemitt_x: float = 1,
    nemitt_y: float = 1,
    nemitt_z: float = 1.0,
) -> dict[str, dict[str, dict[str, Union[np.ndarray, float]]]]:
    """
    Function that returns the approximate separatrix of all regions of phase
    space.

    Input:
        - line: xsuite Line
        - twiss: xsuite TwissTable obtained with the correct delta0, which is
          the same as for the unstable fixed point passed
        - plane: 'H' for horizontal, 'V' for vertical, 'L' for longitudinal
        - ufp: dictionary containing the coordinates of the unstable fixed
          point obtained with xnlbd.visualise.FPFinder
        - sfp: dictionary containing the coordinates of the stable fixed
          point obtained with xnlbd.visualise.FPFinder
        - epsilon: float, small displacement from the unstable fixed point used
          for computing numerical derivative and the initial condition for
          tracking
        - order: integer, order of the resnance and unstable fixed point
        - num_turns: integer, the number of turns to track an initial condition
          from close to the separatrix
        - sampling: integer, frequency in turns of the sampling of separatrix
            points, should be set only if an exciter is
            used to generate the island(s)
        - nemitt_x: float, the normalised emittance in the horizontal plane
            used for converting to and from normalised coordinates, default
            `1e-6`
        - nemitt_y: float, normalised emittance in the vertical plane used
            for converting to and from normalised coordinates, default `1e-6`
        - nemitt_z: float, normalised emittance in the longitudinal plane
            used for converting to and from normalised coordinates, default
            `1.0`

    Output:
        - dictionary with coordinates of points close to the separatrix in real
          space and in normalised space
    """

    if order == 1:
        if sfp is None:
            raise ValueError("Stable fixed point must be provided!")
        island_separatrix = _island_separatrix_2D(
            line,
            twiss,
            plane,
            ufp,
            sfp,
            epsilon,
            order,
            num_turns,
            sampling,
            nemitt_x,
            nemitt_y,
            nemitt_z,
        )

        return island_separatrix
    else:
        # Get core separatrix
        core_separatrix = _core_separatrix_2D(
            line,
            twiss,
            plane,
            ufp,
            epsilon,
            num_turns,
            sampling,
            nemitt_x,
            nemitt_y,
            nemitt_z,
        )

        # If there are stable fixed points, get island separatrices
        if sfp is not None:
            island_separatrix = _island_separatrix_2D(
                line,
                twiss,
                plane,
                ufp,
                sfp,
                epsilon,
                order,
                num_turns,
                sampling,
                nemitt_x,
                nemitt_y,
                nemitt_z,
            )

            island_separatrix["separatrix"]["core"] = core_separatrix["separatrix"]
            island_separatrix["separatrix_norm"]["core"] = core_separatrix[
                "separatrix_norm"
            ]

            return island_separatrix
        else:
            return {
                "separatrix": {"core": core_separatrix["separatrix"]},
                "separatrix_norm": {"core": core_separatrix["separatrix_norm"]},
            }
