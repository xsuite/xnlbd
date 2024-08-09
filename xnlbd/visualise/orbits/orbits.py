import copy

import numpy as np
import scipy.constants as sc  # type: ignore[import-untyped]
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]
from xtrack import Line  # type: ignore[import-untyped]
from xtrack.twiss import TwissTable  # type: ignore[import-untyped]


def get_normalised_coordinates_from_real(
    twiss: TwissTable, particles_dict: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Function to normalise particle coordinates.

    Input:
        - twiss: Twiss table from xsuite
        - particle_dict: dictionary of particle coordinates, can be obtained
          from an xsuite particle object by applying `.to_dict()`; if created
          manually, it must contain the following coordinates (keys): 'x', 'px',
          'y', 'py', 'zeta', 'ptau', 'at_element', 'particle_id'

    Output:
        - dictionary of the normalised particle coordinates containing the
          following variables (keys): 'x_norm', 'px_norm', 'y_norm', 'py_norm',
          'zeta_norm', 'pzeta_norm'
    """

    at_element_particles = np.asarray(particles_dict["at_element"], dtype=int)

    part_id = np.asarray(copy.deepcopy(particles_dict["particle_id"]), dtype=int)
    at_element = (
        np.asarray(copy.deepcopy(part_id), dtype=int) * 0
        + xt.particles.LAST_INVALID_STATE
    )
    x_norm = (
        np.asarray(copy.deepcopy(particles_dict["x"])) * 0
        + xt.particles.LAST_INVALID_STATE
    )
    px_norm = copy.deepcopy(x_norm)
    y_norm = copy.deepcopy(x_norm)
    py_norm = copy.deepcopy(x_norm)
    zeta_norm = copy.deepcopy(x_norm)
    pzeta_norm = copy.deepcopy(x_norm)

    at_element_no_rep = list(
        set(at_element_particles[part_id > xt.particles.LAST_INVALID_STATE])
    )

    for at_ele in at_element_no_rep:
        W = twiss.W_matrix[at_ele]
        W_inv = np.linalg.inv(W)

        mask_at_ele = np.where(at_element_particles == at_ele)[0]

        n_at_ele = len(mask_at_ele)

        # Coordinates wrt to the closed orbit
        XX = np.zeros(shape=(6, n_at_ele), dtype=np.float64)
        XX[0, :] = np.asarray(particles_dict["x"])[mask_at_ele] - twiss.x[at_ele]
        XX[1, :] = np.asarray(particles_dict["px"])[mask_at_ele] - twiss.px[at_ele]
        XX[2, :] = np.asarray(particles_dict["y"])[mask_at_ele] - twiss.y[at_ele]
        XX[3, :] = np.asarray(particles_dict["py"])[mask_at_ele] - twiss.py[at_ele]
        XX[4, :] = np.asarray(particles_dict["zeta"])[mask_at_ele] - twiss.zeta[at_ele]
        XX[5, :] = (
            np.asarray(particles_dict["ptau"])[mask_at_ele] - twiss.ptau[at_ele]
        ) / twiss.particle_on_co.beta0[0]

        XX_norm = np.dot(W_inv, XX)

        x_norm[mask_at_ele] = XX_norm[0, :]
        px_norm[mask_at_ele] = XX_norm[1, :]
        y_norm[mask_at_ele] = XX_norm[2, :]
        py_norm[mask_at_ele] = XX_norm[3, :]
        zeta_norm[mask_at_ele] = XX_norm[4, :]
        pzeta_norm[mask_at_ele] = XX_norm[5, :]
        at_element[mask_at_ele] = at_ele

    return {
        "particle_id": part_id,
        "at_element": at_element,
        "x_norm": x_norm,
        "px_norm": px_norm,
        "y_norm": y_norm,
        "py_norm": py_norm,
        "zeta_norm": zeta_norm,
        "pzeta_norm": pzeta_norm,
    }


def get_real_coordinates_from_normalised(
    twiss: TwissTable, particles_dict: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Function to denormalise particle coordinates.

    Input:
        - twiss: Twiss table from xsuite
        - particle_dict: dictionary of normalised particle coordinates; if
          created manually, it must contain the following coordinates (keys):
          'x_norm', 'px_norm', 'y_norm', 'py_norm', 'zeta_norm', 'pzeta_norm'

    Output:
        - dictionary of the particle coordinates in real space containing the
          following variables (keys): 'x', 'px', 'y', 'py',
          'zeta', 'pzeta'
    """

    at_element_particles = np.asarray(particles_dict["at_element"], dtype=int)

    part_id = np.asarray(copy.deepcopy(particles_dict["particle_id"]), dtype=int)
    at_element = (
        np.asarray(copy.deepcopy(part_id), dtype=int) * 0
        + xt.particles.LAST_INVALID_STATE
    )
    x = (
        np.asarray(copy.deepcopy(particles_dict["x_norm"])) * 0
        + xt.particles.LAST_INVALID_STATE
    )
    px = copy.deepcopy(x)
    y = copy.deepcopy(x)
    py = copy.deepcopy(x)
    zeta = copy.deepcopy(x)
    pzeta = copy.deepcopy(x)

    at_element_no_rep = list(
        set(at_element_particles[part_id > xt.particles.LAST_INVALID_STATE])
    )

    for at_ele in at_element_no_rep:
        W = twiss.W_matrix[at_ele]

        mask_at_ele = np.where(at_element_particles == at_ele)[0]

        n_at_ele = len(mask_at_ele)

        # Coordinates wrt to the closed orbit
        XX_norm = np.zeros(shape=(6, n_at_ele), dtype=np.float64)
        XX_norm[0, :] = np.asarray(particles_dict["x_norm"])[mask_at_ele]
        XX_norm[1, :] = np.asarray(particles_dict["px_norm"])[mask_at_ele]
        XX_norm[2, :] = np.asarray(particles_dict["y_norm"])[mask_at_ele]
        XX_norm[3, :] = np.asarray(particles_dict["py_norm"])[mask_at_ele]
        XX_norm[4, :] = np.asarray(particles_dict["zeta_norm"])[mask_at_ele]
        XX_norm[5, :] = np.asarray(particles_dict["pzeta_norm"])[mask_at_ele]
        XX = np.dot(W, XX_norm)

        x[mask_at_ele] = XX[0, :] + twiss.x[at_ele]
        px[mask_at_ele] = XX[1, :] + twiss.px[at_ele]
        y[mask_at_ele] = XX[2, :] + twiss.y[at_ele]
        py[mask_at_ele] = XX[3, :] + twiss.py[at_ele]
        zeta[mask_at_ele] = XX[4, :] + twiss.zeta[at_ele]
        pzeta[mask_at_ele] = (
            XX[5, :] + twiss.ptau[at_ele] / twiss.particle_on_co.beta0[0]
        )
        at_element[mask_at_ele] = at_ele

    return {
        "particle_id": part_id,
        "at_element": at_element,
        "x": x,
        "px": px,
        "y": y,
        "py": py,
        "zeta": zeta,
        "pzeta": pzeta,
    }


def _get_H_orbit_points(
    line: Line,
    twiss: TwissTable,
    co_coords: dict[str, np.ndarray],
    co_coords_norm: dict[str, np.ndarray],
    num_pts: int,
    num_turns: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the horizontal phase space.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite
        - co_coords: dictionary of closed orbit coordinates in real space
        - co_coords_norm: dictionary of closed orbit coordinates in
          normalised space
        - num_pts: integer, the number of initial conditions up to the limit of
          stability to track
        - num_turns: integer, number of turns to track for, default `2048`;
          note that it has to be large enough to assess the stability of
          initial conditions and to paint the phase space

    Output:
        - dictionary with the following structure:
          {
              "H_orbit_points":
              {
                  "x": np.ndarray((2 x num_pts, num_turns)),
                  "px": np.ndarray((2 x num_pts, num_turns)),
              },
              "H_orbit_points_norm":
              {
                  "x_norm": np.ndarray((2 x num_pts, num_turns)),
                  "px_norm": np.ndarray((2 x num_pts, num_turns)),
              }
          }
    """
    # Find rough estimate for limit of stability
    num_pts_test = num_pts

    coords_test = {
        "x": np.linspace(twiss.x[0], twiss.x[0] + 0.5, num_pts_test),
        "px": np.ones(num_pts_test) * co_coords["px"][0],
        "y": np.ones(num_pts_test) * co_coords["y"][0],
        "py": np.ones(num_pts_test) * co_coords["py"][0],
        "zeta": np.ones(num_pts_test) * co_coords["zeta"][0],
        "ptau": np.ones(num_pts_test) * co_coords["ptau"][0],
    }
    part_test = xp.Particles(
        p0c=twiss.particle_on_co.p0c,
        **coords_test,
    )
    line.track(
        part_test,
        num_turns=num_turns,
    )
    part_test.sort(interleave_lost_particles=True)
    max_x = coords_test["x"][np.min(np.where(part_test.state < 1)[0])]

    coords_test = {
        "x": np.asarray([co_coords["x"][0], max_x]),
        "px": np.asarray([co_coords["px"][0]] * 2),
        "y": np.asarray([co_coords["y"][0]] * 2),
        "py": np.asarray([co_coords["py"][0]] * 2),
        "zeta": np.asarray([co_coords["zeta"][0]] * 2),
        "ptau": np.asarray([co_coords["ptau"][0]] * 2),
        "at_element": np.zeros(2, dtype=int),
        "particle_id": np.arange(2, dtype=int),
    }
    coords_test_norm = get_normalised_coordinates_from_real(twiss, coords_test)
    max_amp_norm = np.sqrt(
        (coords_test_norm["x_norm"][1] - coords_test_norm["x_norm"][0]) ** 2
        + (coords_test_norm["px_norm"][1] - coords_test_norm["px_norm"][0]) ** 2
    )

    # Track several particles within limit of stability
    x_norm = np.hstack(
        (
            np.linspace(
                co_coords_norm["x_norm"][0],
                co_coords_norm["x_norm"][0] + max_amp_norm * np.cos(0),
                num_pts,
            ),
            np.linspace(
                co_coords_norm["x_norm"][0],
                co_coords_norm["x_norm"][0] + max_amp_norm * np.cos(np.pi),
                num_pts,
            ),
        )
    )
    px_norm = np.hstack(
        (
            np.linspace(
                co_coords_norm["px_norm"][0],
                co_coords_norm["px_norm"][0] + max_amp_norm * np.sin(0),
                num_pts,
            ),
            np.linspace(
                co_coords_norm["px_norm"][0],
                co_coords_norm["px_norm"][0] + max_amp_norm * np.sin(np.pi),
                num_pts,
            ),
        )
    )
    sort_idx = np.argsort(x_norm**2 + px_norm**2)

    coords_norm = {
        "x_norm": x_norm[sort_idx],
        "px_norm": px_norm[sort_idx],
        "y_norm": np.ones(2 * num_pts) * co_coords_norm["y_norm"][0],
        "py_norm": np.ones(2 * num_pts) * co_coords_norm["py_norm"][0],
        "zeta_norm": np.ones(2 * num_pts) * co_coords_norm["zeta_norm"][0],
        "pzeta_norm": np.ones(2 * num_pts) * co_coords_norm["pzeta_norm"][0],
        "at_element": np.zeros(2 * num_pts, dtype=int),
        "particle_id": np.arange(2 * num_pts, dtype=int),
    }
    coords = get_real_coordinates_from_normalised(twiss, coords_norm)
    part = xp.Particles(
        p0c=twiss.particle_on_co.p0c,
        **coords,
    )
    line.track(
        part,
        num_turns=num_turns,
        turn_by_turn_monitor=True,
    )
    part.sort(interleave_lost_particles=True)
    state = part.state
    at_turn = part.at_turn

    # Construct data to return
    orbit_points = {
        "x": line.record_last_track.x,
        "px": line.record_last_track.px,
        "y": line.record_last_track.y,
        "py": line.record_last_track.py,
        "zeta": line.record_last_track.zeta,
        "ptau": line.record_last_track.ptau,
        "pzeta": line.record_last_track.pzeta,
        "delta": line.record_last_track.delta,
        "at_element": line.record_last_track.at_element,
        "particle_id": line.record_last_track.particle_id,
    }

    orbit_points_norm = {
        "x_norm": np.zeros((2 * num_pts, num_turns)),
        "px_norm": np.zeros((2 * num_pts, num_turns)),
        "y_norm": np.zeros((2 * num_pts, num_turns)),
        "py_norm": np.zeros((2 * num_pts, num_turns)),
        "zeta_norm": np.zeros((2 * num_pts, num_turns)),
        "pzeta_norm": np.zeros((2 * num_pts, num_turns)),
    }
    for i in range(2 * num_pts):
        curr_coords = {
            "x": orbit_points["x"][i, :],
            "px": orbit_points["px"][i, :],
            "y": orbit_points["y"][i, :],
            "py": orbit_points["py"][i, :],
            "zeta": orbit_points["zeta"][i, :],
            "ptau": orbit_points["ptau"][i, :],
            "at_element": orbit_points["at_element"][i, :],
            "particle_id": orbit_points["particle_id"][i, :],
        }
        curr_coords_norm = get_normalised_coordinates_from_real(twiss, curr_coords)

        orbit_points_norm["x_norm"][i, :] = curr_coords_norm["x_norm"]
        orbit_points_norm["px_norm"][i, :] = curr_coords_norm["px_norm"]
        orbit_points_norm["y_norm"][i, :] = curr_coords_norm["y_norm"]
        orbit_points_norm["py_norm"][i, :] = curr_coords_norm["py_norm"]
        orbit_points_norm["zeta_norm"][i, :] = curr_coords_norm["zeta_norm"]
        orbit_points_norm["pzeta_norm"][i, :] = curr_coords_norm["pzeta_norm"]

    result = {
        "H_orbit_points": {
            "x": orbit_points["x"],
            "px": orbit_points["px"],
        },
        "H_orbit_points_norm": {
            "x_norm": orbit_points_norm["x_norm"],
            "px_norm": orbit_points_norm["px_norm"],
        },
    }
    lost_part_idx = np.where(state < 1)[0]
    for i in lost_part_idx:
        lost_turn = at_turn[i]
        result["H_orbit_points"]["x"][i, lost_turn:] = np.nan
        result["H_orbit_points"]["px"][i, lost_turn:] = np.nan
        result["H_orbit_points_norm"]["x_norm"][i, lost_turn:] = np.nan
        result["H_orbit_points_norm"]["px_norm"][i, lost_turn:] = np.nan

    return result


def _get_V_orbit_points(
    line: Line,
    twiss: TwissTable,
    co_coords: dict[str, np.ndarray],
    co_coords_norm: dict[str, np.ndarray],
    num_pts: int,
    num_turns: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the vertical phase space.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite
        - co_coords: dictionary of closed orbit coordinates in real space
        - co_coords_norm: dictionary of closed orbit coordinates in
          normalised space
        - num_pts: integer, the number of initial conditions up to the limit of
          stability to track
        - num_turns: integer, number of turns to track for, default `2048`;
          note that it has to be large enough to assess the stability of
          initial conditions and to paint the phase space

    Output:
        - dictionary with the following structure:
          {
              "V_orbit_points":
              {
                  "y": np.ndarray((2 x num_pts, num_turns)),
                  "py": np.ndarray((2 x num_pts, num_turns)),
              },
              "V_orbit_points_norm":
              {
                  "y_norm": np.ndarray((2 x num_pts, num_turns)),
                  "py_norm": np.ndarray((2 x num_pts, num_turns)),
              }
          }
    """

    # Find rough estimate for limit of stability
    num_pts_test = num_pts

    coords_test = {
        "x": np.ones(num_pts_test) * co_coords["x"][0],
        "px": np.ones(num_pts_test) * co_coords["px"][0],
        "y": np.linspace(twiss.y[0], twiss.y[0] + 0.5, num_pts_test),
        "py": np.ones(num_pts_test) * co_coords["py"][0],
        "zeta": np.ones(num_pts_test) * co_coords["zeta"][0],
        "ptau": np.ones(num_pts_test) * co_coords["ptau"][0],
    }
    part_test = xp.Particles(
        p0c=twiss.particle_on_co.p0c,
        **coords_test,
    )
    line.track(
        part_test,
        num_turns=num_turns,
    )
    part_test.sort(interleave_lost_particles=True)
    max_y = coords_test["y"][np.min(np.where(part_test.state < 1)[0])]

    coords_test = {
        "x": np.asarray([co_coords["x"][0]] * 2),
        "px": np.asarray([co_coords["px"][0]] * 2),
        "y": np.asarray([co_coords["y"][0], max_y]),
        "py": np.asarray([co_coords["py"][0]] * 2),
        "zeta": np.asarray([co_coords["zeta"][0]] * 2),
        "ptau": np.asarray([co_coords["ptau"][0]] * 2),
        "at_element": np.zeros(2, dtype=int),
        "particle_id": np.arange(2, dtype=int),
    }
    coords_test_norm = get_normalised_coordinates_from_real(twiss, coords_test)
    max_amp_norm = np.sqrt(
        (coords_test_norm["y_norm"][1] - coords_test_norm["y_norm"][0]) ** 2
        + (coords_test_norm["py_norm"][1] - coords_test_norm["py_norm"][0]) ** 2
    )

    # Track several particles within limit of stability
    y_norm = np.hstack(
        (
            np.linspace(
                co_coords_norm["y_norm"][0],
                co_coords_norm["y_norm"][0] + max_amp_norm * np.cos(0),
                num_pts,
            ),
            np.linspace(
                co_coords_norm["y_norm"][0],
                co_coords_norm["y_norm"][0] + max_amp_norm * np.cos(np.pi),
                num_pts,
            ),
        )
    )
    py_norm = np.hstack(
        (
            np.linspace(
                co_coords_norm["py_norm"][0],
                co_coords_norm["py_norm"][0] + max_amp_norm * np.sin(0),
                num_pts,
            ),
            np.linspace(
                co_coords_norm["py_norm"][0],
                co_coords_norm["py_norm"][0] + max_amp_norm * np.sin(np.pi),
                num_pts,
            ),
        )
    )
    sort_idx = np.argsort(y_norm**2 + py_norm**2)

    coords_norm = {
        "x_norm": np.ones(2 * num_pts) * co_coords_norm["x_norm"][0],
        "px_norm": np.ones(2 * num_pts) * co_coords_norm["px_norm"][0],
        "y_norm": y_norm[sort_idx],
        "py_norm": py_norm[sort_idx],
        "zeta_norm": np.ones(2 * num_pts) * co_coords_norm["zeta_norm"][0],
        "pzeta_norm": np.ones(2 * num_pts) * co_coords_norm["pzeta_norm"][0],
        "at_element": np.zeros(2 * num_pts, dtype=int),
        "particle_id": np.arange(2 * num_pts, dtype=int),
    }
    coords = get_real_coordinates_from_normalised(twiss, coords_norm)
    part = xp.Particles(
        p0c=twiss.particle_on_co.p0c,
        **coords,
    )
    line.track(
        part,
        num_turns=num_turns,
        turn_by_turn_monitor=True,
    )
    part.sort(interleave_lost_particles=True)
    state = part.state
    at_turn = part.at_turn

    # Construct data to return
    orbit_points = {
        "x": line.record_last_track.x,
        "px": line.record_last_track.px,
        "y": line.record_last_track.y,
        "py": line.record_last_track.py,
        "zeta": line.record_last_track.zeta,
        "ptau": line.record_last_track.ptau,
        "pzeta": line.record_last_track.pzeta,
        "delta": line.record_last_track.delta,
        "at_element": line.record_last_track.at_element,
        "particle_id": line.record_last_track.particle_id,
    }

    orbit_points_norm = {
        "x_norm": np.zeros((2 * num_pts, num_turns)),
        "px_norm": np.zeros((2 * num_pts, num_turns)),
        "y_norm": np.zeros((2 * num_pts, num_turns)),
        "py_norm": np.zeros((2 * num_pts, num_turns)),
        "zeta_norm": np.zeros((2 * num_pts, num_turns)),
        "pzeta_norm": np.zeros((2 * num_pts, num_turns)),
    }
    for i in range(2 * num_pts):
        curr_coords = {
            "x": orbit_points["x"][i, :],
            "px": orbit_points["px"][i, :],
            "y": orbit_points["y"][i, :],
            "py": orbit_points["py"][i, :],
            "zeta": orbit_points["zeta"][i, :],
            "ptau": orbit_points["ptau"][i, :],
            "at_element": orbit_points["at_element"][i, :],
            "particle_id": orbit_points["particle_id"][i, :],
        }
        curr_coords_norm = get_normalised_coordinates_from_real(twiss, curr_coords)

        orbit_points_norm["x_norm"][i, :] = curr_coords_norm["x_norm"]
        orbit_points_norm["px_norm"][i, :] = curr_coords_norm["px_norm"]
        orbit_points_norm["y_norm"][i, :] = curr_coords_norm["y_norm"]
        orbit_points_norm["py_norm"][i, :] = curr_coords_norm["py_norm"]
        orbit_points_norm["zeta_norm"][i, :] = curr_coords_norm["zeta_norm"]
        orbit_points_norm["pzeta_norm"][i, :] = curr_coords_norm["pzeta_norm"]

    result = {
        "V_orbit_points": {
            "y": orbit_points["y"],
            "py": orbit_points["py"],
        },
        "V_orbit_points_norm": {
            "y_norm": orbit_points_norm["y_norm"],
            "py_norm": orbit_points_norm["py_norm"],
        },
    }
    lost_part_idx = np.where(state < 1)[0]
    for i in lost_part_idx:
        lost_turn = at_turn[i]
        result["V_orbit_points"]["y"][i, lost_turn:] = np.nan
        result["V_orbit_points"]["py"][i, lost_turn:] = np.nan
        result["V_orbit_points_norm"]["y_norm"][i, lost_turn:] = np.nan
        result["V_orbit_points_norm"]["py_norm"][i, lost_turn:] = np.nan

    return result


def _max_bucket_height(line: Line, twiss: TwissTable) -> float:
    """
    Function to compute analytically an approximate bucket height.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite

    Output:
        - a float representing the approximate maximum bucket height
    """

    beta0 = twiss.particle_on_co.beta0[0]
    E = twiss.particle_on_co.energy0[0]
    q = twiss.particle_on_co.q0
    eta = twiss.slip_factor

    cavities = line.get_elements_of_type(xt.Cavity)[1]
    freq = np.asarray([round(line[cav].frequency, 9) for cav in cavities])
    phi = np.array([line[cav].lag for cav in cavities]) * np.pi / 180.0
    V = np.array([line[cav].voltage for cav in cavities])
    L = line.get_length()

    h = freq * L / beta0 / sc.c

    bucket_heights = np.sqrt(
        abs(
            q
            * V
            * beta0**2
            / (np.pi * h * eta * E)
            * (2 * np.cos(phi) + (2 * phi - np.pi) * np.sin(phi))
        )
    )

    return np.max(bucket_heights[np.where(np.isfinite(bucket_heights))[0]])


def _get_L_orbit_points(
    line: Line,
    twiss: TwissTable,
    co_coords: dict[str, np.ndarray],
    num_pts: int,
    num_turns: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the longitudinal phase space.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite
        - co_coords: dictionary of closed orbit coordinates in real space
        - num_pts: integer, the number of initial conditions up to the limit of
          stability to track
        - num_turns: integer, number of turns to track for, default `2048`;
          note that it has to be large enough to assess the stability of
          initial conditions and to paint the phase space

    Output:
        - dictionary with the following structure:
          {
              "L_orbit_points":
              {
                  "zeta": np.ndarray((num_pts, num_turns)),
                  "ptau": np.ndarray((num_pts, num_turns)),
                  "pzeta": np.ndarray((num_pts, num_turns)),
                  "pdelta": np.ndarray((num_pts, num_turns)),
              },
              "L_orbit_points_norm":
              {
                  "zeta_norm": np.ndarray((num_pts, num_turns)),
                  "pzeta_norm": np.ndarray((num_pts, num_turns)),
              }
          }
    """

    # Get relativistic beta from twiss
    beta0 = twiss.particle_on_co.beta0[0]

    # Calculate closed orbit in normalised coordinates
    co_coords = {
        "x": np.asarray([twiss.x[0]]),
        "px": np.asarray([twiss.px[0]]),
        "y": np.asarray([twiss.y[0]]),
        "py": np.asarray([twiss.py[0]]),
        "zeta": np.asarray([twiss.zeta[0]]),
        "ptau": np.asarray([twiss.ptau[0]]),
        "at_element": np.asarray([0]),
        "particle_id": np.asarray([0]),
    }

    # Get max bucket height
    max_h = _max_bucket_height(line, twiss)

    # Track several particles mostly within bucket
    num_turns = 2048

    coords = {
        "x": np.ones(num_pts) * co_coords["x"][0],
        "px": np.ones(num_pts) * co_coords["px"][0],
        "y": np.ones(num_pts) * co_coords["y"][0],
        "py": np.ones(num_pts) * co_coords["py"][0],
        "zeta": np.ones(num_pts) * co_coords["zeta"][0],
        "pzeta": np.linspace(
            co_coords["ptau"][0], co_coords["ptau"][0] + max_h, num_pts
        )
        / beta0,
    }
    part = xp.Particles(
        p0c=twiss.particle_on_co.p0c,
        **coords,
    )
    line.track(
        part,
        num_turns=num_turns,
        turn_by_turn_monitor=True,
    )
    part.sort(interleave_lost_particles=True)
    state = part.state
    at_turn = part.at_turn

    # Construct data to return
    orbit_points = {
        "x": line.record_last_track.x,
        "px": line.record_last_track.px,
        "y": line.record_last_track.y,
        "py": line.record_last_track.py,
        "zeta": line.record_last_track.zeta,
        "ptau": line.record_last_track.ptau,
        "pzeta": line.record_last_track.pzeta,
        "delta": line.record_last_track.delta,
        "at_element": line.record_last_track.at_element,
        "particle_id": line.record_last_track.particle_id,
    }

    orbit_points_norm = {
        "x_norm": np.zeros((2 * num_pts, num_turns)),
        "px_norm": np.zeros((2 * num_pts, num_turns)),
        "y_norm": np.zeros((2 * num_pts, num_turns)),
        "py_norm": np.zeros((2 * num_pts, num_turns)),
        "zeta_norm": np.zeros((2 * num_pts, num_turns)),
        "pzeta_norm": np.zeros((2 * num_pts, num_turns)),
    }
    for i in range(num_pts):
        curr_coords = {
            "x": orbit_points["x"][i, :],
            "px": orbit_points["px"][i, :],
            "y": orbit_points["y"][i, :],
            "py": orbit_points["py"][i, :],
            "zeta": orbit_points["zeta"][i, :],
            "ptau": orbit_points["ptau"][i, :],
            "at_element": orbit_points["at_element"][i, :],
            "particle_id": orbit_points["particle_id"][i, :],
        }
        curr_coords_norm = get_normalised_coordinates_from_real(twiss, curr_coords)

        orbit_points_norm["x_norm"][i, :] = curr_coords_norm["x_norm"]
        orbit_points_norm["px_norm"][i, :] = curr_coords_norm["px_norm"]
        orbit_points_norm["y_norm"][i, :] = curr_coords_norm["y_norm"]
        orbit_points_norm["py_norm"][i, :] = curr_coords_norm["py_norm"]
        orbit_points_norm["zeta_norm"][i, :] = curr_coords_norm["zeta_norm"]
        orbit_points_norm["pzeta_norm"][i, :] = curr_coords_norm["pzeta_norm"]

    result = {
        "L_orbit_points": {
            "zeta": orbit_points["zeta"],
            "pzeta": orbit_points["pzeta"],
            "ptau": orbit_points["ptau"],
            "delta": orbit_points["delta"],
        },
        "L_orbit_points_norm": {
            "zeta_norm": orbit_points_norm["zeta_norm"],
            "pzeta_norm": orbit_points_norm["pzeta_norm"],
        },
    }
    lost_part_idx = np.where(state < 1)[0]
    for i in lost_part_idx:
        lost_turn = at_turn[i]
        result["L_orbit_points"]["zeta"][i, lost_turn:] = np.nan
        result["L_orbit_points"]["pzeta"][i, lost_turn:] = np.nan
        result["L_orbit_points"]["ptau"][i, lost_turn:] = np.nan
        result["L_orbit_points"]["delta"][i, lost_turn:] = np.nan
        result["L_orbit_points_norm"]["zeta_norm"][i, lost_turn:] = np.nan
        result["L_orbit_points_norm"]["pzeta_norm"][i, lost_turn:] = np.nan

    return result


def get_orbit_points(
    line: Line,
    element: str,
    planes: str,
    num_pts: int,
    num_turns: int = 2048,
    delta0: float = 0.0,
    co_guess: dict[str, float] = {
        "x": 0.0,
        "px": 0.0,
        "y": 0.0,
        "py": 0.0,
        "zeta": 0.0,
        "ptau": 0.0,
    },
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the phase space.

    Input:
        - line: xsuite line
        - element: name of the element at which phase space is desired
        - planes: string indicating which plains to obatin the orbits in; can
          be 'H' for horizontal, 'V' for vertical, 'L' for longitudinal, 'HV'
          for both transverse planes, or 'HVL' for all three planes; orbits in
          a given plain correspond to particles on the closed orbit in all
          other planes
        - num_pts: integer, the number of initial conditions up to the limit of
          stability to track
        - num_turns: integer, number of turns to track for, default `2048`;
          note that it has to be large enough to assess the stability of
          initial conditions and to paint the phase space
        - delta0: float, the momentum offset of the reference particle,
          default `0`, will only be considered if the requested planes are
          transverse only
        - co_guess: dictionary containing the closed orbit guess in case it is
          different from 0, default `{'x': 0.0, 'px': 0.0, 'y': 0.0, 'py': 0.0,
          'zeta': 0.0, 'ptau': 0.0}`

    Output:
        - dictionary with the following structure:
          {
              "H_orbit_points":
              {
                  "x": np.ndarray((2 x num_pts, num_turns)),
                  "px": np.ndarray((2 x num_pts, num_turns)),
              },
              "H_orbit_points_norm":
              {
                  "x_norm": np.ndarray((2 x num_pts, num_turns)),
                  "px_norm": np.ndarray((2 x num_pts, num_turns)),
              },
              "V_orbit_points":
              {
                  "y": np.ndarray((2 x num_pts, num_turns)),
                  "py": np.ndarray((2 x num_pts, num_turns)),
              },
              "V_orbit_points_norm":
              {
                  "y_norm": np.ndarray((2 x num_pts, num_turns)),
                  "py_norm": np.ndarray((2 x num_pts, num_turns)),
              },
              "L_orbit_points":
              {
                  "zeta": np.ndarray((2 x num_pts, num_turns)),
                  "ptau": np.ndarray((2 x num_pts, num_turns)),
                  "pzeta": np.ndarray((2 x num_pts, num_turns)),
                  "delta": np.ndarray((2 x num_pts, num_turns)),
              },
              "L_orbit_points_norm":
              {
                  "zeta_norm": np.ndarray((2 x num_pts, num_turns)),
                  "pzeta_norm": np.ndarray((2 x num_pts, num_turns)),
              },
          }
          in case 'HVL' planes are requested, otherwise the data for the
          not requested plane(s) is omitted
    """

    # Copy line
    line_int = copy.deepcopy(line)
    line_int.build_tracker()

    # Twiss
    if planes not in ["H", "V", "L", "HV", "HVL"]:
        raise ValueError(
            "Incorrect plane requested! Must be 'H', 'V', 'L', 'HV' or 'HVL'."
        )
    elif planes in ["H", "V", "HV"]:
        twiss_bc = line_int.twiss(
            continue_on_closed_orbit_error=False, delta0=delta0, co_guess=co_guess
        )
    else:
        twiss_bc = line_int.twiss(
            continue_on_closed_orbit_error=False, co_guess=co_guess
        )

    # Cycle line to requested element
    line_int.cycle(name_first_element=element, inplace=True)

    # Twiss
    if planes in ["H", "V", "HV"]:
        try:
            twiss = line_int.twiss(continue_on_closed_orbit_error=False, delta0=delta0)
        except ValueError:
            part_on_co_at_ele = xt.Particles(
                p0c=twiss_bc.particle_on_co.p0c,
                x=twiss_bc.x[twiss_bc.name == element],
                px=twiss_bc.px[twiss_bc.name == element],
                y=twiss_bc.y[twiss_bc.name == element],
                py=twiss_bc.py[twiss_bc.name == element],
                zeta=twiss_bc.zeta[twiss_bc.name == element],
                ptau=twiss_bc.ptau[twiss_bc.name == element],
            )
            twiss = line_int.twiss(
                continue_on_closed_orbit_error=False,
                delta0=delta0,
                particle_on_co=part_on_co_at_ele,
            )
    else:
        try:
            twiss = line_int.twiss(continue_on_closed_orbit_error=False)
        except ValueError:
            part_on_co_at_ele = xt.Particles(
                p0c=twiss_bc.particle_on_co.p0c,
                x=twiss_bc.x[twiss_bc.name == element],
                px=twiss_bc.px[twiss_bc.name == element],
                y=twiss_bc.y[twiss_bc.name == element],
                py=twiss_bc.py[twiss_bc.name == element],
                zeta=twiss_bc.zeta[twiss_bc.name == element],
                ptau=twiss_bc.ptau[twiss_bc.name == element],
            )
            twiss = line_int.twiss(
                continue_on_closed_orbit_error=False,
                delta0=delta0,
                particle_on_co=part_on_co_at_ele,
            )

    # Calculate closed orbit in normalised coordinates
    co_coords = {
        "x": np.asarray([twiss.x[0]]),
        "px": np.asarray([twiss.px[0]]),
        "y": np.asarray([twiss.y[0]]),
        "py": np.asarray([twiss.py[0]]),
        "zeta": np.asarray([twiss.zeta[0]]),
        "ptau": np.asarray([twiss.ptau[0]]),
        "at_element": np.asarray([0]),
        "particle_id": np.asarray([0]),
    }
    co_coords_norm = get_normalised_coordinates_from_real(twiss, co_coords)

    orbits_H = {}
    orbits_V = {}
    orbits_L = {}
    if "H" in planes:
        orbits_H = _get_H_orbit_points(
            line_int, twiss, co_coords, co_coords_norm, num_pts, num_turns
        )
    if "V" in planes:
        orbits_V = _get_V_orbit_points(
            line_int, twiss, co_coords, co_coords_norm, num_pts, num_turns
        )
    if "L" in planes:
        orbits_L = _get_L_orbit_points(line_int, twiss, co_coords, num_pts, num_turns)

    orbits = orbits_H | orbits_V | orbits_L

    return orbits
