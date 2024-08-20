import copy
from typing import Union

import numpy as np
import scipy.constants as sc  # type: ignore[import-untyped]
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]
from xtrack import Line  # type: ignore[import-untyped]
from xtrack.twiss import TwissTable  # type: ignore[import-untyped]

from xnlbd.tools import NormedParticles


def _get_H_orbit_points(
    line: Line,
    twiss: TwissTable,
    nemitt_x: float,
    nemitt_y: float,
    nemitt_z: float,
    part_on_co: xt.Particles,
    part_on_co_norm: NormedParticles,
    part,
    part_norm,
    num_pts: int,
    num_turns: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the horizontal phase space.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite
        - nemitt_x: normalised emittance in horizontal plane
        - nemitt_y: normalised emittance in vertical plane
        - nemitt_z: normalised emittance in longitudinal plane
        - part_on_co: xsuite particle object representing a particle on the
          closed orbit in real space
        - part_on_co_norm: xsuite particle object representing a particle on the
          closed orbit in normalised space
        - part: xsuite particle object for tracking
        - part_norm: NormedParticles object for normalising
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
    x_test = np.linspace(
        part_on_co.x[0], part_on_co.x[0] + line.config.XTRACK_GLOBAL_XY_LIMIT, num_pts
    )
    part.x = x_test
    part_copy = copy.deepcopy(part)
    line.track(
        part,
        num_turns=num_turns,
    )
    part.sort(interleave_lost_particles=True)
    try:
        max_x_idx = np.where(part.state < 1)[0][0]
    except IndexError:
        max_x_idx = len(x_test) - 1
    part_norm.phys_to_norm(part_copy)
    max_amp_norm = np.sqrt(
        (part_norm.x_norm[max_x_idx] - part_norm.x_norm[0]) ** 2
        + (part_norm.px_norm[max_x_idx] - part_norm.px_norm[0]) ** 2
    )

    # Track several particles within limit of stability
    x_norm = np.hstack(
        (
            np.linspace(
                part_on_co_norm.x_norm[0],
                part_on_co_norm.x_norm[0] + max_amp_norm * np.cos(0),
                int(num_pts / 2.0),
            ),
            np.linspace(
                part_on_co_norm.x_norm[0],
                part_on_co_norm.x_norm[0] + max_amp_norm * np.cos(np.pi),
                num_pts - int(num_pts / 2.0),
            ),
        )
    )
    px_norm = np.hstack(
        (
            np.linspace(
                part_on_co_norm.px_norm[0],
                part_on_co_norm.px_norm[0] + max_amp_norm * np.sin(0),
                int(num_pts / 2.0),
            ),
            np.linspace(
                part_on_co_norm.px_norm[0],
                part_on_co_norm.px_norm[0] + max_amp_norm * np.sin(np.pi),
                num_pts - int(num_pts / 2.0),
            ),
        )
    )
    sort_idx = np.argsort(x_norm**2 + px_norm**2)
    part_norm.x_norm = x_norm[sort_idx]
    part_norm.px_norm = px_norm[sort_idx]
    part = part_norm.norm_to_phys(part_copy)
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
        "x_norm": np.zeros((num_pts, num_turns)),
        "px_norm": np.zeros((num_pts, num_turns)),
        "y_norm": np.zeros((num_pts, num_turns)),
        "py_norm": np.zeros((num_pts, num_turns)),
        "zeta_norm": np.zeros((num_pts, num_turns)),
        "pzeta_norm": np.zeros((num_pts, num_turns)),
    }
    for i in range(num_pts):
        curr_part = xt.Particles(
            p0c=part_on_co.p0c,
            x=orbit_points["x"][i, :],
            px=orbit_points["px"][i, :],
            y=orbit_points["y"][i, :],
            py=orbit_points["py"][i, :],
            zeta=orbit_points["zeta"][i, :],
            ptau=orbit_points["ptau"][i, :],
        )
        curr_part_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=curr_part,
        )

        orbit_points_norm["x_norm"][i, :] = curr_part_norm.x_norm
        orbit_points_norm["px_norm"][i, :] = curr_part_norm.px_norm
        orbit_points_norm["y_norm"][i, :] = curr_part_norm.y_norm
        orbit_points_norm["py_norm"][i, :] = curr_part_norm.py_norm
        orbit_points_norm["zeta_norm"][i, :] = curr_part_norm.zeta_norm
        orbit_points_norm["pzeta_norm"][i, :] = curr_part_norm.pzeta_norm

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
    nemitt_x: float,
    nemitt_y: float,
    nemitt_z: float,
    part_on_co: xt.Particles,
    part_on_co_norm: NormedParticles,
    part,
    part_norm,
    num_pts: int,
    num_turns: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the vertical phase space.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite
        - nemitt_x: normalised emittance in horizontal plane
        - nemitt_y: normalised emittance in vertical plane
        - nemitt_z: normalised emittance in longitudinal plane
        - part_on_co: xsuite particle object representing a particle on the
          closed orbit in real space
        - part_on_co_norm: xsuite particle object representing a particle on the
          closed orbit in normalised space
        - part: xsuite particle object for tracking
        - part_norm: NormedParticles object for normalising
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
    y_test = np.linspace(
        part_on_co.y[0], part_on_co.y[0] + line.config.XTRACK_GLOBAL_XY_LIMIT, num_pts
    )
    part.y = y_test
    part_copy = copy.deepcopy(part)
    line.track(
        part,
        num_turns=num_turns,
    )
    part.sort(interleave_lost_particles=True)
    try:
        max_y_idx = np.where(part.state < 1)[0][0]
    except IndexError:
        max_y_idx = len(y_test) - 1
    part_norm.phys_to_norm(part_copy)
    max_amp_norm = np.sqrt(
        (part_norm.y_norm[max_y_idx] - part_norm.y_norm[0]) ** 2
        + (part_norm.py_norm[max_y_idx] - part_norm.py_norm[0]) ** 2
    )

    # Track several particles within limit of stability
    y_norm = np.hstack(
        (
            np.linspace(
                part_on_co_norm.y_norm[0],
                part_on_co_norm.y_norm[0] + max_amp_norm * np.cos(0),
                int(num_pts / 2.0),
            ),
            np.linspace(
                part_on_co_norm.y_norm[0],
                part_on_co_norm.y_norm[0] + max_amp_norm * np.cos(np.pi),
                num_pts - int(num_pts / 2.0),
            ),
        )
    )
    py_norm = np.hstack(
        (
            np.linspace(
                part_on_co_norm.py_norm[0],
                part_on_co_norm.py_norm[0] + max_amp_norm * np.sin(0),
                int(num_pts / 2.0),
            ),
            np.linspace(
                part_on_co_norm.py_norm[0],
                part_on_co_norm.py_norm[0] + max_amp_norm * np.sin(np.pi),
                num_pts - int(num_pts / 2.0),
            ),
        )
    )
    sort_idx = np.argsort(y_norm**2 + py_norm**2)
    part_norm.y_norm = y_norm[sort_idx]
    part_norm.py_norm = py_norm[sort_idx]
    part = part_norm.norm_to_phys(part_copy)
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
        "x_norm": np.zeros((num_pts, num_turns)),
        "px_norm": np.zeros((num_pts, num_turns)),
        "y_norm": np.zeros((num_pts, num_turns)),
        "py_norm": np.zeros((num_pts, num_turns)),
        "zeta_norm": np.zeros((num_pts, num_turns)),
        "pzeta_norm": np.zeros((num_pts, num_turns)),
    }
    for i in range(num_pts):
        curr_part = xt.Particles(
            p0c=part_on_co.p0c,
            x=orbit_points["x"][i, :],
            px=orbit_points["px"][i, :],
            y=orbit_points["y"][i, :],
            py=orbit_points["py"][i, :],
            zeta=orbit_points["zeta"][i, :],
            ptau=orbit_points["ptau"][i, :],
        )
        curr_part_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=curr_part,
        )

        orbit_points_norm["x_norm"][i, :] = curr_part_norm.x_norm
        orbit_points_norm["px_norm"][i, :] = curr_part_norm.px_norm
        orbit_points_norm["y_norm"][i, :] = curr_part_norm.y_norm
        orbit_points_norm["py_norm"][i, :] = curr_part_norm.py_norm
        orbit_points_norm["zeta_norm"][i, :] = curr_part_norm.zeta_norm
        orbit_points_norm["pzeta_norm"][i, :] = curr_part_norm.pzeta_norm

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
    nemitt_x: float,
    nemitt_y: float,
    nemitt_z: float,
    part_on_co: xt.Particles,
    part,
    num_pts: int,
    num_turns: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Function to track particles and record their coordinates every turn to
    obtain a representation of the longitudinal phase space.

    Input:
        - line: xsuite line
        - twiss: Twiss table from xsuite
        - nemitt_x: normalised emittance in horizontal plane
        - nemitt_y: normalised emittance in vertical plane
        - nemitt_z: normalised emittance in longitudinal plane
        - part_on_co: xsuite particle object representing a particle on the
          closed orbit in real space
        - part: xsuite particle object for tracking
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

    # Get max bucket height
    max_h = _max_bucket_height(line, twiss)

    # Track several particles mostly within bucket
    part.ptau = np.linspace(part_on_co.ptau[0], part_on_co.ptau[0] + max_h, num_pts)
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
        "x_norm": np.zeros((num_pts, num_turns)),
        "px_norm": np.zeros((num_pts, num_turns)),
        "y_norm": np.zeros((num_pts, num_turns)),
        "py_norm": np.zeros((num_pts, num_turns)),
        "zeta_norm": np.zeros((num_pts, num_turns)),
        "pzeta_norm": np.zeros((num_pts, num_turns)),
    }
    for i in range(num_pts):
        curr_part = xt.Particles(
            p0c=part_on_co.p0c,
            x=orbit_points["x"][i, :],
            px=orbit_points["px"][i, :],
            y=orbit_points["y"][i, :],
            py=orbit_points["py"][i, :],
            zeta=orbit_points["zeta"][i, :],
            ptau=orbit_points["ptau"][i, :],
        )
        curr_part_norm = NormedParticles(
            twiss,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            nemitt_z=nemitt_z,
            part=curr_part,
        )

        orbit_points_norm["x_norm"][i, :] = curr_part_norm.x_norm
        orbit_points_norm["px_norm"][i, :] = curr_part_norm.px_norm
        orbit_points_norm["y_norm"][i, :] = curr_part_norm.y_norm
        orbit_points_norm["py_norm"][i, :] = curr_part_norm.py_norm
        orbit_points_norm["zeta_norm"][i, :] = curr_part_norm.zeta_norm
        orbit_points_norm["pzeta_norm"][i, :] = curr_part_norm.pzeta_norm

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
    co_guess: Union[dict[str, float], None] = None,
    nemitt_x: float = 1e-6,
    nemitt_y: float = 1e-6,
    nemitt_z: float = 1,
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
          different from 0, default `None`
        - nemitt_x: normalised emittance in horizontal plane, default `1e-6`
        - nemitt_y: normalised emittance in vertical plane, default `1e-6`
        - nemitt_z: normalised emittance in longitudinal plane, default `1`

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

    # Set closed orbit guess
    if co_guess is None:
        co_guess = {
            "x": 0.0,
            "px": 0.0,
            "y": 0.0,
            "py": 0.0,
            "zeta": 0.0,
            "ptau": 0.0,
        }

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
    part_on_co = xt.Particles(
        p0c=twiss.particle_on_co.p0c,
        x=twiss.x[0],
        px=twiss.px[0],
        y=twiss.y[0],
        py=twiss.py[0],
        zeta=twiss.zeta[0],
        ptau=twiss.ptau[0],
    )
    part_on_co_norm = NormedParticles(
        twiss, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=part_on_co
    )

    orbits_H = {}
    orbits_V = {}
    orbits_L = {}
    if "H" in planes:
        part = xt.Particles(
            p0c=twiss.particle_on_co.p0c,
            x=np.ones(num_pts) * twiss.x[0],
            px=np.ones(num_pts) * twiss.px[0],
            y=np.ones(num_pts) * twiss.y[0],
            py=np.ones(num_pts) * twiss.py[0],
            zeta=np.ones(num_pts) * twiss.zeta[0],
            ptau=np.ones(num_pts) * twiss.ptau[0],
        )
        part_norm = NormedParticles(
            twiss, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=part
        )

        orbits_H = _get_H_orbit_points(
            line_int,
            twiss,
            nemitt_x,
            nemitt_y,
            nemitt_z,
            part_on_co,
            part_on_co_norm,
            part,
            part_norm,
            num_pts,
            num_turns,
        )
    if "V" in planes:
        part = xt.Particles(
            p0c=twiss.particle_on_co.p0c,
            x=np.ones(num_pts) * twiss.x[0],
            px=np.ones(num_pts) * twiss.px[0],
            y=np.ones(num_pts) * twiss.y[0],
            py=np.ones(num_pts) * twiss.py[0],
            zeta=np.ones(num_pts) * twiss.zeta[0],
            ptau=np.ones(num_pts) * twiss.ptau[0],
        )
        part_norm = NormedParticles(
            twiss, nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z, part=part
        )

        orbits_V = _get_V_orbit_points(
            line_int,
            twiss,
            nemitt_x,
            nemitt_y,
            nemitt_z,
            part_on_co,
            part_on_co_norm,
            part,
            part_norm,
            num_pts,
            num_turns,
        )
    if "L" in planes:
        part = xt.Particles(
            p0c=twiss.particle_on_co.p0c,
            x=np.ones(num_pts) * twiss.x[0],
            px=np.ones(num_pts) * twiss.px[0],
            y=np.ones(num_pts) * twiss.y[0],
            py=np.ones(num_pts) * twiss.py[0],
            zeta=np.ones(num_pts) * twiss.zeta[0],
            ptau=np.ones(num_pts) * twiss.ptau[0],
        )

        orbits_L = _get_L_orbit_points(
            line_int,
            twiss,
            nemitt_x,
            nemitt_y,
            nemitt_z,
            part_on_co,
            part,
            num_pts,
            num_turns,
        )

    orbits = orbits_H | orbits_V | orbits_L

    return orbits
