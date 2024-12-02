from typing import List, Optional, Tuple, Union

import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw
from tqdm.auto import tqdm

from ...tools.generic_writer import GenericWriter, LocalWriter
from ...tools.normed_particles import NormedParticles


def reverse_error_method(
    part: xp.Particles,
    turns_to_sample: Union[List[int], np.ndarray],
    line: xt.Line,
    out: Optional[GenericWriter] = None,
    save_all: bool = False,
    twiss: Optional[xtw.TwissTable] = None,
    nemitt: Optional[Tuple[float, float]] = None,
    _context: Union[
        xo.ContextCpu, xo.ContextCupy, xo.ContextPyopencl
    ] = xo.ContextCpu(),
    force_backtrack: bool = False,
    overwrite: bool = True,
):
    """Evaluate the reverse error method for the given values of turns.

    Parameters
    ----------
    part : xp.Particles
        Particle object to be used as reference.
    turns_to_sample : List[int]
        List of turns to be used for the study.
    line : xt.Line
        Line to be used for the study.
    out : GenericWriter, optional
        Writer object to store the results. By default None. If None, the
        results will be saved in a LocalWriter (i.e., a dictionary).
    save_all : bool, optional
        If True, the full particle distribution is saved at the time samples, by default False
    twiss : xtw.TwissTable
        Twiss table of the line to be used for the normalization if provided,
        by default None
    nemitt : Tuple[float, float]
        Normalized emittance in the horizontal and vertical planes, to be
        provided if twiss is provided, by default None
    _context : xo.Context, optional
        xobjects context to be used, by default xo.ContextCPU()
    force_backtrack : bool, optional
        If True, the particles are forced to be backtracked, even if the line does
        not fully support backtracking, by default False
    overwrite : bool, optional
        If True, the output file is overwritten, by default True

    Returns
    -------
    out : GenericWriter
        Writer object containing the results of the study.
    """
    if out is None:
        out = LocalWriter("out")

    turns_to_sample = np.sort(np.unique(turns_to_sample))
    f_part = part.copy()

    if twiss is not None:
        norm_f_part = NormedParticles(
            twiss=twiss,
            nemitt_x=nemitt[0],
            nemitt_y=nemitt[1],
            _context=_context,
            part=f_part,
        )

    particle_idx = _context.nparray_from_context_array(f_part.particle_id)
    argsort_reference = np.argsort(particle_idx)

    # Create a list of attributes to iterate over
    attributes_norm = [
        "x_norm",
        "px_norm",
        "y_norm",
        "py_norm",
        "zeta_norm",
        "pzeta_norm",
    ]
    attributes_phys = ["x", "px", "y", "py", "zeta", "ptau"]

    if twiss is not None:
        for attr in attributes_norm:
            out.write_data(
                f"initial/{attr}",
                _context.nparray_from_context_array(getattr(norm_f_part, attr))[
                    argsort_reference
                ],
            )

    for attr in attributes_phys:
        out.write_data(
            f"initial/{attr}",
            _context.nparray_from_context_array(getattr(f_part, attr))[
                argsort_reference
            ],
        )

    current_t = 0

    pbar = tqdm(total=np.sum(turns_to_sample) * 2)
    for i, t in enumerate(turns_to_sample):
        delta_t = t - current_t
        line.track(f_part, num_turns=delta_t)
        current_t = t
        pbar.update(delta_t)
        r_part = f_part.copy()

        if force_backtrack:
            line.track(r_part, num_turns=t, backtrack="force")
        else:
            line.track(r_part, num_turns=t, backtrack=True)
        pbar.update(t)

        idx_r = _context.nparray_from_context_array(r_part.particle_id)
        argsort_backward = np.argsort(idx_r)

        if twiss is not None:
            norm_r_part = NormedParticles(
                twiss=twiss,
                nemitt_x=nemitt[0],
                nemitt_y=nemitt[1],
                _context=_context,
                part=r_part,
            )

            rem_norm = np.sqrt(
                (
                    norm_f_part.x_norm[argsort_reference]
                    - norm_r_part.x_norm[argsort_backward]
                )
                ** 2
                + (
                    norm_f_part.px_norm[argsort_reference]
                    - norm_r_part.px_norm[argsort_backward]
                )
                ** 2
                + (
                    norm_f_part.y_norm[argsort_reference]
                    - norm_r_part.y_norm[argsort_backward]
                )
                ** 2
                + (
                    norm_f_part.py_norm[argsort_reference]
                    - norm_r_part.py_norm[argsort_backward]
                )
                ** 2
                + (
                    norm_f_part.zeta_norm[argsort_reference]
                    - norm_r_part.zeta_norm[argsort_backward]
                )
                ** 2
                + (
                    norm_f_part.pzeta_norm[argsort_reference]
                    - norm_r_part.pzeta_norm[argsort_backward]
                )
                ** 2
            )

        rem = np.sqrt(
            (part.x[argsort_reference] - r_part.x[argsort_backward]) ** 2
            + (part.px[argsort_reference] - r_part.px[argsort_backward]) ** 2
            + (part.y[argsort_reference] - r_part.y[argsort_backward]) ** 2
            + (part.py[argsort_reference] - r_part.py[argsort_backward]) ** 2
            + (part.zeta[argsort_reference] - r_part.zeta[argsort_backward]) ** 2
            + (part.ptau[argsort_reference] - r_part.ptau[argsort_backward]) ** 2
        )

        lost_mask = r_part.state[argsort_backward] <= 0
        if twiss is not None:
            rem_norm[lost_mask] = np.nan
        rem[lost_mask] = np.nan

        if save_all:

            if twiss is not None:
                for attr in attributes_norm:
                    data = _context.nparray_from_context_array(
                        getattr(norm_r_part, attr)
                    )[argsort_backward]
                    data[lost_mask] = np.nan
                    out.write_data(
                        f"forward-backward/{t}/{attr}",
                        data,
                        overwrite=overwrite,
                    )

            for attr in attributes_phys:
                data = _context.nparray_from_context_array(getattr(r_part, attr))[
                    argsort_backward
                ]
                data[lost_mask] = np.nan
                out.write_data(
                    f"forward-backward/{t}/{attr}",
                    data,
                    overwrite=overwrite,
                )
        if twiss is not None:
            out.write_data(
                f"rem_norm/{t}",
                _context.nparray_from_context_array(rem_norm),
                overwrite=overwrite,
            )
        out.write_data(
            f"rem/{t}", _context.nparray_from_context_array(rem), overwrite=overwrite
        )

    # save final at_turn of the forward particles
    idx_f = _context.nparray_from_context_array(f_part.particle_id)
    argsort_forward = np.argsort(idx_f)
    out.write_data(
        "at_turn",
        _context.nparray_from_context_array(f_part.at_turn)[argsort_forward],
        overwrite=overwrite,
    )
    out.write_data(
        "state",
        _context.nparray_from_context_array(f_part.state)[argsort_forward],
        overwrite=overwrite,
    )

    return out
