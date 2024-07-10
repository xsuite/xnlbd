import warnings

import numpy as np
import xtrack as xt
from tqdm.auto import tqdm

from .generic_writer import GenericWriter
from .ghost_particle_manager import GhostParticleManager
from .tools import birkhoff_weights


def track_displacement(
    gpm: GhostParticleManager,
    line: xt.Line,
    turns_to_sample,
    out: GenericWriter,
    renorm_frequency=100,
    renorm_module=1e-6,
    tqdm_flag=True,
    overwrite=True,
):
    """Track FLI/n and displacement direction of a given set of ghost particles.

    Parameters
    ----------
    line : xt.Line
        The line to track
    turns_to_sample : list
        List of the turns to sample the displacement
    out : GenericWriter
        The writer to write the data
    renorm_frequency : int, optional
        The frequency of distance renormalizations, by default 100
    renorm_module : float, optional
        The custom module to use for renormalizations, by default 1e-6
    tqdm_flag : bool, optional
        If True, show the progress bar, by default True
    overwrite : bool, optional
        If True, overwrite the data in the writer, by default True
    """
    # gpm.save_metadata(out)

    turns_to_sample = np.sort(np.unique(np.asarray(turns_to_sample, dtype=int)))
    max_turn = np.max(turns_to_sample)

    renormalization_turns = np.arange(0, max_turn + 1, renorm_frequency)[1:]

    s_events = [("sample", t, i) for i, t in enumerate(turns_to_sample)]
    r_events = [("renormalize", t, i) for i, t in enumerate(renormalization_turns)]
    events = sorted(
        s_events + r_events, key=lambda x: x[1] + 0.5 if x[0] == "renormalize" else x[1]
    )

    # TODO: check if this implementation ruins GPU performance
    log_module_storage = gpm._context.nplike_array_type(
        (len(gpm._ghost_name), len(gpm._part.particle_id))
    )
    log_module_storage[:] = 0.0

    current_turn = 0
    pbar = tqdm(total=max_turn, disable=not tqdm_flag)
    for event, turn, event_idx in events:
        delta_turn = turn - current_turn
        if delta_turn > 0:
            for part in gpm.particles():
                line.track(part, num_turns=delta_turn)
            current_turn = turn
            pbar.update(delta_turn)

        if event == "sample":
            # Pre-compute normalized coordinates prefix based on gpm._use_norm_coord
            coord_prefix = "norm" if gpm._use_norm_coord else ""

            # Create a mapping of coordinate suffixes and their corresponding attributes
            suffixes_and_attributes = [
                ("x", "displacement_x", "displacement_x_norm"),
                ("px", "displacement_px", "displacement_px_norm"),
                ("y", "displacement_y", "displacement_y_norm"),
                ("py", "displacement_py", "displacement_py_norm"),
                ("zeta", "displacement_zeta", "displacement_zeta_norm"),
                (
                    "ptau" if not coord_prefix else "pzeta",
                    "displacement_ptau",
                    "displacement_pzeta_norm",
                ),
            ]

            for i, (
                stored_log_module,
                name,
            ) in enumerate(
                zip(
                    log_module_storage,
                    gpm.yield_module_and_direction(),
                )
            ):
                log_module_to_save = np.log10(gpm.module / renorm_module) + (
                    stored_log_module
                )
                out.write_data(
                    f"fli_over_n/{name}/{current_turn}",
                    gpm._context.nparray_from_context_array(
                        log_module_to_save / current_turn
                    ),
                    overwrite=overwrite,
                )

                # Use a generator to handle writing the direction data
                for suffix, attr, norm_attr in suffixes_and_attributes:
                    full_suffix = (
                        f"{suffix}_{coord_prefix}"
                        if coord_prefix and suffix != "ptau"
                        else suffix
                    )
                    value = (
                        getattr(gpm, norm_attr)
                        if gpm._use_norm_coord
                        else getattr(gpm, attr)
                    )
                    out.write_data(
                        f"direction/{name}/{full_suffix}/{current_turn}",
                        gpm._context.nparray_from_context_array(value),
                        overwrite=overwrite,
                    )

        elif event == "renormalize":
            for i, (stored_log_module, names) in enumerate(
                zip(log_module_storage, gpm.renormalize_distances_and_yield())
            ):
                stored_log_module += np.log10(gpm.module / renorm_module)

    # save some infos of main particles
    # note: they need sorting
    particle_argsort = np.argsort(gpm._part.particle_id)
    out.write_data(
        "ref_particles_data/at_turn",
        gpm._context.nparray_from_context_array(gpm.part.at_turn)[particle_argsort],
        overwrite=overwrite,
    )
    out.write_data(
        "ref_particles_data/state",
        gpm._context.nparray_from_context_array(gpm.part.state)[particle_argsort],
        overwrite=overwrite,
    )


def track_displacement_birkhoff(
    gpm: GhostParticleManager,
    line: xt.Line,
    turns_to_sample,
    out: GenericWriter,
    renorm_frequency=100,
    renorm_module=None,
    tqdm_flag=True,
    overwrite=True,
):
    """Track the displacement and direction of the ghost particles while using
    the birkhoff weights for the displacement.

    Parameters
    ----------
    line : xt.Line
        The line to track
    turns_to_sample : list
        List of the turns to sample the displacement
    out : GenericWriter
        The writer to write the data
    renorm_frequency : int, optional
        The frequency of realignment, by default 100
    renorm_module : float, optional
        The module to use for realignment, if None, the default module
        set in the GhostParticleManager will be used, by default None
    tqdm_flag : bool, optional
        If True, show the progress bar, by default True
    overwrite : bool, optional
        If True, overwrite the data in the writer, by default True
    """
    if np.any(np.asarray(turns_to_sample, dtype=int) % renorm_frequency != 0):
        warnings.warn(
            "Some of the sampling turns are not multiple of the realign frequency.\n"
            + "The values will be rounded down to the closest multiple.",
            category=UserWarning,
        )

    turns_to_sample = np.unique(
        (((np.asarray(turns_to_sample, dtype=int))) // renorm_frequency)
        * renorm_frequency
    )
    max_turn = np.max(turns_to_sample)

    n_realignments = turns_to_sample // renorm_frequency
    renormalization_turns = np.arange(0, max_turn + 1, renorm_frequency)[1:]

    s_events = [("sample", t, i) for i, t in enumerate(turns_to_sample)]
    r_events = [("renormalize", t, i) for i, t in enumerate(renormalization_turns)]
    events = sorted(
        s_events + r_events, key=lambda x: x[1] + 0.5 if x[0] == "renormalize" else x[1]
    )

    birk_weights_list = [
        gpm._context.nparray_to_context_array(birkhoff_weights(t))
        for t in n_realignments
    ]
    birk_log_module_storage = gpm._context.nplike_array_type(
        (len(turns_to_sample), len(gpm._ghost_name), len(gpm.part.particle_id))
    )
    birk_log_module_storage[:] = 0.0

    # Elements for the no birkhoff case
    log_module_storage = gpm._context.nplike_array_type(
        (len(gpm._ghost_name), len(gpm.part.particle_id))
    )
    log_module_storage[:] = 0.0

    current_turn = 0
    pbar = tqdm(total=max_turn, disable=not tqdm_flag)
    for event, turn, event_idx in events:
        delta_turn = turn - current_turn
        if delta_turn > 0:
            for part in gpm.particles():
                line.track(part, num_turns=delta_turn)
            current_turn = turn
            pbar.update(delta_turn)

        if event == "renormalize":
            for i, name in enumerate(gpm.renormalize_distances_and_yield()):
                log_module_storage[i] += np.log10(gpm.module / renorm_module)

                for s_idx, sample in enumerate(turns_to_sample):
                    if current_turn <= sample:
                        birk_log_module_storage[s_idx][i] += (
                            np.log10(gpm.module / renorm_module)
                            * birk_weights_list[s_idx][event_idx]
                            / renorm_frequency
                        )

        elif event == "sample":
            # due to our forced normalization of sampling turns, we can
            # directly save the values stored in the birk_log_module_storage
            # and in the log_module_storage
            _, direction_list = gpm.get_module_and_direction()

            s_idx = np.where(turns_to_sample == current_turn)[0][0]

            # Pre-compute normalized coordinates prefix based on gpm._use_norm_coord
            coord_prefix = "norm" if gpm._use_norm_coord else ""

            # Create a mapping of coordinate suffixes
            suffixes = [
                "x",
                "px",
                "y",
                "py",
                "zeta",
                "ptau" if coord_prefix else "pzeta",
            ]

            for i, (stored_log_module, direction, name) in enumerate(
                zip(
                    birk_log_module_storage[s_idx],
                    direction_list,
                    gpm._ghost_name,
                )
            ):
                out.write_data(
                    f"fli_over_n_nobirk/{name}/{current_turn}",
                    gpm._context.nparray_from_context_array(
                        log_module_storage[i] / current_turn
                    ),
                    overwrite=overwrite,
                )
                out.write_data(
                    f"fli_over_n_birk/{name}/{current_turn}",
                    gpm._context.nparray_from_context_array(stored_log_module),
                    overwrite=overwrite,
                )

                # Use a generator to handle writing the direction data
                for j, suffix in enumerate(suffixes):
                    full_suffix = (
                        f"{suffix}_{coord_prefix}"
                        if coord_prefix and suffix != "ptau"
                        else suffix
                    )
                    out.write_data(
                        f"direction/{name}/{full_suffix}/{current_turn}",
                        direction[j],
                        overwrite=overwrite,
                    )

    # save nturns of main particles
    # note: they need sorting
    particle_argsort = np.argsort(gpm._part.particle_id)
    out.write_data(
        "ref_particles_data/at_turn",
        gpm._context.nparray_from_context_array(gpm.part.at_turn)[particle_argsort],
        overwrite=overwrite,
    )
    out.write_data(
        "ref_particles_data/state",
        gpm._context.nparray_from_context_array(gpm.part.state)[particle_argsort],
        overwrite=overwrite,
    )
