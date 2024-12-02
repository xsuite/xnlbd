import warnings
from pathlib import Path

import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0

from ...tools.normed_particles import NormedParticles


class GhostParticleManager(xo.HybridClass):
    """Class to manage ghost particles and measure their displacement."""

    _cname = "GhostParticleManagerData"

    operational_vars = (
        (xo.Int64, "_capacity"),
        (xo.Float64, "target_module"),
    )

    managing_vars = (
        (xo.Int64, "idx_a"),
        (xo.Int64, "argsort_a"),
        (xo.Int64, "idx_b"),
        (xo.Int64, "argsort_b"),
        (xo.Float64, "module"),
    )

    phys_vars = (
        (xo.Float64, "displacement_x"),
        (xo.Float64, "displacement_px"),
        (xo.Float64, "displacement_y"),
        (xo.Float64, "displacement_py"),
        (xo.Float64, "displacement_zeta"),
        (xo.Float64, "displacement_ptau"),
    )

    norm_vars = (
        (xo.Float64, "displacement_x_norm"),
        (xo.Float64, "displacement_px_norm"),
        (xo.Float64, "displacement_y_norm"),
        (xo.Float64, "displacement_py_norm"),
        (xo.Float64, "displacement_zeta_norm"),
        (xo.Float64, "displacement_pzeta_norm"),
    )

    _xofields = {
        **{nn: tt for tt, nn in operational_vars},
        **{nn: tt[:] for tt, nn in managing_vars},
        **{nn: tt[:] for tt, nn in phys_vars},
        **{nn: tt[:] for tt, nn in norm_vars},
    }

    _extra_c_sources = [
        Path(__file__).parent.joinpath("src", "get_disp_and_dir.h"),
        Path(__file__).parent.joinpath("src", "get_disp_and_dir_normed.h"),
        Path(__file__).parent.joinpath("src", "renorm_dist.h"),
        Path(__file__).parent.joinpath("src", "renorm_dist_normed.h"),
    ]

    _kernels = {
        "get_disp_and_dir": xo.Kernel(
            args=[
                xo.Arg(xt.Particles._XoStruct, name="part_a"),
                xo.Arg(xt.Particles._XoStruct, name="part_b"),
                xo.Arg(xo.ThisClass, name="manager"),
                xo.Arg(xo.Int64, name="nelem"),
            ],
            n_threads="nelem",
        ),
        "get_disp_and_dir_normed": xo.Kernel(
            args=[
                xo.Arg(NormedParticles._XoStruct, name="part_a"),
                xo.Arg(NormedParticles._XoStruct, name="part_b"),
                xo.Arg(xo.ThisClass, name="manager"),
                xo.Arg(xo.Int64, name="nelem"),
            ],
            n_threads="nelem",
        ),
        "renorm_dist": xo.Kernel(
            args=[
                xo.Arg(xt.Particles._XoStruct, name="ref_part"),
                xo.Arg(xt.Particles._XoStruct, name="part"),
                xo.Arg(xo.ThisClass, name="manager"),
                xo.Arg(xo.Int64, name="nelem"),
            ],
            n_threads="nelem",
        ),
        "renorm_dist_normed": xo.Kernel(
            args=[
                xo.Arg(NormedParticles._XoStruct, name="ref_part"),
                xo.Arg(NormedParticles._XoStruct, name="part"),
                xo.Arg(xo.ThisClass, name="manager"),
                xo.Arg(xo.Int64, name="nelem"),
            ],
            n_threads="nelem",
        ),
    }

    _depends_on = [xt.Particles, NormedParticles]

    def __init__(
        self,
        part: xt.Particles,
        use_norm_coord=True,
        twiss=None,
        nemitt_x=None,
        nemitt_y=None,
        nemitt_z=None,
        idx_pos=0,
        **kwargs,
    ):
        """Initialize the GhostParticleManager.

        Parameters
        ----------
        part : xp.Particles
            Particles to be tracked.
        _context : xo.Context, optional
            Context to be used, by default xo.ContextCpu()
        twiss : xtw.Twiss, optional
            Twiss object, by default None, required if use_norm_coord is True
        nemitt_x : float, optional
            Normalized emittance in x, by default None, required if use_norm_coord
            is True
        nemitt_y : float, optional
            Normalized emittance in y, by default None, required if use_norm_coord
            is True
        nemitt_z : float, optional
            Normalized emittance in z, by default None, if use_norm_coord and nemitt_z is None, a "unitary" emittance is assumed
        idx_pos : int, optional
            Index of the position in the line, used to pick the desired set of
            twiss values from the twiss object, by default 0, required if
            use_norm_coord is True
        use_norm_coord : bool, optional
            If True, normalized coordinates are used, by default True
        """
        if "_xobject" in kwargs.keys():
            # Initialize xobject
            self.xoinitialize(**kwargs)
            return

        self._part = part
        self._use_norm_coord = use_norm_coord

        # allocate xobject
        self.xoinitialize(
            _context=kwargs.pop("_context", None),
            _buffer=kwargs.pop("_buffer", None),
            _offset=kwargs.pop("_offset", None),
            **{field: self._part._capacity for _, field in self.managing_vars},
            **{
                field: self._part._capacity
                for _, field in (
                    self.phys_vars if self._use_norm_coord is False else self.norm_vars
                )
            },
            **{
                field: 0
                for _, field in (
                    self.norm_vars if self._use_norm_coord is False else self.phys_vars
                )
            },
        )

        if self._use_norm_coord:
            if twiss is None:
                raise ValueError("If norm_coord is True, twiss must be given")
            if nemitt_x is None:
                raise ValueError("If norm_coord is True, nemitt_x must be given")
            if nemitt_y is None:
                raise ValueError("If norm_coord is True, nemitt_y must be given")
            if nemitt_z is None:
                # raise warning
                warnings.warn(
                    "Warning: nemitt_z is None, a unitary emittance is assumed"
                )

            self._twiss = twiss
            self._nemitt_x = nemitt_x
            self._nemitt_y = nemitt_y
            self._nemitt_z = nemitt_z
            self._idx_pos = idx_pos

            self._normed_part = NormedParticles(
                self._twiss,
                self._nemitt_x,
                self._nemitt_y,
                self._nemitt_z,
                self._idx_pos,
                part=self._part,
                _context=self._context,
            )
            self._ghost_normed_part = []
        else:
            self._normed_part = None
            self._ghost_normed_part = None

        self._ghost_part = []
        self._ghost_name = []
        self._original_displacement = []
        self._original_direction = []

        self._capacity = self._part._capacity

        self._capacity = self._part._capacity
        self.compile_kernels(only_if_needed=True)

    def add_displacement(
        self, module=1e-6, direction="x", custom_direction=None, ghost_name=None
    ):
        """Add a ghost particle with a displacement in the given direction.

        Parameters
        ----------
        module : float, optional
            Module of the displacement, by default 1e-8
        direction : str, optional
            Direction of the displacement, by default 'x', must be one of: x, px, y,
            py, zeta, pzeta if use_norm_coord is False, or x_norm, px_norm, y_norm,
            py_norm, zeta_norm, pzeta_norm if use_norm_coord is True. Must be custom
            if custom_direction is given.
        custom_direction : np.ndarray, optional
            Custom direction of the displacement, by default None, must be given if
            direction is "custom"
        ghost_name : str, optional
            Name of the ghost particle, by default None, if None, the name is
            automatically generated
        """
        if custom_direction is None:
            disp_vector = np.zeros(6)
            if self._use_norm_coord:
                if direction == "x_norm":
                    disp_vector[0] = module
                elif direction == "px_norm":
                    disp_vector[1] = module
                elif direction == "y_norm":
                    disp_vector[2] = module
                elif direction == "py_norm":
                    disp_vector[3] = module
                elif direction == "zeta_norm":
                    disp_vector[4] = module
                elif direction == "pzeta_norm":
                    disp_vector[5] = module
                else:
                    raise ValueError(
                        "Invalid direction, with use_norm_coord=True, direction must be one of: x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm"
                    )
            else:
                if direction == "x":
                    disp_vector[0] = module
                elif direction == "px":
                    disp_vector[1] = module
                elif direction == "y":
                    disp_vector[2] = module
                elif direction == "py":
                    disp_vector[3] = module
                elif direction == "zeta":
                    disp_vector[4] = module
                elif direction == "ptau":
                    disp_vector[5] = module
                else:
                    raise ValueError(
                        "Invalid direction, with use_norm_coord=False, direction must be one of: x, px, y, py, zeta, ptau"
                    )

            self._original_displacement.append(module)
            self._original_direction.append(disp_vector / module)
        else:
            if direction != "custom":
                raise ValueError(
                    "If custom_direction is given, direction must be custom"
                )
            if np.asarray(custom_direction).shape != (6,):
                raise ValueError(
                    "If custom_direction is given, it must be a 6 element array"
                )
            # normalize custom_direction
            custom_direction = np.asarray(custom_direction)
            custom_direction = custom_direction / np.sum(custom_direction**2) ** 0.5

            self._original_direction.append(
                custom_direction / np.sum(custom_direction**2) ** 0.5
            )

            # add module to normalized custom_direction
            disp_vector = custom_direction * module
            self._original_displacement.append(module)

        # check if ghost_name is already used
        if ghost_name is None:
            ghost_name = f"{direction}"
        if ghost_name in self._ghost_name:
            raise ValueError(f"ghost_name {ghost_name} is already used")

        self._ghost_name.append(ghost_name)

        # make copy of part
        disp_part = self._part.copy()

        if self._use_norm_coord is False:
            disp_part.x += disp_vector[0]
            disp_part.px += disp_vector[1]
            disp_part.y += disp_vector[2]
            disp_part.py += disp_vector[3]
            disp_part.zeta += disp_vector[4]
            disp_part.ptau += disp_vector[5]

            self._ghost_part.append(disp_part)
        else:
            # make copy of normed_part
            disp_normed_part = NormedParticles(
                self._twiss,
                self._nemitt_x,
                self._nemitt_y,
                self._nemitt_z,
                self._idx_pos,
                part=disp_part,
                _context=self._context,
            )
            disp_normed_part.x_norm += disp_vector[0]
            disp_normed_part.px_norm += disp_vector[1]
            disp_normed_part.y_norm += disp_vector[2]
            disp_normed_part.py_norm += disp_vector[3]
            disp_normed_part.zeta_norm += disp_vector[4]
            disp_normed_part.pzeta_norm += disp_vector[5]

            # convert back to disp_part
            disp_part = disp_normed_part.norm_to_phys(disp_part)

            self._ghost_normed_part.append(disp_normed_part)
            self._ghost_part.append(disp_part)

    def particles(self):
        """Yield the particles to be tracked. The first particle is the original."""
        yield self._part
        for ghost_part in self._ghost_part:
            yield ghost_part

    def get_module_and_direction(self):
        """Return the current module and direction of the displacement of the
        ghost particles."""

        module_list = []
        direction_list = []

        # for now, let us evaluate the argsord idx directly with numpy methods
        # TODO: make it more efficient, possibly with direct xobjects methods

        if self._use_norm_coord is False:
            self.idx_a = self._part.particle_id
            self.argsort_a = np.argsort(self.idx_a)
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = ghost_part.particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir(
                    part_a=self._part,
                    part_b=ghost_part,
                    manager=self,
                    nelem=len(self.idx_a),
                )
                module_list.append(
                    self._context.nparray_from_context_array(self.module).copy()
                )
                direction_list.append(
                    np.array(
                        [
                            self._context.nparray_from_context_array(
                                self.displacement_x
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_px
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_y
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_py
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_zeta
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_ptau
                            ).copy(),
                        ]
                    )
                )
        else:
            self._normed_part.phys_to_norm(self._part)
            for pp, pnorm in zip(self._ghost_part, self._ghost_normed_part):
                pnorm.phys_to_norm(pp)
            # get the argsort idx
            self.idx_a = self._normed_part.particle_id
            self.argsort_a = np.argsort(self.idx_a)

            # then, do as above but with normed particles
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = self._ghost_normed_part[i].particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir_normed(
                    part_a=self._normed_part,
                    part_b=self._ghost_normed_part[i],
                    manager=self,
                    nelem=len(self.idx_a),
                )
                module_list.append(
                    self._context.nparray_from_context_array(self.module).copy()
                )
                direction_list.append(
                    np.array(
                        [
                            self._context.nparray_from_context_array(
                                self.displacement_x_norm
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_px_norm
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_y_norm
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_py_norm
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_zeta_norm
                            ).copy(),
                            self._context.nparray_from_context_array(
                                self.displacement_pzeta_norm
                            ).copy(),
                        ]
                    )
                )

        return module_list, direction_list

    def renormalize_distances(self, module, get_data=False):
        """Renormalize the distances of the ghost particles.

        Parameters
        ----------
        module : float
            New module of the displacement
        get_data : bool, optional
            If True, return the module and direction of the displacement of the
            ghost particles, by default False

        Returns
        -------
        tuple
            If get_data is True, return the module and direction of the displacement
            of the ghost particles
        """
        self.target_module = module
        if get_data:
            module_list, direction_list = self.get_module_and_direction()

        if self._use_norm_coord is False:
            self.idx_a = self._part.particle_id
            self.argsort_a = np.argsort(self.idx_a)
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = ghost_part.particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir(
                    part_a=self._part,
                    part_b=ghost_part,
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # self.module = module - self.module
                self._context.kernels.renorm_dist(
                    ref_part=self._part,
                    part=ghost_part,
                    manager=self,
                    nelem=len(self.idx_a),
                )
        else:
            self._normed_part.phys_to_norm(self._part)
            for pp, pnorm in zip(self._ghost_part, self._ghost_normed_part):
                pnorm.phys_to_norm(pp)
            self.idx_a = self._normed_part.particle_id
            self.argsort_a = np.argsort(self.idx_a)
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = self._ghost_normed_part[i].particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir_normed(
                    part_a=self._normed_part,
                    part_b=self._ghost_normed_part[i],
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # self.module = module - self.module
                self._context.kernels.renorm_dist_normed(
                    ref_part=self._normed_part,
                    part=self._ghost_normed_part[i],
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # convert back to phys
                self._ghost_part[i] = self._ghost_normed_part[i].norm_to_phys(
                    self._ghost_part[i]
                )

        if get_data:
            return module_list, direction_list

    def yield_module_and_direction(self):
        """Yield progressively the current module and direction of the
        displacement of the various ghost particles, by populating the module
        and direction xobjects accordingly.

        The method yields the name of the ghost particle, but the data will be
        actually stored in the module and direction xobjects.

        Yield
        -----
        str
            Name of the ghost particle

        """
        if self._use_norm_coord is False:
            self.idx_a = self._part.particle_id
            self.argsort_a = np.argsort(self.idx_a)
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = ghost_part.particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir(
                    part_a=self._part,
                    part_b=ghost_part,
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # Now the arrays are populated, yield the name of the ghost particle
                yield self._ghost_name[i]

        else:
            self._normed_part.phys_to_norm(self._part)
            for pp, pnorm in zip(self._ghost_part, self._ghost_normed_part):
                pnorm.phys_to_norm(pp)
            # get the argsort idx
            self.idx_a = self._normed_part.particle_id
            self.argsort_a = np.argsort(self.idx_a)

            # then, do as above but with normed particles
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = self._ghost_normed_part[i].particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir_normed(
                    part_a=self._normed_part,
                    part_b=self._ghost_normed_part[i],
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # Now the arrays are populated, yield the name of the ghost particle
                yield self._ghost_name[i]

    def renormalize_distances_and_yield(self, module):
        """Renormalize the distances of the ghost particles. The method yields
        progressively the current module and direction of the displacement of the
        various ghost particles, by populating the module and direction xobjects
        accordingly.

        The method yields the name of the ghost particle, but the data will be
        actually stored in the module and direction xobjects.

        Parameters
        ----------
        module : float
            New module of the displacement

        Yield
        -----
        str
            Name of the ghost particle

        """
        self.target_module = module
        if self._use_norm_coord is False:
            self.idx_a = self._part.particle_id
            self.argsort_a = np.argsort(self.idx_a)
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = ghost_part.particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir(
                    part_a=self._part,
                    part_b=ghost_part,
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # Now the arrays are populated, yield the name of the ghost particle
                # self.module = module - self.module
                self._context.kernels.renorm_dist(
                    ref_part=self._part,
                    part=ghost_part,
                    manager=self,
                    nelem=len(self.idx_a),
                )
                yield self._ghost_name[i]
        else:
            self._normed_part.phys_to_norm(self._part)
            for pp, pnorm in zip(self._ghost_part, self._ghost_normed_part):
                pnorm.phys_to_norm(pp)
            self.idx_a = self._normed_part.particle_id
            self.argsort_a = np.argsort(self.idx_a)
            for i, ghost_part in enumerate(self._ghost_part):
                self.idx_b = self._ghost_normed_part[i].particle_id
                self.argsort_b = np.argsort(self.idx_b)
                self._context.kernels.get_disp_and_dir_normed(
                    part_a=self._normed_part,
                    part_b=self._ghost_normed_part[i],
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # Now the arrays are populated, yield the name of the ghost particle
                # self.module = module - self.module
                self._context.kernels.renorm_dist_normed(
                    ref_part=self._normed_part,
                    part=self._ghost_normed_part[i],
                    manager=self,
                    nelem=len(self.idx_a),
                )
                # convert back to phys
                self._ghost_part[i] = self._ghost_normed_part[i].norm_to_phys(
                    self._ghost_part[i]
                )
                yield self._ghost_name[i]
