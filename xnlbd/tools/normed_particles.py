from pathlib import Path
from typing import Optional

import numpy as np
import xobjects as xo  # type: ignore[import-untyped]
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]
import xtrack.twiss as xtw  # type: ignore[import-untyped]


class NormedParticles(xo.HybridClass):
    """Class to store particles in normalized coordinates, with the possibility
    of transforming to and from physical coordinates without the need of breaking
    GPU parallelism.
    """

    _cname = "NormedParticlesData"

    size_vars = (
        (xo.Int64, "_capacity"),
        (xo.Int64, "_start_tracking_at_element"),
    )

    geom_vars = (
        (xo.Float64[:], "twiss_data"),
        (xo.Float64[:], "w"),
        (xo.Float64[:], "w_inv"),
        (xo.Float64, "nemitt_x"),
        (xo.Float64, "nemitt_y"),
        (xo.Float64, "nemitt_z"),
    )

    per_particle_vars = (
        (xo.Float64, "zeta_norm"),
        (xo.Float64, "pzeta_norm"),
        (xo.Float64, "x_norm"),
        (xo.Float64, "y_norm"),
        (xo.Float64, "px_norm"),
        (xo.Float64, "py_norm"),
        (xo.Int64, "particle_id"),
        (xo.Int64, "state"),
    )

    _xofields = {
        "part_reference": xo.Ref(xt.Particles),
        **{nn: tt for tt, nn in size_vars},
        **{nn: tt for tt, nn in geom_vars},
        **{nn: tt[:] for tt, nn in per_particle_vars},
    }

    _extra_c_sources = [
        Path(__file__).parent.joinpath("src", "norm_to_phys.h"),
        Path(__file__).parent.joinpath("src", "phys_to_norm.h"),
    ]

    _kernels = {
        "norm_to_phys": xo.Kernel(
            args=[
                xo.Arg(xt.Particles._XoStruct, name="part"),
                xo.Arg(xo.ThisClass, name="norm_part"),
                xo.Arg(xo.Int64, name="nelem"),
            ],
            n_threads="nelem",
        ),
        "phys_to_norm": xo.Kernel(
            args=[
                xo.Arg(xt.Particles._XoStruct, name="part"),
                xo.Arg(xo.ThisClass, name="norm_part"),
                xo.Arg(xo.Int64, name="nelem"),
            ],
            n_threads="nelem",
        ),
    }

    _depends_on = [xt.Particles]

    def __len__(self):
        return self._capacity

    def __init__(
        self,
        twiss: xtw.TwissTable,
        nemitt_x: float,
        nemitt_y: float,
        nemitt_z: Optional[float] = None,
        idx_pos: int = 0,
        part: Optional[xt.Particles] = None,
        _capacity: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the NormedParticles object.

        Parameters
        ----------
        twiss : xtw.TwissTable
            Twiss table of the line to be used for the normalization.
        nemitt_x : float
            Normalized emittance in the horizontal plane.
        nemitt_y : float
            Normalized emittance in the vertical plane.
        nemitt_z : float, optional
            Normalized emittance in the longitudinal plane, by default None.
            If None, a unitary emittance is assumed.
        idx_pos : int, optional
            Index to the element wanted for the normalization, by default 0
        part : _type_, optional
            Particle object to be used as base, by default None
        _capacity : int, optional
            If no part is given, a storage of size _capacity is allocated.
        _context : xo.Context, optional
            xobjects context to be used.

        Raises
        ------
        ValueError
            If neither part nor _capacity is given.
        """
        if "_xobject" in kwargs.keys():
            # Initialize xobject
            self.xoinitialize(**kwargs)
            return

        # validate _capacity, is it given or is it to be taken from part?
        if part is not None:
            _capacity = part._capacity
        else:
            if _capacity is None:
                raise ValueError("Either part or _capacity must be given")

        # Allocate the xobject of the right size
        self.xoinitialize(
            _context=kwargs.pop("_context", None),
            _buffer=kwargs.pop("_buffer", None),
            _offset=kwargs.pop("_offset", None),
            **{field: _capacity for _, field in self.per_particle_vars},
            **{
                "twiss_data": 9,
                "w": 6 * 6,
                "w_inv": 6 * 6,
            },
        )

        self._capacity = _capacity

        # Get the twiss data for the given twiss object and the given
        # normalized emittance values
        self.twiss_data = self._context.nparray_to_context_array(
            np.array(
                [
                    nemitt_x,
                    nemitt_y,
                    twiss.x[idx_pos],
                    twiss.px[idx_pos],
                    twiss.y[idx_pos],
                    twiss.py[idx_pos],
                    twiss.zeta[idx_pos],
                    twiss.ptau[idx_pos],
                    nemitt_z if nemitt_z is not None else np.nan,
                ]
            )
        )
        self._start_tracking_at_element = idx_pos
        # GPUs like flattened arrays more...
        self.w = self._context.nparray_to_context_array(
            twiss.W_matrix[idx_pos].flatten()
        )
        self.w_inv = self._context.nparray_to_context_array(
            np.linalg.inv(twiss.W_matrix[idx_pos]).flatten()
        )

        self.compile_kernels(only_if_needed=True)

        if part is not None:
            self.phys_to_norm(part)

    def phys_to_norm(self, part: xp.Particles):
        """Transform the physical coordinates to normalized coordinates.
        Updates the normed_part attribute.

        Parameters
        ----------
        part : xp.Particles
            Particles object
        """
        self._context.kernels.phys_to_norm(
            part=part,
            norm_part=self,
            nelem=self._capacity,
        )

    def norm_to_phys(self, part: xp.Particles):
        """Transform the normalized coordinates to physical coordinates.
        Updates the given Particles object.

        Parameters
        ----------
        part : xp.Particles
            Target Particles object

        Returns
        -------
        part : xp.Particles
            Particles object
        """
        self._context.kernels.norm_to_phys(
            part=part,
            norm_part=self,
            nelem=self._capacity,
        )

        return part
