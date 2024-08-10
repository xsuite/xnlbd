import pathlib

import numpy as np
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]
from xobjects.test_helpers import for_all_test_contexts  # type: ignore[import-untyped]

from xnlbd.tools import NormedParticles


@for_all_test_contexts
def test_normed_particles(test_context):
    # First load the example collider
    collider = xt.Multiline.from_json(
        pathlib.Path(__file__).parent.joinpath("test_data/example_hl_lhc_collider.json")
    )
    line = collider["lhcb1"]
    line.build_tracker(_context=test_context)

    # evaluate the twiss
    twiss = line.twiss()

    x_norm_coord = np.linspace(0, 2)
    y_norm_coord = np.linspace(0, 0.5)
    zeta_norm_coord = np.linspace(0, 1)

    # build some particles
    part = line.build_particles(
        x_norm=x_norm_coord,
        y_norm=y_norm_coord,
        zeta_norm=zeta_norm_coord,
        nemitt_x=2.5e-6,
        nemitt_y=2.5e-6,
        _context=test_context,
    )
    ref_part = part.copy()

    # make the normalized particles object
    part_norm = NormedParticles(twiss, 2.5e-6, 2.5e-6, part=part, _context=test_context)

    # do a couple of iterations
    for _ in range(10):
        part_norm.phys_to_norm(part)
        part = part_norm.norm_to_phys(part)

    # check that the particles are still the same
    assert np.allclose(
        test_context.nparray_from_context_array(part.x),
        test_context.nparray_from_context_array(ref_part.x),
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part.y),
        test_context.nparray_from_context_array(ref_part.y),
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part.zeta),
        test_context.nparray_from_context_array(ref_part.zeta),
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part.px),
        test_context.nparray_from_context_array(ref_part.px),
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part.py),
        test_context.nparray_from_context_array(ref_part.py),
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part.delta),
        test_context.nparray_from_context_array(ref_part.delta),
        rtol=1e-12,
    )

    # check that the normalized particles are still the same
    assert np.allclose(
        test_context.nparray_from_context_array(part_norm.x_norm),
        x_norm_coord,
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part_norm.y_norm),
        y_norm_coord,
        rtol=1e-12,
    )
    assert np.allclose(
        test_context.nparray_from_context_array(part_norm.zeta_norm),
        zeta_norm_coord,
        rtol=1e-12,
    )
