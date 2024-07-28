import pathlib

import numpy as np
import xpart as xp  # type: ignore[import-untyped]
import xtrack as xt  # type: ignore[import-untyped]

from xnlbd.visualise.orbits import (
    get_normalised_coordinates_from_real,
    get_real_coordinates_from_normalised,
)

test_data_folder = pathlib.Path(__file__).parent.joinpath("test_data").absolute()


def test_norm_vs_xt_internal():
    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tunes and chromaticities
    line.vv["qh_setvalue"] = 26.3371
    line.vv["qv_setvalue"] = 26.1311
    line.vv["qph_setvalue"] = 0.419287
    line.vv["qpv_setvalue"] = 0.0755032

    # Set RF
    line.vv["v200"] = 2e6

    # Twiss
    twiss = line.twiss(continue_on_closed_orbit_error=False)

    # Create particles
    num_particles = 10000

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)
    zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution="gaussian", sigma_z=0.2, line=line
    )

    beam = line.build_particles(
        x_norm=x_norm,
        px_norm=px_norm,
        y_norm=y_norm,
        py_norm=py_norm,
        zeta=zeta,
        delta=delta,
        method="6d",
        nemitt_x=2e-6,
        nemitt_y=0.5e-6,
    )
    beam_dict = beam.to_dict()

    # Normalise using xtrack internal method
    coords_norm_int = twiss.get_normalized_coordinates(beam)

    # Normalise using xnlbd method
    coords_norm = get_normalised_coordinates_from_real(twiss, beam_dict)

    # Compare
    assert np.all(np.isclose(coords_norm_int.x_norm, coords_norm["x_norm"], atol=1e-15))
    assert np.all(
        np.isclose(coords_norm_int.px_norm, coords_norm["px_norm"], atol=1e-15)
    )
    assert np.all(np.isclose(coords_norm_int.y_norm, coords_norm["y_norm"], atol=1e-15))
    assert np.all(
        np.isclose(coords_norm_int.py_norm, coords_norm["py_norm"], atol=1e-15)
    )
    assert np.all(
        np.isclose(coords_norm_int.zeta_norm, coords_norm["zeta_norm"], atol=1e-15)
    )
    assert np.all(
        np.isclose(coords_norm_int.pzeta_norm, coords_norm["pzeta_norm"], atol=1e-15)
    )


def test_norm_denorm():
    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tunes and chromaticities
    line.vv["qh_setvalue"] = 26.3371
    line.vv["qv_setvalue"] = 26.1311
    line.vv["qph_setvalue"] = 0.419287
    line.vv["qpv_setvalue"] = 0.0755032

    # Set RF
    line.vv["v200"] = 2e6

    # Twiss
    twiss = line.twiss(continue_on_closed_orbit_error=False)

    # Create particles
    num_particles = 10000

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)
    zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution="gaussian", sigma_z=0.2, line=line
    )

    beam = line.build_particles(
        x_norm=x_norm,
        px_norm=px_norm,
        y_norm=y_norm,
        py_norm=py_norm,
        zeta=zeta,
        delta=delta,
        method="6d",
        nemitt_x=2e-6,
        nemitt_y=0.5e-6,
    )
    beam_dict = beam.to_dict()

    # Normalise
    coords_norm = get_normalised_coordinates_from_real(twiss, beam_dict)

    # Denormalise
    coords_denorm = get_real_coordinates_from_normalised(twiss, coords_norm)

    # Compare
    assert np.all(np.isclose(beam_dict["x"], coords_denorm["x"], atol=1e-15))
    assert np.all(np.isclose(beam_dict["px"], coords_denorm["px"], atol=1e-15))
    assert np.all(np.isclose(beam_dict["y"], coords_denorm["y"], atol=1e-15))
    assert np.all(np.isclose(beam_dict["py"], coords_denorm["py"], atol=1e-15))
    assert np.all(np.isclose(beam_dict["zeta"], coords_denorm["zeta"], atol=1e-15))
    assert np.all(
        np.isclose(
            beam_dict["ptau"] / twiss.particle_on_co.beta0,
            coords_denorm["pzeta"],
            atol=1e-15,
        )
    )
