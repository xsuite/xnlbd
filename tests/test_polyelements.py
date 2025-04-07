import numpy as np
import xpart as xp  # type: ignore[import-untyped, import-not-found]
import xtrack as xt  # type: ignore[import-untyped, import-not-found]

from xnlbd.analyse.normal_forms import *
from xnlbd.track import Henonmap


def test_PolyReferenceEnergyIncrease4D():
    element = xt.ReferenceEnergyIncrease(Delta_p0c=1)
    line = xt.Line(elements=[element], element_names=["ReferenceEnergyIncrease"])
    line.particle_ref = xt.Particles(p0c=450e9, q0=1, mass0=xp.PROTON_MASS_EV)

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.random.uniform(-1e-2, 1e-2, 100)
    delta_vals = np.random.uniform(-1.5e-3, 1.5e-3, 100)

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyReferenceEnergyIncrease4D()

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyMarker4D():
    element = xt.Marker()
    line = xt.Line(elements=[element], element_names=["Marker"])
    line.particle_ref = xt.Particles(p0c=450e9, q0=1, mass0=xp.PROTON_MASS_EV)

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.random.uniform(-1e-2, 1e-2, 100)
    delta_vals = np.random.uniform(-1.5e-3, 1.5e-3, 100)

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyMarker4D()

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyDrift4D():
    element = xt.Drift(length=1.0)
    line = xt.Line(elements=[element], element_names=["Drift"])
    line.particle_ref = xt.Particles(
        p0c=450e9, q0=1, zeta=0.637e-2, delta=1.5e-3, mass0=xp.PROTON_MASS_EV
    )

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.ones(100) * 0.637e-2
    delta_vals = np.ones(100) * 1.5e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyDrift4D(part=line.particle_ref, length=1.0)

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyCavity4D():
    element = xt.Cavity()
    line = xt.Line(elements=[element], element_names=["Cavity"])
    line.particle_ref = xt.Particles(p0c=450e9, q0=1, mass0=xp.PROTON_MASS_EV)

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.random.uniform(-1e-2, 1e-2, 100)
    delta_vals = np.random.uniform(-1.5e-3, 1.5e-3, 100)

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyCavity4D()

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolySRotation4D():
    sin_z = np.sin(np.pi / 7)
    cos_z = np.cos(np.pi / 7)

    element = xt.SRotation(sin_z=sin_z, cos_z=cos_z)
    line = xt.Line(elements=[element], element_names=["SRotation"])
    line.particle_ref = xt.Particles(p0c=450e9, q0=1, mass0=xp.PROTON_MASS_EV)

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.random.uniform(-1e-2, 1e-2, 100)
    delta_vals = np.random.uniform(-1.5e-3, 1.5e-3, 100)

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolySRotation4D(sin_z=sin_z, cos_z=cos_z)

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyZetaShift4D():
    element = xt.ZetaShift()
    line = xt.Line(elements=[element], element_names=["ZetaShift"])
    line.particle_ref = xt.Particles(p0c=450e9, q0=1, mass0=xp.PROTON_MASS_EV)

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.random.uniform(-1e-2, 1e-2, 100)
    delta_vals = np.random.uniform(-1.5e-3, 1.5e-3, 100)

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyZetaShift4D()

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyMultipole4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=1.5e-3, p0c=450e9, mass0=xp.PROTON_MASS_EV
    )

    element = xt.Multipole(
        knl=[0.008, 0.014, 0.2, 8],
        ksl=[0.008, 0.014, 0.2, 8],
        hxl=np.pi / 100,
        length=1.7,
    )
    line = xt.Line(elements=[element], element_names=["Multipole"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 1.5e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyMultipole4D(
        part=ref_part,
        order=element._order,
        inv_factorial_order=element.inv_factorial_order,
        knl=line.element_refs["Multipole"].knl._get_value(),
        ksl=line.element_refs["Multipole"].ksl._get_value(),
        hxl=line.element_refs["Multipole"].hxl._get_value(),
        length=element.length,
        max_order=10,
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolySimpleThinQuadrupole4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=0.9e-3, p0c=450e9, mass0=xp.PROTON_MASS_EV
    )

    element = xt.SimpleThinQuadrupole(
        knl=[0.0, 0.014],
    )
    line = xt.Line(elements=[element], element_names=["SimpleThinQuadrupole"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 0.9e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolySimpleThinQuadrupole4D(
        part=ref_part,
        knl=line.element_refs["SimpleThinQuadrupole"].knl._get_value(),
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolySimpleThinBend4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=0.74e-3, p0c=450e9, mass0=xp.PROTON_MASS_EV
    )

    element = xt.SimpleThinBend(
        knl=[0.1],
        hxl=0.2,
        length=7,
    )
    line = xt.Line(elements=[element], element_names=["SimpleThinBend"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 0.74e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolySimpleThinBend4D(
        part=ref_part,
        knl=line.element_refs["SimpleThinBend"].knl._get_value(),
        hxl=line.element_refs["SimpleThinBend"].hxl._get_value(),
        length=line.element_refs["SimpleThinBend"].length._get_value(),
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolySextupole4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=0.9e-3, p0c=450e9, mass0=xp.PROTON_MASS_EV
    )

    element = xt.Sextupole(knl=[0.0, 0.0, 0.4], edge_entry_active=0, edge_exit_active=0)
    line = xt.Line(elements=[element], element_names=["Sextupole"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 0.9e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolySextupole4D(
        part=ref_part,
        k2=line.element_refs["Sextupole"].k2._get_value(),
        k2s=line.element_refs["Sextupole"].k2s._get_value(),
        length=line.element_refs["Sextupole"].length._get_value(),
        order=line.element_refs["Sextupole"]._order._get_value(),
        inv_factorial_order=line.element_refs[
            "Sextupole"
        ].inv_factorial_order._get_value(),
        knl=line.element_refs["Sextupole"].knl._get_value(),
        ksl=line.element_refs["Sextupole"].ksl._get_value(),
        edge_entry_active=line.element_refs["Sextupole"].edge_entry_active._get_value(),
        edge_exit_active=line.element_refs["Sextupole"].edge_exit_active._get_value(),
        max_order=10,
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyOctupole4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=0.9e-3, p0c=450e9, mass0=xp.PROTON_MASS_EV
    )

    element = xt.Octupole(knl=[0.0, 0.0, 0.4], edge_entry_active=0, edge_exit_active=0)
    line = xt.Line(elements=[element], element_names=["Octupole"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 0.9e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyOctupole4D(
        part=ref_part,
        k3=line.element_refs["Octupole"].k3._get_value(),
        k3s=line.element_refs["Octupole"].k3s._get_value(),
        length=line.element_refs["Octupole"].length._get_value(),
        order=line.element_refs["Octupole"]._order._get_value(),
        inv_factorial_order=line.element_refs[
            "Octupole"
        ].inv_factorial_order._get_value(),
        knl=line.element_refs["Octupole"].knl._get_value(),
        ksl=line.element_refs["Octupole"].ksl._get_value(),
        edge_entry_active=line.element_refs["Octupole"].edge_entry_active._get_value(),
        edge_exit_active=line.element_refs["Octupole"].edge_exit_active._get_value(),
        max_order=10,
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyDipoleEdge4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=1.2e-3, p0c=450e9, mass0=xp.PROTON_MASS_EV
    )

    element = xt.DipoleEdge(k=0.00135, e1=0.005, e1_fd=0, fint=0, model="linear")
    line = xt.Line(elements=[element], element_names=["DipoleEdge"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 1.2e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyDipoleEdge4D(
        part=ref_part,
        r21=line.element_refs["DipoleEdge"]._r21._get_value(),
        r43=line.element_refs["DipoleEdge"]._r43._get_value(),
        model=0,
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))


def test_PolyHenonmap4D():
    ref_part = xt.Particles(
        x=0, px=0, y=0, py=0, zeta=0, delta=1.2e-3, p0c=400e9, mass0=xp.PROTON_MASS_EV
    )

    element = Henonmap(
        omega_x=2 * np.pi * 0.36,
        omega_y=2 * np.pi * 0.213,
        n_turns=1,
        twiss_params=[0.1, 100.0, 0.3, 20.0],
        dqx=10.0,
        dqy=-3.0,
        dx=0.001,
        ddx=10e-6,
        multipole_coeffs=[0.4, 6.0],
        norm=False,
    )
    line = xt.Line(elements=[element], element_names=["Henonmap"])
    line.particle_ref = ref_part

    x_vals = np.random.uniform(-1e-3, 1e-3, 100)
    px_vals = np.random.uniform(-5e-6, 5e-6, 100)
    y_vals = np.random.uniform(-1e-3, 1e-3, 100)
    py_vals = np.random.uniform(-5e-6, 5e-6, 100)
    zeta_vals = np.zeros(100)
    delta_vals = np.ones(100) * 1.2e-3

    part = xt.Particles(
        x=x_vals,
        px=px_vals,
        y=y_vals,
        py=py_vals,
        zeta=zeta_vals,
        delta=delta_vals,
        p0c=line.particle_ref.p0c,
    )

    line.build_tracker()
    line.track(part)

    poly_element = PolyHenonmap4D(
        part=ref_part,
        cos_omega_x=line.element_refs["Henonmap"].cos_omega_x._get_value(),
        sin_omega_x=line.element_refs["Henonmap"].sin_omega_x._get_value(),
        cos_omega_y=line.element_refs["Henonmap"].cos_omega_y._get_value(),
        sin_omega_y=line.element_refs["Henonmap"].sin_omega_y._get_value(),
        domegax=line.element_refs["Henonmap"].domegax._get_value(),
        domegay=line.element_refs["Henonmap"].domegay._get_value(),
        n_turns=line.element_refs["Henonmap"].n_turns._get_value(),
        twiss_params=line.element_refs["Henonmap"].twiss_params._get_value(),
        dx=line.element_refs["Henonmap"].dx._get_value(),
        ddx=line.element_refs["Henonmap"].ddx._get_value(),
        fx_coeffs=line.element_refs["Henonmap"].fx_coeffs._get_value(),
        fx_x_exps=line.element_refs["Henonmap"].fx_x_exps._get_value(),
        fx_y_exps=line.element_refs["Henonmap"].fx_y_exps._get_value(),
        fy_coeffs=line.element_refs["Henonmap"].fy_coeffs._get_value(),
        fy_x_exps=line.element_refs["Henonmap"].fy_x_exps._get_value(),
        fy_y_exps=line.element_refs["Henonmap"].fy_y_exps._get_value(),
        norm=line.element_refs["Henonmap"].norm._get_value(),
        max_order=100,
    )

    x_tracked = poly_element.ele_map.x_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    px_tracked = poly_element.ele_map.px_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )
    y_tracked = poly_element.ele_map.y_poly.substitute(x_vals, px_vals, y_vals, py_vals)
    py_tracked = poly_element.ele_map.py_poly.substitute(
        x_vals, px_vals, y_vals, py_vals
    )

    assert np.all(np.isclose(part.x, x_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.px, px_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.y, y_tracked, rtol=1e-14, atol=1e-16))
    assert np.all(np.isclose(part.py, py_tracked, rtol=1e-14, atol=1e-16))
