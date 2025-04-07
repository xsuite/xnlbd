import os
import pathlib
import re
import time

import numpy as np
import xpart as xp  # type: ignore[import-untyped, import-not-found]
import xtrack as xt  # type: ignore[import-untyped, import-not-found]
from pymadng import MAD  # type: ignore[import-untyped, import-not-found]

from xnlbd.analyse.normal_forms import *

test_data_folder = pathlib.Path(__file__).parent.joinpath("test_data").absolute()


def parse_section(lines, section_name):
    section_data = []
    for line in lines:
        match = re.match(
            r"\s*\d+\s+([-+]?\d*\.\d+E[-+]?\d+)\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
            line,
        )
        if match:
            section_data.append(
                [float(match.group(1))] + [int(match.group(i)) for i in range(2, 8)]
            )
        else:
            continue
    return section_data


def parse_madng_map(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data_sections = {"X": [], "PX": [], "Y": [], "PY": [], "T": [], "PT": []}
    current_section = None

    for line in lines:
        line = line.strip()

        match = re.match(r"^\s*(X|PX|Y|PY|T|PT)\s*:", line)
        if match:
            current_section = match.group(1)
            continue
        elif current_section:
            data_sections[current_section].append(line)

    for key in data_sections:
        if not data_sections[key]:
            print(f"⚠️ Warning: No data found for section {key}")

        data_sections[key] = parse_section(data_sections[key], key)

    x_terms = []
    for i in range(len(data_sections["X"])):
        if data_sections["X"][i][5] != 0 or data_sections["X"][i][6] != 0:
            continue
        else:
            x_terms.append(
                Term(
                    coeff=data_sections["X"][i][0],
                    x_exp=data_sections["X"][i][1],
                    px_exp=data_sections["X"][i][2],
                    y_exp=data_sections["X"][i][3],
                    py_exp=data_sections["X"][i][4],
                )
            )
    px_terms = []
    for i in range(len(data_sections["PX"])):
        if data_sections["PX"][i][5] != 0 or data_sections["PX"][i][6] != 0:
            continue
        else:
            px_terms.append(
                Term(
                    coeff=data_sections["PX"][i][0],
                    x_exp=data_sections["PX"][i][1],
                    px_exp=data_sections["PX"][i][2],
                    y_exp=data_sections["PX"][i][3],
                    py_exp=data_sections["PX"][i][4],
                )
            )
    y_terms = []
    for i in range(len(data_sections["Y"])):
        if data_sections["Y"][i][5] != 0 or data_sections["Y"][i][6] != 0:
            continue
        else:
            y_terms.append(
                Term(
                    coeff=data_sections["Y"][i][0],
                    x_exp=data_sections["Y"][i][1],
                    px_exp=data_sections["Y"][i][2],
                    y_exp=data_sections["Y"][i][3],
                    py_exp=data_sections["Y"][i][4],
                )
            )
    py_terms = []
    for i in range(len(data_sections["PY"])):
        if data_sections["PY"][i][5] != 0 or data_sections["PY"][i][6] != 0:
            continue
        else:
            py_terms.append(
                Term(
                    coeff=data_sections["PY"][i][0],
                    x_exp=data_sections["PY"][i][1],
                    px_exp=data_sections["PY"][i][2],
                    y_exp=data_sections["PY"][i][3],
                    py_exp=data_sections["PY"][i][4],
                )
            )

    return Map(
        x_poly=Polynom(terms=x_terms),
        px_poly=Polynom(terms=px_terms),
        y_poly=Polynom(terms=y_terms),
        py_poly=Polynom(terms=py_terms),
    )


def test_exact_drift_map():
    Lmb = 6.26
    Lqf = 3.085

    line = xt.Line(
        elements=[xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0))],
        element_names=["drift0"],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    map_xnlbd = PolyDrift4D(
        part=line.particle_ref,
        length=line.element_refs["drift0"].length._get_value(),
        exact=1,
        max_order=6,
    ).ele_map

    seq = line.to_madx_sequence(sequence_name="test")

    with open(
        test_data_folder.joinpath("normal_forms/test_exact_drift.seq"), "w"
    ) as file:
        file.write(seq)

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_exact_drift.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_exact_drift.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath("normal_forms/test_exact_drift_map_MADNG_O6")
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng


def test_fodo_map():
    kmb = 0.008445141542
    kqf = 0.014508048734097173
    kqd = -0.014435866477457426

    Lmb = 6.26
    Lqf = 3.085
    Lqd = 3.085
    Lmult = 0.42

    line = xt.Line(
        elements=[
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
            xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1357 + Lmb / 2.0 + Lmult / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(0.787 + Lqd / 2.0 + Lmult / 2.0)),
            xt.Multipole(knl=[0.0, kqd * Lqd], ksl=[0.0, 0.0], length=Lqd, hxl=0.0),
            xt.Drift(length=(0.35 + Lqd / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1457 + Lmult / 2.0 + Lmb / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(2.3295 + Lmult / 2.0 + Lqf / 2.0)),
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
        ],
        element_names=[
            "qf1",
            "drift0",
            "mb1",
            "drift1",
            "mb2",
            "drift2",
            "mb3",
            "drift3",
            "mb4",
            "drift4",
            "multipole1",
            "drift5",
            "qd",
            "drift6",
            "mb5",
            "drift7",
            "mb6",
            "drift8",
            "mb7",
            "drift9",
            "mb8",
            "drift10",
            "multipole2",
            "drift11",
            "qf2",
        ],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    seq = line.to_madx_sequence(sequence_name="test")

    with open(test_data_folder.joinpath("normal_forms/test_fodo.seq"), "w") as file:
        file.write(seq)

    poly_line = PolyLine4D(
        line=line,
        part=line.particle_ref,
        max_ele_order=8,
        max_map_order=6,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )
    poly_line.calculate_one_turn_map()

    map_xnlbd = poly_line.one_turn_map_real

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_fodo.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_fodo.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath("normal_forms/test_fodo_map_MADNG_O6")
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng


def test_fodo_1sext_map():
    kmb = 0.008445141542
    kqf = 0.014508048734097173
    kqd = -0.014435866477457426

    Lmb = 6.26
    Lqf = 3.085
    Lqd = 3.085
    Lmult = 0.42

    line = xt.Line(
        elements=[
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
            xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1357 + Lmb / 2.0 + Lmult / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.3, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(0.787 + Lqd / 2.0 + Lmult / 2.0)),
            xt.Multipole(knl=[0.0, kqd * Lqd], ksl=[0.0, 0.0], length=Lqd, hxl=0.0),
            xt.Drift(length=(0.35 + Lqd / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1457 + Lmult / 2.0 + Lmb / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(2.3295 + Lmult / 2.0 + Lqf / 2.0)),
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
        ],
        element_names=[
            "qf1",
            "drift0",
            "mb1",
            "drift1",
            "mb2",
            "drift2",
            "mb3",
            "drift3",
            "mb4",
            "drift4",
            "multipole1",
            "drift5",
            "qd",
            "drift6",
            "mb5",
            "drift7",
            "mb6",
            "drift8",
            "mb7",
            "drift9",
            "mb8",
            "drift10",
            "multipole2",
            "drift11",
            "qf2",
        ],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    seq = line.to_madx_sequence(sequence_name="test")

    with open(
        test_data_folder.joinpath("normal_forms/test_fodo_1sext.seq"), "w"
    ) as file:
        file.write(seq)

    poly_line = PolyLine4D(
        line=line,
        part=line.particle_ref,
        max_ele_order=8,
        max_map_order=6,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )
    poly_line.calculate_one_turn_map()

    map_xnlbd = poly_line.one_turn_map_real

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_fodo_1sext.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_fodo_1sext.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath("normal_forms/test_fodo_1sext_map_MADNG_O6")
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng


def test_fodo_2sext_map():
    kmb = 0.008445141542
    kqf = 0.014508048734097173
    kqd = -0.014435866477457426

    Lmb = 6.26
    Lqf = 3.085
    Lqd = 3.085
    Lmult = 0.42

    line = xt.Line(
        elements=[
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
            xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1357 + Lmb / 2.0 + Lmult / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.3, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(0.787 + Lqd / 2.0 + Lmult / 2.0)),
            xt.Multipole(knl=[0.0, kqd * Lqd], ksl=[0.0, 0.0], length=Lqd, hxl=0.0),
            xt.Drift(length=(0.35 + Lqd / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1457 + Lmult / 2.0 + Lmb / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, -0.24, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(2.3295 + Lmult / 2.0 + Lqf / 2.0)),
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
        ],
        element_names=[
            "qf1",
            "drift0",
            "mb1",
            "drift1",
            "mb2",
            "drift2",
            "mb3",
            "drift3",
            "mb4",
            "drift4",
            "multipole1",
            "drift5",
            "qd",
            "drift6",
            "mb5",
            "drift7",
            "mb6",
            "drift8",
            "mb7",
            "drift9",
            "mb8",
            "drift10",
            "multipole2",
            "drift11",
            "qf2",
        ],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    seq = line.to_madx_sequence(sequence_name="test")

    with open(
        test_data_folder.joinpath("normal_forms/test_fodo_2sext.seq"), "w"
    ) as file:
        file.write(seq)

    poly_line = PolyLine4D(
        line=line,
        part=line.particle_ref,
        max_ele_order=8,
        max_map_order=6,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )
    poly_line.calculate_one_turn_map()

    map_xnlbd = poly_line.one_turn_map_real

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_fodo_2sext.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_fodo_2sext.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath("normal_forms/test_fodo_2sext_map_MADNG_O6")
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng


def test_fodo_1oct_map():
    kmb = 0.008445141542
    kqf = 0.014508048734097173
    kqd = -0.014435866477457426

    Lmb = 6.26
    Lqf = 3.085
    Lqd = 3.085
    Lmult = 0.42

    line = xt.Line(
        elements=[
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
            xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1357 + Lmb / 2.0 + Lmult / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, -6.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(0.787 + Lqd / 2.0 + Lmult / 2.0)),
            xt.Multipole(knl=[0.0, kqd * Lqd], ksl=[0.0, 0.0], length=Lqd, hxl=0.0),
            xt.Drift(length=(0.35 + Lqd / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1457 + Lmult / 2.0 + Lmb / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(2.3295 + Lmult / 2.0 + Lqf / 2.0)),
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
        ],
        element_names=[
            "qf1",
            "drift0",
            "mb1",
            "drift1",
            "mb2",
            "drift2",
            "mb3",
            "drift3",
            "mb4",
            "drift4",
            "multipole1",
            "drift5",
            "qd",
            "drift6",
            "mb5",
            "drift7",
            "mb6",
            "drift8",
            "mb7",
            "drift9",
            "mb8",
            "drift10",
            "multipole2",
            "drift11",
            "qf2",
        ],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    seq = line.to_madx_sequence(sequence_name="test")

    with open(
        test_data_folder.joinpath("normal_forms/test_fodo_1oct.seq"), "w"
    ) as file:
        file.write(seq)

    poly_line = PolyLine4D(
        line=line,
        part=line.particle_ref,
        max_ele_order=8,
        max_map_order=6,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )
    poly_line.calculate_one_turn_map()

    map_xnlbd = poly_line.one_turn_map_real

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_fodo_1oct.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_fodo_1oct.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath("normal_forms/test_fodo_1oct_map_MADNG_O6")
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng


def test_fodo_2oct_map():
    kmb = 0.008445141542
    kqf = 0.014508048734097173
    kqd = -0.014435866477457426

    Lmb = 6.26
    Lqf = 3.085
    Lqd = 3.085
    Lmult = 0.42

    line = xt.Line(
        elements=[
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
            xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1357 + Lmb / 2.0 + Lmult / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, -6.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(0.787 + Lqd / 2.0 + Lmult / 2.0)),
            xt.Multipole(knl=[0.0, kqd * Lqd], ksl=[0.0, 0.0], length=Lqd, hxl=0.0),
            xt.Drift(length=(0.35 + Lqd / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1457 + Lmult / 2.0 + Lmb / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, 4.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(2.3295 + Lmult / 2.0 + Lqf / 2.0)),
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
        ],
        element_names=[
            "qf1",
            "drift0",
            "mb1",
            "drift1",
            "mb2",
            "drift2",
            "mb3",
            "drift3",
            "mb4",
            "drift4",
            "multipole1",
            "drift5",
            "qd",
            "drift6",
            "mb5",
            "drift7",
            "mb6",
            "drift8",
            "mb7",
            "drift9",
            "mb8",
            "drift10",
            "multipole2",
            "drift11",
            "qf2",
        ],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    seq = line.to_madx_sequence(sequence_name="test")

    with open(
        test_data_folder.joinpath("normal_forms/test_fodo_2oct.seq"), "w"
    ) as file:
        file.write(seq)

    poly_line = PolyLine4D(
        line=line,
        part=line.particle_ref,
        max_ele_order=8,
        max_map_order=6,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )
    poly_line.calculate_one_turn_map()

    map_xnlbd = poly_line.one_turn_map_real

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_fodo_2oct.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_fodo_2oct.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath("normal_forms/test_fodo_2oct_map_MADNG_O6")
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng


def test_fodo_1sext1oct_map():
    kmb = 0.008445141542
    kqf = 0.014508048734097173
    kqd = -0.014435866477457426

    Lmb = 6.26
    Lqf = 3.085
    Lqd = 3.085
    Lmult = 0.42

    line = xt.Line(
        elements=[
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
            xt.Drift(length=(1.9025 + Lqf / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1357 + Lmb / 2.0 + Lmult / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.3, 0.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(0.787 + Lqd / 2.0 + Lmult / 2.0)),
            xt.Multipole(knl=[0.0, kqd * Lqd], ksl=[0.0, 0.0], length=Lqd, hxl=0.0),
            xt.Drift(length=(0.35 + Lqd / 2.0 + Lmb / 2.0)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.38 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.39 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(0.40 + Lmb)),
            xt.Multipole(knl=[kmb], ksl=[0.0], length=Lmb, hxl=kmb),
            xt.Drift(length=(1.1457 + Lmult / 2.0 + Lmb / 2.0)),
            xt.Multipole(
                knl=[0.0, 0.0, 0.0, -6.0],
                ksl=[0.0, 0.0, 0.0, 0.0],
                length=Lmult,
                hxl=0.0,
            ),
            xt.Drift(length=(2.3295 + Lmult / 2.0 + Lqf / 2.0)),
            xt.Multipole(
                knl=[0.0, kqf * Lqf / 2.0], ksl=[0.0, 0.0], length=Lqf / 2.0, hxl=0.0
            ),
        ],
        element_names=[
            "qf1",
            "drift0",
            "mb1",
            "drift1",
            "mb2",
            "drift2",
            "mb3",
            "drift3",
            "mb4",
            "drift4",
            "multipole1",
            "drift5",
            "qd",
            "drift6",
            "mb5",
            "drift7",
            "mb6",
            "drift8",
            "mb7",
            "drift9",
            "mb8",
            "drift10",
            "multipole2",
            "drift11",
            "qf2",
        ],
    )
    line.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=1e11)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.twiss_default["method"] = "4d"

    seq = line.to_madx_sequence(sequence_name="test")

    with open(
        test_data_folder.joinpath("normal_forms/test_fodo_1sext1oct.seq"), "w"
    ) as file:
        file.write(seq)

    poly_line = PolyLine4D(
        line=line,
        part=line.particle_ref,
        max_ele_order=8,
        max_map_order=6,
        nemitt_x=1e-6,
        nemitt_y=1e-6,
        nemitt_z=1,
    )
    poly_line.calculate_one_turn_map()

    map_xnlbd = poly_line.one_turn_map_real

    mad = MAD()
    madx_seq = test_data_folder.joinpath("normal_forms/test_fodo_1sext1oct.seq")
    madng_seq = test_data_folder.joinpath("normal_forms/test_fodo_1sext1oct.mad")
    mad.MADX.load(f"'{madx_seq}'", f"'{madng_seq}'")
    mad.load("MADX", "test")
    mad.send("test.beam = beam {energy = 100}")
    mad.send("mtbl, mflw = track{sequence=test, mapdef=6}")
    madng_out = test_data_folder.joinpath(
        "normal_forms/test_fodo_1sext1oct_map_MADNG_O6"
    )
    mad.send(f'mflw[1]:write("{madng_out}")')

    time.sleep(5)

    map_madng = parse_madng_map(f"{madng_out}.dat")

    os.remove(madx_seq)
    os.remove(madng_seq)
    os.remove(f"{madng_out}.dat")

    assert map_xnlbd == map_madng
