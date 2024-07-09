import pathlib

import numpy as np
import xtrack as xt  # type: ignore[import-not-found]
from scipy.special import factorial  # type: ignore[import-untyped]

from xnlbd.analyse.rdt import calculate_rdts

test_data_folder = pathlib.Path(__file__).parent.joinpath("test_data").absolute()


def test_f3000():
    # Define tunes
    qx = 26.33
    qy = 26.13

    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tunes
    line.vv["qh_setvalue"] = qx
    line.vv["qv_setvalue"] = qy

    # Set chromatic sextupole strengths to 0
    line.vv["klsfa"] = 0.0
    line.vv["klsfb"] = 0.0
    line.vv["klsfc"] = 0.0
    line.vv["klsda"] = 0.0
    line.vv["klsdb"] = 0.0

    # Turn on single extraction sextupole
    k2 = 0.2
    L = 0.74
    line.vv["klse40602"] = k2

    # Twiss for analytical calculation
    twiss = line.twiss(continue_on_closed_orbit_error=False)

    # Define elements at which to calculate
    locations = ["veqf.10010.a", "lse.40602", "drift_10806"]
    loc_idx = []
    for i in range(len(locations)):
        loc_idx.append(np.where(twiss.name == locations[i])[0][0])

    # Calculate analytically the f3000 RDT
    h3000 = -k2 * L * twiss["betx"][loc_idx[1]] ** (3 / 2) / (factorial(3) * 2**3)
    f3000_analytical = [
        h3000
        * np.exp(
            3j
            * 2
            * np.pi
            * (twiss["mux"][loc_idx[0]] - twiss["mux"][loc_idx[1]] + twiss.qx)
        )
        / (1 - np.exp(6j * np.pi * twiss.qx)),
        h3000
        * np.exp(3j * 2 * np.pi * (twiss["mux"][loc_idx[1]] - twiss["mux"][loc_idx[1]]))
        / (1 - np.exp(6j * np.pi * twiss.qx)),
        h3000
        * np.exp(3j * 2 * np.pi * (twiss["mux"][loc_idx[2]] - twiss["mux"][loc_idx[1]]))
        / (1 - np.exp(6j * np.pi * twiss.qx)),
    ]

    # Calculate the RDT with the function
    rdt_table = calculate_rdts(
        line=line,
        feeddown=0,
        rdts=["f3000"],
        locations=["veqf.10010.a", "lse.40602", "drift_10806"],
    )

    assert np.isclose(rdt_table.loc[locations[0]], f3000_analytical[0], atol=1e-10)
    assert np.isclose(rdt_table.loc[locations[1]], f3000_analytical[1], atol=1e-10)
    assert np.isclose(rdt_table.loc[locations[2]], f3000_analytical[2], atol=1e-10)


def test_f1120():
    # Define tunes
    qx = 26.251
    qy = 26.13

    # Load xsuite line
    line = xt.Line.from_json(test_data_folder.joinpath("sps_100GeV_lhc_q26.json"))

    # Set tunes
    line.vv["qh_setvalue"] = qx
    line.vv["qv_setvalue"] = qy

    # Set chromatic sextupole strengths to 0
    line.vv["klsfa"] = 0.0
    line.vv["klsfb"] = 0.0
    line.vv["klsfc"] = 0.0
    line.vv["klsda"] = 0.0
    line.vv["klsdb"] = 0.0

    # Turn on single extraction octupole
    k3 = 6.0
    L = 0.74
    line.vv["kloe22002"] = k3

    # Twiss for analytical calculation
    twiss = line.twiss(continue_on_closed_orbit_error=False)

    # Define elements at which to calculate
    locations = ["veqf.10010.a", "loe.22002", "drift_10806"]
    loc_idx = []
    for i in range(len(locations)):
        loc_idx.append(np.where(twiss.name == locations[i])[0][0])

    # Calculate analytically the f1120 RDT
    h1120 = (
        k3
        * L
        * twiss["betx"][loc_idx[1]]
        * twiss["bety"][loc_idx[1]]
        / (factorial(2) * 2**4)
    )
    f1120_analytical = [
        h1120
        * np.exp(
            2j
            * 2
            * np.pi
            * (twiss["muy"][loc_idx[0]] - twiss["muy"][loc_idx[1]] + twiss.qy)
        )
        / (1 - np.exp(4j * np.pi * twiss.qy)),
        h1120
        * np.exp(2j * 2 * np.pi * (twiss["muy"][loc_idx[1]] - twiss["muy"][loc_idx[1]]))
        / (1 - np.exp(4j * np.pi * twiss.qy)),
        h1120
        * np.exp(2j * 2 * np.pi * (twiss["muy"][loc_idx[2]] - twiss["muy"][loc_idx[1]]))
        / (1 - np.exp(4j * np.pi * twiss.qy)),
    ]

    # Calculate the RDT with the function
    rdt_table = calculate_rdts(
        line=line,
        feeddown=0,
        rdts=["f1120"],
        locations=["veqf.10010.a", "loe.22002", "drift_10806"],
    )

    assert np.isclose(rdt_table.loc[locations[0]], f1120_analytical[0], atol=1e-10)
    assert np.isclose(rdt_table.loc[locations[1]], f1120_analytical[1], atol=1e-10)
    assert np.isclose(rdt_table.loc[locations[2]], f1120_analytical[2], atol=1e-10)
