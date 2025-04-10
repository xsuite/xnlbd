import re
import warnings
from typing import Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.special import factorial  # type: ignore[import-untyped]
from xtrack import Line  # type: ignore[import-untyped, import-not-found]


def _i_pow(n: int) -> complex:
    """
    Function to compute nth power of the complex number i.

    Input:
        - n: integer number

    Output:
        - complex number, nth power of i
    """

    return 1j ** (n % 4)


def _dmus_at_element(
    twiss_data: dict[str, Union[float, np.ndarray]], loc: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the phase advance differences between all line
    elements and a given element at an integer location. If the differences
    were negative, the tune is added.

    Input:
        - twiss_data: dictionary containing all relevant data from the
          Twiss table, including phase advances and tunes in both planes
        - loc: integer indicating the location to which other elements
          are compared

    Output:
        - tuple of numpy arrays containing the relative phase advances
    """

    mux = twiss_data["mux"]
    muy = twiss_data["muy"]

    if not isinstance(mux, np.ndarray) or not isinstance(muy, np.ndarray):
        raise ValueError("Phase advances must be defined for all elements!")

    dmux = np.where(
        mux[loc] - mux >= 0,
        mux[loc] - mux,
        mux[loc] - mux + twiss_data["qx"],
    )
    dmuy = np.where(
        muy[loc] - muy >= 0,
        muy[loc] - muy,
        muy[loc] - muy + twiss_data["qy"],
    )

    return dmux, dmuy


def _create_twiss_data(line: Line) -> dict[str, Union[float, np.ndarray]]:
    """
    Function to create a dictionary containing all necessary data
    from the Twiss.

    Input:
        - line: xsuite line

    Output:
        - dictionary storing the tunes, beta functions, phase
          advances and orbits in both planes for all line
          elements (where applicable)
    """

    twiss_data = {}

    try:
        twiss_tab = line.twiss(strengths=True)
        twiss_data["name"] = twiss_tab.name[:-1]
        twiss_data["qx"] = twiss_tab.qx
        twiss_data["qy"] = twiss_tab.qy
        twiss_data["betx"] = twiss_tab.betx[:-1]
        twiss_data["bety"] = twiss_tab.bety[:-1]
        twiss_data["mux"] = twiss_tab.mux[:-1]
        twiss_data["muy"] = twiss_tab.muy[:-1]
        twiss_data["x"] = twiss_tab.x[:-1]
        twiss_data["y"] = twiss_tab.y[:-1]

        twiss_keys = list(twiss_tab.keys())
        k_pattern = re.compile(r"k\d+l")
        k_keys = [s for s in twiss_keys if k_pattern.fullmatch(s)]
        j_pattern = re.compile(r"k\d+sl")
        ks_keys = [s for s in twiss_keys if j_pattern.fullmatch(s)]
        j_keys = ["j" + s[1:-2] + "l" for s in ks_keys]
        for i in range(len(k_keys)):
            twiss_data[k_keys[i]] = twiss_tab[k_keys[i]][:-1]
        for i in range(len(ks_keys)):
            twiss_data[j_keys[i]] = twiss_tab[ks_keys[i]][:-1]
    except ValueError:
        raise ValueError("Line cannot be twissed, fix issue and try again!")

    return twiss_data


def _calc_single_rdt_single_loc(
    twiss_data: dict[str, Union[float, np.ndarray]],
    feeddown: int,
    pqrt: Tuple[int, int, int, int],
    loc: int,
) -> complex:
    """
    Function to compute a single resonance driving term at a single
    location in the line, taking into account the desired level of
    feeddown. The function implements equation (A8) from
    https://arxiv.org/abs/1711.06589.

    Input:
        - twiss_data: dictionary storing the tunes, beta functions, phase
          advances and orbits in both planes for all line
          elements (where applicable)
        - feeddown: integer indicating the level of feeddown to take into
          account
        - pqrt: tuple of integers indicating the resonance driving term to
          be computed
        - loc: integer indicating the location at which the resonance
          driving term is to be computed

    Output:
        - a single complex number, the value of the resonance driving term
          at the desired location
    """

    p, q, r, t = pqrt

    n = p + q + r + t

    f_denom = 1.0 - np.exp(
        2 * np.pi * 1j * ((p - q) * twiss_data["qx"] + (r - t) * twiss_data["qy"])
    )

    dx_idy = twiss_data["x"] + 1j * twiss_data["y"]
    if not isinstance(dx_idy, np.ndarray):
        raise ValueError("x and y orbits must be defined for all elements!")
    kl_ijl = twiss_data[f"k{n-1:d}l"] + 1j * twiss_data[f"j{n-1:d}l"]
    if not isinstance(kl_ijl, np.ndarray):
        raise ValueError("Strengths must be defined for all elements!")
    for i in range(1, feeddown + 1):
        n_mad = n + i - 1
        try:
            curr_kl_ijl = twiss_data[f"k{n_mad:d}l"] + 1j * twiss_data[f"j{n_mad:d}l"]
            kl_ijl += (curr_kl_ijl * (dx_idy**i)) / factorial(q)
        except KeyError:
            raise KeyError("Feeddown order {:d} too high with given line, would need magnets of order {:d}"\
                           .format(feeddown, n_mad))
    kljl_real = np.real(kl_ijl * _i_pow(r + t))
    sources = np.where(kljl_real != 0)[0]

    betx = twiss_data["betx"]
    bety = twiss_data["bety"]
    if not isinstance(betx, np.ndarray) or not isinstance(bety, np.ndarray):
        raise ValueError("Beta functions must be defined for all elements!")

    h_fact = -1.0 / (factorial(p) * factorial(q) * factorial(r) * factorial(t) * 2**n)
    h_terms = (
        kljl_real[sources]
        * (betx[sources]) ** ((p + q) / 2.0)
        * (bety[sources]) ** ((r + t) / 2.0)
    )

    dmux, dmuy = _dmus_at_element(twiss_data, loc)
    dmux = dmux[sources]
    dmuy = dmuy[sources]

    fpqrt = (
        h_fact
        * np.sum(h_terms * np.exp(2 * np.pi * 1j * ((p - q) * dmux + (r - t) * dmuy)))
        / f_denom
    )

    return fpqrt


def calculate_rdts(
    line: Line, feeddown: int, rdts: list[str], locations: Union[str, list[str]]
) -> DataFrame:
    """
    Function to evaluate one or more resonance driving terms at one or
    more locations around the ring, including the desired level of
    feeddown. The function implements equation (A8) from
    https://arxiv.org/abs/1711.06589.

    Input:
        - line: xsuite line
        - feeddown: integer indicating the level of feeddown to take
          into account
        - rdts: list of strings indicating the desired resonance
          driving terms; the RDTs should be of the following form:
          "f3000", i.e. always 5 character, starting with "f",
          followed by 4 integersd
        - locations: list of strings, names of line elements where the
          resonance driving terms should be computed, alternatively
          passing "all" returns the resonance driving terms at all
          elements in the line

    Output:
        - Pandas DataFrame, indices will be the locations (element names),
          columns will be the resonance driving term names
    """

    twiss_data = _create_twiss_data(line)

    names = twiss_data["name"]
    if not isinstance(names, np.ndarray):
        raise ValueError("Names must be defined for all elements!")

    num_rdts = len(rdts)
    if isinstance(locations, list):
        num_locs = len(locations)
    elif isinstance(locations, str) and locations == "all":
        num_locs = len(names)
    else:
        raise ValueError(
            "Incorrect locations requested, must be 'all' or list of element names!"
        )

    rdt_data = np.full([num_locs, num_rdts], np.nan, dtype=complex)

    for i in range(num_rdts):
        if (len(rdts[i]) != 5) or (rdts[i][0] != "f"):
            warnings.warn(
                "Invalid RDT requested, ignoring and moving on. Will return NaN."
            )
            continue
        p = int(rdts[i][1])
        q = int(rdts[i][2])
        r = int(rdts[i][3])
        t = int(rdts[i][4])
        if (p + q + r + t) <= 1:
            warnings.warn(
                "Invalid RDT requested, ignoring and moving on. Will return NaN."
            )
            continue

        if locations == "all":
            for j in range(num_locs):
                rdt_data[j][i] = _calc_single_rdt_single_loc(
                    twiss_data, feeddown, (p, q, r, t), j
                )
        else:
            for j in range(num_locs):
                loc_idxs = np.where(names == locations[j])[0]
                if len(loc_idxs) == 0:
                    warnings.warn(
                        "Element not in line, ignoring and moving on. Will return NaN."
                    )
                    continue
                loc_idx = int(loc_idxs[0])

                rdt_data[j][i] = _calc_single_rdt_single_loc(
                    twiss_data, feeddown, (p, q, r, t), loc_idx
                )

    if isinstance(locations, str):
        rdt_df = pd.DataFrame(data=rdt_data, index=names, columns=rdts)
    else:
        rdt_df = pd.DataFrame(data=rdt_data, index=locations, columns=rdts)

    return rdt_df
