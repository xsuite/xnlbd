import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit
from tqdm.auto import tqdm

from ...tools.generic_writer import GenericWriter, H5pyWriter, LocalWriter


@njit
def _svd_values_product(gali_matrix):
    if np.any(np.isnan(gali_matrix)):
        return np.nan
    else:
        _, s, _ = np.linalg.svd(gali_matrix)
        return np.prod(s)


@njit
def _gali_eval(gali_matrix):
    gali_matrix = np.transpose(gali_matrix, (2, 0, 1))
    gali = []
    for m in gali_matrix:
        gali.append(_svd_values_product(m))
    gali = np.asarray(gali)
    return gali


COORD_LIST = ["x", "px", "y", "py", "zeta", "ptau"]
NORM_COORD_LIST = ["x_norm", "px_norm", "y_norm", "py_norm", "zeta_norm", "pzeta_norm"]


def _make_gali_combos(coord_list):
    return {
        "2": list(itertools.combinations(coord_list, 2)),
        "3": list(itertools.combinations(coord_list, 3)),
        "4": list(itertools.combinations(coord_list, 4)),
        "5": list(itertools.combinations(coord_list, 5)),
        "6": list(itertools.combinations(coord_list, 6)),
        "all": list(itertools.combinations(coord_list, 2))
        + list(itertools.combinations(coord_list, 3))
        + list(itertools.combinations(coord_list, 4))
        + list(itertools.combinations(coord_list, 5))
        + list(itertools.combinations(coord_list, 6)),
    }


def gali_extractor(
    input_writer: GenericWriter,
    output_writer: GenericWriter,
    times: Union[List[int], np.ndarray],
    coord_list: Union[Tuple[str], List[str]] = tuple(NORM_COORD_LIST),
    which_gali: str = "all",
    custom_combos: Optional[List[Tuple[str]]] = None,
    preload_data: bool = False,
    coord_list_nested: Optional[List[str]] = None,
    overwrite: bool = False,
):
    """Calculate the Generalized Alignment Index (GALI) for a set of
    coordinates managed by the GhostParticleManager class and tracked with
    the displacement_tracker module.

    Parameters
    ----------
    input_writer : GenericWriter
        The writer containing the data to be converted.
    output_writer : GenericWriter
        The writer to which the converted data will be written.
    times : list
        The times at which to calculate GALI.
    coord_list : list, optional
        The coordinates to use in the GALI calculation. Defaults to
        NORM_COORD_LIST.
    which_gali : str, optional
        Which GALI to calculate. Options are '2', '3', '4', '5', '6', 'all',
        and 'custom'. Defaults to 'all'.
    custom_combos : list, optional
        A list of tuples containing the coordinates to use in the GALI
        calculation. Only used if which_gali is 'custom'. Defaults to None.
    preload_data : bool, optional
        Whether to preload the data from the input_writer. Only works if the
        input_writer is an H5pyWriter. Defaults to False.
    coord_list_nested : list, optional
        The second list of coordinates to use in the GALI calculation. Defaults to None. If it is None, then coord_list_nested is set to coord_list.
    overwrite : bool, optional
        Whether to overwrite the output file. Defaults to False.
    """
    if preload_data and not isinstance(input_writer, H5pyWriter):
        raise ValueError("preload_data only works with H5pyWriter")
    elif isinstance(input_writer, H5pyWriter):
        print("preloading data")
        input_writer = input_writer.convert_to_localwriter()

    if custom_combos is not None:
        if which_gali != "custom":
            raise ValueError("which_gali must be 'custom' if using custom_combos")
        combo_list = custom_combos
    else:
        combo_list = _make_gali_combos(coord_list)[which_gali]

    if coord_list_nested is None:
        coord_list_nested = tuple(coord_list)

    nested_dict = dict(zip(coord_list, coord_list_nested))

    print("evaluating gali")
    for combo in tqdm(combo_list):
        for t in tqdm(times):
            dataset_name = f"gali{len(combo)}/{'_'.join(combo)}/{t}"
            if not overwrite and output_writer.dataset_exists(dataset_name):
                print(
                    f"Skipping {dataset_name} as it already exists in the output file"
                )
                continue

            gali_matrix = np.asarray(
                [
                    [
                        input_writer.get_data(f"direction/{a}/{nested_dict[x]}/{t}")
                        for x in coord_list
                    ]
                    for a in combo
                ]
            )
            disp = _gali_eval(gali_matrix)

            output_writer.write_data(dataset_name, data=disp, overwrite=overwrite)
