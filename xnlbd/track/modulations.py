import numpy as np

# REF: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.57.3432
SPS_COEFFICIENTS = np.array(
    [1.000e-4, 0.218e-4, 0.708e-4, 0.254e-4, 0.100e-4, 0.078e-4, 0.218e-4]
)
SPS_MODULATION = np.array(
    [
        1 * (2 * np.pi / 868.12),
        2 * (2 * np.pi / 868.12),
        3 * (2 * np.pi / 868.12),
        6 * (2 * np.pi / 868.12),
        7 * (2 * np.pi / 868.12),
        10 * (2 * np.pi / 868.12),
        12 * (2 * np.pi / 868.12),
    ]
)


def make_sps_modulation(base_tune, epsilon, times=np.arange(86812)):
    """Get the SPS modulation for the given tune and epsilon. Units are in
    radians. REF: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.57.3432

    Arguments:
    ----------
    base_tune : float
        The base tune for the modulation.
    epsilon : float
        The modulation amplitude. Usually between 0 (no modulation) and 
        64.0 (strong modulation).
    times : np.ndarray, optional
        The times at which to calculate the modulation. Default is np.arange(86812).

    Returns:
    --------
    tunes : np.ndarray
        The modulated tunes in radians.
    """
    tunes = np.zeros_like(times, dtype=float)
    for i, modulation in enumerate(SPS_MODULATION):
        tunes += SPS_COEFFICIENTS[i] * np.cos(modulation * times)
    tunes = base_tune * (1 + epsilon * tunes)
    return tunes
