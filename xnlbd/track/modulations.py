import numpy as np

# Amplitude of different 50 Hz harmonics measured in the SPS tune modulation
# as described in
# REF: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.57.3432

SPS_REV_FREQ_IN_TURNS = 868.12

SPS_COEFFICIENTS = np.array(
    [1.000e-4, 0.218e-4, 0.708e-4, 0.254e-4, 0.100e-4, 0.078e-4, 0.218e-4]
)
SPS_MODULATION = np.array(
    [
        1 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
        2 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
        3 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
        6 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
        7 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
        10 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
        12 * (2 * np.pi / SPS_REV_FREQ_IN_TURNS),
    ]
)


def make_sps_modulation(base_tune, epsilon, turns=np.arange(86812)):
    """Get the SPS modulation for the given tune and epsilon. Units are in
    radians. REF: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.57.3432

    Arguments:
    ----------
    base_tune : float
        The base tune for the modulation.
    epsilon : float
        The modulation amplitude. Usually between 0 (no modulation) and
        64.0 (strong modulation).
    turns : np.ndarray, optional
        The given turns at which to calculate the modulation. Default is
        np.arange(86812).

    Returns:
    --------
    tunes : np.ndarray
        The modulated tunes in radians.
    """
    tunes = np.zeros_like(turns, dtype=float)
    for i, modulation in enumerate(SPS_MODULATION):
        tunes += SPS_COEFFICIENTS[i] * np.cos(modulation * turns)
    tunes = base_tune * (1 + epsilon * tunes)
    return tunes
