import numpy as np


def birkhoff_weights(n: int):
    """Get the Birkhoff weights for a given number of samples.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Array of Birkhoff weights.
    """
    weights = np.arange(n, dtype=np.float64)
    weights /= n
    weights = np.exp(-1 / (weights * (1 - weights)))
    return weights / np.sum(weights)
