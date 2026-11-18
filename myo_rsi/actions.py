"""Native hand action adapter for the verified normalized sigmoid path."""

import numpy as np


def _array(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite control")
    return values


def excitation(native):
    action = np.clip(_array(native), -1, 1)
    return 1.0 / (1.0 + np.exp(-5.0 * (action - 0.5)))


def native_action(desired):
    """Return legal native action and signed excitation projection error."""
    desired = _array(desired)
    low, high = excitation([-1, 1])
    projected = np.clip(desired, low, high)
    native = 0.5 + (np.log(projected) - np.log1p(-projected)) / 5.0
    return np.clip(native, -1, 1), projected - desired
