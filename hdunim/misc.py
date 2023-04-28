"""Implementation of various auxiliary functions."""
import random

import numpy as np
import torch


def assert_correct_input_size(arr: np.ndarray) -> None:
    """Assert that the tensor is a 2D array.
    Args:
        arr: A 2D numpy array with the first dimension being the number of different
            datapoints and the second being the features' size.
    Raises:
        AssertionError: if the tensor has a different shape.
    """
    shape = arr.shape
    assert len(shape) == 2, (
        f'The tensor should be a 2D array! It has this shape instead: {shape}!'
    )


def ordinal(n: int):
    """Get the ordinal of the number."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def set_seed(s: int):
    """Set the random seed.
    Args:
        s: the random seed to be set
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
