"""Implementation of various auxiliary functions."""
import random

import numpy as np


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


def set_seed(s: int):
    """Set the random seed.
    Args:
        s: the random seed to be set
    """
    np.random.seed(s)
    random.seed(s)
