"""Implementation of various auxiliary functions."""
from dataclasses import dataclass
from os import getpid
from time import time
import random


import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
import torch


@dataclass
class Distance:
    """A wrapper class for distances."""

    dtype: str
    # The distance type.

    allowed_dtypes: frozenset = frozenset(['euclidean', 'mahalanobis'])
    # The allowed distances' types.

    def __post_init__(self):
        assert self.dtype in self.allowed_dtypes, (
            f'The type {self.dtype} is not supported!'
        )

    def compute(self, arr: np.ndarray, o: np.ndarray) -> np.ndarray:
        """Computes the distances from a given point.

        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                    datapoints and the second being the features' size.
            o: A 2D numpy array with the first dimension always equal to 1 and
                    the second being the features' size.
        Returns:
            The distances with respect to the observer `o`.
        """
        def dist(x: np.ndarray) -> float:
            if self.dtype == 'euclidean':
                return euclidean(x, o)
            if self.dtype == 'mahalanobis':
                c = np.cov(arr.T)
                return mahalanobis(x, o, c)

        return np.apply_along_axis(
            dist,
            0,
            arr.T
        )


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
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def get_random_seed() -> int:
    """Get a pseudo random seed."""
    return (getpid() * int(time())) % 123456789
