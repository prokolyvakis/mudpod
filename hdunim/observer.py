"""Observers definition."""
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import Protocol

import numpy as np

from hdunim.misc import assert_correct_input_size
from hdunim.misc import Distance


class Observer(Protocol):
    """The Observer interface."""

    def get(self, arr: np.ndarray) -> np.ndarray:
        """Get the observer.

        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            An observer with the same features' size.
        """


@dataclass
class PercentileObserver(Observer):
    """Sample uniformly an observer from a certain percentile."""

    percentile: float
    # The percentile to be sampled over.

    dtype: InitVar[str] = 'mahalanobis'
    # The distance type.

    distance: Distance = field(init=False, repr=True)
    # The distance.

    def __post_init__(self, dtype: str):
        assert 0 < self.percentile < 1, 'The percentile should lie in (0, 1) interval.'
        self.distance = Distance(dtype=dtype)

    def get(self, arr: np.ndarray) -> np.ndarray:
        """Get the observer.
        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            An observer with the same features' size.
        """
        assert_correct_input_size(arr)
        m = np.mean(arr, axis=0)

        ds = self.distance.compute(arr, m)

        t = np.percentile(
            ds,
            int(100 * self.percentile)
        )

        ps = np.argwhere(
            ds > t
        ).ravel()

        o_i = np.random.choice(ps, size=1, replace=False)[0]

        o = arr[o_i, :].T
        return o


@dataclass
class RandomObserver(Observer):
    """Sample uniformly an observer from the dataset."""

    def get(self, arr: np.ndarray) -> np.ndarray:
        """Get the observer.
        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            An observer with the same features' size.
        """
        assert_correct_input_size(arr)

        n = arr.shape[0]

        o_i = np.random.choice(n, size=1, replace=False)[0]

        o = arr[o_i, :].T
        return o
