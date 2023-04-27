"""Observers definition."""
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
import numpy as np
from scipy.spatial.distance import mahalanobis

from hdunim.misc import assert_correct_input_size
from hdunim.misc import ordinal


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

    def __post_init__(self):
        assert 0 < self.percentile < 1, 'The percentile should lie in (0, 1) interval.'

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
        c = np.cov(arr.T)

        ds = np.apply_along_axis(
            lambda a: mahalanobis(a, m, c),
            0, arr.T
        )

        t = np.percentile(
            ds,
            int(100 * self.percentile)
        )

        logger.debug(f'The percentile threshold is {t}.')

        ps = np.argwhere(
            ds > t
        ).ravel()

        o = np.random.choice(ps, size=1, replace=False)[0]

        logger.debug(f'The observer chosen was the {ordinal(o)} datapoint.')
        return o
