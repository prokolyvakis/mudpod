"""Projections defintion."""
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import Protocol

from loguru import logger
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from hdunim.misc import assert_correct_input_size
from hdunim.misc import Distance
from hdunim.observer import Observer


class Projector(Protocol):
    """The Projection Dim interface."""

    def estimate_dim(self, n: int, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            n: the number of data samples.
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """


class IdentityProjector(Projector):
    """An identity transform that retains the initial dim."""

    def estimate_dim(self, n: int, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            n: the number of data samples.
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """
        return dim


@dataclass
class JohnsonLindenstrauss(Projector):
    """An estimate of the projection dim based on the Johnson-Lindenstrauss lemma."""
    
    eps: float = 0.99
    # the distortion factor.

    def estimate_dim(self, n: int, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            n: the number of data samples.
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """
        projection_dim = np.log2(n)
        projection_dim = max(
            int(projection_dim * 8 / (self.eps * self.eps)),
            1
        )
        return projection_dim


@dataclass
class View:
    """A View wrapper class."""

    projector: Projector
    # A projector that yields the dimension over which the projection will be performed.

    observer: Observer
    # The policy on how to pick an observer.

    dtype: InitVar[str] = 'mahalanobis'
    # The distance type.

    distance: Distance = field(init=False, repr=True)
    # The distance.

    def __post_init__(self, dtype: str):
        self.distance = Distance(dtype=dtype)

    def project(self, arr: np.ndarray) -> np.ndarray:
        """Get the projected data.

        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                    datapoints and the second being the features' size.
        Returns:
            A 2D numpy array with the projected data.
        """
        assert_correct_input_size(arr)

        if self.projector == IdentityProjector:
            return arr

        arr_n = arr.shape[0]
        arr_d = arr.shape[1]
        d = self.projector.estimate_dim(arr_n, arr_d)
        p = GaussianRandomProjection(n_components=d)

        return p.fit_transform(arr)

    def distances(self, arr: np.ndarray) -> np.ndarray:
        """Compute the distances from an observer.

        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                    datapoints and the second being the features' size.
        Returns:
            A 1D numpy array with the distances from a picked observer.
        """

        x = self.project(arr)
        o = self.observer.get(x)

        return self.distance.compute(x, o)
