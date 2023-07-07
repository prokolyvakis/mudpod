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

    @classmethod
    def estimate_dim(cls, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """


class IdentityProjector(Projector):
    """An identity transform that retains the initial dim."""

    @classmethod
    def estimate_dim(cls, dim: int) -> int:
        """Returns the initial dimension.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            The original dimension of the Euclidean space that the data lie on.
        """
        return dim


class JohnsonLindenstrauss(Projector):
    """An estimate of the projection dim based on the Johnson-Lindenstrauss lemma."""

    @classmethod
    def estimate_dim(cls, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """
        projection_dim = max(
            np.log2(dim).astype(int),
            1
        )
        return projection_dim


class ExponentiallyDescendOrAscend(JohnsonLindenstrauss):

    @classmethod
    def estimate_dim(cls, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """

        projection_dim = super().estimate_dim(dim)

        if projection_dim == 1:
            projection_dim = np.ceil(np.exp2(dim)).astype(int)

            logger.debug(
                f'Ascend Case: the data will be projected into '
                f'the {projection_dim}-D space'
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

        arr_d = arr.shape[1]
        d = self.projector.estimate_dim(arr_d)
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
