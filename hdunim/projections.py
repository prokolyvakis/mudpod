"""Projections defintion."""
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial.distance import mahalanobis

from hdunim.misc import assert_correct_input_size
from hdunim.observer import Observer


class ProjectionDim(Protocol):
    """The Projection Dim interface."""

    @classmethod
    def estimate(cls, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """


class IdentityDim(ProjectionDim):
    """An identity transform that retains the initial dim."""

    @classmethod
    def estimate(cls, dim: int) -> int:
        """Returns the initial dimension.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            The original dimension of the Euclidean space that the data lie on.
        """

        logger.debug('No projection is performed.')
        return dim


class JohnsonLindenstraussDim(ProjectionDim):
    """An estimate of the projection dim based on the Johnson-Lindenstrauss lemma."""

    @classmethod
    def estimate(cls, dim: int) -> int:
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

        logger.debug(f'Based on the Johnson-Lindenstrauss lemma, the data will be '
                     f'projected in the {projection_dim}-D space.')

        return projection_dim


class JohnsonLindenstraussOrAscend(JohnsonLindenstraussDim):

    @classmethod
    def estimate(cls, dim: int) -> int:
        """Estimate the dimension over which the projection will be performed.

        Args:
            dim: the original dimension of the Euclidean space that the data lie on.
        Returns:
            An integer indicating the projection dimension.
        """

        projection_dim = super().estimate(dim)

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

    projection_dim: ProjectionDim
    # The dimension over which the projection will be performed.

    observer: Observer
    # The policy on how to pick an observer.

    def project(self, arr: np.ndarray) -> np.ndarray:
        """Get the projected data.

        Args:
            arr: A 2D numpy array with the first dimension being the number of different
                    datapoints and the second being the features' size.
        Returns:
            A 2D numpy array with the projected data.
        """
        assert_correct_input_size(arr)

        if self.projection_dim == IdentityDim:
            return arr

        arr_d = arr.shape[1]
        d = self.projection_dim.estimate(arr_d)
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
        c = np.cov(x.T)

        return np.apply_along_axis(
            lambda a: mahalanobis(a, o, c),
            0,
            x.T
        )
