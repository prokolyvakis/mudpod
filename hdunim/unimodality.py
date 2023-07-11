"""Definition of unimodality tests."""
from dataclasses import dataclass

from diptest import diptest
from loguru import logger
from mpire import WorkerPool
import numpy as np

from hdunim.projections import View


@dataclass
class UnimodalityTest:
    """A wrapper class for checking the unimodality of a dataset using views."""

    view: View
    # The view over which the dip test will be performed.

    pval: float
    # A float in the interval (0, 1) indicating the p_value.

    boot_pval: bool = False
    # A boolean flag indicating whether to boostrap the dip value.

    def __post_init__(self):
        assert 0. < self.pval < 1., f'The p_value {self.pval} does not lie in (0, 1).'

    def test(self, x: np.ndarray, parallel: bool = False) -> bool:
        """A test that assesses the \alpha-unimodality of a dataset using a random view.

        Args:
        x: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        parallel: a boolean flag indicating whether the test is part of a parallel
                Monte Carlo test [default: False]
        Returns:
            A boolean indicating whether the data follow a \alpha-unimodal distribution.
        """
        ds = self.view.distances(x, parallel)

        _, pv = diptest(ds, boot_pval=self.boot_pval)

        logger.debug(f'The unimodality statistic is: {pv}.')

        return pv > self.pval


@dataclass
class MonteCarloUnimodalityTest:
    """
    A wrapper class for checking the unimodality of a dataset using Monte Carlo views.
    """

    tester: UnimodalityTest
    # A unimodality tester.

    sim_num: int
    # The number of Monte Carlo simulations.

    workers_num: int
    # The number of workers.

    def __post_init__(self):
        if self.workers_num < 0:
            raise ValueError('The number of works must be non-negative!')
        if self.sim_num <= 1:
            raise ValueError("The simulations' number must be greater than 1!")

    def estimate(self, x: np.ndarray) -> float:
        """Estimate the ecdf of the Monte Carlo \alpha-unimodality statistical test.

        Args:
            x: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            The estimated probability.
        """
        def generator(n):
            i = 0
            p = self.workers_num > 1
            while i < n:
                yield (x, p)
                i += 1

        func = self.tester.test
        with WorkerPool(n_jobs=self.workers_num) as pool:
            tests = pool.map(
                func,
                generator(self.sim_num),
                iterable_len=self.sim_num
            )

        logger.debug(f'The result of the Monte Carlo simulations is: {tests}')

        pv = (1. * np.count_nonzero(tests)) / self.sim_num

        return 1. - pv

    def test(self, x: np.ndarray) -> bool:
        """A test that assesses the \alpha-unimodality of a dataset using a random view.

        Args:
            x: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            A boolean indicating whether the data follow a \alpha-unimodal distribution.
        """
        pv = self.estimate(x)

        return pv < self.tester.pval
