"""Definition of unimodality tests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil

from diptest import diptest
from loguru import logger
import numpy as np

from mudpod.projections import View


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
        if self.workers_num <= 0:
            raise ValueError('The number of works must be positive!')
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
        def task_ranges(total, batch_size):
            for start in range(0, total, batch_size):
                yield (start, min(start + batch_size, total))

        def task(start_index, end_index):
            results = [self.tester.test(x, self.workers_num > 1) for _ in range(start_index, end_index)]
            return results
        
        batch_size = ceil(self.sim_num / self.workers_num)
        with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            futures = [executor.submit(task, start, end) for start, end in task_ranges(self.sim_num, batch_size)]
            tests = [result for future in futures for result in future.result()]

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
