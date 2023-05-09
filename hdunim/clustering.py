"""Clustering algorithms."""
from dataclasses import dataclass
from dataclasses import field
from numbers import Integral
from numbers import Real
from typing import ClassVar
from typing import Optional
from typing import Union

from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils._param_validation import Interval

from hdunim.projections import View
from hdunim.unimodality import UnimodalityTester
from hdunim.unimodality import MonteCarloUnimodalityTester


@dataclass
class DipMeans(KMeans):
    """Dip-means clustering algorithm."""

    view: View
    # The view over which the dip test will be performed.

    pval: float
    # A float in the interval (0, 1) indicating the p_value.

    sim_num: int
    # The number of Monte Carlo simulations.

    workers_num: int
    # The numbers of workers.

    boot_pval: bool = False
    # A boolean flag indicating whether to boostrap the dip value.

    n_clusters: int = 1

    init: Union[str, np.ndarray] = 'k-means++'

    n_init: str = 'warn'

    max_iter: int = 300

    tol: float = 1e-4

    verbose: int = 8

    random_state: Optional[int] = None

    copy_x: bool = True

    algorithm: str = 'lloyd'

    mc_test: MonteCarloUnimodalityTester = field(init=False, repr=False)

    _parameter_constraints: dict = field(default_factory=lambda: {
            **KMeans._parameter_constraints,
            'pval': [Interval(Real, 0, 1, closed="neither")],
            'sim_num': [Interval(Integral, 1, None, closed="left")],
            'workers_num': [Interval(Integral, 0, None, closed="left")],
        }
    )

    _min_input_size: ClassVar[int] = 5
    # The minimum input size that is considered unimodal without further checking.

    def __post_init__(self):
        super().__init__(
            n_clusters=1,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm
        )

        t = UnimodalityTester(self.view, self.pval, self.boot_pval)
        self.mc_test = MonteCarloUnimodalityTester(t, self.sim_num, self.workers_num)

    def _is_unimodal(self, x: np.ndarray) -> bool:
        """Method that checks whether the input is unimodal.

        Args:
            x: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            A boolean value indicating whether the input is unimodal.
        """
        return True if x.shape[0] < self._min_input_size else self.mc_test.test(x)

    def _estimate_unimodality(self, x: np.ndarray) -> float:
        """Estimate the ecdf of the Monte Carlo \alpha-unimodality statistical test.

        Args:
            x: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        Returns:
            The estimated probability.
        """
        return 0. if x.shape[0] < self._min_input_size else self.mc_test.estimate(x)

    def fit(self, X, y=None, sample_weight=None):
        """Compute dip-means clustering.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training instances to cluster. It must be noted that the data
                will be converted to C ordering, which will cause a memory
                copy if the given data is not C-contiguous.
                If a sparse matrix is passed, a copy will be made if it's not in
                CSR format.
            y : Ignored
                Not used, present here for API consistency by convention.
            sample_weight : array-like of shape (n_samples,), default=None
                The weights for each observation in X. If None, all observations
                are assigned equal weight.

                .. versionadded:: 0.20
        Returns:
            self : object
                Fitted estimator.
        """
        self.n_clusters = 1
        self.init = 'k-means++'
        if self._is_unimodal(X):
            logger.debug('The initial data were unimodal!')
            # ToDO: fill in the missing attributes!
            return self

        self.n_clusters = 2
        super().fit(X, y, sample_weight)
        self.init = self.cluster_centers_

        while True:
            logger.debug(f'The number of clusters has been increased! '
                         f'The current estimate is: {self.n_clusters}')

            labels = self.labels_.copy()

            ests = np.array([
                self._estimate_unimodality(X[labels == i])
                for i in range(self.n_clusters)
            ])

            if np.all(ests < self.mc_test.tester.pval):
                logger.info(f"The final number of clusters is: {self.n_clusters}.")
                break

            cluster_centers = self.cluster_centers_.copy()
            n_clusters = self.n_clusters
            i_max = np.argmax(ests)
            logger.debug(f'The maximum estimate is: {ests[i_max]}.')
            l_max = np.max(labels)

            self.n_clusters = 2
            self.init = np.vstack((cluster_centers[i_max], cluster_centers[i_max]))
            super().fit(
                X[labels == i_max],
                y,
                None if sample_weight is None else sample_weight[labels == i_max]
            )

            self.cluster_centers_ = np.vstack((
                cluster_centers[:i_max],
                self.cluster_centers_,
                cluster_centers[i_max+1:]
            ))

            self.labels_[self.labels_ == 0] = i_max
            self.labels_[self.labels_ == 1] = l_max + 1
            labels[labels == i_max] = self.labels_.copy()
            self.labels_ = labels
            self.n_clusters = n_clusters + 1

        return self
