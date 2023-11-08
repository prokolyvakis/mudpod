"""Tests written for the clustering module."""
import pytest

from sklearn.datasets import make_blobs

from mudpod.misc import set_seed
from mudpod.clustering import DipMeans
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.projections import View

set_seed(42)


@pytest.mark.parametrize("projector", [JohnsonLindenstrauss()])
@pytest.mark.parametrize("observer_fn", [lambda a: PercentileObserver(0.95, alpha=a)])
@pytest.mark.parametrize("alpha", [1, 2, 4])
@pytest.mark.parametrize("n_features", [200, 400])
def test_dip_means_fit(projector, observer_fn, alpha, n_features) -> None:
    """Test that the dip means class is properly initialized"""

    v = View(projector, observer_fn(alpha), alpha=alpha)
    dp = DipMeans(view=v, pval=0.05, sim_num=10, workers_num=10)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=4,
        cluster_std=1.,
        random_state=42
    )

    dp.fit(x)

    assert dp.n_clusters == 4
