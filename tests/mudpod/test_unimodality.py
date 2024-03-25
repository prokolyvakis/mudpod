"""Tests written for the unimodality module."""
import pytest

from sklearn.datasets import make_blobs

from mudpod.misc import set_seed
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.projections import View
from mudpod.unimodality import UnimodalityTest
from mudpod.unimodality import MonteCarloUnimodalityTest

set_seed(42)


@pytest.mark.parametrize("projector", [JohnsonLindenstrauss()])
@pytest.mark.parametrize("observer_fn", [lambda a: PercentileObserver(0.95, alpha=a)])
@pytest.mark.parametrize("alpha", [1, 4])
@pytest.mark.parametrize("n_features", [200, 400])
def test_unimodality_tester(projector, observer_fn, alpha, n_features) -> None:
    """Test that the unimodality tester works properly."""

    v = View(projector, observer_fn(alpha), alpha=alpha)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=2,
        cluster_std=1.,
        random_state=42
    )

    t = UnimodalityTest(v, 0.05)

    assert not t.test(x)


@pytest.mark.parametrize("projector", [JohnsonLindenstrauss()])
@pytest.mark.parametrize("observer_fn", [lambda a: PercentileObserver(0.95, alpha=a)])
@pytest.mark.parametrize("alpha", [1, 4])
@pytest.mark.parametrize("n_features", [200])
@pytest.mark.parametrize("workers_num", [1, 5])
@pytest.mark.parametrize("sim_num", [10])
def test_monte_carlo_unimodality_tester(
        projector, observer_fn, alpha, n_features, workers_num, sim_num) -> None:
    """Test that the Monte Carlo unimodality tester works properly."""

    v = View(projector, observer_fn(alpha), alpha=alpha)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=2,
        cluster_std=1.,
        random_state=42
    )

    t = UnimodalityTest(v, 0.05)

    mct = MonteCarloUnimodalityTest(t, sim_num, workers_num)

    assert not mct.test(x)


def test_monte_carlo_unimodality_tester_assertions() -> None:
    """Test that the assertions of the MonteCarloUnimodalityTest work properly."""

    o = PercentileObserver(0.95)

    v = View(JohnsonLindenstrauss(), o)
    t = UnimodalityTest(v, 0.05)
    with pytest.raises(ValueError):
        _ = MonteCarloUnimodalityTest(t, -1, 42)

    for i in (-42, 0, 1):
        with pytest.raises(ValueError):
            _ = MonteCarloUnimodalityTest(t, i, 42)
