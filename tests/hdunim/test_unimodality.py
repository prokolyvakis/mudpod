"""Tests written for the unimodality module."""
import pytest

from sklearn.datasets import make_blobs

from hdunim.misc import set_seed
from hdunim.projections import JohnsonLindenstrauss
from hdunim.projections import IdentityProjector
from hdunim.observer import PercentileObserver
from hdunim.projections import View
from hdunim.unimodality import UnimodalityTester
from hdunim.unimodality import MonteCarloUnimodalityTester

set_seed(42)


@pytest.mark.parametrize("projector", [JohnsonLindenstrauss])
@pytest.mark.parametrize("observer", [PercentileObserver(0.95)])
@pytest.mark.parametrize("n_features", [200, 400])
def test_unimodality_tester(projector, observer, n_features) -> None:
    """Test that the unimodality tester works properly."""

    v = View(projector, observer)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=2,
        cluster_std=1.,
        random_state=42
    )

    t = UnimodalityTester(v, 0.05)

    assert not t.test(x)


@pytest.mark.parametrize("projector", [JohnsonLindenstrauss])
@pytest.mark.parametrize("observer", [PercentileObserver(0.95)])
@pytest.mark.parametrize("n_features", [200, 400])
@pytest.mark.parametrize("workers_num", [0, 1, 10])
@pytest.mark.parametrize("sim_num", [10, 20])
def test_monte_carlo_unimodality_tester(
        projector, observer, n_features, workers_num, sim_num) -> None:
    """Test that the Monte Carlo unimodality tester works properly."""

    v = View(projector, observer)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=2,
        cluster_std=1.,
        random_state=42
    )

    t = UnimodalityTester(v, 0.05)

    mct = MonteCarloUnimodalityTester(t, sim_num, workers_num)

    assert not mct.test(x)


def test_monte_carlo_unimodality_tester_assertions() -> None:
    """Test that the assertions of the MonteCarloUnimodalityTester."""

    o = PercentileObserver(0.95)

    v = View(JohnsonLindenstrauss, o)
    t = UnimodalityTester(v, 0.05)
    with pytest.raises(ValueError):
        _ = MonteCarloUnimodalityTester(t, -1, 42)

    for i in (-42, 0, 1):
        with pytest.raises(ValueError):
            _ = MonteCarloUnimodalityTester(t, i, 42)
