"""Tests written for the Observer module."""
import pytest

import numpy as np

from mudpod.misc import set_seed
from mudpod.observer import PercentileObserver

set_seed(42)


@pytest.mark.parametrize("alpha", [1, 2, 4])
@pytest.mark.parametrize("dtype", sorted(['euclidean', 'mahalanobis']))
def test_percentile_observer(alpha: float, dtype: str) -> None:
    """Test that the percentile observer works smoothly."""
    os = PercentileObserver(percentile=0.12, alpha=alpha, dtype=dtype)

    m = np.array([0, 0])
    c = np.array([[6, -3], [-3, 3.5]])
    x = np.random.multivariate_normal(m, c, size=42)

    o = os.get(x)

    assert o is not None


def test_percentile_observer_assertions() -> None:
    """Test that the assertions of the PercentileObserver work properly."""

    with pytest.raises(AssertionError):
        _ = PercentileObserver(percentile=-1)
    
    with pytest.raises(AssertionError):
        _ = PercentileObserver(percentile=12)

    with pytest.raises(AssertionError):
        _ = PercentileObserver(percentile=0.42, alpha=-1)
