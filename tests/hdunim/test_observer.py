"""Tests written for the Observer module."""
import pytest

import numpy as np

from hdunim.misc import set_seed
from hdunim.observer import PercentileObserver

set_seed(42)


@pytest.mark.parametrize("dtype", sorted(['euclidean', 'mahalanobis']))
def test_percentile_observer(dtype: str) -> None:
    """Test that the percentile observer works smoothly."""
    os = PercentileObserver(percentile=0.12, dtype=dtype)

    m = np.array([0, 0])
    c = np.array([[6, -3], [-3, 3.5]])
    x = np.random.multivariate_normal(m, c, size=42)

    o = os.get(x)

    assert o is not None
