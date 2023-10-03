"""Tests written for the Projections module."""
import pytest

import numpy as np
from sklearn.datasets import make_blobs

from mudpod.misc import set_seed
from mudpod.projections import IdentityProjector
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.projections import View

set_seed(42)


def test_identity_dim() -> None:
    """Test that IdentityProjector works smoothly."""
    idp = IdentityProjector()

    assert 42 == idp.estimate_dim(256, 42)


@pytest.mark.parametrize(
    "samples_num,projection_dim", 
    [(32, 8 * 5), (128, 8 * 7), (1024, 8 * 10)]
)
def test_johnson_lindenstrauss_dim(samples_num, projection_dim) -> None:
    """Test that JohnsonLindenstrauss works smoothly."""
    jlp = JohnsonLindenstrauss(eps=1.)

    assert projection_dim == jlp.estimate_dim(samples_num, 128)


@pytest.mark.parametrize(
    "projector",
    [IdentityProjector(), JohnsonLindenstrauss()]
)
@pytest.mark.parametrize("observer", [PercentileObserver(0.95)])
@pytest.mark.parametrize("dtype", sorted(['euclidean', 'mahalanobis']))
@pytest.mark.parametrize("n_features", [200, 400])
def test_view(projector, observer, dtype, n_features) -> None:
    """A functional test for the View class."""

    v = View(projector, observer, dtype)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=2,
        cluster_std=1.,
        random_state=42
    )

    ds = v.distances(x)

    assert not np.isnan(ds).any()
