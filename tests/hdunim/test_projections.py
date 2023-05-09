"""Tests written for the Projections module."""
import pytest

import numpy as np
from sklearn.datasets import make_blobs

from hdunim.misc import set_seed
from hdunim.projections import JohnsonLindenstrauss
from hdunim.projections import ExponentiallyDescendOrAscend
from hdunim.projections import IdentityProjector
from hdunim.observer import PercentileObserver
from hdunim.projections import View

set_seed(42)


def test_identity_dim() -> None:
    """Test that IdentityProjector works smoothly."""

    assert 42 == IdentityProjector.estimate_dim(42)


@pytest.mark.parametrize("input_dim,projection_dim", [(32, 5), (128, 7), (1024, 10)])
def test_johnson_lindenstrauss_dim(input_dim, projection_dim) -> None:
    """Test that JohnsonLindenstrauss works smoothly."""

    assert projection_dim == JohnsonLindenstrauss.estimate_dim(input_dim)


@pytest.mark.parametrize("inp_dim,projection_dim", [(32, 5), (64, 6), (2, 4), (3, 8)])
def test_johnson_lindenstrauss_or_ascend_dim(inp_dim: int, projection_dim: int) -> None:
    """Test that ExponentiallyDescendOrAscend works smoothly."""

    assert projection_dim == ExponentiallyDescendOrAscend.estimate_dim(inp_dim)

@pytest.mark.parametrize(
    "projector",
    [IdentityProjector, JohnsonLindenstrauss, ExponentiallyDescendOrAscend]
)
@pytest.mark.parametrize("observer", [PercentileObserver(0.95)])
@pytest.mark.parametrize("n_features", [200, 400])
def test_view(projector, observer, n_features) -> None:
    """A functional test for the View class."""

    v = View(projector, observer)

    x, _ = make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=2,
        cluster_std=1.,
        random_state=42
    )

    ds = v.distances(x)

    assert not np.isnan(ds).any()
