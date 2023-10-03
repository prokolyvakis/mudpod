"""Tests written for the synthetic experiments module."""
import pytest

from experiments.common import get_dip_means
from experiments.synthetic.two_gaussians_mix import run as run_gaussians
from experiments.synthetic.unimodality import run as run_unimodality


@pytest.mark.parametrize("projector", ['jl', 'i'])
@pytest.mark.parametrize("distance", ['euclidean', 'mahalanobis'])
@pytest.mark.parametrize("observer", ['percentile', 'random'])
def test_get_dip_means(projector: str, distance: str, observer: str) -> None:
    """Test that the factory for DipMeans works properly."""

    args = {
        '<pj>': projector,
        '--dist': distance,
        '--obs': observer,
        '<pv>': 0.05,
        '<sims>': 5,
    }

    get_dip_means(args, seed=42)


@pytest.mark.parametrize("projector", ['jl', 'i'])
@pytest.mark.parametrize("distance", ['euclidean', 'mahalanobis'])
@pytest.mark.parametrize("observer", ['percentile', 'random'])
def test_run_gaussians(projector: str, distance: str, observer: str) -> None:
    """Test that the unimodality experiments on 2D gaussians mixture work properly."""

    args = {
        '--seed': 42,
        '--samples': 200,
        '--noise': 0.5,
        '<pj>': projector,
        '--dist': distance,
        '--obs': observer,
        '<pv>': 0.05,
        '<sims>': 5,
        '--help': False,
        '--plot': 'False',
    }

    run_gaussians(args)


@pytest.mark.parametrize("projector", ['jl', 'i'])
@pytest.mark.parametrize("distance", ['euclidean', 'mahalanobis'])
@pytest.mark.parametrize("observer", ['percentile', 'random'])
def test_run_unimodality(projector: str, distance: str, observer: str) -> None:
    """Test that the unimodality experiments work properly."""

    args = {
        '--seed': 42,
        '<d>': 'circles',
        '--samples': 200,
        '--noise': 0.5,
        '<pj>': projector,
        '--dist': distance,
        '--obs': observer,
        '<pv>': 0.05,
        '<sims>': 5,
        '--help': False,
        '--plot': 'False',
    }

    run_unimodality(args)
