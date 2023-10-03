"""Unimodality hypothesis testing experiments with synthetic datasets.

Usage:
  unimodality.py <d> <pj> <pv> <sims> [--samples=<s> --noise=<n> --seed=<sd> --dist=<ds> --obs=<o> --plot=<f>]
  unimodality.py -h | --help

Options:
  -h --help         Show this screen.
  --samples=<s>     The number of samples [default: 200].
  --noise=<n>       The standard deviation inside the clusters [default: 0].
  --seed=<sd>       The seed [default: 42].
  --dist=<ds>       The type of distance [default: mahalanobis].
  --obs=<o>         The type of the observer [default: percentile].
  --plot=<f>        Whether to produce a plot or not [default: False].
"""
import sys
from typing import Callable
import warnings

from docopt import docopt
from loguru import logger
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris

from experiments.common import get_monte_carlo_test
from experiments.common import plot_clustered_data
from mudpod.misc import set_seed

logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


def get_dataset(name: str) -> Callable:
    """Get a sklearn dataset according to name."""
    if name == 'circles':
        return make_circles
    elif name == 'moons':
        return make_moons
    elif name == 'swiss_roll':
        return make_swiss_roll
    else:
        msg = f'Unknown dataset name: {name}'
        logger.error(msg)
        raise ValueError(msg)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    SEED = int(arguments['--seed'])
    set_seed(SEED)

    data_func = get_dataset(str(arguments['<d>']))
    n_samples = int(arguments['--samples'])
    noise = float(arguments['--noise'])
    x, y = data_func(n_samples=n_samples, noise=noise, random_state=SEED)

    mct = get_monte_carlo_test(arguments=arguments, workers_num=1)

    msg = dict(arguments)
    msg['result'] = 'unimodal' if mct.test(x) else 'multimodal'
    msg.pop('--help')

    if eval(msg['--plot']):
       plot_clustered_data(x, y)
    
    msg.pop('--plot')

    logger.info(
        'The inputs and the output of the experiments is: '
        f'{msg}'
    )

