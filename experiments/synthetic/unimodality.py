"""Unimodality hypothesis testing experiments with synthetic datasets.

Usage:
  unimodality.py <d> <pj> <pv> <sims> [--samples=<s> --noise=<n> --seed=<sd>]
  unimodality.py -h | --help

Options:
  -h --help         Show this screen.
  --samples=<s>     The number of samples [default: 200].
  --noise=<n>       The standard deviation inside the clusters [default: 0].
  --seed=<sd>       The seed [default: 42].
"""
import sys
from typing import Callable

from docopt import docopt
from loguru import logger
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris

from experiments.common import plot_clustered_data
from hdunim.misc import set_seed
from hdunim.projections import IdentityProjector
from hdunim.projections import JohnsonLindenstrauss
from hdunim.observer import PercentileObserver
from hdunim.projections import View
from hdunim.unimodality import UnimodalityTest
from hdunim.unimodality import MonteCarloUnimodalityTest

logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")


def get_dataset(name: str) -> Callable:
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

    pt = str(arguments['<pj>'])
    p = JohnsonLindenstrauss if pt == 'jl' else IdentityProjector
    v = View(p, PercentileObserver(0.99))
    t = UnimodalityTest(v, float(arguments['<pv>']))
    mct = MonteCarloUnimodalityTest(
        t,
        sim_num=int(arguments['<sims>']),
        workers_num=10
    )

    data_func = get_dataset(str(arguments['<d>']))
    n_samples = int(arguments['--samples'])
    noise = float(arguments['--noise'])
    x, y = data_func(n_samples=n_samples, noise=noise, random_state=SEED)
    # mask = np.isin(y, [0])
    # x = x[mask]
    # y = y[mask]

    msg = dict(arguments)
    msg['result'] = 'unimodal' if mct.test(x) else 'multimodal'
    logger.info(
        'The inputs and the output of the experiments is: '
        f'{msg}'
    )

    plot_clustered_data(x, y)
