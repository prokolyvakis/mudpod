"""Unimodality hypothesis testing experiments with a mixture of 2D gaussians.

Usage:
  two_gaussians_mix.py <pj> <pv> <sims> [--samples=<s> --noise=<n> --seed=<sd>]
  two_gaussians_mix.py -h | --help

Options:
  -h --help         Show this screen.
  --samples=<s>     The number of samples [default: 200].
  --noise=<n>       The standard deviation inside the clusters [default: 0].
  --seed=<sd>       The seed [default: 42].
"""
import sys

from docopt import docopt
from loguru import logger

from experiments.common import plot_clustered_data
from experiments.synthetic.misc import TwoDimGaussianSumGenerator
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

    n_samples = int(arguments['--samples'])
    std = float(arguments['--noise'])
    g = TwoDimGaussianSumGenerator(
      n=n_samples,
      cluster_std=std, 
      random_state=SEED
    )

    tr = 'unimodal' if mct.test(g.x) else 'bimodal'
    logger.info(f'The statistical test says {tr} and the data were {g.t}!')
    msg = dict(arguments)
    msg['groundtruth'] = g.t
    msg['result'] = tr
    logger.info(
        'The inputs and the output of the experiments is: '
        f'{msg}'
    )

    plot_clustered_data(g.x, g.y)
