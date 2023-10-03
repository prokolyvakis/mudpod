"""Unimodality hypothesis testing experiments with a mixture of 2D gaussians.

Usage:
  two_gaussians_mix.py <pj> <pv> <sims> [--samples=<s> --noise=<n> --seed=<sd> --dist=<ds> --obs=<o> --plot=<f>]
  two_gaussians_mix.py -h | --help

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
import warnings

from docopt import docopt
from loguru import logger

from experiments.common import get_monte_carlo_test
from experiments.common import plot_clustered_data
from experiments.synthetic.misc import TwoDimGaussianSumGenerator
from mudpod.misc import set_seed


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    arguments = docopt(__doc__)

    SEED = int(arguments['--seed'])
    set_seed(SEED)

    n_samples = int(arguments['--samples'])
    std = float(arguments['--noise'])
    g = TwoDimGaussianSumGenerator(
      n=n_samples,
      cluster_std=std, 
      random_state=SEED
    )

    mct = get_monte_carlo_test(arguments=arguments, workers_num=1)

    tr = 'unimodal' if mct.test(g.x) else 'bimodal'
    msg = dict(arguments)
    msg['groundtruth'] = g.t
    msg['result'] = tr
    msg.pop('--help')
    msg['parity'] = int(tr == g.t)

    if eval(msg['--plot']):
       plot_clustered_data(g.x, g.y)
    
    msg.pop('--plot')

    logger.info(
        'The inputs and the output of the experiments is: '
        f'{msg}'
    )
