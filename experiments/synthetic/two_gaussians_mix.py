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

from experiments.common import plot_clustered_data
from experiments.synthetic.misc import TwoDimGaussianSumGenerator
from mudpod.misc import set_seed
from mudpod.projections import IdentityProjector
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.observer import RandomObserver
from mudpod.projections import View
from mudpod.unimodality import UnimodalityTest
from mudpod.unimodality import MonteCarloUnimodalityTest


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    arguments = docopt(__doc__)

    SEED = int(arguments['--seed'])
    set_seed(SEED)

    pt = str(arguments['<pj>'])
    if pt == 'jl':
        p = JohnsonLindenstrauss()
    elif pt == 'i':
        p = IdentityProjector()
    else:
       raise ValueError(f'The projection type: {pt} is not supported!')
    
    dt = str(arguments['--dist'])
    ot = str(arguments['--obs'])
    if ot == 'percentile':
        o = PercentileObserver(0.99, dt)
    elif ot == 'random':
        o = RandomObserver()
    else:
       raise ValueError(f'The observer type: {ot} is not supported!')

    v = View(p, o, dt)
    t = UnimodalityTest(v, float(arguments['<pv>']))
    mct = MonteCarloUnimodalityTest(
        t,
        sim_num=int(arguments['<sims>']),
        workers_num=1
    )

    n_samples = int(arguments['--samples'])
    std = float(arguments['--noise'])
    g = TwoDimGaussianSumGenerator(
      n=n_samples,
      cluster_std=std, 
      random_state=SEED
    )

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
