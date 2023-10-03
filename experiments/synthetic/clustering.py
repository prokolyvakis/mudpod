"""Clustering experiments with synthetic datasets.

Usage:
  clustering.py <d> <pj> <pv> <sims> [--samples=<s> --noise=<n> --seed=<sd>  --dist=<ds> --obs=<o>]
  clustering.py -h | --help

Options:
  -h --help         Show this screen.
  --samples=<s>     The number of samples [default: 200].
  --noise=<n>       The standard deviation inside the clusters [default: 0].
  --seed=<sd>       The seed [default: 42].
  --dist=<ds>       The type of distance [default: mahalanobis].
  --obs=<o>         The type of the observer [default: percentile].
"""
import sys
import warnings

from docopt import docopt
from loguru import logger
from sklearn.metrics import normalized_mutual_info_score

from experiments.common import plot_clustered_data
from experiments.synthetic.misc import load
from mudpod.clustering import DipMeans
from mudpod.projections import IdentityProjector
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.observer import RandomObserver
from mudpod.projections import View
from mudpod.misc import set_seed


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    arguments = docopt(__doc__)

    x, y = load(str(arguments['<d>']))

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

    dm = DipMeans(
        view=v,
        pval=float(arguments['<pv>']),
        sim_num=int(arguments['<sims>']),
        workers_num=1,
        random_state=SEED
    )

    clusters = dm.fit(x).labels_

    logger.info(f'The NMI score is {normalized_mutual_info_score(y, clusters)}')

    plot_clustered_data(x, clusters)
