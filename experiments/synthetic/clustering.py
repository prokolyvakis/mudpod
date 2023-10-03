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

from experiments.common import get_dip_means
from experiments.common import plot_clustered_data
from experiments.synthetic.misc import load
from mudpod.misc import set_seed


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


def run(args: dict) -> None:
    """Main runner."""
    SEED = int(args['--seed'])
    set_seed(SEED)

    x, y = load(str(args['<d>']))
    
    dm = get_dip_means(
        arguments=args,
        seed=SEED
    )

    clusters = dm.fit(x).labels_

    logger.info(f'The NMI score is {normalized_mutual_info_score(y, clusters)}')

    plot_clustered_data(x, clusters)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    run(args=arguments)
