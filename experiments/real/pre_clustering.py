"""Clustering experiments with real datasets with representations obtained from
   pre-trained embeddings stored in Numpy saved arrays.

Usage:
  pre_clustering.py <p> <pj> <pv> <sims> [--samples=<s> --seed=<sd> --dist=<ds> --obs=<o> --plot=<f>]
  pre_clustering.py -h | --help

Options:
  -h --help         Show this screen.
  --samples=<s>     Optional number of samples [default: ].
  --seed=<sd>       The seed [default: 42].
  --dist=<ds>       The type of distance [default: mahalanobis].
  --obs=<o>         The type of the observer [default: percentile].
  --plot=<f>        Whether to produce a plot or not [default: False].
"""
from pathlib import Path
import sys
from typing import Optional
import warnings

from docopt import docopt
from loguru import logger
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from umap import UMAP

from experiments.common import get_dip_means
from experiments.common import plot_clustered_data
from mudpod.misc import set_seed


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


def get_data(
    path: Path,
    samples: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.load(path / 'embeddings.npy')
        y = np.load(path / 'labels.npy')
    except FileNotFoundError as e:
        logger.error(
            'Either the embeddings or the labels do not confront to the naming'
            ' convention, i.e., the embeddings to be stored in a file named:'
            ' `embeddings.npy` and the labels in a file named: `labels.npy`!'
            ' Original error: %s', str(e)
        )
        raise
    
    if samples is None:
        return x, y

    idx = np.random.choice(x.shape[0], size=samples, replace=False)
    xs = x[idx]
    ys = y[idx]

    return xs, ys


if __name__ == "__main__":
    arguments = docopt(__doc__)

    SEED = int(arguments['--seed'])
    set_seed(SEED)

    n_samples = arguments['--samples'] or None
    if n_samples is not None:
        n_samples = int(n_samples)
    x, y = get_data(Path(arguments['<p>']), samples=n_samples)

    dm = get_dip_means(
        arguments=arguments,
        seed=SEED
    )

    clusters = dm.fit(x).labels_
    nmi = normalized_mutual_info_score(y, clusters)

    msg = dict(arguments)
    msg['result'] = f'The NMI score is {nmi}'
    msg.pop('--help')

    if eval(msg['--plot']):
        reducer = UMAP(random_state=SEED)
        reducer.fit(x)
        embeddings = reducer.transform(x)

        plot_clustered_data(embeddings, y)
    
    msg.pop('--plot')

    logger.info(
        'The inputs and the output of the experiment is: '
        f'{msg}'
    )
