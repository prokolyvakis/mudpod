"""Clustering experiments with real datasets with representations obtained from
   pre-trained embeddings stored in Numpy saved arrays.

Usage:
  pre_clustering.py <p> <pj> <pv> <sims> [--samples=<s> --seed=<sd>]
  pre_clustering.py -h | --help

Options:
  -h --help         Show this screen.
  --samples=<s>     The number of samples [default: 200].
  --seed=<sd>       The seed [default: 42].
"""
from pathlib import Path
import sys
from typing import Optional

from docopt import docopt
from loguru import logger
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from umap import UMAP

from experiments.common import plot_clustered_data
from hdunim.misc import set_seed
from hdunim.clustering import DipMeans
from hdunim.projections import IdentityProjector
from hdunim.projections import JohnsonLindenstrauss
from hdunim.observer import PercentileObserver
from hdunim.projections import View


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")


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
        )
        raise FileNotFoundError(e)
    
    if samples is None:
        return x, y

    classes = np.unique(y)
    xs = []
    ys = []

    # calculate the number of samples per class
    samples_per_class = samples // len(classes)

    for c in classes:
        i = np.where(y == c)[0]
        if len(i) > samples_per_class:
            ids = np.random.choice(i, size=samples_per_class, replace=False)
            xs.append(x[ids])
            ys.append(y[ids])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    # Shuffle embeddings and labels, retaining their correct order
    indices = np.random.permutation(len(xs))
    xs = xs[indices]
    ys = ys[indices]

    return xs, ys


if __name__ == "__main__":
    arguments = docopt(__doc__)

    SEED = int(arguments['--seed'])
    set_seed(SEED)

    n_samples = int(arguments['--samples'])
    x, y = get_data(Path(arguments['<p>']), samples=n_samples)

    pt = str(arguments['<pj>'])
    p = JohnsonLindenstrauss if pt == 'jl' else IdentityProjector
    v = View(p, PercentileObserver(0.99))
    
    dm = DipMeans(
        view=v,
        pval=float(arguments['<pv>']),
        sim_num=int(arguments['<sims>']),
        workers_num=10,
        random_state=SEED
    )

    clusters = dm.fit(x).labels_
    nmi = normalized_mutual_info_score(y, clusters)

    msg = dict(arguments)
    msg['result'] = f'The NMI score is {nmi}'
    logger.info(
        'The inputs and the output of the experiment is: '
        f'{msg}'
    )

    reducer = UMAP(random_state=SEED)
    reducer.fit(x)
    embeddings = reducer.transform(x)

    plot_clustered_data(embeddings, clusters)
