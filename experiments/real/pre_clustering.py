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

from experiments.common import plot_clustered_data
from mudpod.misc import set_seed
from mudpod.clustering import DipMeans
from mudpod.projections import IdentityProjector
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.observer import RandomObserver
from mudpod.projections import View


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

    n_samples = arguments['--samples'] or None
    if n_samples is not None:
        n_samples = int(n_samples)
    x, y = get_data(Path(arguments['<p>']), samples=n_samples)

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
