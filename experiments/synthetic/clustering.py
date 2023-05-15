"""Clustering experiments with synthetic datasets."""
from loguru import logger
import numpy as np
import scipy.io as sio
from sklearn.metrics import normalized_mutual_info_score

from experiments.common import plot_clustered_data
from hdunim.clustering import DipMeans
from hdunim.projections import IdentityProjector
from hdunim.projections import JohnsonLindenstrauss
from hdunim.observer import PercentileObserver
from hdunim.observer import RandomObserver
from hdunim.projections import View
from hdunim.misc import set_seed

SEED = 128

set_seed(SEED)


if __name__ == "__main__":
    mat = sio.loadmat('synthetic.mat')
    x = np.array(mat['X'])
    y = np.array(mat['C']).ravel()
    # mask = np.isin(y, [5, 8])
    # x = x[mask]
    # y = y[mask]

    v = View(JohnsonLindenstrauss, PercentileObserver(0.99))
    # v = View(JohnsonLindenstrauss, RandomObserver())
    # v = View(IdentityProjector, RandomObserver())

    dm = DipMeans(view=v, pval=0.001, sim_num=100, workers_num=10, random_state=SEED)

    clusters = dm.fit(x).labels_

    logger.info(f'The NMI score is {normalized_mutual_info_score(y, clusters)}')

    plot_clustered_data(x, clusters)
