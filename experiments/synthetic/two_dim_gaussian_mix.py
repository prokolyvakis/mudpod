"""Unimodality hypothesis testing experiments with a mixture of 2D gaussians."""
import sys

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

SEED = 12

set_seed(SEED)

logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")


if __name__ == "__main__":
    v = View(JohnsonLindenstrauss, PercentileObserver(0.99))
    t = UnimodalityTest(v, 0.001)
    mct = MonteCarloUnimodalityTest(t, sim_num=100, workers_num=10)

    n_samples = 2000
    std = 1.60
    g = TwoDimGaussianSumGenerator(n=n_samples, cluster_std=std, random_state=SEED)

    tr = 'unimodal' if mct.test(g.x) else 'bimodal'
    logger.info(f'The statistical test says {tr} and the data were {g.t}!')

    plot_clustered_data(g.x, g.y)
