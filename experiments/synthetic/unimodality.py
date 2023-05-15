"""Unimodality hypothesis testing experiments with synthetic datasets.s."""
from loguru import logger
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris


from hdunim.misc import set_seed
from hdunim.projections import IdentityProjector
from hdunim.projections import JohnsonLindenstrauss
from hdunim.observer import PercentileObserver
from hdunim.projections import View
from hdunim.unimodality import UnimodalityTester
from hdunim.unimodality import MonteCarloUnimodalityTester

SEED = 42

set_seed(SEED)


if __name__ == "__main__":
    v = View(JohnsonLindenstrauss, PercentileObserver(0.99))
    t = UnimodalityTester(v, 0.01)
    mct = MonteCarloUnimodalityTester(t, sim_num=1000, workers_num=10)

    data_func = make_swiss_roll
    n_samples = 200
    noise = 0
    x, y = data_func(n_samples=n_samples, noise=noise, random_state=SEED)

    logger.info(f'The statistical test says that {mct.test(x)}')