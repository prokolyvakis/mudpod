"""A helper module containing functions used across the synthetic data experiments."""
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

import numpy as np
import scipy.io as sio
from sklearn.datasets import make_blobs


def load(fname: str, data_path: str = 'data/') -> tuple[np.ndarray, np.ndarray]:
    """Loads the data and the clusters.

    Args:
        fname: the filename containing the suffix.
        data_path: the data path.
    Returns:
        a tuple containing the data and the labels.
    """
    suffix = fname.split(sep='.')[-1]

    if suffix == 'mat':
        mat = sio.loadmat(data_path + fname)

        d_key = 'data'
        c_key = 'label'
        if fname == 'synthetic.mat':
            d_key = 'X'
            c_key = 'C'

        d = np.array(mat[d_key])
        c = np.array(mat[c_key])
    elif suffix == 'arff':
        mat = sio.arff.loadarff(data_path + fname)[0]
        d = np.array([list(t)[:-1] for t in mat])
        c = np.array([list(t)[-1] for t in mat])
    else:
        raise ValueError(f'Unknown data file type: {suffix}')

    return d, c.ravel()


@dataclass
class TwoDimGaussianSumGenerator:
    """ Generates data from a sum of two gaussians while assessing its unimodality.

    To assess the unimodality, it performs the Konstantelos et al. test, i.e.,

        Konstantellos, A. (1980). Unimodality conditions for Gaussian sums.
        IEEE Transactions on Automatic Control, 25(4), 838-839.
    """

    n: InitVar[int]
    # The number of data that will be generated.

    cluster_std: InitVar[float]
    # The standard deviation of the clusters.

    random_state: InitVar[int]
    # The random state

    x: np.ndarray = field(init=False, repr=True)
    # The generated data.

    y: np.ndarray = field(init=False, repr=True)
    # The generated labels.

    t: str = field(init=False, repr=True)
    # The type of the dataset, i.e., either 'unimodal' or 'bimodal'.

    def __post_init__(self, n: int, cluster_std: float, random_state: int):
        self.x, self.y = make_blobs(
            n_samples=n,
            n_features=2,
            centers=2,
            cluster_std=cluster_std,
            random_state=random_state
        )

        # Mean & Covariance computation
        f_m = np.mean(self.x[self.y == 0], axis=0)
        s_m = np.mean(self.x[self.y == 1], axis=0)
        f_c = np.cov((self.x[self.y == 0]).T)
        s_c = np.cov((self.x[self.y == 1]).T)

        # Compute the argmax of the inverse covariances:
        f_c_inv = np.linalg.inv(f_c)
        s_c_inv = np.linalg.inv(s_c)
        d = f_c_inv - s_c_inv
        c = np.all(np.linalg.eigvals(d) >= 0)
        m_c_inv = f_c_inv if c else s_c_inv

        # Equation (3) from the Konstantellos et al. paper
        m_diff = f_m - s_m
        u_c = np.matmul(
            np.matmul(m_diff.T, m_c_inv),
            m_diff
        ) < 4.

        self.t = 'unimodal' if u_c else 'bimodal'
