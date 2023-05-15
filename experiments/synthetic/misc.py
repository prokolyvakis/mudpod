"""A helper module containing functions used across the synthetic data experiments."""
import numpy as np
import scipy.io as sio


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
