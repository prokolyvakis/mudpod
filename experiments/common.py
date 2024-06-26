"""Common utility functions used across the different experiments."""
import numpy as np
import plotly.graph_objects as go

from mudpod.clustering import DipMeans
from mudpod.projections import IdentityProjector
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.observer import RandomObserver
from mudpod.projections import View
from mudpod.unimodality import UnimodalityTest
from mudpod.unimodality import MonteCarloUnimodalityTest


def plot_clustered_data(data: np.ndarray, labels: np.ndarray) -> None:
    """Plots clustered data.

    Args:
        data: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        labels: a 1D numpy array containing the labels for each datapoint.
    """
    if data.shape[1] != 2:
        raise ValueError("Input array must have 2 columns for 2D plotting.")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Data and labels arrays must have the same number of rows.")

    unique_labels = np.unique(labels)

    # Generate colors for each cluster
    # ToDO: Validate that each cluster is assigned to a unique cluster, i.e., sample w/o
    # replacement
    colors = np.random.randint(0, 255, (unique_labels.shape[0], 3))

    fig = go.Figure()

    for i, label in enumerate(unique_labels):
        cluster_data = data[labels == label]
        fig.add_trace(go.Scatter(
            x=cluster_data[:, 0],
            y=cluster_data[:, 1],
            mode='markers',
            marker=dict(
                color=f'rgb({colors[i, 0]}, {colors[i, 1]}, {colors[i, 2]})'
            ), name=f'Cluster {label}')
        )

    fig.update_layout(
        title="2D Clustered Data Plot",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True
    )

    fig.show()


def group_data_points(data: np.ndarray, clusters: np.ndarray) -> list[np.ndarray]:
    """Group data points into disjoint sets based on the cluster index they belong to.

    Args:
        data: a 2D numpy array with the first dimension being the number of different
                datapoints and the second being the features' size.
        clusters: a 1D numpy array indicating the cluster indices.
    Returns:
        A list of 1D numpy arrays where each list element corresponds to the datapoints
            belong to the same cluster.
    """
    m = np.hstack((data, clusters[:, None]))
    m = m[m[:, -1].argsort()]
    m = np.split(m[:, :-1], np.unique(m[:, -1], return_index=True)[1][1:])
    return m


def get_view(arguments: dict) -> View:
    """Get a view based on the config parameters existing in arguments.

    Args:
        arguments: a dict containing the config parameters.
    Returns:
        The parametrized view.
    """
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

    return v


def get_monte_carlo_test(arguments: dict, workers_num: int = 1) -> MonteCarloUnimodalityTest:
    """Get a Monte Carlo unimodality test.

    Args:
        arguments: a dict containing the config parameters.
        workers_num: an integer indicating the number of workers.
    Returns:
        A parametrized Monte Carlo unimodality test.
    """
    v = get_view(arguments)

    t = UnimodalityTest(v, float(arguments['<pv>']))
    mct = MonteCarloUnimodalityTest(
        t,
        sim_num=int(arguments['<sims>']),
        workers_num=workers_num
    )

    return mct


def get_dip_means(arguments: dict, seed: int, workers_num: int = 1) -> DipMeans:
    """Get a DipMeans clustering instance.

    Args:
        arguments: a dict containing the config parameters.
        seed: a random seed.
        workers_num: an integer indicating the number of workers.
    Returns:
        A parametrized DipMeans instance.
    """
    v = get_view(arguments)

    dm = DipMeans(
        view=v,
        pval=float(arguments['<pv>']),
        sim_num=int(arguments['<sims>']),
        workers_num=workers_num,
        random_state=seed
    )

    return dm
