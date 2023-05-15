"""Common utility functions used across the different experiments."""
import numpy as np
import plotly.graph_objects as go


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
