"""Clustering experiments with real datasets with representations obtained from
   pre-trained hugging face models.
"""
import sys
import warnings

from loguru import logger
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from umap import UMAP

from experiments.common import plot_clustered_data
from experiments.real.utils import DataHandler
from experiments.real.utils import SplitMode
from mudpod.clustering import DipMeans
from mudpod.projections import IdentityProjector
from mudpod.projections import JohnsonLindenstrauss
from mudpod.observer import PercentileObserver
from mudpod.observer import RandomObserver
from mudpod.projections import View
from mudpod.misc import set_seed

SEED = 120

set_seed(SEED)


logger.remove()
# add a new handler with level set to INFO
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    max_samples = 1000
    image_key, label_key = "image", "label"
    dataset_name = "fashion_mnist"
    model_name = "abhishek/autotrain_fashion_mnist_vit_base"

    data_handler = DataHandler(
        dataset_name=dataset_name,
        image_key=image_key,
        label_key=label_key,
        model_name=model_name,
        max_samples=max_samples,
        seed=SEED,
    )

    x, y = data_handler.get(SplitMode.TEST)

    v = View(JohnsonLindenstrauss, PercentileObserver(0.99))

    dm = DipMeans(view=v, pval=0.05, sim_num=100, workers_num=10, random_state=SEED)

    clusters = dm.fit(x).labels_

    logger.info(f'The NMI score is {normalized_mutual_info_score(y, clusters)}')

    reducer = UMAP(random_state=SEED)
    reducer.fit(x)
    embeddings = reducer.transform(x)

    plot_clustered_data(embeddings, clusters)
