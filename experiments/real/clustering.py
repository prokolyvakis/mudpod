"""Clustering experiments with real datasets."""
from loguru import logger
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score

from experiments.real.utils import DataHandler
from experiments.real.utils import SplitMode
from hdunim.clustering import DipMeans
from hdunim.projections import JohnsonLindenstrauss
from hdunim.observer import PercentileObserver
from hdunim.projections import View
from hdunim.misc import set_seed

SEED = 120

set_seed(SEED)


# logger.remove()
# logger.add(
#     sys.stderr,
#     level="INFO"
# )


if __name__ == "__main__":
    max_samples = 1000
    do_resize = True
    image_key, label_key, is_rgb = "image", "label", False
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
    # ms = group_data_points(x, y)
    # i, j = 0, 1
    # x = np.concatenate([ms[i], ms[j]], axis=0)
    # logger.info(
    #     f'The 1st modality has {ms[i].shape[0]} samples, while '
    #     f'the 2d modality has {ms[j].shape[0]} samples!'
    # )
    # y = np.array(
    #     ([0.] * ms[i].shape[0]) +
    #     ([1.] * ms[j].shape[0])
    # )

    v = View(JohnsonLindenstrauss, PercentileObserver(0.99))

    dm = DipMeans(view=v, pval=0.05, sim_num=10, workers_num=10)

    clusters = dm.fit(x).predict(x)

    logger.info(f'The NMI score is {normalized_mutual_info_score(y, clusters)}')