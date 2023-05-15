"""Utility functions used in the `real` experiments."""
from dataclasses import dataclass
from dataclasses import field
from enum import Enum

import torch
from datasets import IterableDataset
from datasets import load_dataset
import numpy as np
from transformers import AutoFeatureExtractor
from transformers import pipeline
from transformers import Pipeline


class SplitMode(str, Enum):
    """The split types."""
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'


@dataclass
class DataHandler:
    """A wrapper class for the data handling."""

    dataset_name: str
    # The name of the dataset.

    image_key: str
    # The key of the image.

    label_key: str
    # The key of the labels.

    model_name: str
    # The name of the model.

    max_samples: int
    # The maximum number of samples.

    seed: int
    # The random seed.

    dataset: IterableDataset = field(init=False, repr=False)
    # The loaded dataset.

    extractor: Pipeline = field(init=False, repr=False)
    # A feature extractor.

    def __post_init__(self):
        if self.max_samples <= 0:
            raise ValueError('The `max_samples` should be positive!')
        if self.seed < 0:
            raise ValueError('The `seed` should be non-negative!')

        self.dataset = load_dataset(self.dataset_name, streaming=True)

        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.extractor = pipeline(
            model=self.model_name,
            tokenizer=feature_extractor,
            task="feature-extraction"
        )

    def _gather(self, dataset_iter: IterableDataset) -> tuple[np.ndarray, np.ndarray]:
        data, labels = [], []

        for x in dataset_iter:
            data.append(x[self.image_key].convert("RGB"))
            labels.append(x[self.label_key])

        data = self.extractor(data, return_tensors="pt")
        # ToDo also add max pooling as an option!
        data = torch.concatenate([torch.mean(d, dim=1) for d in data], dim=0).numpy()

        labels = np.array(labels)
        return data, labels

    def get(self, mode: SplitMode) -> tuple[np.ndarray, np.ndarray]:
        m = mode.value
        dataset = self.dataset[m].shuffle(seed=self.seed).take(self.max_samples)
        data, labels = self._gather(dataset)

        return data, labels
