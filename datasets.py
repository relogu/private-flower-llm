import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from models import LEAF_CHARACTERS

SHAKESPEARE_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "index": "string",
    "label": np.int64,
}


class SHAKESPEARE_LOADED(Dataset):  # NOSONAR
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: Path,
        client_id=None,
        dataset="train",
        transform=None,
        target_transform=None,
    ):
        self.characters = LEAF_CHARACTERS
        self.num_letters = len(self.characters)  # 80

        self.client_id = client_id
        self.name = dataset  # 'train', 'test', 'validation'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # load data and targets
        self.data, self.targets = self.load_file()

    def word_to_indices(self, word):
        """Converts a sequence of characters into position indices in the
        reference string `self.characters`.
        Args:
            word (str): Sequence of characters to be converted.
        Returns:
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.data[index]
        y = self.data[index + 1][0]
        y = str(self.targets[index])

        sentence_indices = np.array(self.word_to_indices(x))
        next_word_index = np.array(self.characters.find(y))

        return sentence_indices, next_word_index

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    @property
    def path_to_data(self):
        return Path(self.processed_folder, "data")

    @property
    def path_to_mapping(self):
        return Path(self.processed_folder, "client_data_mapping")

    def _check_exists(self):
        return self.path_to_data.exists()

    def load_meta_data(self, path):
        dataframe = pd.read_csv(
            path,
            # engine="pyarrow", # NOSONAR
            dtype=SHAKESPEARE_DTYPES,
            names=list(SHAKESPEARE_DTYPES.keys()),
            sep=",",
            header=0,
        )

        # if self.client_id is not None:
        #     dataframe = dataframe[dataframe['client_id'] == self.client_id]

        return dataframe["sample_path"], dataframe["label"]

    def load_file(self):
        # load meta file to get labels
        samples, labels = self.load_meta_data(
            # Path(self.processed_folder, 'client_data_mapping', self.name + '.csv'))
            # (self.path_to_mapping/self.name).with_suffix('.csv'))
            self.path_to_mapping
            / self.name
            / f"{self.client_id}.csv"
        )
        data = []
        labels = []
        for _, sample in enumerate(samples):
            sample_path = self.path_to_data / self.name / sample
            with open(sample_path, "rb") as f:
                d = pickle.load(f)
                data.append(d["x"])
                labels.append(d["y"])

        return data, labels
