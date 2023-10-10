from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)

SHAKESPEARE_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "index": "string",
    "label": np.int64,
}


class ShakespeareDataset(Dataset):  # NOSONAR
    def __init__(
        self,
        root: Path,
        clint_id: int,
        dataset: str = "train",
        transform=None,
        target_transform=None,
    ):
        self.name: str = dataset  # 'train', 'test', 'validation'
        self.root: Path = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = pd.read_parquet(f"{self.root}/{dataset}.parquet")

        # load data and targets
        self.data, self.target = [], []
        self.load_client(client_id=clint_id)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index.

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        x = self.data[index]
        y = self.target[index][0]

        return x, y

    def __len__(self):
        return len(self.data)

    def load_client(self, client_id: int):
        # load meta file to get labels
        self.data = self.dataset.loc[int(client_id)]["x"]
        self.target = self.dataset.loc[int(client_id)]["y"]
