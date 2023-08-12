from __future__ import print_function

import os
import os.path
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

OPENIMAGE_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "label_name": "string",
    "label_id": np.int64,
}

train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class OpenImage(Dataset):
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

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
        root,
        client_id=None,
        dataset="train",
        transform=train_transform,
        target_transform=torch.tensor,
        imgview=False,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.client_id = client_id
        self.data_file = dataset  # 'train', 'test', 'validation'

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You have to download it")

        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data, self.targets = self.load_file()
        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.path, imgName))

        # avoid channel error
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def load_meta_data(self, path):
        dataframe = pd.read_csv(
            path,
            # engine="pyarrow", # NOSONAR
            dtype=OPENIMAGE_DTYPES,
            names=list(OPENIMAGE_DTYPES.keys()),
            sep=",",
            header=0,
        )

        # if self.client_id is not None:
        #     dataframe = dataframe[dataframe['client_id'] == self.client_id]

        return dataframe["sample_path"].tolist(), dataframe["label_id"].tolist()

    def load_file(self):
        # load meta file to get labels
        datas, labels = self.load_meta_data(
            os.path.join(
                self.processed_folder,
                "client_data_mapping",
                # self.data_file+'.csv'
                self.data_file,
                f"{self.client_id}.csv",
            )
        )

        return datas, labels


if __name__ == "__main__":
    img_root = Path("/datasets/FedScale/openImg/")
    cid_csv_root = Path("/datasets/FedScale/openImg/client_data_mapping/clean_ids")
    train_dataset = OpenImage(
        root=img_root, transform=train_transform, target_transform=torch.tensor
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )

    for img, lbl in train_dataloader:
        print(img.shape)
        print(lbl)
