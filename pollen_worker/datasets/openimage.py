from __future__ import print_function

import os
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
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


def _chunks_idx(list, n_chunks):
    d, r = divmod(len(list), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


class OpenImage(Dataset):
    """Open Image dataset object."""

    classes = []

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
        self.data_file = dataset  # 'train', 'test', 'val'

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You have to download it")

        self.path = os.path.join(self.root, self.data_file)
        # load data and targets
        self.data, self.targets = self._load_file()
        self.imgview = imgview

    def __getitem__(self, index):
        """Return the sample at current `index`."""
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
        """Return the length of the dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.data_file))

    def _load_meta_data(self, path):
        dataframe = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(OPENIMAGE_DTYPES[col])

        return dataframe["sample_path"].tolist(), dataframe["label_id"].tolist()

    def _load_file(self):
        path = Path(
            self.root
            / "client_data_mapping"
            / self.data_file
            / f"{self.client_id}.parquet"
        )
        # Load meta file to get samples path and labels
        datas, labels = self._load_meta_data(path)

        return datas, labels


def _dump_info(worker_idx, client_ids, dataset):
    clients = []
    time.time()
    for _i, client_id in enumerate(client_ids):
        ds = OpenImage(
            root=Path("/datasets/FedScale/openImg"),
            client_id=client_id,
            dataset=dataset,
        )
        clients.append([client_id, len(ds)])
    return clients


def _create_parquet_clients_dict(dataset: str = "train", n_jobs: int = 100):
    log(INFO, f"Creating client data mapping for {dataset} dataset")

    dataframe = pd.read_csv(
        Path(f"/datasets/FedScale/openImg/client_data_mapping/{dataset}.csv"),
        engine="pyarrow",
        dtype=OPENIMAGE_DTYPES,
        names=list(OPENIMAGE_DTYPES.keys()),
        sep=",",
        header=0,
    )

    pool_inputs = []
    pool = Pool(n_jobs)
    client_ids = pd.unique(dataframe["client_id"])
    cnt = 0
    for begin, end in _chunks_idx(
        range(len(pd.unique(dataframe["client_id"]))), n_jobs
    ):
        pool_inputs.append([cnt, client_ids[begin:end], dataset])
        cnt += 1
    pool_outputs = pool.starmap(_dump_info, pool_inputs)
    pool.close()
    pool.join()
    log(INFO, f"Pool outputs length: {len(pool_outputs)}")
    clients = []
    [clients.extend(out) for out in pool_outputs]
    log(INFO, f"Pool outputs concat length: {len(clients)}")

    df = pd.DataFrame(clients, columns=["client_id", "samples"])
    log(INFO, f"Dataframe: {df.head()}")
    df.to_parquet(
        f"/datasets/FedScale/openImg/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    s_t = time.time()
    df = pd.read_parquet(
        f"/datasets/FedScale/openImg/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    log(INFO, f"Dataframe: {df.head()}")
    log(INFO, f"Read parquet file in {time.time()-s_t} seconds")
    s_t = time.time()
    samples = []
    for i in client_ids:
        samples.append(int(df[df["client_id"] == i]["samples"]))
    log(INFO, f"Getting all the samples took {time.time()-s_t} seconds")


def _create_parquet_client_samples_map(dataset: str = "train"):
    dataframe = pd.read_csv(
        Path(f"/datasets/FedScale/openImg/client_data_mapping/{dataset}.csv"),
        engine="pyarrow",
        dtype=OPENIMAGE_DTYPES,
        names=list(OPENIMAGE_DTYPES.keys()),
        sep=",",
        header=0,
    )
    Path(f"/datasets/FedScale/openImg/client_data_mapping/{dataset}").mkdir(
        parents=True, exist_ok=True
    )
    client_ids = pd.unique(dataframe["client_id"])
    for client_id in tqdm(client_ids):
        if not Path(
            f"/datasets/FedScale/openImg/client_data_mapping/{dataset}/{client_id}.parquet"
        ).exists():
            tmp: pd.DataFrame = dataframe[dataframe["client_id"] == client_id]
            tmp.to_parquet(
                f"/datasets/FedScale/openImg/client_data_mapping/{dataset}/{client_id}.parquet"
            )


if __name__ == "__main__":
    import time
    from logging import INFO
    from multiprocessing import Pool

    import psutil
    from flwr.common.logger import log
    from tqdm import tqdm

    # Set the number of jobs
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus

    dataset = "train"

    for dataset in ["train", "test", "val"]:
        _create_parquet_client_samples_map(dataset=dataset)
        if not Path(
            f"/datasets/FedScale/openImg/clients_data_mapping/{dataset}_clients_dict.parquet"
        ).exists():
            _create_parquet_clients_dict(dataset=dataset, n_jobs=n_jobs)
