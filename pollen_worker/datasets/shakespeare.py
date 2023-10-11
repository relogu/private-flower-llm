"""The SHAKESPEARE dataset and afferent functions.

Based on the implementation of FedScale: Benchmarking Model and System Performance of
Federated Learning at Scale. ICML 2022: 11814-11827 with repo:
https://github.com/SymbioticLab/FedScale . Originially from: LEAF: A Benchmark for
Federated Settings. CoRR abs/1812.01097 (2018).
"""

import csv
import os
import pickle
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


def _chunks_idx(list, n_chunks):
    d, r = divmod(len(list), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


class SHAKESPEARE(Dataset):
    """Shakespeare dataset object."""

    classes = []

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
        self.path_to_mapping = Path(self.root, "client_data_mapping")
        self.path_to_data = Path(self.root, "data")

        # load data and targets
        self.data, self.targets = self._load_file()

    def _word_to_indices(self, word):
        """Convert a sequence of characters into position indices.

        The reference string `self.characters` is used.

        Args:
            word (str): Sequence of characters to be converted.

        Returns
        -------
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, index):
        """Return the sample at current `index`."""
        with open(self.path_to_data / self.name / self.data.iloc[index], "rb") as f:
            d = pickle.load(f)
            x = d["x"]
            y = d["y"]

        sentence_indices = np.array(self._word_to_indices(x))
        next_word_index = np.array(self.characters.find(y))

        return sentence_indices, next_word_index

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.path_to_data)

    def _old__load_meta_data(self, path):
        data, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    data.append((row[1]))
                    labels.append(row[-1])
                line_count += 1

        return data, labels

    def _load_meta_data(self, path):
        dataframe = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(SHAKESPEARE_DTYPES[col])

        return dataframe["sample_path"], dataframe["label"]

    def _load_file(self):
        path = Path(self.path_to_mapping / self.name / f"{self.client_id}.parquet")
        # Load meta file to get samples path and labels
        data, labels = self._load_meta_data(path)

        return data, labels


class SHAKESPEARE_LOADED(Dataset):  # NOSONAR
    """Shakespeare dataset object stored in the RAM."""

    classes = []

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
        self.path_to_mapping = Path(self.root, "client_data_mapping")
        self.path_to_data = Path(self.root, "data")

        # load data and targets
        self.data, self.targets = self._load_file()

    def _word_to_indices(self, word):
        """Convert a sequence of characters into position indices.

        The reference string `self.characters` is used.

        Args:
            word (str): Sequence of characters to be converted.

        Returns
        -------
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, index):
        """Return the sample at current `index`."""
        x = self.data[index]
        # y = self.data[index+1][0]
        y = str(self.targets[index])

        sentence_indices = np.array(self._word_to_indices(x))
        next_word_index = np.array(self.characters.find(y))

        return sentence_indices, next_word_index

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.path_to_data)

    def _old__load_meta_data(self, path):
        data, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    data.append((row[1]))
                    labels.append(row[-1])
                line_count += 1

        return data, labels

    def _load_meta_data(self, path):
        dataframe = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(SHAKESPEARE_DTYPES[col])

        return dataframe["sample_path"], dataframe["label"]

    def _load_file(self):
        path = Path(self.path_to_mapping / self.name / f"{self.client_id}.parquet")
        # Load meta file to get samples path and labels
        samples, labels = self._load_meta_data(path)
        data = []
        labels = []
        for _i, sample in enumerate(samples):
            sample_path = self.path_to_data / self.name / sample
            with open(sample_path, "rb") as f:
                d = pickle.load(f)
                data.append(d["x"])
                labels.append(d["y"])

        return data, labels


def _dump_info(worker_idx, client_ids, dataset):
    clients = []
    time.time()
    for _i, client_id in enumerate(client_ids):
        ds = SHAKESPEARE_LOADED(
            root=Path("/datasets/FedScale/leaf_shakespeare"),
            client_id=client_id,
            dataset=dataset,
        )
        clients.append([client_id, len(ds)])
    return clients


def _create_parquet_clients_dict(dataset: str = "train", n_jobs: int = 100):
    log(INFO, f"Creating client data mapping for {dataset} dataset")

    dataframe = pd.read_csv(
        Path(f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}.csv"),
        engine="pyarrow",
        dtype=SHAKESPEARE_DTYPES,
        names=list(SHAKESPEARE_DTYPES.keys()),
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
        f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    s_t = time.time()
    df = pd.read_parquet(
        f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}_clients_dict.parquet"
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
        Path(f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}.csv"),
        engine="pyarrow",
        dtype=SHAKESPEARE_DTYPES,
        names=list(SHAKESPEARE_DTYPES.keys()),
        sep=",",
        header=0,
    )
    Path(f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}").mkdir(
        parents=True, exist_ok=True
    )
    client_ids = pd.unique(dataframe["client_id"])
    for client_id in tqdm(client_ids):
        if not Path(
            f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}/{client_id}.parquet"
        ).exists():
            tmp: pd.DataFrame = dataframe[dataframe["client_id"] == client_id]
            tmp.to_parquet(
                f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}/{client_id}.parquet"
            )


if __name__ == "__main__":
    import time
    from logging import INFO
    from multiprocessing import Pool
    from pathlib import Path

    import psutil
    from flwr.common.logger import log
    from tqdm import tqdm

    # Set the number of jobs
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus

    dataset = "train"

    for dataset in ["train", "test"]:
        _create_parquet_client_samples_map(dataset=dataset)
        if not Path(
            f"/datasets/FedScale/leaf_shakespeare/clients_data_mapping/{dataset}_clients_dict.parquet"
        ).exists():
            _create_parquet_clients_dict(dataset=dataset, n_jobs=n_jobs)
