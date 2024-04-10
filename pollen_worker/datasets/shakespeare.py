"""The SHAKESPEARE dataset and afferent functions.

Based on the implementation of FedScale: Benchmarking Model and System Performance of
Federated Learning at Scale. ICML 2022: 11814-11827 with repo:
https://github.com/SymbioticLab/FedScale . Originially from: LEAF: A Benchmark for
Federated Settings. CoRR abs/1812.01097 (2018).
"""

import csv
import pickle
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pollen_worker.utils import chunks_idx

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)

SHAKESPEARE_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "index": "string",
    "label": np.int64,
}


class Shakespeare(Dataset):
    """Shakespeare dataset object."""

    def __init__(
        self,
        root: Path,
        client_id: int | None = None,
        dataset: str = "train",
        transform: Callable[[Any], torch.Tensor] | None = None,
        target_transform: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
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

    def _word_to_indices(self, word: str) -> list[int]:
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

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the sample at current `index`."""
        with open(self.path_to_data / self.name / self.data.iloc[index], "rb") as f:
            d = pickle.load(f)
            x = d["x"]
            y = d["y"]

        sentence_indices = np.array(self._word_to_indices(x))
        next_word_index = np.array(self.characters.find(y))

        return sentence_indices, next_word_index

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def _check_exists(self) -> bool:
        return Path.exists(self.path_to_data)

    def _old__load_meta_data(self, path: Path) -> tuple[list, list]:
        data, labels = [], []

        with open(path, encoding="locale") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for line_count, row in enumerate(csv_reader):
                if line_count != 0:
                    data.append(row[1])
                    labels.append(row[-1])
                line_count += 1

        return data, labels

    def _load_meta_data(self, path: Path) -> tuple[pd.Series, pd.Series]:
        return _load_shakespeare_meta_data(path, SHAKESPEARE_DTYPES)

    def _load_file(self) -> tuple[pd.Series, pd.Series]:
        path = Path(self.path_to_mapping / self.name / f"{self.client_id}.parquet")
        # Load meta file to get samples path and labels
        data, labels = self._load_meta_data(path)

        return data, labels


class ShakespeareLoaded(Dataset):
    """Shakespeare dataset object stored in the RAM."""

    def __init__(
        self,
        root: Path,
        data_targets: tuple[list, list] | None = None,
        client_id: int | None = None,
        dataset: str = "train",
        transform: Callable[[Any], torch.Tensor] | None = None,
        target_transform: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
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
        if data_targets is not None:
            self.data, self.targets = data_targets
        else:
            self.data, self.targets = self._load_file()

    def _word_to_indices(self, word: str) -> list[int]:
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

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the sample at current `index`."""
        x = self.data[index]
        # y = self.data[index+1][0]
        y = str(self.targets[index])

        sentence_indices = np.array(self._word_to_indices(x))
        next_word_index = np.array(self.characters.find(y))

        return sentence_indices, next_word_index

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def _check_exists(self) -> bool:
        return Path.exists(self.path_to_data)

    def _old__load_meta_data(self, path: Path) -> tuple[list, list]:
        data, labels = [], []

        with open(path, encoding="locale") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for line_count, row in enumerate(csv_reader):
                if line_count != 0:
                    data.append(row[1])
                    labels.append(row[-1])

        return data, labels

    def _load_meta_data(self, path: Path) -> tuple[pd.Series, pd.Series]:
        # Filter deprecation warning from incompatible pandas and pyarrow versions
        warnings.filterwarnings(
            action="ignore",
            category=DeprecationWarning,
            message="Passing a BlockManager to DataFrame*",
        )
        dataframe = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(SHAKESPEARE_DTYPES[col])

        return dataframe["sample_path"], dataframe["label"]

    def _load_file(self) -> tuple[list, list]:
        path = Path(self.path_to_mapping / self.name / f"{self.client_id}.parquet")
        # Load meta file to get samples path and labels
        samples, labels = self._load_meta_data(path)
        data = []
        labels = []
        for sample in samples:
            sample_path = self.path_to_data / self.name / sample
            with open(sample_path, "rb") as f:
                d = pickle.load(f)
                data.append(d["x"])
                labels.append(d["y"])

        return data, labels


def _dump_info(worker_idx: int, client_ids: list[int], dataset: str) -> list:
    clients = []
    time.time()
    for client_id in client_ids:
        ds = ShakespeareLoaded(
            root=Path("/datasets/FedScale/leaf_shakespeare"),
            client_id=client_id,
            dataset=dataset,
        )
        clients.append([client_id, len(ds)])
    return clients


def _create_parquet_clients_dict(dataset: str = "train", n_jobs: int = 100) -> None:
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
    for cnt, (begin, end) in enumerate(
        chunks_idx(range(len(pd.unique(dataframe["client_id"]))), n_jobs)
    ):
        pool_inputs.append([cnt, client_ids[begin:end], dataset])
    pool_outputs = pool.starmap(_dump_info, pool_inputs)
    pool.close()
    pool.join()
    log(INFO, f"Pool outputs length: {len(pool_outputs)}")
    clients = []
    for out in pool_outputs:
        clients.extend(out)
    log(INFO, f"Pool outputs concat length: {len(clients)}")

    clients_df = pd.DataFrame(clients, columns=["client_id", "samples"])
    log(INFO, f"Dataframe: {clients_df.head()}")
    clients_df.to_parquet(
        f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    s_t = time.time()
    parquet_client_df = pd.read_parquet(
        f"/datasets/FedScale/leaf_shakespeare/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    log(INFO, f"Dataframe: {parquet_client_df.head()}")
    log(INFO, f"Read parquet file in {time.time() - s_t} seconds")
    s_t = time.time()
    samples = []
    for i in client_ids:
        samples.append(
            int(parquet_client_df[parquet_client_df["client_id"] == i]["samples"])
        )
    log(INFO, f"Getting all the samples took {time.time() - s_t} seconds")


def _create_parquet_client_samples_map(dataset: str = "train") -> None:
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
        cpus = len(psutil.Process().cpu_affinity())
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


def _load_shakespeare_meta_data(
    path: Path, shakespeare_dtypes: dict
) -> tuple[pd.Series, pd.Series]:
    """Load the Shakespeare dataset from the parquet file."""
    # Filter deprecation warning from incompatible pandas and pyarrow versions
    warnings.filterwarnings(
        action="ignore",
        category=DeprecationWarning,
        message="Passing a BlockManager to DataFrame*",
    )
    dataframe = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    for col in dataframe.columns:
        dataframe[col] = dataframe[col].astype(shakespeare_dtypes[col])

    return dataframe["sample_path"], dataframe["label"]


def load_shakespeare_file(
    path: Path, path_to_data: Path, dataset_name: str, shakespeare_dtypes: dict
) -> tuple[list, list]:
    """Load the Shakespeare dataset from the parquet file."""
    samples, labels = _load_shakespeare_meta_data(path, shakespeare_dtypes)
    data = []
    labels = []
    for sample in samples:
        sample_path = path_to_data / dataset_name / sample
        with open(sample_path, "rb") as f:
            d = pickle.load(f)
            data.append(d["x"])
            labels.append(d["y"])

    return data, labels
