import csv
import os
import pickle
import warnings
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


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


class SHAKESPEARE(Dataset):
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
        """Converts a sequence of characters into position indices in the reference
        string `self.characters`.

        Args:
            word (str): Sequence of characters to be converted.

        Returns
        -------
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index.

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        with open(self.path_to_data / self.name / self.data.iloc[index], "rb") as f:  # type: ignore
            d = pickle.load(f)
            x = d["x"]
            y = d["y"]

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
        return os.path.exists(self.path_to_data)

    def old_load_meta_data(self, path):
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

    def load_meta_data(self, path):
        dataframe = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(SHAKESPEARE_DTYPES[col])

        return dataframe["sample_path"], dataframe["label"]

    def load_file(self):
        path = Path(self.path_to_mapping / self.name / f"{self.client_id}.parquet")
        # Load meta file to get samples path and labels
        data, labels = self.load_meta_data(path)

        return data, labels


class SHAKESPEARE_LOADED(Dataset):  # NOSONAR
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
        """Converts a sequence of characters into position indices in the reference
        string `self.characters`.

        Args:
            word (str): Sequence of characters to be converted.

        Returns
        -------
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index.

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        x = self.data[index]
        # y = self.data[index+1][0]
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
        return os.path.exists(self.path_to_data)

    def old_load_meta_data(self, path):
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

    def load_meta_data(self, path):
        dataframe = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(SHAKESPEARE_DTYPES[col])

        return dataframe["sample_path"], dataframe["label"]

    def load_file(self):
        path = Path(self.path_to_mapping / self.name / f"{self.client_id}.parquet")
        # Load meta file to get samples path and labels
        samples, labels = self.load_meta_data(path)
        data = []
        labels = []
        for _i, sample in enumerate(samples):
            sample_path = self.path_to_data / self.name / sample
            with open(sample_path, "rb") as f:
                d = pickle.load(f)
                data.append(d["x"])
                labels.append(d["y"])

        return data, labels


def dump_info(worker_idx, client_ids, dataset):
    clients = []
    time.time()
    for _i, client_id in enumerate(client_ids):
        ds = SHAKESPEARE_LOADED(
            root=Path("/datasets/FedScale/leaf_shakespeare"),
            client_id=client_id,
            dataset=dataset,
        )
        clients.append([client_id, len(ds)])
        # if i % 10 == 0:
        #     log(
        #         INFO,
        #         f"Worker {worker_idx}: {len(client_ids)-i} client_ids left, {i} client_ids complete, remaining time {(time.time()-start_time)/(i+1)*(len(client_ids)-i)}",
        #     )
    return clients


def create_parquet_clients_dict(dataset: str = "train", n_jobs: int = 100):
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
    for begin, end in chunks_idx(range(len(pd.unique(dataframe["client_id"]))), n_jobs):
        pool_inputs.append([cnt, client_ids[begin:end], dataset])
        cnt += 1
    pool_outputs = pool.starmap(dump_info, pool_inputs)
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


def create_parquet_client_samples_map(dataset: str = "train"):
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

    import pandas as pd
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
        create_parquet_client_samples_map(dataset=dataset)
        if not Path(
            f"/datasets/FedScale/leaf_shakespeare/clients_data_mapping/{dataset}_clients_dict.parquet"
        ).exists():
            create_parquet_clients_dict(dataset=dataset, n_jobs=n_jobs)
