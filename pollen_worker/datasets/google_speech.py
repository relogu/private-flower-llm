"""The google speech dataset and afferent functions.

Based on the implementation of FedScale: Benchmarking Model and System Performance of
Federated Learning at Scale. ICML 2022: 11814-11827 with repo:
https://github.com/SymbioticLab/FedScale
"""
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from pollen_worker.datasets.transforms_stft import (
    AddBackgroundNoiseOnSTFT,
    DeleteSTFT,
    FixSTFTDimension,
    StretchAudioOnSTFT,
    TimeshiftAudioOnSTFT,
    ToMelSpectrogramFromSTFT,
    ToSTFT,
)
from pollen_worker.datasets.transforms_wav import (
    ChangeAmplitude,
    ChangeSpeedAndPitchAudio,
    FixAudioLength,
    LoadAudio,
    ToMelSpectrogram,
    ToTensor,
)
from pollen_worker.utils import chunks_idx

CLASSES = [
    "up",
    "two",
    "sheila",
    "zero",
    "yes",
    "five",
    "one",
    "happy",
    "marvin",
    "no",
    "go",
    "seven",
    "eight",
    "tree",
    "stop",
    "down",
    "forward",
    "learn",
    "house",
    "three",
    "six",
    "backward",
    "dog",
    "cat",
    "wow",
    "left",
    "off",
    "on",
    "four",
    "visual",
    "nine",
    "bird",
    "right",
    "follow",
    "bed",
]


GOOGLE_SPEECH_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "label_name": "string",
    "label_id": np.int64,
}


class BackgroundNoiseDataset:
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if d.endswith(".wav")]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[: r * c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return the sample at current `index`."""
        data = {
            "samples": self.samples[index],
            "sample_rate": self.sample_rate,
            "target": 1,
            "path": self.path,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class SPEECH(Dataset):
    """Google Speech dataset object."""

    classes = []

    def __init__(
        self,
        root,
        client_id=None,
        dataset="train",
        transform=None,
        target_transform=None,
        classes=CLASSES,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.client_id = client_id
        self.data_file = dataset  # 'train', 'test', 'validation'
        if self.transform is None:
            self._set_default_transform()

        self.classMapping = {classes[i]: i for i in range(len(classes))}

        # load data and targets
        self.data, self.targets = self._load_file()

        self.data_dir = self.root / self.data_file

    def __getitem__(self, index):
        """Return the sample at current `index`."""
        path, target = self.data[index], int(self.targets[index])
        data = {"path": os.path.join(self.data_dir, path), "target": target}

        if self.transform is not None:
            data = self.transform(data)

        return data["input"], data["target"]

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
            dataframe[col] = dataframe[col].astype(GOOGLE_SPEECH_DTYPES[col])

        dataframe["label_name"] = dataframe["label_name"].map(
            lambda x: self.classMapping[x]
        )

        return dataframe["sample_path"].tolist(), dataframe["label_name"].tolist()

    def _load_file(self):
        sample_paths, label_names = [], []
        filename = Path(
            self.root
            / "client_data_mapping"
            / self.data_file
            / f"{self.client_id}.parquet"
        )
        # Load meta file to get sample paths and labels
        sample_paths, label_names = self._load_meta_data(filename)

        return sample_paths, label_names

    def _set_default_transform(self):
        if self.data_file == "train":
            bkg = "_background_noise_"
            data_aug_transform = transforms.Compose(
                [
                    ChangeAmplitude(),
                    ChangeSpeedAndPitchAudio(),
                    FixAudioLength(),
                    ToSTFT(),
                    StretchAudioOnSTFT(),
                    TimeshiftAudioOnSTFT(),
                    FixSTFTDimension(),
                ]
            )
            bg_dataset = BackgroundNoiseDataset(
                os.path.join(self.root, bkg), data_aug_transform
            )
            add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = transforms.Compose(
                [
                    ToMelSpectrogramFromSTFT(n_mels=32),
                    DeleteSTFT(),
                    ToTensor("mel_spectrogram", "input"),
                ]
            )
            self.transform = transforms.Compose(
                [LoadAudio(), data_aug_transform, add_bg_noise, train_feature_transform]
            )
        elif self.data_file == "test":
            valid_feature_transform = transforms.Compose(
                [ToMelSpectrogram(n_mels=32), ToTensor("mel_spectrogram", "input")]
            )
            self.transform = transforms.Compose(
                [LoadAudio(), FixAudioLength(), valid_feature_transform]
            )


def _dump_info(worker_idx, client_ids, dataset):
    clients = []
    time.time()
    for _i, client_id in enumerate(client_ids):
        ds = SPEECH(
            root=Path("/datasets/FedScale/google_speech/google_speech"),
            client_id=client_id,
            dataset=dataset,
        )
        clients.append([client_id, len(ds)])
    return clients


def _create_parquet_clients_dict(dataset: str = "train", n_jobs: int = 100):
    log(INFO, f"Creating client data mapping for {dataset} dataset")

    dataframe = pd.read_csv(
        Path(
            f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}.csv"
        ),
        engine="pyarrow",
        dtype=GOOGLE_SPEECH_DTYPES,
        names=list(GOOGLE_SPEECH_DTYPES.keys()),
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
        f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    s_t = time.time()
    df = pd.read_parquet(
        f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}_clients_dict.parquet"
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
        Path(
            f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}.csv"
        ),
        engine="pyarrow",
        dtype=GOOGLE_SPEECH_DTYPES,
        names=list(GOOGLE_SPEECH_DTYPES.keys()),
        sep=",",
        header=0,
    )
    Path(
        f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}"
    ).mkdir(parents=True, exist_ok=True)
    client_ids = pd.unique(dataframe["client_id"])
    for client_id in tqdm(client_ids):
        if not Path(
            f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}/{client_id}.parquet"
        ).exists():
            tmp: pd.DataFrame = dataframe[dataframe["client_id"] == client_id]
            tmp.to_parquet(
                f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}/{client_id}.parquet"
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
            f"/datasets/FedScale/google_speech/google_speech/client_data_mapping/{dataset}_clients_dict.parquet"
        ).exists():
            _create_parquet_clients_dict(dataset=dataset, n_jobs=n_jobs)
