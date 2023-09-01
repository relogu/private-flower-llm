import os
import warnings

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms_stft import (
    AddBackgroundNoiseOnSTFT,
    DeleteSTFT,
    FixSTFTDimension,
    StretchAudioOnSTFT,
    TimeshiftAudioOnSTFT,
    ToMelSpectrogramFromSTFT,
    ToSTFT,
)
from datasets.transforms_wav import (
    ChangeAmplitude,
    ChangeSpeedAndPitchAudio,
    FixAudioLength,
    LoadAudio,
    ToMelSpectrogram,
    ToTensor,
)

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
        return len(self.samples)

    def __getitem__(self, index):
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
            self.set_default_transform()

        self.classMapping = {classes[i]: i for i in range(len(classes))}

        # load data and targets
        self.data, self.targets = self.load_file()

        self.data_dir = self.root / self.data_file

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.data[index], int(self.targets[index])
        data = {"path": os.path.join(self.data_dir, path), "target": target}

        if self.transform is not None:
            data = self.transform(data)

        return data["input"], data["target"]

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.data_file))

    def load_meta_data(self, path):
        dataframe = pd.read_csv(
            path,
            # NOTE: we may want to add in the future: engine="pyarrow", # NOSONAR
            dtype=GOOGLE_SPEECH_DTYPES,
            names=list(GOOGLE_SPEECH_DTYPES.keys()),
            sep=",",
            header=0,
        )

        dataframe["label_name"] = dataframe["label_name"].map(
            lambda x: self.classMapping[x]
        )

        return dataframe["sample_path"].tolist(), dataframe["label_name"].tolist()

    def load_file(self):
        rawData, rawTags = [], []
        filename = os.path.join(
            self.root, "client_data_mapping", self.data_file, f"{self.client_id}.csv"
        )
        # load meta file to get labels
        rawData, rawTags = self.load_meta_data(filename)

        return rawData, rawTags

    def set_default_transform(self):
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

if __name__ == "__main__":
    # import sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    import pandas as pd
    import time
    from flwr.common.logger import log
    from logging import INFO
    from multiprocessing import Pool
    import psutil
    from pathlib import Path

    def chunks_idx(l, n):
        d, r = divmod(len(l), n)
        for i in range(n):
            si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            yield si, si + (d + 1 if i < r else d)
    
    # Set the number of jobs
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus
    
    def dump_info(worker_idx, client_ids):
        clients = []
        start_time = time.time()
        for i, client_id in enumerate(client_ids):
            ds = SPEECH(
                root=Path("/datasets/FedScale/google_speech/google_speech"),
                client_id=client_id,
            )
            clients.append([client_id, len(ds)])
            if i % 10 == 0:
                log(INFO,
                    f"Worker {worker_idx}: {len(client_ids)-i} client_ids left, {i} client_ids complete, remaining time {(time.time()-start_time)/(i+1)*(len(client_ids)-i)}"
                )
        return clients
    
    
    dataframe = pd.read_csv(
        Path("/datasets/FedScale/google_speech/google_speech/client_data_mapping/train.csv"),
        engine="pyarrow", # NOSONAR
        dtype=GOOGLE_SPEECH_DTYPES,
        names=list(GOOGLE_SPEECH_DTYPES.keys()),
        sep=",",
        header=0,
    )
        
    # Parallelise the tokenisation
    pool_inputs = []
    pool = Pool(n_jobs)
    client_ids = pd.unique(dataframe["client_id"])
    cnt = 0
    for begin, end in chunks_idx(range(len(pd.unique(dataframe["client_id"]))), n_jobs):
        pool_inputs.append(
            [cnt, client_ids[begin:end]]
        )
        cnt += 1
    pool_outputs = pool.starmap(dump_info, pool_inputs)
    pool.close()
    pool.join()
    log(INFO,
        f"Pool outputs length: {len(pool_outputs)}"
    )
    clients = []
    [clients.extend(out) for out in pool_outputs]
    log(INFO,
        f"Pool outputs concat length: {len(clients)}"
    )
    
    df = pd.DataFrame(clients, columns=["client_id", "samples"])
    log(INFO,
        f"Dataframe: {df.head()}"
    )
    df.to_parquet("/datasets/FedScale/google_speech/google_speech/client_data_mapping/clients_dict.parquet")
    s_t = time.time()
    df = pd.read_parquet("/datasets/FedScale/google_speech/google_speech/client_data_mapping/clients_dict.parquet")
    log(INFO,
        f"Dataframe: {df.head()}"
    )
    log(INFO, f"Read parquet file in {time.time()-s_t} seconds")
    s_t = time.time()
    samples = []
    for i in client_ids:
        samples.append(int(df[df['client_id'] == i]['samples']))
    log(INFO,
        f"Getting samples of clients from 0 to 100: {samples}"
    )
    log(INFO, f"Getting samples took {time.time()-s_t} seconds")
