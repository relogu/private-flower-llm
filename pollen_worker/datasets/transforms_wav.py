"""Transforms on raw wav samples."""

__author__ = "Yuan Xu"

import random
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

random.seed(233)


def should_apply_transform(prob: float = 0.5) -> bool:
    """Transform are only randomly applied with the given probability."""
    return random.random() < prob


class LoadAudio:
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        path = data["path"]
        if path:
            samples, sample_rate = librosa.load(path, sr=self.sample_rate)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data["samples"] = samples
        data["sample_rate"] = sample_rate
        return data


class FixAudioLength:
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time: int = 1) -> None:
        self.time = time

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        samples = data["samples"]
        sample_rate = data["sample_rate"]
        length = int(self.time * sample_rate)
        if length < len(samples):
            data["samples"] = samples[:length]
        elif length > len(samples):
            data["samples"] = np.pad(samples, (0, length - len(samples)), "constant")
        return data


class ChangeAmplitude:
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range: tuple[float, float] = (0.7, 1.1)) -> None:
        self.amplitude_range = amplitude_range

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        data["samples"] = data["samples"] * random.uniform(*self.amplitude_range)
        return data


class ChangeSpeedAndPitchAudio:
    """Change the speed of an audio.

    This transform also changes the pitch of the audio.
    """

    def __init__(self, max_scale: float = 0.2) -> None:
        self.max_scale = max_scale

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        samples = data["samples"]
        data["sample_rate"]
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0 / (1 + scale)
        data["samples"] = np.interp(
            np.arange(0, len(samples), speed_fac), np.arange(0, len(samples)), samples
        ).astype(np.float32)
        return data


class StretchAudio:
    """Stretches an audio randomly."""

    def __init__(self, max_scale: float = 0.2) -> None:
        self.max_scale = max_scale

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data["samples"] = librosa.effects.time_stretch(data["samples"], rate=1 + scale)
        return data


class TimeshiftAudio:
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds: float = 0.2) -> None:
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        samples = data["samples"]
        sample_rate = data["sample_rate"]
        max_shift = sample_rate * self.max_shift_seconds
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        samples = np.pad(samples, (a, b), "constant")
        data["samples"] = samples[: len(samples) - a] if a else samples[b:]
        return data


class AddBackgroundNoise(Dataset):
    """Adds a random background noise."""

    def __init__(self, bg_dataset: Any, max_percentage: float = 0.45) -> None:
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        samples = data["samples"]
        noise = random.choice(self.bg_dataset)["samples"]
        percentage = random.uniform(0, self.max_percentage)
        data["samples"] = samples * (1 - percentage) + noise * percentage
        return data


class ToMelSpectrogram:
    """Creates the mel spectrogram from an audio.

    The result is a 32x32 matrix.
    """

    def __init__(self, n_mels: int = 32) -> None:
        self.n_mels = n_mels

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        samples = data["samples"]
        sample_rate = data["sample_rate"]
        s = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate, n_mels=self.n_mels
        )
        data["mel_spectrogram"] = librosa.power_to_db(s, ref=np.max)
        return data


class ToTensor:
    """Converts into a tensor."""

    def __init__(
        self,
        np_name: str,
        tensor_name: str,
        normalize: tuple[float, float] | None = None,
    ) -> None:
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data: Any) -> torch.Tensor:
        """Implement the execution function."""
        tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data
