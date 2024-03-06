"""Transforms on the short time fourier transforms of wav samples."""

__author__ = "Erdene-Ochir Tuguldur"

import random
from typing import Any

import librosa
import numpy as np
from torch.utils.data import Dataset

from pollen_worker.datasets.transforms_wav import should_apply_transform

random.seed(233)


class ToSTFT:
    """Applies on an audio the short time fourier transform."""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        samples = data["samples"]
        data["sample_rate"]
        data["n_fft"] = self.n_fft
        data["hop_length"] = self.hop_length
        data["stft"] = librosa.stft(
            samples, n_fft=self.n_fft, hop_length=self.hop_length
        )
        data["stft_shape"] = data["stft"].shape
        return data


class StretchAudioOnSTFT:
    """Stretches an audio on the frequency domain."""

    def __init__(self, max_scale: float = 0.2) -> None:
        self.max_scale = max_scale

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        stft = data["stft"]
        data["sample_rate"]
        hop_length = data["hop_length"]
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = librosa.core.phase_vocoder(
            stft, rate=1 + scale, hop_length=hop_length
        )
        data["stft"] = stft_stretch
        return data


class TimeshiftAudioOnSTFT:
    """A simple timeshift on the frequency domain without multiplying with exp."""

    def __init__(self, max_shift: int = 8) -> None:
        self.max_shift = max_shift

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        stft = data["stft"]
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        stft = stft[:, b:] if a == 0 else stft[:, 0:-a]
        data["stft"] = stft
        return data


class AddBackgroundNoiseOnSTFT(Dataset):
    """Adds a random background noise on the frequency domain."""

    def __init__(self, bg_dataset: Any, max_percentage: float = 0.45) -> None:
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        if not should_apply_transform():
            return data

        noise = random.choice(self.bg_dataset)["stft"]
        percentage = random.uniform(0, self.max_percentage)
        data["stft"] = data["stft"] * (1 - percentage) + noise * percentage
        return data


class FixSTFTDimension:
    """Pad or truncate in the time axis on the frequency domain.

    This is applied after stretching, time shifting etc.
    """

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        stft = data["stft"]
        t_len = stft.shape[1]
        orig_t_len = data["stft_shape"][1]
        if t_len > orig_t_len:
            stft = stft[:, 0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len - t_len)), "constant")

        data["stft"] = stft
        return data


class ToMelSpectrogramFromSTFT:
    """Create the mel spectrogram from the short time fourier transform of a file.

    The result is a 32x32 matrix.
    """

    def __init__(self, n_mels: int = 32) -> None:
        self.n_mels = n_mels

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        stft = data["stft"]
        sample_rate = data["sample_rate"]
        n_fft = data["n_fft"]
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=self.n_mels)
        s = np.dot(mel_basis, np.abs(stft) ** 2.0)
        data["mel_spectrogram"] = librosa.power_to_db(s, ref=np.max)
        return data


class DeleteSTFT:
    """Remove STFT after computing the mel spectrogram.

    Pytorch doesn't like complex numbers.
    """

    def __call__(self, data: Any) -> Any:
        """Remove the stft from the data."""
        del data["stft"]
        return data


class AudioFromSTFT:
    """Inverse short time fourier transform."""

    def __call__(self, data: Any) -> Any:
        """Implement the execution function."""
        stft = data["stft"]
        data["istft_samples"] = librosa.core.istft(stft, dtype=data["samples"].dtype)
        return data
