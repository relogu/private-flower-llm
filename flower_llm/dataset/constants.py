"""Constants for datasets used in our work."""

from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConcatMode(Enum):
    """Describe concatenation modes."""

    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


@dataclass
class DataSplitConstants:
    """Describe constants for a dataset split."""

    path: str  # Path or name of the dataset
    name: str  # Defining the name of the dataset configuration.
    split: str  # Split
    folder_split: str  # Custom split (and folder name)

    raw_samples: int
    truncated_samples: int | None
    denominator: int | None = None


@dataclass
class DatasetConstants:
    """Describe constants for a dataset."""

    chars_per_sample: int  # Computed over validation set
    chars_per_token: int  # TODO: Describe how to compute
    splits: dict[str, DataSplitConstants]

    def __iter__(self) -> Generator[Any, Any, None]:
        """Iterate over splits."""
        yield from self.splits.values()


"""------------------- The Pile -------------------"""


pile_constants = DatasetConstants(
    chars_per_sample=6212,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
    splits={},
)
pile_constants.splits["train"] = DataSplitConstants(
    path="",
    name="",
    split="train",
    folder_split="train",
    raw_samples=210607728,
    truncated_samples=None,
)
pile_constants.splits["train_small"] = DataSplitConstants(
    path="",
    name="",
    split="train",
    folder_split="train_small",
    raw_samples=1000000,
    truncated_samples=100000,
)
pile_constants.splits["val"] = DataSplitConstants(
    path="",
    name="",
    split="validation",
    folder_split="val",
    raw_samples=214670,
    truncated_samples=None,
)
pile_constants.splits["val_small"] = DataSplitConstants(
    path="",
    name="",
    split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
pile_constants.splits["val_xsmall"] = DataSplitConstants(
    path="",
    name="",
    split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)


"""------------------- C4 (en) -------------------"""

c4_constants = DatasetConstants(
    chars_per_sample=2163,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
    splits={},
)
c4_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="train",
    folder_split="train",
    raw_samples=364868892,
    truncated_samples=None,
    denominator=85336729,
)
c4_constants.splits["train_small"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="train",
    folder_split="train_small",
    raw_samples=1000000,
    truncated_samples=100000,
)
c4_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val",
    raw_samples=364608,
    truncated_samples=None,
    denominator=85039,
)
c4_constants.splits["val_small"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
c4_constants.splits["val_xsmall"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)
c4_constants.splits["val_xxsmall"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val_xxsmall",
    raw_samples=100,
    truncated_samples=100,
)


"""------------------- C4 (sr) -------------------"""

c4_sr_constants = DatasetConstants(
    chars_per_sample=3756,  # Computed over validation set
    # chars_per_sample=3686,  # Computed over train set
    chars_per_token=0,
    splits={},
)
c4_sr_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="sr",
    split="train",
    folder_split="train",
    raw_samples=3398483,
    truncated_samples=None,
    denominator=0,
)
c4_sr_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="sr",
    split="validation",
    folder_split="val",
    raw_samples=3443,
    truncated_samples=None,
    denominator=0,
)


"""------------------- C4 (la) -------------------"""

c4_la_constants = DatasetConstants(
    chars_per_sample=2604,  # Computed over validation set
    chars_per_token=0,
    splits={},
)
c4_la_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="la",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
    denominator=0,
)
c4_la_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="la",
    split="validation",
    folder_split="val",
    raw_samples=1654,
    truncated_samples=None,
    denominator=0,
)


"""------------------- C4 (sw) -------------------"""

c4_sw_constants = DatasetConstants(
    chars_per_sample=3349,  # Computed over validation set
    chars_per_token=0,
    splits={},
)
c4_sw_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="sw",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
    denominator=0,
)
c4_sw_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="sw",
    split="validation",
    folder_split="val",
    raw_samples=994,
    truncated_samples=None,
    denominator=0,
)


"""------------------- C4 (ur) -------------------"""

c4_ur_constants = DatasetConstants(
    chars_per_sample=3418,  # Computed over validation set
    chars_per_token=0,
    splits={},
)
c4_ur_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="ur",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
    denominator=0,
)
c4_ur_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="ur",
    split="validation",
    folder_split="val",
    raw_samples=1885,
    truncated_samples=None,
    denominator=0,
)


"""------------------- C4 (ms) -------------------"""

c4_ms_constants = DatasetConstants(
    chars_per_sample=3556,  # Computed over validation set
    chars_per_token=0,
    splits={},
)
c4_ms_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="ms",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
    denominator=0,
)
c4_ms_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="ms",
    split="validation",
    folder_split="val",
    raw_samples=13391,
    truncated_samples=None,
    denominator=0,
)


"""------------------- C4 (zh) -------------------"""

c4_zh_constants = DatasetConstants(
    chars_per_sample=1144,  # Computed over validation set
    chars_per_token=0,
    splits={},
)
c4_zh_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="zh",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
    denominator=0,
)
c4_zh_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="zh",
    split="validation",
    folder_split="val",
    raw_samples=54656,
    truncated_samples=None,
    denominator=0,
)


"""------------------- C4 (it) -------------------"""

c4_it_constants = DatasetConstants(
    chars_per_sample=2991,  # Computed over validation set
    chars_per_token=0,
    splits={},
)
c4_it_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="it",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
    denominator=0,
)
c4_it_constants.splits["val"] = DataSplitConstants(
    path="allenai/c4",
    name="it",
    split="validation",
    folder_split="val",
    raw_samples=186030,
    truncated_samples=None,
    denominator=0,
)

"""------------------- Constants dict -------------------"""

CONSTANTS: dict[str, DatasetConstants] = {
    "c4": c4_constants,
    "c4_it": c4_it_constants,
    "c4_zh": c4_zh_constants,
    "c4_ms": c4_ms_constants,
    "c4_ur": c4_ur_constants,
    "c4_sw": c4_sw_constants,
    "c4_la": c4_la_constants,
    "c4_sr": c4_sr_constants,
    "the_pile": pile_constants,
}
