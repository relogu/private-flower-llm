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


@dataclass
class DatasetConstants:
    """Describe constants for a dataset."""

    chars_per_sample: int  # Computed over validation set
    chars_per_token: int  # TODO: Describe how to compute
    splits: dict[str, DataSplitConstants]

    def __iter__(self) -> Generator[Any, Any, None]:
        """Iterate over splits."""
        yield from self.splits.values()


THE_PILE_CLIENT_MAP = {
    "wikipedia": 0,
    "arxiv": 1,
    "gutenberg": 2,
    "hackernews": 3,
    "pubmedcentral": 4,
    "freelaw": 5,
    "philpapers": 6,
    "dmmathematics": 7,
    "enronemails": 8,
    "europarl": 9,
    "nihexporter": 10,
    "github": 11,
    "pilecc": 12,
    "pubmedabstract": 13,
    "stackexchange": 14,
    "usptobackgrounds": 15,
}


THE_PILE_CLIENT_NAMING_MAP = {
    "wikipedia": "Wikipedia (en)",
    "arxiv": "ArXiv",
    "gutenberg": "Gutenberg (PG-19)",
    "hackernews": "HackerNews",
    "pubmedcentral": "PubMed Central",
    "freelaw": "FreeLaw",
    "philpapers": "PhilPapers",
    "dmmathematics": "DM Mathematics",
    "enronemails": "Enron Emails",
    "europarl": "EuroParl",
    "nihexporter": "NIH ExPorter",
    "github": "Github",
    "pilecc": "Pile-CC",
    "pubmedabstract": "PubMed Abstracts",
    "stackexchange": "StackExchange",
    "usptobackgrounds": "USPTO Backgrounds",
}


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
pile_constants.splits["validation"] = DataSplitConstants(
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

c4_en_constants = DatasetConstants(
    chars_per_sample=2163,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
    splits={},
)
c4_en_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="train",
    folder_split="train",
    raw_samples=364868892,
    truncated_samples=None,
)
c4_en_constants.splits["train_small"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="train",
    folder_split="train_small",
    raw_samples=1000000,
    truncated_samples=100000,
)
c4_en_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val",
    raw_samples=364608,
    truncated_samples=None,
)
c4_en_constants.splits["val_small"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
c4_en_constants.splits["val_xsmall"] = DataSplitConstants(
    path="allenai/c4",
    name="en",
    split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)
c4_en_constants.splits["val_xxsmall"] = DataSplitConstants(
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
)
c4_sr_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="sr",
    split="validation",
    folder_split="val",
    raw_samples=3443,
    truncated_samples=None,
)


"""------------------- C4 (la) -------------------"""

c4_la_constants = DatasetConstants(
    chars_per_sample=2604,  # Computed over validation set
    # chars_per_sample=2503,  # Computed over train set
    chars_per_token=0,
    splits={},
)
c4_la_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="la",
    split="train",
    folder_split="train",
    raw_samples=1674463,
    truncated_samples=None,
)
c4_la_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="la",
    split="validation",
    folder_split="val",
    raw_samples=1654,
    truncated_samples=None,
)


"""------------------- C4 (sw) -------------------"""

c4_sw_constants = DatasetConstants(
    chars_per_sample=3349,  # Computed over validation set
    # chars_per_sample=3317,  # Computed over train set
    chars_per_token=0,
    splits={},
)
c4_sw_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="sw",
    split="train",
    folder_split="train",
    raw_samples=985654,
    truncated_samples=None,
)
c4_sw_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="sw",
    split="validation",
    folder_split="val",
    raw_samples=994,
    truncated_samples=None,
)


"""------------------- C4 (ur) -------------------"""

c4_ur_constants = DatasetConstants(
    chars_per_sample=3418,  # Computed over validation set
    # chars_per_sample=3209,  # Computed over train set
    chars_per_token=0,
    splits={},
)
c4_ur_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="ur",
    split="train",
    folder_split="train",
    raw_samples=1950124,
    truncated_samples=None,
)
c4_ur_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="ur",
    split="validation",
    folder_split="val",
    raw_samples=1885,
    truncated_samples=None,
)


"""------------------- C4 (ms) -------------------"""

c4_ms_constants = DatasetConstants(
    chars_per_sample=3556,  # Computed over validation set
    # chars_per_sample=3564,  # Computed over train set
    chars_per_token=0,
    splits={},
)
c4_ms_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="ms",
    split="train",
    folder_split="train",
    raw_samples=13180647,
    truncated_samples=None,
)
c4_ms_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="ms",
    split="validation",
    folder_split="val",
    raw_samples=13391,
    truncated_samples=None,
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
)
c4_zh_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="zh",
    split="validation",
    folder_split="val",
    raw_samples=54656,  # 1000x validation = 54,656,000, 11000000
    truncated_samples=None,
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
    raw_samples=0,  # 1000x validation = 186,030,000, 37000000
    truncated_samples=None,
)
c4_it_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="it",
    split="validation",
    folder_split="val",
    raw_samples=186030,
    truncated_samples=None,
)


"""------------------- C4 (es) -------------------"""

c4_es_constants = DatasetConstants(
    chars_per_sample=0,
    chars_per_token=0,
    splits={},
)
c4_es_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="es",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
)
c4_es_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="es",
    split="validation",
    folder_split="val",
    raw_samples=0,
    truncated_samples=None,
)


"""------------------- C4 (de) -------------------"""

c4_de_constants = DatasetConstants(
    chars_per_sample=0,
    chars_per_token=0,
    splits={},
)
c4_de_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="de",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
)
c4_de_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="de",
    split="validation",
    folder_split="val",
    raw_samples=0,
    truncated_samples=None,
)


"""------------------- C4 (el) -------------------"""

c4_el_constants = DatasetConstants(
    chars_per_sample=0,
    chars_per_token=0,
    splits={},
)
c4_el_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="el",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
)
c4_el_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="el",
    split="validation",
    folder_split="val",
    raw_samples=0,
    truncated_samples=None,
)


"""------------------- C4 (ru) -------------------"""

c4_ru_constants = DatasetConstants(
    chars_per_sample=0,
    chars_per_token=0,
    splits={},
)
c4_ru_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="ru",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
)
c4_ru_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="ru",
    split="validation",
    folder_split="val",
    raw_samples=0,
    truncated_samples=None,
)


"""------------------- C4 (hi) -------------------"""

c4_hi_constants = DatasetConstants(
    chars_per_sample=0,
    chars_per_token=0,
    splits={},
)
c4_hi_constants.splits["train"] = DataSplitConstants(
    path="allenai/c4",
    name="hi",
    split="train",
    folder_split="train",
    raw_samples=0,
    truncated_samples=None,
)
c4_hi_constants.splits["validation"] = DataSplitConstants(
    path="allenai/c4",
    name="hi",
    split="validation",
    folder_split="val",
    raw_samples=0,
    truncated_samples=None,
)

"""------------------- Constants dict -------------------"""

CONSTANTS: dict[str, DatasetConstants] = {
    "c4_en": c4_en_constants,
    "c4_it": c4_it_constants,
    "c4_zh": c4_zh_constants,
    "c4_ms": c4_ms_constants,
    "c4_ur": c4_ur_constants,
    "c4_sw": c4_sw_constants,
    "c4_la": c4_la_constants,
    "c4_sr": c4_sr_constants,
    "c4_es": c4_es_constants,
    "c4_de": c4_de_constants,
    "c4_el": c4_el_constants,
    "c4_ru": c4_ru_constants,
    "c4_hi": c4_hi_constants,
    "the_pile": pile_constants,
}
