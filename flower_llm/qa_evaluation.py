"""
Question Answering Model Evaluation.

This module provides functionality for training and evaluating question answering models
using the SQuAD dataset. It includes functions for preprocessing the dataset, setting up
data loaders, initializing the model, and computing evaluation metrics such as exact
match and F1 score.

Imports
-------
- logging.INFO : Logging level for informational messages.
- datasets : Library for loading and processing datasets.
- transformers : Hugging Face library for transformer models.
- torch.utils.data.DataLoader : DataLoader class for creating data loaders.
- llmfoundry.utils.builders.build_tokenizer : Function to build a tokenizer.
- os : Module for interacting with the operating system.
- typing : Module for type hints.
- uuid : Module for generating unique identifiers.
- omegaconf.DictConfig : Configuration class for OmegaConf.
- composer.Trainer : Trainer class from the Composer library.
- flower_llm.server.s3_utils.load_pretrained_model_from_path : Function to load a
    pretrained model from S3.
- flower_llm.conf.base_schema.S3CommConfig : Configuration class for S3 communication.
- collections.abc.Callable, Mapping : Abstract base classes for callable and mapping
    types.
- torch : PyTorch library.
- torchmetrics.Metric : Base class for metrics in PyTorch.
- datasets.load_metric : Function to load evaluation metrics.
- copy.deepcopy : Function to create deep copies of objects.
- llmfoundry.utils.builders.build_composer_model : Function to build a Composer model.
- llmfoundry.utils.config_utils.process_init_device : Function to process the
    initialization device.

Functions
---------
- main() -> None
    Main function to train and evaluate a question answering model using the SQuAD
    dataset.

Classes
-------
- ExactMatchMetric(Metric)
    Exact Match Metric for Question Answering.
- F1ScoreMetric(Metric)
    F1 Score Metric for Question Answering.

Dependencies
------------
- logging
- datasets
- transformers
- torch
- torchmetrics
- llmfoundry
- omegaconf
- composer
- flower_llm
- os
- typing
- uuid
- collections
- copy
"""

from logging import INFO
import datasets
import transformers

from torch.utils.data import DataLoader

from llmfoundry.utils.builders import (
    build_tokenizer,
)

import os
from typing import Any, cast
import uuid
from omegaconf import DictConfig
from composer import Trainer, Evaluator
from flower_llm.server.s3_utils import load_pretrained_model_from_path
from flower_llm.conf.base_schema import S3CommConfig

from collections.abc import Callable, Mapping
import torch
from torchmetrics import Metric
from datasets import load_metric

from copy import deepcopy

from llmfoundry.utils.builders import (
    build_composer_model,
)
from llmfoundry.utils.config_utils import (
    process_init_device,
)

from flwr.common import log

from flower_llm.models import MPTForQuestionAnswering

from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR

from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

from datasets.formatting.formatting import LazyBatch


def get_preprocess_training_examples_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
    train_example_ids_map: dict[str, int],
) -> Callable[[LazyBatch], BatchEncoding]:
    """
    Create a preprocessing function for training examples for question answering tasks.

    This function returns a preprocessing function that tokenizes question and context,
    handles overflow tokens, and computes start and end positions for answers. It also
    maps example IDs to unique integers and tokenizes the answers.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use for tokenizing the questions and contexts.
    max_length : int
        The maximum length of the tokenized input sequences.
    stride : int
        The stride to use when handling overflow tokens.
    train_example_ids_map : dict[str, int]
        A dictionary mapping example IDs to unique integers.

    Returns
    -------
    Callable[[dict], dict]
        A function that pre-processes a batch of training examples.

    Example
    -------
    >>> tokenizer = PreTrainedTokenizerBase.from_pretrained("bert-base-uncased")
    >>> preprocess_fn = get_preprocess_training_examples_fn(tokenizer, 384, 128, {})
    >>> examples = {
    ...     "question": ["What is the capital of France?"],
    ...     "context": ["Paris is the capital of France."],
    ...     "answers": [{"text": ["Paris"], "answer_start": [0]}],
    ...     "id": ["1"]
    ... }
    >>> inputs = preprocess_fn(examples)
    >>> print(inputs)
    """

    def preprocess_training_examples(examples: LazyBatch) -> BatchEncoding:
        """
        Preprocess a batch of training examples.

        This function tokenizes the questions and contexts, handles overflow tokens,
        computes start and end positions for answers, maps example IDs to unique
        integers, and tokenizes the answers.

        Parameters
        ----------
        examples : dict
            A dictionary containing the training examples with keys "question",
            "context", "answers", and "id".

        Returns
        -------
        dict
            A dictionary containing the tokenized inputs and additional information such
            as start and end positions, example IDs, and tokenized answers.
        """
        examples_question = examples["question"]
        assert examples_question is not None
        # Strip whitespace from questions
        questions = [str(q).strip() for q in examples_question]

        # Tokenize questions and contexts
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Extract offset mapping and sample map
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        input_examples_id = examples["id"]
        assert answers is not None
        assert input_examples_id is not None

        # Initialize lists for start/end positions, example IDs, and tokenized answers
        start_positions: list[list[int]] = []
        end_positions: list[list[int]] = []
        example_ids: list[list[int]] = []
        tokenized_answers: list[list[int]] = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            text_answer: str = answer["text"]

            # Tokenize the answer text
            tokenized_answer = tokenizer(
                text_answer[0], max_length=100, padding="max_length"
            )["input_ids"]
            tokenized_answers.append(tokenized_answer)  # type: ignore[reportArgumentType]

            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            example_id = input_examples_id[sample_idx]

            # Map example ID to a unique integer
            if example_id not in train_example_ids_map:
                train_example_ids_map[example_id] = len(train_example_ids_map)
            example_ids.append([train_example_ids_map[example_id]])

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
                if idx >= len(sequence_ids):
                    break
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append([0])
                end_positions.append([0])
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append([idx - 1])

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append([idx + 1])

        # Add computed positions, example IDs, and tokenized answers to inputs
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["example_ids"] = example_ids
        inputs["answers"] = tokenized_answers
        return inputs

    return preprocess_training_examples


def get_preprocess_validation_examples_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
    validation_example_ids_map: dict[str, int],
) -> Callable[[LazyBatch], BatchEncoding]:
    """
    Create a preprocessing function for eval examples for question answering tasks.

    This function returns a preprocessing function that tokenizes question and context,
    handles overflow tokens, and maps example IDs to unique integers. It also tokenizes
    the answers and adjusts the offset mapping to exclude non-context tokens.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use for tokenizing the questions and contexts.
    max_length : int
        The maximum length of the tokenized input sequences.
    stride : int
        The stride to use when handling overflow tokens.
    validation_example_ids_map : dict[str, int]
        A dictionary mapping example IDs to unique integers.

    Returns
    -------
    Callable[[dict], dict]
        A function that pre-processes a batch of validation examples.

    Example
    -------
    >>> tokenizer = PreTrainedTokenizerBase.from_pretrained("bert-base-uncased")
    >>> preprocess_fn = get_preprocess_validation_examples_fn(tokenizer, 384, 128, {})
    >>> examples = {
    ...     "question": ["What is the capital of France?"],
    ...     "context": ["Paris is the capital of France."],
    ...     "answers": [{"text": ["Paris"], "answer_start": [0]}],
    ...     "id": ["1"]
    ... }
    >>> inputs = preprocess_fn(examples)
    >>> print(inputs)
    """

    def preprocess_validation_examples(examples: LazyBatch) -> BatchEncoding:
        """
        Preprocess a batch of validation examples.

        This function tokenizes the questions and contexts, handles overflow tokens,
        maps example IDs to unique integers, tokenizes the answers, and adjusts the
        offset mapping to exclude non-context tokens.

        Parameters
        ----------
        examples : dict
            A dictionary containing the validation examples with keys "question",
            "context", "answers", and "id".

        Returns
        -------
        dict
            A dictionary containing the tokenized inputs and additional information such
            as example IDs and tokenized answers.
        """
        # Strip whitespace from questions
        examples_question = examples["question"]
        assert examples_question is not None
        questions = [str(q).strip() for q in examples_question]

        # Tokenize questions and contexts
        inputs: BatchEncoding = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Extract sample map and answers
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        input_examples_id = examples["id"]
        assert answers is not None
        assert input_examples_id is not None
        example_ids: list[list[int]] = []
        tokenized_answers: list[list[int]] = []

        for i in range(len(inputs["input_ids"])):  # type: ignore[reportArgumentType]
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            text_answer: str = answer["text"]

            # Tokenize the answer text
            tokenized_answer = tokenizer(
                text_answer[0], max_length=100, padding="max_length"
            )["input_ids"]
            tokenized_answers.append(tokenized_answer)  # type: ignore[reportArgumentType]

            # Map example ID to a unique integer
            example_id = input_examples_id[sample_idx]
            if example_id not in validation_example_ids_map:
                validation_example_ids_map[example_id] = len(validation_example_ids_map)
            example_ids.append([validation_example_ids_map[example_id]])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]  # type: ignore[reportIndexIssue]

            # Adjust offset mapping to exclude non-context tokens
            inputs["offset_mapping"][i] = [  # type: ignore[reportIndexIssue]
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        # Add example IDs and tokenized answers to inputs
        inputs["example_ids"] = example_ids
        inputs["answers"] = tokenized_answers

        # Remove offset mapping to avoid issues with PyTorch's DataLoader
        inputs.pop("offset_mapping")
        return inputs

    return preprocess_validation_examples


class ExactMatchMetric(Metric):
    """
    Exact Match Metric for Question Answering.

    This class implements the exact match metric for evaluating question answering
    models. It compares the predicted answers with the ground truth answers and
    computes the exact match accuracy.

    Attributes
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used to decode the predicted and ground truth answers.
    exact_match : torch.Tensor
        A tensor to store the count of exact matches.
    total : torch.Tensor
        A tensor to store the total number of examples.

    Methods
    -------
    __init__(tokenizer: PreTrainedTokenizerBase)
        Initialize the ExactMatchMetric with the given tokenizer.
    update(outputs: Mapping | torch.Tensor, labels: torch.Tensor)
        Update the state with the predicted and ground truth answers.
    compute() -> torch.Tensor
        Compute the exact match accuracy.

    Example
    -------
    >>> tokenizer = PreTrainedTokenizerBase.from_pretrained("bert-base-uncased")
    >>> metric = ExactMatchMetric(tokenizer)
    >>> outputs = (start_logits, end_logits, batch)
    >>> labels = torch.tensor([...])
    >>> metric.update(outputs, labels)
    >>> accuracy = metric.compute()
    >>> print(accuracy)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Initialize the ExactMatchMetric with the given tokenizer.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            The tokenizer used to decode the predicted and ground truth answers.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.add_state("exact_match", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Mapping | torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state with the predicted and ground truth answers.

        This function processes the model outputs and ground truth answers, decodes
        them, and updates the exact match count and total count.

        Parameters
        ----------
        outputs : Mapping | torch.Tensor
            The model outputs, including start logits, end logits, and the input batch.
        labels : torch.Tensor
            The ground truth labels (not used in this function).

        Raises
        ------
        AssertionError
            If the outputs or batch do not contain the expected keys.
        """
        # Verify outputs
        start_logits = outputs[0]
        end_logits = outputs[1]
        assert start_logits is not None and end_logits is not None
        batch = outputs[2]
        assert type(batch) is dict
        answers = batch.get("answers", None)
        assert answers is not None

        # Convert logits to predictions
        start_preds = torch.argmax(start_logits, dim=-1)
        end_preds = torch.argmax(end_logits, dim=-1)

        # Decode predictions and ground truth answers
        pred_texts = []
        for i in range(len(batch["input_ids"])):
            pred_ids = batch["input_ids"][i][start_preds[i] : end_preds[i] + 1]
            pred_texts.append(self.tokenizer.decode(pred_ids, skip_special_tokens=True))

        target_texts = [
            self.tokenizer.decode(answer, skip_special_tokens=True)
            for answer in answers
        ]

        # Compute exact matches
        exact_match_count = sum(
            1
            for pred, target in zip(pred_texts, target_texts, strict=True)
            if pred.strip() == target.strip()
        )

        # Update the state
        self.exact_match += exact_match_count
        self.total += len(target_texts)

    def compute(self) -> Any:
        """
        Compute the exact match accuracy.

        This function computes the exact match accuracy by dividing the exact match
        count by the total count.

        Returns
        -------
        torch.Tensor
            The exact match accuracy as a float tensor.
        """
        return self.exact_match.float() / self.total.float()


class F1ScoreMetric(Metric):
    """
    F1 Score Metric for Question Answering.

    This class implements the F1 score metric for evaluating question answering models.
    It compares the predicted answers with the ground truth answers and computes the F1
    score using the SQuAD evaluation metric.

    Attributes
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used to decode the predicted and ground truth answers.
    squad_metric : Metric
        The SQuAD metric used to compute the F1 score.
    f1_sum : torch.Tensor
        A tensor to store the sum of F1 scores.
    total_items : torch.Tensor
        A tensor to store the total number of examples.

    Methods
    -------
    __init__(tokenizer: PreTrainedTokenizerBase, dist_sync_on_step: bool = False)
        Initialize the F1ScoreMetric with the given tokenizer.
    update(outputs: Mapping | torch.Tensor, labels: torch.Tensor) -> None
        Update the state with the predicted and ground truth answers.
    compute() -> torch.Tensor
        Compute the average F1 score.

    Example
    -------
    >>> tokenizer = PreTrainedTokenizerBase.from_pretrained("bert-base-uncased")
    >>> metric = F1ScoreMetric(tokenizer)
    >>> outputs = (start_logits, end_logits, batch)
    >>> labels = torch.tensor([...])
    >>> metric.update(outputs, labels)
    >>> f1_score = metric.compute()
    >>> print(f1_score)
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, dist_sync_on_step: bool = False
    ) -> None:
        """
        Initialize the F1ScoreMetric with the given tokenizer.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            The tokenizer used to decode the predicted and ground truth answers.
        dist_sync_on_step : bool, optional
            Whether to synchronize the metric state across processes at each forward
            step (default is False).
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        # Load the SQuAD metric
        self.squad_metric: Metric = load_metric("squad")  # type: ignore[reportAttributeAccessIssue]
        self.add_state("f1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Mapping | torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state with the predicted and ground truth answers.

        This function processes the model outputs and ground truth answers, decodes
        them, and updates the F1 score sum and total count.

        Parameters
        ----------
        outputs : Mapping | torch.Tensor
            The model outputs, including start logits, end logits, and the input batch.
        labels : torch.Tensor
            The ground truth labels (not used in this function).

        Raises
        ------
        AssertionError
            If the outputs or batch do not contain the expected keys.
        """
        # Verify outputs
        start_logits = outputs[0]
        end_logits = outputs[1]
        assert start_logits is not None and end_logits is not None
        batch = outputs[2]
        assert type(batch) is dict
        input_ids = batch.get("input_ids", None)
        assert input_ids is not None
        answers = batch.get("answers", None)
        assert answers is not None
        example_ids = batch.get("example_ids", None)
        assert example_ids is not None

        # Convert logits to predictions
        start_preds = torch.argmax(start_logits, dim=-1)
        end_preds = torch.argmax(end_logits, dim=-1)

        # Decode predictions and ground truth answers
        pred_texts = []
        for i in range(len(input_ids)):
            pred_ids = input_ids[i][start_preds[i] : end_preds[i] + 1]
            pred_texts.append(self.tokenizer.decode(pred_ids, skip_special_tokens=True))

        target_texts = [
            self.tokenizer.decode(answer, skip_special_tokens=True)
            for answer in answers
        ]

        # Compute F1 scores
        f1_score_sum = 0
        for pred_text, target_text, example_id in zip(
            pred_texts, target_texts, example_ids, strict=True
        ):
            predictions_dict = {
                "id": example_id,
                "prediction_text": pred_text,
            }
            references_dict = {
                "id": example_id,
                "answers": {
                    "text": [target_text],
                },
            }
            f1_score_sum += self.squad_metric.compute(
                predictions=[predictions_dict], references=[references_dict]  # type: ignore[call-arg]
            )["f1"]

        # Update the state
        self.f1_sum += f1_score_sum
        self.total_items += len(target_texts)

    def compute(self) -> torch.Tensor:
        """
        Compute the average F1 score.

        This function computes the average F1 score by dividing the sum of F1 scores
        by the total count.

        Returns
        -------
        torch.Tensor
            The average F1 score as a float tensor.
        """
        return self.f1_sum / self.total_items.float()


def main() -> None:
    """
    Train and evaluate a question answering model using the SQuAD dataset.

    This function performs the following steps:
    1. Initializes the tokenizer.
    2. Loads and pre-processes the SQuAD dataset.
    3. Creates data loaders for training and evaluation.
    4. Initializes the model and loads pretrained weights.
    5. Sets up the optimizer and learning rate scheduler.
    6. Creates a Trainer object and performs evaluation and training.
    7. Logs the final metrics and saves the trained model.

    Example
    -------
    >>> main()
    """
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer_kwargs = {"model_max_length": 2048}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)
    # NOTE: We shouldn't add tokens but map them to known token with similar
    # functionality
    tokenizer.pad_token = "<|padding|>"
    # log(INFO, "Tokenizer loaded %s.", tokenizer)

    # Load and pre-process SQuAD dataset
    raw_datasets = datasets.load_dataset("squad")
    # max_length = 384
    max_length = 2048
    stride = 128
    train_example_ids_map: dict[str, int] = {}
    validation_example_ids_map: dict[str, int] = {}
    preprocess_training_examples_fn = get_preprocess_training_examples_fn(
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        train_example_ids_map=train_example_ids_map,
    )
    preprocess_validation_examples_fn = get_preprocess_validation_examples_fn(
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        validation_example_ids_map=validation_example_ids_map,
    )
    train_dataset = raw_datasets["train"].map(  # type: ignore[reportIndexIssue,reportAttributeAccessIssue]
        preprocess_training_examples_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,  # type: ignore[reportIndexIssue,reportAttributeAccessIssue]
    )
    validation_dataset = raw_datasets["validation"].map(  # type: ignore[reportIndexIssue,reportAttributeAccessIssue]
        preprocess_validation_examples_fn,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,  # type: ignore[reportIndexIssue,reportAttributeAccessIssue]
    )

    # Creating Dataloaders
    data_collator = transformers.data.data_collator.default_data_collator
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore[reportArgumentType]
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        validation_dataset,  # type: ignore[reportArgumentType]
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )

    # Instantiate metrics
    exact_match_metric = ExactMatchMetric(tokenizer=tokenizer)
    f1_score_metric = F1ScoreMetric(tokenizer=tokenizer)
    metrics = [exact_match_metric, f1_score_metric]
    # Get model config  - 125M
    model_config = {
        "name": "mpt_causal_lm",
        "init_device": "cpu",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "expansion_ratio": 4,
        "max_seq_len": 2048,
        # "max_seq_len": max_length,
        "vocab_size": 50368,
        "attn_config": {
            "attn_impl": "torch",  # "flash"
        },
        "output_hidden_states": True,
    }
    # # Get model config  - 3B
    # model_config = {
    #     "name": "mpt_causal_lm",
    #     "init_device": "cpu",
    #     "d_model": 2560,
    #     "n_heads": 20,
    #     "n_layers": 32,
    #     "expansion_ratio": 4,
    #     "max_seq_len": 2048,
    #     "vocab_size": 50368,
    #     "attn_config": {
    #         # "attn_impl": "torch"
    #         "attn_impl": "flash"
    #     },
    #     "output_hidden_states": True,
    # }
    # # Get model config  - 350M
    # model_config = {
    #     "name": "mpt_causal_lm",
    #     "init_device": "cpu",
    #     "d_model": 1024,
    #     "n_heads": 16,
    #     "n_layers": 24,
    #     "expansion_ratio": 4,
    #     "max_seq_len": 2048,
    #     "vocab_size": 50368,
    #     "attn_config": {
    #         "attn_impl": "torch",  # "flash"
    #     },
    #     "output_hidden_states": True,
    # }
    # Get model while forcing cpu to prevent any GPU allocation
    model = build_composer_model(
        name=model_config["name"],
        cfg=model_config,
        tokenizer=tokenizer,
        init_context=process_init_device(model_config, None),
        master_weights_dtype=None,
    )

    # Load the model from a checkpoint
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.0:9000"
    pretrained_model_path = "/path/to/pretrained/model"
    pretrained_model_path = (
        "/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/"
        "fed-3B-20240702_141112/server/25/current_server_parameters.npz"
    )
    pretrained_model_path = (
        "/nfs-share/ls985/projects/flower_llm /flower_llm_checkpoints/"
        "fed-350M-2024505_100605/server/19/current_server_parameters.npz"
    )
    pretrained_model_path = "s3://checkpoints/G1kgg-centB-125M-p-20240919/server/0/current_server_parameters.npz"
    pretrained_model_path = (
        "s3://checkpoints/GF-pers-125M-p-20240821_tle/ep1-ba1200-rank0.pt"
    )
    pretrained_model_path = (
        "s3://checkpoints/G1kgg-centB-125M-p-20240919/ep0-ba100-rank0.pt"
    )
    s3_comm_config = {
        "bucket_name": "checkpoints",
        "num_attempts": 3,
        "backend_kwargs": {
            "client_config": {
                "connect_timeout": 3600,
                "read_timeout": 3600,
            }
        },
    }

    # Wrap the original model with the Composer-ready fine-tuning model
    assert tokenizer.pad_token_id is not None
    composer_model = MPTForQuestionAnswering(
        model,  # type: ignore[reportArgumentType]
        train_metrics=metrics,  # type: ignore[reportArgumentType]
        eval_metrics=deepcopy(metrics),
        hidden_size=model_config["d_model"],  # type: ignore[arg-type]
        dropout_rate=0.1,
    )

    # Optimizers and Learning Rate Schedulers
    optimizer = AdamW(
        params=composer_model.parameters(),
        lr=3e-5,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=3e-6,
    )
    linear_lr_decay = LinearLR(
        # NOTE: The `total_iters` parameter should be set to the number of iterations
        # that the model is supposed to train for, e.g., if we train for 3 epochs on the
        # full dataset, then `total_iters` should be set to 3*(number of batches in the
        # full dataset).
        optimizer,
        start_factor=1.0,
        end_factor=0,
        total_iters=3000,
    )

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=Evaluator(
            label="eval/SQuAD",
            dataloader=eval_dataloader,
            metric_names=[],  # we will add these after model is created
            # NOTE: This is to enable automatic micro-batching so we can select the best
            # batch size for machine learning purposes and make sure we are nice to the
            # resources no matter what.
            device_eval_microbatch_size="auto",
        ),
        # NOTE: Set here the number of epochs
        max_duration="1ep",
        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        device="gpu" if torch.cuda.is_available() else "cpu",
        # NOTE: Setting to -1 means using all batches
        train_subset_num_batches=-1,
        eval_subset_num_batches=-1,
        # NOTE: This precision parameter is used to set the floating point precision of
        # the model weights. Note that Ampere GPUs or later can also use the brain
        # float16 (by setting `precision="amp_bf16"`) format, which is supposed to be
        # more efficient. A40s and H100s are Ampere or later, V100s are not!
        precision="amp_fp16",
        seed=17,
        load_path=pretrained_model_path if ".pt" in pretrained_model_path else None,
        load_weights_only=True,
        load_strict_model_weights=False,
        is_model_finetune=True,
        # NOTE: This is to enable automatic micro-batching so we can select the best
        # batch size for machine learning purposes and make sure we are nice to the
        # resources no matter what.
        device_train_microbatch_size="auto",
    )
    # Eval w/o training
    trainer.eval()

    if ".npz" in pretrained_model_path:
        load_pretrained_model_from_path(
            trainer=trainer,
            pretrained_model_path=pretrained_model_path,
            run_uuid=str(uuid.uuid4()),
            s3_comm_config=cast(S3CommConfig, DictConfig(s3_comm_config)),
        )
    # Start training
    trainer.fit()
    # Log the final metrics
    log(INFO, "Final training metrics: %s", trainer.state.train_metric_values)
    log(INFO, "Final validation metrics: %s", trainer.state.eval_metric_values)
    # NOTE: Save the final model locally. Please change the filename or comment this out
    # if you don't want to save the finetuned model.
    torch.save(trainer.state.model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
