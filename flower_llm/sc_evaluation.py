"""
Sequence Classification Model Evaluation.

This module provides functionality for training and evaluating sequence classification
models using datasets such as MultiNLI. It includes functions for tokenizing the
dataset, setting up data loaders, initializing the model, and computing evaluation
metrics such as accuracy and cross-entropy loss.

Imports
-------
- collections.abc.Callable : Abstract base class for callable types.
- transformers : Hugging Face library for transformer models.
- llmfoundry.utils.builders.build_tokenizer : Function to build a tokenizer.
- datasets : Library for loading and processing datasets.
- os : Module for interacting with the operating system.
- torch.utils.data.DataLoader : DataLoader class for creating data loaders.
- torchmetrics.classification.MulticlassAccuracy : Metric for computing multiclass
    accuracy.
- composer.metrics.CrossEntropy : Metric for computing cross-entropy loss.
- copy.deepcopy : Function to create deep copies of objects.
- llmfoundry.utils.builders.build_composer_model : Function to build a Composer model.
- llmfoundry.utils.config_utils.process_init_device : Function to process the
    initialization device.
- flower_llm.models.MPTForSequenceClassification : Model class for sequence
    classification.
- torch.optim.adamw.AdamW : AdamW optimizer.
- torch.optim.lr_scheduler.LinearLR : Linear learning rate scheduler.
- typing.cast : Function for type casting.
- uuid : Module for generating unique identifiers.
- torch : PyTorch library.
- omegaconf.DictConfig : Configuration class for OmegaConf.
- composer.Trainer : Trainer class from the Composer library.
- composer.Evaluator : Evaluator class from the Composer library.
- flower_llm.server.s3_utils.load_pretrained_model_from_path : Function to load a
    pretrained model from S3.
- flower_llm.conf.base_schema.S3CommConfig : Configuration class for S3 communication.
- transformers.PreTrainedTokenizerBase : Base class for pretrained tokenizers.
- transformers.tokenization_utils_base.BatchEncoding : Class for batch encoding.
- datasets.formatting.formatting.LazyBatch : Class for lazy batch formatting.
- logging.INFO : Logging level for informational messages.
- flwr.common.log : Logging function from Flower.

Functions
---------
- get_tokenization_fn(tokenizer: PreTrainedTokenizerBase, max_length: int) -> Callable[
    [LazyBatch], BatchEncoding
    ]
    Create a tokenization function for sequence classification tasks.
- main() -> None
    Main function to train and evaluate a sequence classification model using the
    MultiNLI dataset.

Dependencies
------------
- collections
- transformers
- llmfoundry
- datasets
- os
- torch
- torchmetrics
- composer
- copy
- typing
- uuid
- omegaconf
- flower_llm
- logging
- flwr
"""

from collections.abc import Callable
import transformers

from llmfoundry.utils.builders import (
    build_tokenizer,
)

import datasets
import os

from torch.utils.data import DataLoader

from torchmetrics.classification import MulticlassAccuracy

from composer.metrics import CrossEntropy
from copy import deepcopy

from llmfoundry.utils.builders import (
    build_composer_model,
)
from llmfoundry.utils.config_utils import (
    process_init_device,
)

from flower_llm.models import MPTForSequenceClassification

from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR

from typing import cast
import uuid
import torch
from omegaconf import DictConfig
from composer import Trainer, Evaluator
from flower_llm.server.s3_utils import load_pretrained_model_from_path
from flower_llm.conf.base_schema import S3CommConfig

from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from datasets.formatting.formatting import LazyBatch

from logging import INFO
from flwr.common import log


def get_tokenization_fn(
    tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Callable[[LazyBatch], BatchEncoding]:
    """
    Create a tokenization function for sequence classification tasks.

    This function returns a tokenization function that tokenizes pairs of sentences
    (premise and hypothesis) for sequence classification tasks. The tokenization
    function uses the provided tokenizer and ensures that the tokenized sequences
    are padded and truncated to the specified maximum length. It also includes the
    labels in the tokenized output.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use for tokenizing the sentences.
    max_length : int
        The maximum length of the tokenized input sequences.

    Returns
    -------
    Callable[[LazyBatch], BatchEncoding]
        A function that tokenizes a batch of input samples.

    Example
    -------
    >>> tokenizer = PreTrainedTokenizerBase.from_pretrained("bert-base-uncased")
    >>> tokenization_fn = get_tokenization_fn(tokenizer, max_length=128)
    >>> sample = {
    ...     "premise": "The cat sat on the mat.",
    ...     "hypothesis": "The cat is sitting on the mat.",
    ...     "label": 1
    ... }
    >>> tokenized_sample = tokenization_fn(sample)
    >>> print(tokenized_sample)
    """

    def _tokenize_function(sample: LazyBatch) -> BatchEncoding:
        """Tokenize a sentence."""
        sample_premise = sample["premise"]
        samples_hypothesis = sample["hypothesis"]
        samples_labels = sample["label"]
        assert sample_premise is not None
        assert samples_hypothesis is not None
        inputs = tokenizer(
            sample_premise,
            samples_hypothesis,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        inputs["labels"] = samples_labels
        return inputs

    return _tokenize_function


def main() -> None:
    """
    Train and evaluate a sequence classification model using the MultiNLI dataset.

    This function performs the following steps:
    1. Initializes the tokenizer.
    2. Loads and preprocesses the MultiNLI dataset.
    3. Creates data loaders for training and evaluation.
    4. Initializes the model and loads pretrained weights.
    5. Sets up the optimizer and learning rate scheduler.
    6. Creates a Trainer object and performs evaluation and training.
    7. Logs the final metrics and saves the trained model.

    Steps
    -----
    1. Initialize the tokenizer with the specified model and configuration.
    2. Load the MultiNLI dataset and preprocess it using the tokenizer.
    3. Split the dataset into training and validation sets and create data loaders.
    4. Define the model configuration and build the model.
    5. Load pretrained weights from a specified checkpoint.
    6. Set up the optimizer and learning rate scheduler.
    7. Create a Trainer object with the model, data loaders, optimizer, and other
        settings.
    8. Perform evaluation without training.
    9. Start the training process.
    10. Perform evaluation after training.
    11. Log the final training and validation metrics.
    12. Save the trained model to a local file.

    Example
    -------
    >>> main()
    """
    # Create the tokenizer
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer_kwargs = {"model_max_length": 2048}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)
    # NOTE: We shouldn't add tokens but map them to known token with similar
    # functionality
    tokenizer.pad_token = "<|padding|>"
    # log(INFO, "Tokenizer loaded %s.", tokenizer)

    # Load and pre-process MultiNLI dataset

    # Get tokenization function
    tokenization_function = get_tokenization_fn(
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
    )

    # Tokenize MultiNLI
    multi_nli = datasets.load_dataset("nyu-mll/multi_nli", num_proc=os.cpu_count() - 1)  # type: ignore[reportOptionalOperand, operator]
    tokenized_multi_nli = multi_nli.map(
        tokenization_function,
        batched=True,
        batch_size=100,
        remove_columns=[
            "promptID",
            "pairID",
            "premise_binary_parse",
            "premise_parse",
            "hypothesis_binary_parse",
            "hypothesis_parse",
            "genre",
        ],
    )
    # Split dataset into train and validation sets
    train_dataset = tokenized_multi_nli["train"]  # type: ignore[reportIndexIssue]
    eval_matched_dataset = tokenized_multi_nli["validation_matched"]  # type: ignore[reportIndexIssue]
    eval_mismatched_dataset = tokenized_multi_nli["validation_mismatched"]  # type: ignore[reportIndexIssue]

    # Create DataLoaders
    data_collator = transformers.data.data_collator.default_data_collator
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore[reportArgumentType]
        batch_size=128,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )
    eval_matched_dataloader = DataLoader(
        eval_matched_dataset,  # type: ignore[reportArgumentType]
        batch_size=128,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )
    eval_mismatched_dataloader = DataLoader(
        eval_mismatched_dataset,  # type: ignore[reportArgumentType]
        batch_size=128,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )

    # Instantiate metrics
    metrics = [CrossEntropy(), MulticlassAccuracy(num_classes=3, average="micro")]
    # Get model config  - 125M
    model_config = {
        "name": "mpt_causal_lm",
        "init_device": "cpu",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "expansion_ratio": 4,
        "max_seq_len": 2048,
        "vocab_size": 50368,
        "attn_config": {
            # NOTE: Only Ampere GPUs or later generations support the `flash`
            # implementation. A40s and H100s are Ampere or later, V100s are not!
            "attn_impl": "torch",
            # "attn_impl": "flash",
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
    #         "attn_impl": "torch",  # "flash"
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
    # Package as a trainer-friendly Composer model
    assert tokenizer.pad_token_id is not None
    composer_model = MPTForSequenceClassification(
        model,  # type: ignore[reportArgumentType]
        train_metrics=metrics,
        eval_metrics=deepcopy(metrics),
        num_labels=3,
        hidden_size=model_config["d_model"],  # type: ignore[arg-type]
        pad_token_id=tokenizer.pad_token_id,
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
        # full dataset).§
        optimizer,
        start_factor=1.0,
        end_factor=0,
        total_iters=100,
    )

    # Composer Trainer
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.0:9000"
    pretrained_model_path = "/path/to/pretrained/model"
    pretrained_model_path = (
        "/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/"
        "fed-3B-20240702_141112/server/25/current_server_parameters.npz"
    )
    pretrained_model_path = (
        "/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/"
        "fed-350M-2024505_100605/server/19/current_server_parameters.npz"
    )
    pretrained_model_path = "s3://checkpoints/G1kgg-centB-125M-p-20240919/server/0/current_server_parameters.npz"
    pretrained_model_path = (
        "s3://checkpoints/G1kgg-centB-125M-p-20240919/ep0-ba100-rank0.pt"
    )
    pretrained_model_path = (
        "s3://checkpoints/GF-pers-125M-p-20240821_tle/ep1-ba1200-rank0.pt"
    )

    # Create Trainer Object
    assert composer_model.val_metrics is not None
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=[
            Evaluator(
                label="eval/MultiNLI_matched",
                dataloader=eval_matched_dataloader,
                metric_names=list(composer_model.val_metrics.keys()),
                # NOTE: This is to enable automatic micro-batching so we can select the
                # best batch size for machine learning purposes and make sure we are
                # nice to the resources no matter what.
                device_eval_microbatch_size="auto",
            ),
            Evaluator(
                label="eval/MultiNLI_mismatched",
                dataloader=eval_mismatched_dataloader,
                metric_names=list(composer_model.val_metrics.keys()),
                # NOTE: This is to enable automatic micro-batching so we can select the
                # best batch size for machine learning purposes and make sure we are
                # nice to the resources no matter what.
                device_eval_microbatch_size="auto",
            ),
        ],
        # NOTE: Set here the number of epochs
        max_duration="5ep",
        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        device="gpu" if torch.cuda.is_available() else "cpu",
        # NOTE: Setting to -1 means using all batches
        train_subset_num_batches=100,
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

    if ".npz" in pretrained_model_path:
        load_pretrained_model_from_path(
            trainer=trainer,
            pretrained_model_path=pretrained_model_path,
            run_uuid=str(uuid.uuid4()),
            s3_comm_config=cast(S3CommConfig, DictConfig(s3_comm_config)),
        )
    # Eval w/o training
    trainer.eval()
    # Start training
    trainer.fit()
    # Eval w/o training
    trainer.eval()
    # Log the final metrics
    log(INFO, "Final training metrics: %s", trainer.state.train_metric_values)
    log(INFO, "Final validation metrics: %s", trainer.state.eval_metric_values)
    # NOTE: Save the final model locally. Please change the filename or comment this out
    # if you don't want to save the finetuned model.
    torch.save(trainer.state.model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
