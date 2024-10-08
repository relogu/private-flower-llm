"""
Model Definitions for Sequence Classification and Question Answering.

This module provides implementations of sequence classification and question answering
models using the MPT (Multi-Document Processing Transformer) architecture. It extends
the HuggingFaceModel class and provides functionality for training and evaluating these
tasks.

Classes
-------
- MPTForSequenceClassification
    A sequence classification model using the MPT architecture.
- MPTForQuestionAnswering
    A question answering model using the MPT architecture.

Functions
---------
- forward(
    batch: MutableMapping
) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast
    Perform a forward pass on the input batch and return the classification output.
- eval_forward(batch: MutableMapping, outputs: Any | None = None) -> torch.Tensor
    Perform a forward pass during evaluation and return the logits.

Dependencies
------------
- logging
- typing
- collections.abc
- composer.models
- llmfoundry
- torch
- torch.utils.checkpoint
- torch.nn
- torchmetrics
- transformers.modeling_outputs
"""

import logging
from typing import Any, cast
from collections.abc import MutableMapping

from composer.models import HuggingFaceModel
from llmfoundry import ComposerMPTCausalLM, MPTForCausalLM
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchmetrics import Metric
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    QuestionAnsweringModelOutput,
    TokenClassifierOutput,
)

logger = logging.getLogger(__name__)


class MPTForSequenceClassification(HuggingFaceModel):
    """
    MPT Model for Sequence Classification.

    This class implements a sequence classification model using the MPT (Multi-Document
    Processing Transformer) architecture. It extends the HuggingFaceModel class and
    provides functionality for training and evaluating sequence classification tasks.

    Attributes
    ----------
    num_labels : int
        The number of labels for classification.
    problem_type : str | None
        The type of classification problem (e.g., "regression",
        "single_label_classification", "multi_label_classification").
    pad_token_id : int
        The ID of the padding token.
    transformer : ComposerMPTCausalLM
        The transformer model used for sequence classification.
    score : nn.Linear
        A linear layer to compute the classification scores.

    Methods
    -------
    __init__(
        model,
        num_labels,
        hidden_size,
        pad_token_id,
        problem_type=None,
        train_metrics=None,
        eval_metrics=None
    )
        Initialize the MPTForSequenceClassification model.
    forward(batch) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast
        Perform a forward pass on the input batch and return the classification output.
    eval_forward(batch, outputs=None)
        Perform a forward pass on the input batch during evaluation and return the
        logits.

    Parameters
    ----------
    model : ComposerMPTCausalLM
        The transformer model to be used for sequence classification.
    num_labels : int
        The number of labels for classification.
    hidden_size : int
        The size of the hidden layer in the transformer model.
    pad_token_id : int
        The ID of the padding token.
    problem_type : str | None, optional
        The type of classification problem (default is None).
    train_metrics : list[Metric] | None, optional
        A list of metrics to be used during training (default is None).
    eval_metrics : list[Metric] | None, optional
        A list of metrics to be used during evaluation (default is None).

    Example
    -------
    >>> model = ComposerMPTCausalLM(...)
    >>> classifier = MPTForSequenceClassification(
    ...     model=model,
    ...     num_labels=2,
    ...     hidden_size=768,
    ...     pad_token_id=0,
    ...     problem_type="single_label_classification",
    ...     train_metrics=[...],
    ...     eval_metrics=[...]
    ... )
    >>> batch = {"input_ids": ..., "labels": ...}
    >>> output = classifier.forward(batch)
    >>> print(output)
    """

    def __init__(
        self,
        model: ComposerMPTCausalLM,
        num_labels: int,
        hidden_size: int,
        pad_token_id: int,
        problem_type: str | None = None,
        train_metrics: list[Metric] | None = None,
        eval_metrics: list[Metric] | None = None,
    ) -> None:
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.pad_token_id = pad_token_id

        super().__init__(
            model=model.model,
            tokenizer=model.tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=cast(MPTForCausalLM, model.model).transformer.shift_labels,
            allow_embedding_resizing=True,
        )
        self.transformer = model
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(
        self,
        batch: MutableMapping,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast:
        """
        Perform a forward pass on the input batch and return the classification output.

        This function processes a batch of input data, performs a forward pass using the
        transformer model, and returns the classification output. It supports different
        problem types, including regression, single-label classification, and multi-
        label classification. The function also handles padding tokens and computes the
        loss if labels are provided.

        Parameters
        ----------
        batch : MutableMapping
            A dictionary containing the input data for the model. It must include the
            following keys:
            - "input_ids" (torch.Tensor): The input token IDs.
            - "inputs_embeds" (torch.Tensor): The input embeddings.
            - "labels" (torch.Tensor): The labels for the input data.
            - "return_dict" (bool): Whether to return a dictionary of outputs.

        Returns
        -------
        tuple[torch.Tensor] | SequenceClassifierOutputWithPast
            The classification output. If `return_dict` is True, a
            SequenceClassifierOutputWithPast object is returned. Otherwise, a tuple
            containing the pooled logits and additional outputs is returned. If labels
            are provided, the loss is included in the output.

        Raises
        ------
        ValueError
            If the batch size is greater than 1 and no padding token is defined.

        Example
        -------
        >>> batch = {
        ...     "input_ids": torch.tensor([[101, 102, 103], [104, 105, 106]]),
        ...     "inputs_embeds": None,
        ...     "labels": torch.tensor([1, 0]),
        ...     "return_dict": True
        ... }
        >>> model = MPTForSequenceClassification(...)
        >>> output = model.forward(batch)
        >>> print(output)
        """
        input_ids = batch.get("input_ids", None)
        inputs_embeds = batch.get("inputs_embeds", None)
        labels = batch.pop("labels", None)
        return_dict = batch.get("return_dict", None)
        output_hidden_states = self.transformer.model.config.output_hidden_states  # type: ignore[reportAttributeAccessIssue]

        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )
        transformer_outputs: CausalLMOutputWithPast = self.transformer.model(
            **batch,
            output_hidden_states=output_hidden_states,
        )
        assert transformer_outputs is not None
        assert transformer_outputs.hidden_states is not None

        hidden_states = transformer_outputs.hidden_states[-1]
        logits = self.score(hidden_states)

        batch_size: int | None = None
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        assert batch_size is not None

        if self.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            # If no pad token has been found, use modulo instead of reverse indexing for
            # ONNX compatibility
            sequence_lengths = (
                torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths %= input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in {torch.long, torch.int}):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits, labels)
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def eval_forward(self, batch, outputs: Any | None = None):  # noqa: ANN201, ANN001
        """
        Perform a forward pass during evaluation and return the logits.

        This function processes a batch of input data, performs a forward pass using the
        model, and returns the logits for evaluation. It ensures that the "labels"
        component is removed from the batch to avoid computing loss during evaluation.

        Parameters
        ----------
        batch : dict
            A dictionary containing the input data for the model. It must include the
            "labels" key, which will be removed before the forward pass.
        outputs : Any | None, optional
            An optional parameter for additional outputs. Default is None.

        Returns
        -------
        torch.Tensor
            The logits obtained from the forward pass. If the model output is of type
            SequenceClassifierOutputWithPast, the logits are extracted from the output.
            Otherwise, the appropriate tensor is selected based on the output shape.

        Raises
        ------
        AssertionError
            If the "labels" key is not present in the batch.

        Example
        -------
        >>> batch = {
        ...     "input_ids": ...,
        ...     "attention_mask": ...,
        ...     "labels": ...
        ... }
        >>> model = MPTForSequenceClassification(...)
        >>> logits = model.eval_forward(batch)
        >>> print(logits)
        """
        # Pop "labels" component first to avoid computing loss
        self.labels = batch.pop("labels", None)
        assert self.labels is not None
        output = self.forward(batch)

        if type(output) is SequenceClassifierOutputWithPast:
            output = output.logits
        else:
            assert len(output) > 1
            output = output[1] if len(output[0].shape) == 0 else output[0]  # type: ignore[misc]
        # If we are in the single class case, then remove the classes dimension
        if output.ndim == 2 and output.shape[1] == 1:  # noqa: PLR2004
            output = output.squeeze(dim=1)

        return output


class MPTForTokenClassification(HuggingFaceModel):  # noqa: D101

    def __init__(
        self,
        model: ComposerMPTCausalLM,
        num_labels: int,
        hidden_size: int,
        classifier_dropout: float | None = None,
        hidden_dropout: float | None = None,
        train_metrics: list[Metric] | None = None,
        eval_metrics: list[Metric] | None = None,
    ) -> None:
        self.num_labels = num_labels

        super().__init__(
            model=model.model,
            tokenizer=model.tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=cast(MPTForCausalLM, model.model).transformer.shift_labels,
            allow_embedding_resizing=True,
        )

        self.transformer = model
        if classifier_dropout is not None:
            dropout = classifier_dropout
        elif hidden_dropout is not None:
            dropout = hidden_dropout
        else:
            dropout = 0.1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(  # noqa: D102
        self,
        batch: MutableMapping,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:

        labels = batch.pop("labels", None)
        output_hidden_states = batch.get("output_hidden_states", None)
        return_dict = batch.get("return_dict", None)

        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )

        transformer_outputs: CausalLMOutputWithPast = self.transformer.model(
            **batch,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels),
                labels.view(batch_size * seq_length),
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class MPTForQuestionAnswering(HuggingFaceModel):
    """
    MPT Model for Question Answering.

    This class implements a question answering model using the MPT (Multi-Document
    Processing Transformer) architecture. It extends the HuggingFaceModel class and
    provides functionality for training and evaluating question answering tasks.

    Attributes
    ----------
    transformer : ComposerMPTCausalLM
        The transformer model used for question answering.
    qa_outputs : nn.Linear
        A linear layer to compute the start and end logits for the answer span.

    Methods
    -------
    __init__(model, hidden_size, train_metrics=None, eval_metrics=None)
        Initialize the MPTForQuestionAnswering model.
    forward(batch) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput
        Perform a forward pass on the input batch and return the question answering
        output.

    Parameters
    ----------
    model : ComposerMPTCausalLM
        The transformer model to be used for question answering.
    hidden_size : int
        The size of the hidden layer in the transformer model.
    train_metrics : list[Metric] | None, optional
        A list of metrics to be used during training (default is None).
    eval_metrics : list[Metric] | None, optional
        A list of metrics to be used during evaluation (default is None).

    Example
    -------
    >>> model = ComposerMPTCausalLM(...)
    >>> qa_model = MPTForQuestionAnswering(
    ...     model=model,
    ...     hidden_size=768,
    ...     train_metrics=[...],
    ...     eval_metrics=[...]
    ... )
    >>> batch = {
    ...     "input_ids": ...,
    ...     "attention_mask": ...,
    ...     "start_positions": ...,
    ...     "end_positions": ...
    ... }
    >>> output = qa_model.forward(batch)
    >>> print(output)
    """

    def __init__(
        self,
        model: ComposerMPTCausalLM,
        hidden_size: int,
        train_metrics: list[Metric] | None = None,
        eval_metrics: list[Metric] | None = None,
    ) -> None:
        self.transformer = model
        self.qa_outputs = nn.Linear(hidden_size, 2)

        super().__init__(
            model=model.model,
            tokenizer=model.tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=cast(MPTForCausalLM, model.model).transformer.shift_labels,
            allow_embedding_resizing=True,
        )

    def forward(
        self,
        batch: MutableMapping,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput:
        """
        Perform a forward pass on the input batch, return the question answering output.

        This function processes a batch of input data, performs a forward pass using the
        transformer model, and returns the start and end logits for the answer span. It
        supports different return types based on the `return_dict` parameter and
        computes the loss if start and end positions are provided.

        Parameters
        ----------
        batch : MutableMapping
            A dictionary containing the input data for the model. It must include the
            following keys:
            - "input_ids" (torch.Tensor): The input token IDs.
            - "inputs_embeds" (torch.Tensor): The input embeddings.
            - "attention_mask" (torch.Tensor): The attention mask for the input data.
            - "start_positions" (torch.Tensor): The start positions of the answer spans.
            - "end_positions" (torch.Tensor): The end positions of the answer spans.
            - "return_dict" (bool): Whether to return a dictionary of outputs.

        Returns
        -------
        tuple[torch.Tensor] | QuestionAnsweringModelOutput
            The question answering output. If `return_dict` is True, a
            QuestionAnsweringModelOutput object is returned. Otherwise, a tuple
            containing the start logits, end logits, and additional outputs is returned.
            If start and end positions are provided, the loss is included in the output.

        Example
        -------
        >>> batch = {
        ...     "input_ids": torch.tensor([[101, 102, 103], [104, 105, 106]]),
        ...     "inputs_embeds": None,
        ...     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        ...     "start_positions": torch.tensor([1, 0]),
        ...     "end_positions": torch.tensor([2, 1]),
        ...     "return_dict": True
        ... }
        >>> model = MPTForQuestionAnswering(...)
        >>> output = model.forward(batch)
        >>> print(output)
        """
        _labels = batch.pop("labels", None)
        start_positions = batch.get("start_positions", None)
        end_positions = batch.get("end_positions", None)
        return_dict = batch.get("return_dict", None)
        output_hidden_states = self.transformer.model.config.output_hidden_states  # type: ignore[reportAttributeAccessIssue]

        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )

        transformer_outputs: CausalLMOutputWithPast = self.transformer.model(
            **batch,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = transformer_outputs.hidden_states

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # Sometimes the start/end positions are outside our model inputs, we ignore
            # these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
