"""Allows comparing between models with different vocabularies."""

from collections.abc import Mapping

import torch
from torch import Tensor
from torchmetrics import Metric


class UnigramNormalizedLanguageCrossEntropy(Metric):
    """Torchmetric that computes cross entropy on language modeling outputs.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        lookup_probabilities: Tensor,
        dist_sync_on_step: bool = False,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.lookup_probabilities = lookup_probabilities
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="sum"
        )
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        """Update the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type
                    that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """
        if isinstance(output, Mapping):
            logits = output["logits"]
        elif isinstance(output, Tensor):
            logits = output
        else:
            raise TypeError(f"Type {type(output)} for the output is unsupported.")

        target = target.view(-1)
        logits = logits.view(target.shape[0], -1)

        losses = self.loss_fn(logits, target)

        probabilities = self.lookup_probabilities.to(target.device)[target].unsqueeze(
            2
        )  # [B,S,1]

        unigram_cross_entropy = -torch.log(probabilities).squeeze(2)  # [B,S]

        total_items = (target != self.ignore_index).sum()
        self.total_items += total_items

        # Mask out the ignored indices
        mask = (target != self.ignore_index).float()
        unigram_cross_entropy = (unigram_cross_entropy * mask).sum()

        # accumulate loss over all batches
        self.sum_loss += losses - unigram_cross_entropy

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns
        -------
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items


class UnigramNormalizedLanguagePerplexity(UnigramNormalizedLanguageCrossEntropy):
    """Implements unigram-normalized perplexity."""

    def compute(self) -> Tensor:
        """Return torch.exp() of the UnigramNormalizedLanguageCrossEntropy."""
        avg_loss = super().compute()
        return torch.exp(avg_loss)
