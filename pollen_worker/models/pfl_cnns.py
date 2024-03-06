"""CNN models for PFL training on multi-label and single-label classification tasks.

The script has been slightly modified from: TODO: add link to original script
"""

# Copyright © 2023-2024 Apple Inc.
import torch
from torch import nn
import torchvision.models
from torchvision import transforms
from pollen_worker.models.layers import Transpose2D

from pollen_worker.models.module_modifications import (
    convert_batchnorm_modules,
    freeze_batchnorm_modules,
    validate_no_batchnorm,
)

torchvision_models = torchvision.models.__dict__


class MultiLabelCNN(nn.Module):
    """
    Wrapper of torchvision.models used for PFL training.

    The task is multi-label classification, e.g. on FLAIR dataset.
    """

    def __init__(
        self,
        torchvision_model_type: str,
        num_outputs: int,
        channel_mean: list[float],
        channel_stddevs: list[float],
        pretrained: bool,
    ) -> None:
        super().__init__()
        self._num_outputs = num_outputs

        # input image transformation, same as standard ImageNet training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(channel_mean, channel_stddevs),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(channel_mean, channel_stddevs),
        ])

        # per-class binary cross-entropy for multi-label classification
        # learning objective
        self.loss_fct = nn.BCEWithLogitsLoss()

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        base_model = torchvision_models[torchvision_model_type](
            num_classes=self._num_outputs
        )
        if pretrained:
            pretrained_model = torchvision_models[torchvision_model_type](
                pretrained=True
            )
            pretrained_state = pretrained_model.state_dict()
            # Pretrained models typically use Batch Normalization. Since we do
            # not want to collect channel statistics in private learning, we
            # freeze the trained statistics in all batch norm modules, i.e.,
            # the statistics will be from pretrained dataset (ImageNet) instead
            # of private data on device.
            base_model = freeze_batchnorm_modules(base_model)
            base_state = base_model.state_dict()
            state_to_load = {}
            # skip loading the final classifier layer's weight and bias
            for k, v in list(pretrained_state.items())[:-2]:
                assert k in base_state and v.size() == base_state[k].size()
                state_to_load[k] = v
            base_model.load_state_dict(state_to_load, strict=False)
            self.base_model = base_model
        else:
            # convert all batch norm module to group norm if not using
            # pretrained models
            self.base_model = convert_batchnorm_modules(base_model)

        # assert there is no batch norm module in current model
        validate_no_batchnorm(self)

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        """Execute internal transform."""
        images = (images.float() / 255.0).permute(0, 3, 1, 2)
        if self.training:
            return self.train_transform(images)
        else:
            return self.eval_transform(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.transform(images)
        return self.base_model(x)


def simple_cnn(input_shape: tuple[int, ...], num_outputs: int) -> nn.Module:
    """Return a simple CNN with 2 convolutional layers and one dense hidden layer.

    :param input_shape:
        The shape of the input images, e.g. (32,32,3).
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A PyTorch CNN model.
    """
    in_channels = input_shape[-1]
    maxpool_output_size = (input_shape[0] - 4) // 2
    flatten_size = maxpool_output_size * maxpool_output_size * 64

    model = nn.Sequential(*[
        Transpose2D(),
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(flatten_size, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_outputs),
    ])

    # Apply Glorot (Xavier) uniform initialization to match TF2 model.
    for m in model.modules():
        if isinstance(m, nn.Conv2d | nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    return model
