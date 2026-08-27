"""Image Dataset.
Samples as numpy memory mapped files.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from ..types import PathOrStr


class SafeTensorsDataset(VisionDataset):
    """Image Dataset from samples"""

    classes: tuple[str, ...] | None  # as torchvision Datasets examples
    class_to_idx: dict[str, int] | None  # as torchvision Datasets examples

    def __init__(
        self,
        *,
        root: PathOrStr | None = None,  # only for compatibility
        samples: Tensor,
        targets: Tensor,
        classes: tuple[str, ...] | None = None,
        class_to_idx: dict[str, int] | None = None,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Image Dataset from samples - samples as tuple - filename and target"""
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.samples = samples
        self.targets = targets
        assert len(self.samples) == len(self.targets)
        self.classes = classes
        self.class_to_idx = class_to_idx
        if transform is None:
            self.transform = lambda x: x
        if target_transform is None:
            self.target_transform = lambda x: x
        self._num_samples = len(self.samples)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.transform(self.samples[index]),
            self.target_transform(self.targets[index]),
        )
