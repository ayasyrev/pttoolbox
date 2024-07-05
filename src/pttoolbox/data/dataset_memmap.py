""" Image Dataset.
Samples as numpy memory mapped files.
"""

from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset

from ..typing import PathOrStr


class MemmapDataset(VisionDataset):
    """Image Dataset from samples"""

    classes: Optional[tuple[str, ...]]  # as torchvision Datasets examples
    class_to_idx: Optional[dict[str, int]]  # as torchvision Datasets examples

    def __init__(
        self,
        *,
        root: Optional[PathOrStr] = None,  # only for compatibility
        samples: np.array,
        targets: np.array,
        classes: Optional[tuple[str, ...]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
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
            # self.transform = torch.tensor
            self.transform = lambda x: x
        if target_transform is None:
            # self.target_transform = torch.tensor
            self.target_transform = lambda x: x
        self._num_samples = len(self.samples)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.transform(torch.tensor(self.samples[index])),
            self.target_transform(self.targets[index]),
        )
