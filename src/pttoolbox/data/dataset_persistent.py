"""Persistent Image Dataset.
"""
from typing import Callable, Optional

import pandas as pd
import torch

from ..typing import PathOrStr

# from .get_files import get_files
from .imagedataset import ImageDataset, samples_from_df
from .transforms import TrainPersistentTransform


class DatasetPersistent(ImageDataset):
    """Image Dataset from samples, persistent."""

    transform_args: Optional[list[tuple[str, ...]]] = None

    def __init__(
        self,
        root: PathOrStr,
        samples: tuple[tuple[str, int], ...],
        indexes: Optional[list[list[int]]] = None,
        transform_indexes: Optional[list[list[int]]] = None,
        epochs: Optional[int] = None,
        classes: Optional[tuple[str, ...]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable[[torch.Tensor, int, int, int], torch.Tensor]] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        classes_as_imagenet: bool = False,
    ):
        """Dataset with persistent sampler."""
        if transform is None:
            transform = TrainPersistentTransform()
        super().__init__(
            root=root,
            samples=samples,
            classes=classes,
            class_to_idx=class_to_idx,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            # classes_as_imagenet=classes_as_imagenet,
        )
        if indexes is None:
            self.indexes = [list(range(len(self.samples)))]
        else:
            self.indexes = indexes
        self.transform_indexes = transform_indexes
        if self.transform_indexes is not None:
            self.create_transform_args(0)
        self.epochs = epochs or len(self.indexes)
        self.epoch = 0

    def create_transform_args(self, epoch: int) -> None:
        epoch_index_list = [item for item in self.transform_indexes]  # permute it by epochs
        num_args = len(epoch_index_list)
        self.transform_args = [
            
            (epoch_index_list[j][i] for j in range(num_args)) for i in range(self._num_samples)
        ]

    def step_epoch(self) -> None:
        self.epoch += 1
        if self.epoch == self.epochs:
            self.epoch = 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample_index = self.indexes[self.epoch][index]
        if self.transform_args is not None:
            sample_transforms = self.transform_args[index]
        else:
            sample_transforms = (0, 16, 16)
        return (
            self.transform(self.loader(self.samples[sample_index][0]), *sample_transforms),
            self.samples[sample_index][1],
        )

    @classmethod
    def from_folder(cls, root: PathOrStr, **kwargs) -> "DatasetPersistent":
        return cls(root, **kwargs)


def persistent_dataset_from_df(
    root: PathOrStr,
    df: pd.DataFrame,
    indexes: Optional[list[list[int]]] = None,
    transform_indexes: Optional[list[list[int]]] = None,
    epochs: Optional[int] = None,
    num_samples: int = 0,
    classes_as_imagenet: bool = False,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Optional[Callable] = None,
) -> "DatasetPersistent":
    samples, class_to_idx = samples_from_df(
        df, num_samples=num_samples, classes_as_imagenet=classes_as_imagenet
    )
    return DatasetPersistent(
        root=root,
        samples=samples,
        indexes=indexes,
        transform_indexes=transform_indexes,
        classes_as_imagenet=classes_as_imagenet,
        classes=class_to_idx.keys(),
        class_to_idx=class_to_idx,
        epochs=epochs,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
    )
