"""Persistent Image Dataset.
"""
from typing import Callable, Optional

import pandas as pd
import torch

from ..typing import PathOrStr

# from .get_files import get_files
from .imagedataset import ImageDataset, samples_from_df


class DatasetPersistent(ImageDataset):
    """Image Dataset from samples, persistent."""

    def __init__(
        self,
        root: PathOrStr,
        samples: tuple[tuple[str, int], ...],
        indexes: Optional[list[list[int]]] = None,
        epochs: Optional[int] = None,
        classes: Optional[tuple[str, ...]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        classes_as_imagenet: bool = False,
    ):
        """Dataset with persistent sampler."""
        super().__init__(
            root=root,
            samples=samples,
            classes=classes,
            class_to_idx=class_to_idx,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            classes_as_imagenet=classes_as_imagenet,
        )
        if indexes is None:
            self.indexes = [list(range(len(self.samples)))]
        else:
            self.indexes = indexes
        self.epochs = epochs or len(self.indexes)
        self.epoch = 0

    def step_epoch(self) -> None:
        self.epoch += 1
        if self.epoch == self.epochs:
            self.epoch = 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # if index == 0:
        #     self.step_epoch()
        sample_index = self.indexes[self.epoch][index]
        return (
            self.transform(self.loader(self.samples[sample_index][0])),
            self.samples[sample_index][1],
        )

    @classmethod
    def from_folder(cls, root: PathOrStr, **kwargs) -> "DatasetPersistent":
        return cls(root, **kwargs)


def persistent_dataset_from_df(
    root: PathOrStr,
    df: pd.DataFrame,
    indexes: Optional[list[list[int]]] = None,
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
        classes_as_imagenet=classes_as_imagenet,
        classes=class_to_idx.keys(),
        class_to_idx=class_to_idx,
        epochs=epochs,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
    )
