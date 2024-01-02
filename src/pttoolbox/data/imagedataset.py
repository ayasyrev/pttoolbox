""" Image Dataset.
As ImageFolderDataset -> base use from given samples.
Use classes from imagenet, samples from dataframe.
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

from ..typing import PathOrStr
from .get_files import get_files
from .image_loader import accimage_loader, pil_loader
from .imagenet1k_classes import SYNSET2TARGET, synset2target
from .transforms import ImageClassification


class ImageDataset(VisionDataset):
    """Image Dataset from samples"""

    classes: Optional[tuple[str, ...]]  # as torchvision Datasets examples
    class_to_idx: dict[str, int]  # as torchvision Datasets examples

    def __init__(
        self,
        *,
        root: Optional[PathOrStr] = None,  # only for compatibility
        samples: tuple[tuple[str, int], ...],
        classes: Optional[tuple[str, ...]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
    ):
        """Image Dataset from samples - samples as tuple - filename and target"""
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.loader = loader or default_loader
        assert samples
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        if transform is None:
            self.transform = ImageClassification(crop_size=224)
        if target_transform is None:
            self.target_transform = torch.tensor
        self._num_samples = len(self.samples)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return (
            self.transform(self.loader(self.samples[index][0])),
            self.target_transform(self.samples[index][1]),
        )


def imagedataset_from_folder(
    root: PathOrStr,
    *,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Optional[Callable] = None,
    classes_as_imagenet: bool = False,
    num_samples: int = 0,
) -> ImageDataset:
    """Create dataset from folder structure. Folders as classes."""
    filenames = get_files(root, num_samples=num_samples)
    synsets = [fn.parent.name for fn in filenames]
    classes = tuple(sorted(set(synsets)))
    if classes_as_imagenet:
        class_to_idx = {synset: SYNSET2TARGET[synset] for synset in classes}
    else:
        class_to_idx = {key: num for num, key in enumerate(classes)}
    return ImageDataset(
        root=root,
        samples=tuple(zip(map(str, filenames), map(synset2target, synsets))),
        classes=classes,
        class_to_idx=class_to_idx,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
    )


def imagedataset_from_df(
    df: pd.DataFrame,
    *,
    root: Optional[PathOrStr] = None,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Optional[Callable] = None,
    # image_backend: str = "accimage",
    classes_as_imagenet: bool = False,
    num_samples: int = 0,
) -> ImageDataset:
    """Create dataset from dataframe.
    Dataframe should have columns 'path' and 'synset' columns"""
    samples, class_to_idx = samples_from_df(df, num_samples=num_samples, classes_as_imagenet=classes_as_imagenet)
    return ImageDataset(
        root=root,
        samples=samples,
        class_to_idx=class_to_idx,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
    )


def df_add_path(df: pd.DataFrame, root: PathOrStr) -> pd.DataFrame:
    """Add path column to dataframe with 'ds', 'split', 'synset', 'filename' columns."""
    df["path"] = Path(root) / df.ds / df.split / df.synset / df.filename
    assert df.path.iloc[0].exists()
    return df


def samples_from_df(
    df: pd.DataFrame,
    num_samples: Optional[int] = None,
    classes_as_imagenet: bool = False,
) -> tuple[tuple[tuple[str, int], ...], dict[str, int]]:
    """Generate samples and class_to_idx from dataframe.

    Args:
        df (pd.DataFrame): Dataframe with columns 'path' and 'synset'
        num_samples (Optional[int], optional): Number of samples. Defaults to None.

    Returns:
        tuple[tuple[tuple[str, int], ...], dict[str, int]]: Samples and class_to_idx
    """
    assert "synset" in df.columns
    assert "path" in df.columns
    classes = tuple(sorted(df.synset.unique()))
    if classes_as_imagenet:
        class_to_idx = {synset: SYNSET2TARGET[synset] for synset in classes}
    else:
        class_to_idx = {synset: num for num, synset in enumerate(classes)}
    samples = tuple(zip(df.path.apply(str), df.synset.apply(lambda x: class_to_idx[x])))
    if num_samples:
        samples = samples[:num_samples]
    return samples, class_to_idx
