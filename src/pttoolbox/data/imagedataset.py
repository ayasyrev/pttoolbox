""" Image Dataset.
As ImageFolderDataset -> base use from given samples.
Use classes from imagenet, samples from dataframe.
"""
from typing import Callable, Optional

import pandas as pd
import torch
from torchvision import set_image_backend
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms._presets import ImageClassification

from .get_files import get_files
from .imagenet1k_classes import SYNSET2TARGET, synset2target
from .typing import PathOrStr


class ImageDataset(VisionDataset):
    """Image Dataset from samples, folders (classes by folders), dataframe."""
    classes: Optional[tuple[str, ...]]  # as torchvision Datasets examples
    class_to_idx: dict[str, int]  # as torchvision Datasets examples

    def __init__(
        self,
        root: PathOrStr,
        samples: tuple[tuple[str, int], ...],
        classes: Optional[tuple[str, ...]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        image_backend: str = "accimage",
        classes_as_imagenet: bool = False,
    ):
        """Nette / woof ds."""
        super().__init__(
            root=str(root),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        if loader is None:
            self.loader = default_loader
            set_image_backend(image_backend)
        else:
            self.loader = loader
        assert samples
        self.samples = samples
        self.classes = classes or tuple(
            sorted(set(sample[0].split("/")[-2] for sample in self.samples))
        )
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        elif classes_as_imagenet:
            self.class_to_idx = {
                synset: SYNSET2TARGET[synset] for synset in self.classes
            }
        else:
            self.class_to_idx = {key: num for num, key in enumerate(self.classes)}
        if transform is None:
            self.transform = ImageClassification(crop_size=224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return (
            self.transform(self.loader(self.samples[index][0])),
            self.samples[index][1],
        )

    @classmethod
    def from_folder(
        cls,
        root: PathOrStr,
        *,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        image_backend: str = "accimage",
        classes_as_imagenet: bool = False,
        num_samples: int = 0,
    ):
        """Create dataset from folder structure. Folders as classes."""
        filenames = get_files(root, num_samples=num_samples)
        synsets = [fn.parent.name for fn in filenames]
        classes = tuple(sorted(set(synsets)))
        if classes_as_imagenet:
            class_to_idx = {synset: SYNSET2TARGET[synset] for synset in classes}
        else:
            class_to_idx = {key: num for num, key in enumerate(classes)}
        # samples = tuple((str(fn), synset2target[fn.parent.name]) for fn in filenames)
        return cls(
            root=root,
            samples=tuple(zip(map(str, filenames), map(synset2target, synsets))),
            classes=classes,
            class_to_idx=class_to_idx,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            image_backend=image_backend,
            classes_as_imagenet=classes_as_imagenet,
        )

    @classmethod
    def from_df(
        cls,
        root: PathOrStr,
        df: pd.DataFrame,
        *,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        image_backend: str = "accimage",
        classes_as_imagenet: bool = False,
        num_samples: int = 0,
    ):
        """Create dataset from dataframe.
        Dataframe should have columns 'path' and 'synset' columns"""
        assert "path" in df.columns
        assert "synset" in df.columns
        classes = tuple(sorted(df.synset.unique()))
        if classes_as_imagenet:
            class_to_idx = {synset: SYNSET2TARGET[synset] for synset in classes}
        else:
            class_to_idx = {synset: num for num, synset in enumerate(classes)}
        samples = tuple(zip(df.path.apply(str), df.synset.apply(lambda x: class_to_idx[x])))
        if num_samples:
            samples = samples[:num_samples]
        return cls(
            root=root,
            samples=samples,
            classes=classes,
            class_to_idx=class_to_idx,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            image_backend=image_backend,
            classes_as_imagenet=classes_as_imagenet,
        )


def df_add_path(df: pd.DataFrame, root: PathOrStr) -> pd.DataFrame:
    """Add path column to dataframe with 'ds', 'split', 'synset', 'filename' columns."""
    df["path"] = root / df.ds / df.split / df.synset / df.filename
    assert df.path.iloc[0].exists()
    return df
