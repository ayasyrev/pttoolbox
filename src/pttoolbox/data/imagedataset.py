""" Image Dataset.
As ImageFolderDataset -> base use from given samples.
Use classes from imagenet, samples from dataframe.
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torchvision import set_image_backend
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms._presets import ImageClassification

from .get_files import get_files
from .imagenet_1k_classes import synset2target
from .typing import PathOrStr


class ImageDataset(VisionDataset):
    """Image Dataset from samples, folders (classes by folders), dataframe."""

    def __init__(
        self,
        root: PathOrStr,
        samples: tuple[tuple[str, int], ...],
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
        classes = tuple(
            sorted(set(sample[0].split("/")[-2] for sample in self.samples))
        )
        if classes_as_imagenet:
            self.class_to_idx = {
                key: value for key, value in synset2target.items() if key in classes
            }
        else:
            self.class_to_idx = {key: num for num, key in enumerate(sorted(classes))}
        self.classes = classes
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
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        image_backend: str = "accimage",
        limit_dataset: int = 0,
        classes_as_imagenet: bool = True,
    ):
        filenames = get_files(root, num_samples=limit_dataset)
        samples = tuple((str(fn), synset2target[fn.parent.name]) for fn in filenames)
        return cls(
            root=root,
            samples=samples,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            image_backend=image_backend,
            classes_as_imagenet=classes_as_imagenet,
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        *,
        root: PathOrStr,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        image_backend: str = "accimage",
        limit_dataset: int = 0,
        classes_as_imagenet: bool = False,
    ):
        if classes_as_imagenet:
            df["target"] = df.synsetid.apply(lambda x: synset2target[x])
        else:
            synset2class = {
                synset: num for num, synset in enumerate(sorted(df.synsetid.unique()))
            }
            df["target"] = df.synsetid.apply(lambda x: synset2class[x])
        root = Path(root)
        df["path"] = root / df.ds / df.split / df.synsetid / df.filename
        df["path"] = df.path.apply(str)
        samples = tuple(zip(df.path, df.target))
        if limit_dataset:
            samples = samples[:limit_dataset]
        return cls(
            root=root,
            samples=samples,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            image_backend=image_backend,
            classes_as_imagenet=classes_as_imagenet,
        )
