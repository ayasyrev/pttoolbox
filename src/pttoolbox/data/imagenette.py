"""Create ImageDataset for Imagenette2 / Imagewoof2."""

import os
from importlib import resources
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from ..typing import PathOrStr
from .dataset_memmap import MemmapDataset
from .dataset_persistent import persistent_dataset_from_df
from .dataset_safetensors import SafeTensorsDataset
from .imagedataset import ImageDataset, df_add_path, imagedataset_from_df


def load_df(
    filename: Optional[str] = None,
    dataset: Optional[Literal["imagenette2", "imagewoof2"]] = None,
    split: Optional[Literal["train", "val"]] = None,
) -> pd.DataFrame:
    """Load dataframe with information about dataset from parquet file.

    Args:
        filename: path to parquet file. If no name is given, used prepared data.
        dataset: dataset name, default: None - load full data
        split: split name, default: None
    """
    if filename is None:
        filename = (
            resources.files("pttoolbox.data.data_info") / "imagenette2.parquet.gzip"
        )
    if dataset is None and split is None:
        return pd.read_parquet(filename)
    ds_filter = [("ds", "==", dataset)] if dataset else []
    split_filter = [("split", "==", split)] if split else []
    return pd.read_parquet(
        filename,
        filters=ds_filter + split_filter,
    )


def get_imagenette_dataset(
    root: Optional[PathOrStr] = None,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    classes_as_imagenet: bool = False,
    **kwargs,
) -> ImageDataset:
    """Create ImageDataset for Imagenette2 / Imagewoof2."""
    df = load_df(dataset=dataset, split=split)
    df_add_path(df, root)
    return imagedataset_from_df(
        df,
        root=root,
        num_samples=num_samples,
        classes_as_imagenet=classes_as_imagenet,
        **kwargs,
    )


def get_imagenette_dataloader(
    root: PathOrStr,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    batch_size: int = 32,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Optional[Callable] = None,
    sampler: Optional[Callable] = None,
    # image_backend: str = "accimage",
    classes_as_imagenet: bool = False,
    num_workers: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for Imagenette2 / Imagewoof2."""
    dataset = get_imagenette_dataset(
        root=root,
        dataset=dataset,
        split=split,
        num_samples=num_samples,
        classes_as_imagenet=classes_as_imagenet,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
    )
    if split == "train":
        shuffle = sampler is None  # if sampler -> no shuffle
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    if num_workers is None:
        num_workers = os.cpu_count()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
        **kwargs,
    )


def get_persistent_imagenette_dataloader(
    root: PathOrStr,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    batch_size: int = 32,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Optional[Callable] = None,
    sampler: Optional[Callable] = None,
    classes_as_imagenet: bool = False,
    num_workers: Optional[int] = None,
    indexes: Optional[list[list[int]]] = None,
    transform_indexes: Optional[list[list[int]]] = None,
    epochs: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create persistent DataLoader for Imagenette2 / Imagewoof2."""
    df = load_df(dataset=dataset, split=split)
    df_add_path(df, root)
    dataset = persistent_dataset_from_df(
        root=root,
        df=df,
        num_samples=num_samples,
        indexes=indexes,
        transform_indexes=transform_indexes,
        epochs=epochs,
        classes_as_imagenet=classes_as_imagenet,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
    )
    if num_workers is None:
        num_workers = os.cpu_count()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=split == "train",
        num_workers=num_workers,
        sampler=sampler,
        **kwargs,
    )


def get_imagenette_memmap_dataset(
    root: Optional[PathOrStr] = None,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    classes_as_imagenet: bool = False,
    **kwargs,
) -> ImageDataset:
    """Create memory mapped Dataset for Imagenette2 / Imagewoof2."""
    data_path = Path(root) / dataset
    if split == "train":
        samples_path = data_path / "train_256.dat"
        shape = (9025, 3, 256, 256)
        targets_shape = (9025,)
    else:
        samples_path = data_path / f"{split}_224.dat"
        shape = (3929, 3, 224, 224)
        targets_shape = (3929,)
    samples = np.memmap(
        samples_path,
        # data_path / f"{split}_224_float.dat",
        dtype=np.uint8,
        # dtype=np.float32,
        mode="r",
        shape=shape,
    )
    targets = np.memmap(
        data_path / f"{split}_targets.dat",
        dtype=np.uint16,
        mode="r",
        shape=targets_shape,
    )
    if num_samples:
        samples = samples[:num_samples]
        targets = targets[:num_samples]
    return MemmapDataset(
        samples=samples,
        targets=targets,
        **kwargs,
    )


def get_imagenette_memmap_dataloader(
    root: PathOrStr,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    batch_size: int = 32,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    sampler: Optional[Callable] = None,
    classes_as_imagenet: bool = False,
    num_workers: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create memory mapped DataLoader for Imagenette2 / Imagewoof2."""
    dataset = get_imagenette_memmap_dataset(
        root=root,
        dataset=dataset,
        split=split,
        num_samples=num_samples,
        classes_as_imagenet=classes_as_imagenet,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
    )
    if split == "train":
        shuffle = sampler is None  # if sampler -> no shuffle
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    if num_workers is None:
        num_workers = os.cpu_count()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
        **kwargs,
    )


def get_imagenette_safetensor_dataset(
    root: Optional[PathOrStr] = None,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    classes_as_imagenet: bool = False,
    **kwargs,
) -> ImageDataset:
    """Create memory mapped Dataset for Imagenette2 / Imagewoof2."""
    data_path = Path(root) / dataset
    samples_path = data_path / f"{split}_data.st"
    data = load_file(samples_path)
    samples = data["samples"]
    targets = data["targets"]
    if num_samples:
        samples = samples[:num_samples]
        targets = targets[:num_samples]
    return SafeTensorsDataset(
        samples=samples,
        targets=targets,
        **kwargs,
    )


def get_imagenette_safetensor_dataloader(
    root: PathOrStr,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int = 0,
    batch_size: int = 32,
    transforms: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    sampler: Optional[Callable] = None,
    classes_as_imagenet: bool = False,
    num_workers: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create safetensors DataLoader for Imagenette2 / Imagewoof2."""
    dataset = get_imagenette_safetensor_dataset(
        root=root,
        dataset=dataset,
        split=split,
        num_samples=num_samples,
        classes_as_imagenet=classes_as_imagenet,
        transforms=transforms,
        transform=transform,
        target_transform=target_transform,
    )
    if split == "train":
        shuffle = sampler is None  # if sampler -> no shuffle
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    if num_workers is None:
        num_workers = os.cpu_count()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        **kwargs,
    )
