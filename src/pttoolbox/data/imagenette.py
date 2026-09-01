"""Create ImageDataset for Imagenette2 / Imagewoof2."""

import hashlib
import hmac
import os
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from ..types import PathOrStr
from .dataset_memmap import MemmapDataset
from .dataset_persistent import persistent_dataset_from_df
from .dataset_safetensors import SafeTensorsDataset
from .imagedataset import ImageDataset, df_add_path, imagedataset_from_df

IMAGENETTE_ARCHIVE_FILENAME = "imagenette2.tgz"
IMAGENETTE_ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMAGENETTE_ARCHIVE_SHA256 = (
    "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd"
)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def verify_imagenette_archive(
    archive_path: PathOrStr,
    *,
    expected_sha256: str = IMAGENETTE_ARCHIVE_SHA256,
) -> bool:
    """Return whether an Imagenette archive matches the expected SHA-256."""
    path = Path(archive_path)
    if not path.is_file():
        return False

    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(_DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return hmac.compare_digest(digest.hexdigest(), expected_sha256.lower())


def download_imagenette_archive(
    root: PathOrStr,
    *,
    timeout: float = 30.0,
) -> Path:
    """Download and verify the full Imagenette archive without extracting it."""
    root_path = Path(root)
    archive_path = root_path / IMAGENETTE_ARCHIVE_FILENAME
    if verify_imagenette_archive(
        archive_path,
        expected_sha256=IMAGENETTE_ARCHIVE_SHA256,
    ):
        return archive_path

    root_path.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    digest = hashlib.sha256()

    try:
        with requests.get(
            IMAGENETTE_ARCHIVE_URL,
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            total_size = int(content_length) if content_length else None

            with (
                Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                ) as progress,
                partial_path.open("wb") as archive,
            ):
                task = progress.add_task(
                    f"Downloading {IMAGENETTE_ARCHIVE_FILENAME}...",
                    total=total_size,
                )
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    archive.write(chunk)
                    digest.update(chunk)
                    progress.update(task, advance=len(chunk))

        actual_sha256 = digest.hexdigest()
        if not hmac.compare_digest(actual_sha256, IMAGENETTE_ARCHIVE_SHA256):
            raise ValueError(
                "Imagenette archive SHA-256 mismatch: "
                f"expected {IMAGENETTE_ARCHIVE_SHA256}, got {actual_sha256}"
            )
        partial_path.replace(archive_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    return archive_path


def load_df(
    filename: str | None = None,
    dataset: Literal["imagenette2", "imagewoof2"] | None = None,
    split: Literal["train", "val"] | None = None,
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
    root: PathOrStr | None = None,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int | None = None,
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
    num_samples: int | None = None,
    batch_size: int = 32,
    transforms: Callable | None = None,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    loader: Callable | None = None,
    sampler: Callable | None = None,
    classes_as_imagenet: bool = False,
    num_workers: int | None = None,
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
    num_samples: int | None = None,
    batch_size: int = 32,
    transforms: Callable | None = None,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    loader: Callable | None = None,
    sampler: Callable | None = None,
    classes_as_imagenet: bool = False,
    num_workers: int | None = None,
    indexes: list[list[int]] | None = None,
    transform_indexes: list[list[int]] | None = None,
    epochs: int | None = None,
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
    root: PathOrStr | None = None,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int | None = None,
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
    num_samples: int | None = None,
    batch_size: int = 32,
    transforms: Callable | None = None,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    sampler: Callable | None = None,
    classes_as_imagenet: bool = False,
    num_workers: int | None = None,
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
    root: PathOrStr | None = None,
    dataset: Literal["imagenette2", "imagewoof2"] = "imagenette2",
    split: Literal["train", "val"] = "train",
    num_samples: int | None = None,
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
    num_samples: int | None = None,
    batch_size: int = 32,
    transforms: Callable | None = None,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    sampler: Callable | None = None,
    classes_as_imagenet: bool = False,
    num_workers: int | None = None,
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
