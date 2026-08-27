import importlib
import importlib.metadata
import pkgutil
import tomllib
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

import pttoolbox
from pttoolbox.data.image_loader import io_loader, pil_loader
from pttoolbox.version import __version__ as compatibility_version


def test_version_has_single_source() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared_version = pyproject["project"]["version"]

    assert declared_version == "0.2.0"
    assert importlib.metadata.version("pttoolbox") == declared_version
    assert pttoolbox.__version__ == declared_version
    assert compatibility_version == declared_version


def test_all_modules_import() -> None:
    modules = pkgutil.walk_packages(pttoolbox.__path__, f"{pttoolbox.__name__}.")
    for module in modules:
        importlib.import_module(module.name)


def test_image_loaders(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(image_path)

    pil_image = pil_loader(str(image_path))
    tensor_image = io_loader(str(image_path))

    assert pil_image.mode == "RGB"
    assert pil_image.size == (3, 2)
    assert tensor_image.shape == (3, 2, 3)
    assert tensor_image.dtype == torch.uint8


@pytest.mark.parametrize(
    "filename",
    [
        "imagenette2.parquet.gzip",
        "imagenet1k_train.parquet.gzip",
        "imagenet1k_val.parquet.gzip",
    ],
)
def test_bundled_parquet_files(filename: str) -> None:
    resource = resources.files("pttoolbox.data.data_info") / filename
    with resources.as_file(resource) as path:
        dataframe = pd.read_parquet(path)
    assert not dataframe.empty
