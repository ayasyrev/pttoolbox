"""Create ImageDataset for Imagenet 1k."""

from importlib import resources
from typing import Literal, Optional

import pandas as pd


def load_df(
    filename: Optional[str] = None,
    split: Optional[Literal["train", "val"]] = "val",
) -> pd.DataFrame:
    """Load dataframe with information about dataset from parquet file.

    Args:
        filename: path to parquet file. If no name is given, used prepared data.
        split: split name, default: val
    """
    if filename is None:
        filename = (
            resources.files("pttoolbox.data.data_info")
            / f"imagenet1k_{split}.parquet.gzip"
        )
    print(filename)
    return pd.read_parquet(filename)
