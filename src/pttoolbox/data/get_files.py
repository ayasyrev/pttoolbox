from pathlib import Path, PosixPath
from typing import List, Optional, Union


IMG_EXT = (".JPEG", ".JPG", ".jpeg", ".jpg", ".PNG", ".png")


PathOrStr = Union[str, PosixPath, Path]


def get_files(
    data_dir: PathOrStr,
    num_samples: int = 0,
    sort: bool = True,
    images: bool = True,
    ext: Optional[tuple[str]] = None,
) -> List[Path]:
    """Return list of num_samples filenames from data_dir.
    If num_samples == 0 return list of ALL images.
    Sorted by default, use sorted = False for unsorted list.

    Args:
        data_dir (str | PosixPath | Path):
        num_samples (int, optional): Number of samples to return. Defaults to 0.
            If num_samples == 0 return list of ALL images.
        sort (bool, optional): Sort list. Defaults to True.
        images (bool, optional): Return only images. Defaults to True.
        ext (tuple[str], optional): Filter extensions. Defaults to None.

    Returns:
        List[Path]: List of filenames
    """
    if images:
        filter_ext = ext or IMG_EXT
        filenames = [
            Path(fn) for fn in Path(data_dir).rglob("*.*") if fn.suffix in filter_ext
        ]
    else:
        filenames = [Path(fn) for fn in Path(data_dir).rglob("*.*")]
        if ext is not None:
            filenames = [fn for fn in filenames if fn.suffix in ext]
    if sort:
        filenames.sort()
    if num_samples != 0:
        filenames = filenames[:num_samples]

    return filenames
