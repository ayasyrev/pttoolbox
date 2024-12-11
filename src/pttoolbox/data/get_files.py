from pathlib import Path
from typing import List, Optional

from ..types import PathOrStr


__all__ = ["get_files", "get_image_files"]

IMG_EXT = (".jpeg", ".jpg", ".jfif", ".pjpeg", ".pjp", ".png", ".bmp", ".tif", ".tiff")


def get_files(
    data_dir: PathOrStr,
    num_samples: Optional[int] = None,
    sort: bool = True,
    images: bool = True,
    ext: Optional[tuple[str]] = None,
) -> List[Path]:
    """Return list of num_samples filenames from data_dir.
    If num_samples is None (default) return list of ALL images.
    Sorted by default, use sorted = False for unsorted list.

    Args:
        data_dir (str | PosixPath | Path):
        num_samples (int, optional): Number of samples to return. Defaults to None.
            If num_samples is None return list of ALL images.
        sort (bool, optional): Sort list. Defaults to True.
        images (bool, optional): Return only images. Defaults to True.
        ext (tuple[str], optional): Filter extensions. Defaults to None.

    Returns:
        List[Path]: List of filenames
    """
    if images:
        filter_ext = ext or IMG_EXT
        filenames = [
            Path(fn)
            for fn in Path(data_dir).rglob("*.*")
            if fn.suffix.lower() in filter_ext
        ]
    else:
        filenames = [Path(fn) for fn in Path(data_dir).rglob("*.*")]
        if ext is not None:
            filenames = [fn for fn in filenames if fn.suffix in ext]
    if sort:
        filenames.sort()
    num_samples = num_samples or len(filenames)

    return filenames[:num_samples]


def get_image_files(
    data_dir: PathOrStr,
    num_samples: Optional[int] = None,
    sort: bool = True,
    ext: Optional[tuple[str]] = None,
) -> List[Path]:
    """Return list of num_samples image filenames from data_dir.s

    Args:
        data_dir (str | PosixPath | Path):
        num_samples (int, optional): Number of samples to return. Defaults to None.
            If num_samples is None return list of ALL images.
        sort (bool, optional): Sort list. Defaults to True.
        ext (tuple[str], optional): Filter extensions. Defaults to None.

    Returns:
        List[Path]: List of filenames
    """
    return get_files(data_dir, num_samples, sort, images=True, ext=ext)
