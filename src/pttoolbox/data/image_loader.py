# copied from torchvision and refactored
import accimage
from PIL import Image
from torchvision.io import ImageReadMode, read_image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader_safe(path: str) -> accimage.Image:
    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


# @lru_cache(maxsize=None)
def accimage_loader(path: str) -> accimage.Image:
    return accimage.Image(path)


def io_loader(path: str):
    return read_image(path, mode=ImageReadMode.RGB)
