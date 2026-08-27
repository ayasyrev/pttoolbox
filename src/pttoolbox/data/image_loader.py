# copied from torchvision and refactored
from PIL import Image
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def io_loader(path: str) -> Tensor:
    return decode_image(path, mode=ImageReadMode.RGB)
