from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torchvision.transforms import functional as F, InterpolationMode


class ImageClassificationNoNorm(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are rescaled to ``[0.0, 1.0]``."
        )


class Normalize(nn.Module):
    def __init__(
        self,
        *,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.mean = list(mean)
        self.std = list(std)

    def forward(self, img: Tensor) -> Tensor:
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += "\n)"
        return format_string


class TrainPersistentTransform(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int = 224,
        resize_size: int = 256,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self,
        img: Tensor,
        flip: Union[bool, int] = 0,
        # rotate: int = 0,  # -60 -- 60
        shift_x: int = 16,  # 0 -- 32
        shift_y: int = 16,  # 0 -- 32
    ) -> Tensor:
        img = F.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        if flip:
            img = F.hflip(img)
        # img = F.rotate(img, rotate / 10)
        img = F.crop(img, shift_x, shift_y, self.crop_size[0], self.crop_size[0])
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string
