from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.v2 import functional as Fv2
from torchvision.utils import _log_api_usage_once


class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        """Basic transforms, copied and adapted from torchvision.transforms._presets"""
        super().__init__()
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.mean = mean
        self.std = std
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
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


class ImageClassificationV2(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        """Basic V2 transforms, copied and adapted from torchvision.transforms._presets"""
        super().__init__()
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = Fv2.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        img = Fv2.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = Fv2.pil_to_tensor(img)
        img = Fv2.convert_image_dtype(img, torch.float)
        img = Fv2.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " V2 ("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


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
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
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


class ImageClassificationNoNormV2(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        super().__init__()
        self.crop_size = (crop_size,)
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = Fv2.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        img = Fv2.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = Fv2.pil_to_tensor(img)
        img = Fv2.convert_image_dtype(img, torch.float)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " V2 ("
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
        self.mean = mean
        self.std = std

    def forward(self, img: Tensor) -> Tensor:
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += "\n)"
        return format_string


class NormalizeV2(nn.Module):
    def __init__(
        self,
        *,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img: Tensor) -> Tensor:
        img = Fv2.normalize(img, mean=self.mean, std=self.std)
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
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self,
        img: Tensor,
        flip: int = 0,
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


class TrainPersistentTransformV2(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int = 224,
        resize_size: int = 256,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        super().__init__()
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self,
        img: Tensor,
        flip: int = 0,
        # rotate: int = 0,  # -60 -- 60
        shift_x: int = 16,  # 0 -- 32
        shift_y: int = 16,  # 0 -- 32
    ) -> Tensor:
        img = Fv2.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        if flip:
            img = Fv2.hflip(img)
        # img = F.rotate(img, rotate / 10)
        img = Fv2.crop(img, shift_x, shift_y, self.crop_size[0], self.crop_size[0])
        if not isinstance(img, Tensor):
            img = Fv2.pil_to_tensor(img)
        img = Fv2.convert_image_dtype(img, torch.float)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string


class ResizeCenterCropV2(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        super().__init__()
        self.crop_size = (crop_size,)
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = Fv2.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        return Fv2.center_crop(img, self.crop_size)


class TrainPersistentTrfmV2(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int = 224,
        resize_size: int = 256,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ) -> None:
        super().__init__()
        self.crop_size = (crop_size,)
        self.resize_size = (resize_size,)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self,
        img: Tensor,
        flip: int = 0,
        # rotate: int = 0,  # -60 -- 60
        shift_x: int = 16,  # 0 -- 32
        shift_y: int = 16,  # 0 -- 32
    ) -> Tensor:
        img = Fv2.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        if flip:
            img = Fv2.hflip(img)
        # img = F.rotate(img, rotate / 10)
        return Fv2.crop(img, shift_x, shift_y, self.crop_size[0], self.crop_size[0])


class ConvertNormalizeV2(nn.Module):
    def __init__(
        self,
        *,
        mean: Tuple[float, ...] = (123.6750, 116.2800, 103.5300),
        std: Tuple[float, ...] = (58.3950, 57.1200, 57.3750),
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img: Tensor) -> Tensor:
        img = img.to(torch.float)
        img = Fv2.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += "\n)"
        return format_string


class ConvertNormV2(nn.Module):
    def __init__(
        self,
        *,
        mean: Tuple[float, ...] = (123.6750, 116.2800, 103.5300),
        std: Tuple[float, ...] = (58.3950, 57.1200, 57.3750),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.mean = torch.as_tensor(mean, dtype=torch.float, device=device).view(
            -1, 1, 1
        )
        self.std = torch.as_tensor(std, dtype=torch.float, device=device).view(-1, 1, 1)

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.to(torch.float).sub_(self.mean).div_(self.std)
