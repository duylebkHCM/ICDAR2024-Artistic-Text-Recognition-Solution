import itertools
from collections.abc import Sized, Iterable
from typing import Union, Tuple

import torch
from torch import Tensor


__all__ = [
    "geometric_mean",
    "harmonic_mean",
    "harmonic1p_mean",
    "logodd_mean",
    "log1p_mean",
    "pad_image_tensor",
    "pad_tensor_to_size",
    "torch_fliplr",
    "torch_flipud",
    "torch_none",
    "torch_rot180",
    "torch_rot90_ccw",
    "torch_rot90_ccw_transpose",
    "torch_rot90_cw",
    "torch_rot90_cw_transpose",
    "torch_transpose",
    "torch_transpose2",
    "torch_transpose_",
    "torch_transpose_rot90_ccw",
    "torch_transpose_rot90_cw",
    "unpad_image_tensor",
    "unpad_xyxy_bboxes",
]


def torch_none(x: Tensor) -> Tensor:
    """
    Return input argument without any modifications
    :param x: input tensor
    :return: x
    """
    return x


def torch_rot90_ccw(x):
    return x.rot90(k=1, dims=(2, 3))


def torch_rot90_cw(x):
    return x.rot90(k=-1, dims=(2, 3))


def torch_rot90_ccw_transpose(x):
    return x.rot90(k=1, dims=(2, 3)).transpose(2, 3)


def torch_rot90_cw_transpose(x):
    return x.rot90(k=-1, dims=(2, 3)).transpose(2, 3)


def torch_transpose_rot90_ccw(x: Tensor):
    return x.transpose(2, 3).rot90(k=1, dims=(2, 3))


def torch_transpose_rot90_cw(x):
    return x.transpose(2, 3).rot90(k=-1, dims=(2, 3))



def torch_rot180(x: Tensor):
    """
    Rotate 4D image tensor by 180 degrees
    :param x:
    :return:
    """
    return torch.rot90(x, k=2, dims=(2, 3))


def torch_rot180_transpose(x):
    return x.rot90(k=2, dims=(2, 3)).transpose(2, 3)


def torch_transpose_rot180(x):
    return x.transpose(2, 3).rot90(k=2, dims=(2, 3))



def torch_flipud(x: Tensor):
    """
    Flip 4D image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x: Tensor):
    """
    Flip 4D image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)


def torch_transpose(x: Tensor):
    """
    Transpose 4D image tensor by main image diagonal
    :param x:
    :return:
    """
    return x.transpose(2, 3)


def torch_transpose_(x: Tensor):
    return x.transpose_(2, 3)


def torch_transpose2(x: Tensor):
    """
    Transpose 4D image tensor by second image diagonal
    :param x:
    :return:
    """
    return x.transpose(3, 2)


def pad_tensor_to_size(x: Tensor, size: Tuple[int, ...], mode="constant", value=0) -> Tuple[Tensor, Tuple[slice, ...]]:
    """
    Pad tensor to given size by appending elements to the beginning and end of each axis.

    :param x: Input tensor of shape [B, C, *num_spatial_dims]
    :param size: Target tensor size defined as an array of [num_spatial_dims] elements
    :param mode: Padding mode, see torch.nn.functional.pad
    :param value: Padding value, see torch.nn.functional.pad
    :return: Tuple of padded tensor and crop parameters. Second argument can be used to reverse pad operation of model output
    """
    num_spatial_dims = len(size)
    if num_spatial_dims != len(x.shape) - 2:
        raise ValueError(f"Expected {num_spatial_dims} spatial dimensions, got {len(x.shape) - 2}")

    spatial_dims = x.shape[-num_spatial_dims:]
    padding = torch.tensor(size) - torch.tensor(spatial_dims)
    padding_before = padding // 2
    padding_after = padding - padding_before

    padding_pairs = tuple(zip(padding_before.tolist(), padding_after.tolist()))
    padding_params = tuple(itertools.chain(*reversed(padding_pairs)))

    x = torch.nn.functional.pad(x, pad=padding_params, mode=mode, value=value)

    crop_params = [slice(None), slice(None)] + [
        slice(before, before + total_size)
        for (before, after, total_size) in zip(padding_before, padding_after, spatial_dims)
    ]
    return x, crop_params


def pad_image_tensor(
    image_tensor: Tensor, pad_size: Union[int, Tuple[int, int]] = 32
) -> Tuple[Tensor, Tuple[int, int, int, int]]:
    """Pad input tensor to make it's height and width dividable by @pad_size

    :param image_tensor: 4D image tensor of shape NCHW
    :param pad_size: Pad size
    :return: Tuple of output tensor and pad params. Second argument can be used to reverse pad operation of model output
    """
    if len(image_tensor.size()) != 4:
        raise ValueError("Tensor must have rank 4 ([B,C,H,W])")

    rows, cols = image_tensor.size(2), image_tensor.size(3)
    if isinstance(pad_size, Sized) and isinstance(pad_size, Iterable) and len(pad_size) == 2:
        pad_height, pad_width = [int(val) for val in pad_size]
    elif isinstance(pad_size, int):
        pad_height = pad_width = pad_size
    else:
        raise ValueError(
            f"Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar."
        )

    if rows > pad_height:
        pad_rows = rows % pad_height
        pad_rows = pad_height - pad_rows if pad_rows > 0 else 0
    else:
        pad_rows = pad_height - rows

    if cols > pad_width:
        pad_cols = cols % pad_width
        pad_cols = pad_width - pad_cols if pad_cols > 0 else 0
    else:
        pad_cols = pad_width - cols

    if pad_rows == 0 and pad_cols == 0:
        return image_tensor, (0, 0, 0, 0)

    pad_top = pad_rows // 2
    pad_btm = pad_rows - pad_top

    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    pad = [pad_left, pad_right, pad_top, pad_btm]
    image_tensor = torch.nn.functional.pad(image_tensor, pad)
    return image_tensor, pad


def unpad_image_tensor(image_tensor: Tensor, pad) -> Tensor:
    if len(image_tensor.size()) != 4:
        raise ValueError("Tensor must have rank 4 ([B,C,H,W])")

    pad_left, pad_right, pad_top, pad_btm = pad
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    return image_tensor[..., pad_top : rows - pad_btm, pad_left : cols - pad_right]


def unpad_xyxy_bboxes(bboxes_tensor: torch.Tensor, pad, dim=-1):
    pad_left, pad_right, pad_top, pad_btm = pad
    pad = torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype).to(bboxes_tensor.device)

    if dim == -1:
        dim = len(bboxes_tensor.size()) - 1

    expand_dims = list(set(range(len(bboxes_tensor.size()))) - {dim})
    for i, dim in enumerate(expand_dims):
        pad = pad.unsqueeze(dim)

    return bboxes_tensor - pad


def geometric_mean(x: Tensor, dim: int) -> Tensor:
    """
    Compute geometric mean along given dimension.
    This implementation assume values are in range (0...1) (Probabilities)
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    return x.log().mean(dim=dim).exp()


def harmonic_mean(x: Tensor, dim: int, eps: float = 1e-6) -> Tensor:
    """
    Compute harmonic mean along given dimension.

    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    x = torch.reciprocal(x.clamp_min(eps))
    x = torch.mean(x, dim=dim)
    x = torch.reciprocal(x.clamp_min(eps))
    return x


def harmonic1p_mean(x: Tensor, dim: int) -> Tensor:
    """
    Compute harmonic mean along given dimension.

    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    x = torch.reciprocal(x + 1)
    x = torch.mean(x, dim=dim)
    x = torch.reciprocal(x) - 1
    return x


def logodd_mean(x: Tensor, dim: int, eps: float = 1e-6) -> Tensor:
    """
    Compute log-odd mean along given dimension.
    logodd = log(p / (1 - p))

    This implementation assume values are in range [0, 1] (Probabilities)
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    x = x.clamp(min=eps, max=1.0 - eps)
    x = torch.log(x / (1 - x))
    x = torch.mean(x, dim=dim)
    x = torch.exp(x) / (1 + torch.exp(x))
    return x


def log1p_mean(x: Tensor, dim: int) -> Tensor:
    """
    Compute average log(x+1) and them compute exp.
    Requires all inputs to be non-negative

    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    x = torch.log1p(x)
    x = torch.mean(x, dim=dim)
    x = torch.exp(x) - 1
    return x