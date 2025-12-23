"""Create cached multi-scale pyramids for facies, seismic and well data.

This module exposes helpers that build multi-resolution tensors for the
training pipeline. The functions are cached via ``joblib.Memory`` so repeated
calls with the same scale list are fast. All pyramid functions accept a
``scale_list`` that is a tuple of shape descriptors produced by
``ops.generate_scales`` (each element is ``(batch, channels, height, width)``)
and return a list of PyTorch tensors, one tensor per scale, with shape
``(N, C, H, W)``.
"""

import mlx.core as mx
import datasets.torch.utils as torch_utils
import utils


def to_facies_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    channels_last: bool = False,
) -> tuple[mx.array, ...]:
    """Generate multi-scale pyramid tensors for facies images using a neural interpolator.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors as produced by ``ops.generate_scales``. Each
        element must be a 4-tuple ``(batch, channels, height, width)`` describing
        the target resolution.

    Returns
    -------
    tuple[mx.array, ...]
        A tuple where each element is a MLX array containing all facies at
        that scale with shape ``(N, H, W, C)`` (N images, C channels).
    """
    torch_pyramids = torch_utils.to_facies_pyramids(
        scale_list, channels_last=channels_last
    )
    # Vectorized: stack and convert in one go if possible
    if len(torch_pyramids) > 0:
        mlx_pyramids = tuple(mx.array(p.numpy()) for p in torch_pyramids)
        return mlx_pyramids
    else:
        return tuple()


def to_seismic_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    channels_last: bool = False,
) -> tuple[mx.array, ...]:
    """Generate multi-scale pyramid tensors for seismic images using nearest interpolation.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, height, width, channels)``.

    Returns
    -------
    tuple[mx.array, ...]
        A tuple of tensors (one per scale) with shape ``(N, H, W, C)``.

    Notes
    -----
    Uses NearestInterpolator and caches results in ``./.cache``.
    """
    torch_pyramids = torch_utils.to_seismic_pyramids(
        scale_list, channels_last=channels_last
    )
    if len(torch_pyramids) > 0:
        mlx_pyramids = tuple(mx.array(p.numpy()) for p in torch_pyramids)
        return mlx_pyramids
    else:
        return tuple()


def to_wells_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    channels_last: bool = False,
) -> tuple[mx.array, ...]:
    """Generate multi-scale pyramid tensors for well location data.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, height, width, channels)``.

    Returns
    -------
    tuple[mx.array, ...]
        A tuple of tensors (one per scale) with shape ``(N, H, W, C)``. Well
        locations are represented as sparse vertical traces at the appropriate
        column indices.

    Notes
    -----
    Uses ``WellInterpolator`` and caches results in ``./.cache``.
    """
    torch_pyramids = torch_utils.to_wells_pyramids(
        scale_list, channels_last=channels_last
    )
    if len(torch_pyramids) > 0:
        mlx_pyramids = tuple(mx.array(p.numpy()) for p in torch_pyramids)
        return mlx_pyramids
    else:
        return tuple()


def norm(x: mx.array) -> mx.array:
    """Normalize tensor from [0, 1] to [-1, 1] range.

    Parameters
    ----------
    x : mx.array
        Input tensor with values in [0, 1] range.

    Returns
    -------
    mx.array
        Normalized tensor with values clamped to [-1, 1].
    """
    out = (x - 0.5) * 2
    return utils.clamp(out)
