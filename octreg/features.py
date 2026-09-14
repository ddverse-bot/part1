"""Two-class representation (spec step 5): one soft bright/dark map per modality, its channels and the level pyramid.

Score S(T) = 1/2 sum_c NCC_w(u_c, v_c o T). Since NCC_w(1 - u, v) = -NCC_w(u, v) for any weight, swapping the OCT classes
gives exactly -S: relative polarity is the sign of one score map.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def two_class(img, mask, h, modality, params: Params = Params(), device="cpu"):
    """Soft two-class map, the same function for OCT and MRI: normalize.flatten, Gaussian blur of sigma class_blur_h x h,
    t = Otsu of the foreground values of [::class_sample_stride] clipped at p[class_clip_percentile],
    p = sigmoid((I - t) / (class_sigmoid_std x std of those values)); chunked on device for whole hemispheres.
    img: float array [D, H, W] at h; mask: bool foreground at h; h: mm; modality: 'oct' | 'mri'.
    compat.v1_oct_unflattened (modality 'oct'): no flattening, Otsu and std on all mask values unclipped.
    -> (p float16 numpy [D, H, W] in [0, 1], 1 = bright class; {'threshold', 'scale', 'flattened': bool})."""
    raise NotImplementedError


def oct_channels(p, mask, params: Params = Params()):
    """OCT channels u = (p, 1 - p), NOT multiplied by the mask ('same' polarity; 'inverted' is u[::-1]); weight w = mask.
    compat.v1_masked_oct_channels: u = (p m, (1 - p) m) as v1. p, mask: arrays [D, H, W] at h.
    -> (u float16 [2, D, H, W], w float32 [D, H, W])."""
    raise NotImplementedError


def mri_channels(p, mask) -> np.ndarray:
    """MRI channels v = (p M, (1 - p) M); the foreground factor is kept on purpose (it carries the specimen outline).
    p, mask: arrays [D, H, W] at h. -> float16 [2, D, H, W]."""
    raise NotImplementedError


def pyramid(x, affine, factors) -> dict:
    """Box-average pyramid (geometry.box_pool) of maps x (numpy [C, D, H, W] or [D, H, W]) on a grid with voxel -> world affine;
    factors: integer box sizes relative to that grid, e.g. (4, 2, 1). -> {factor: (float32 array, affine 4x4)}, factor 1 = x."""
    raise NotImplementedError
