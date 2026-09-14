"""Refinement (spec step 7): penalised rigid -> similarity -> affine fit of each hypothesis, handedness fixed.

Model x_mri = R Sh diag(exp(ls)) [mirror] (x_oct - c) + t (geometry.compose), c = OCT foreground centroid (mm).
Loss L = 1 - S + refine_lambda (sum ls^2 + sum sh^2), |ls|, |sh| <= refine_clamp (absolute).
"""
from __future__ import annotations

import numpy as np

from .params import Params


class Refiner:
    """Fits one hypothesis on the level pyramid."""

    def __init__(self, mri_levels, oct_levels, hypothesis, params: Params = Params(), device="cuda"):
        """mri_levels: {factor: (v [2, D, H, W], affine 4x4)} MRI channels per pyramid level (numpy, may stay on the CPU; below
        the search level only a crop of block radius + refine_crop_mm around the start pose goes to device).
        oct_levels: {factor: (u [2, d, h, w], w [d, h, w], affine 4x4)} OCT channels ('same' polarity) and weight per level;
        the loss uses the voxels with w > refine_mask_min. hypothesis: a name in types.HYPOTHESES ('inverted' swaps the OCT
        channels; every T0 must have the hypothesis's handedness). c = weighted OCT foreground centroid at the finest level."""
        raise NotImplementedError

    def fit(self, T0, dof, iters, level):
        """Adam with a cosine schedule from T0 (4x4 numpy) at pyramid factor level; dof 'rigid' | 'similarity' (one isotropic
        log-scale) | 'affine'; learning rates refine_lr_rot (rad), refine_lr_t (mm), refine_lr_ls, refine_lr_sh; clamps applied
        after every step. -> (T 4x4 numpy of the best-L iterate, S, L) at that level."""
        raise NotImplementedError

    def evaluate(self, T, level):
        """(S, L) of pose T (4x4 numpy) at pyramid factor level, no optimisation."""
        raise NotImplementedError

    def prior(self, T) -> float:
        """refine_lambda (sum ls^2 + sum sh^2) of geometry.decompose(T, c); dimensionless."""
        raise NotImplementedError

    def points(self, level):
        """OCT points entering the loss at level: (world mm numpy [N, 3], weights [N])."""
        raise NotImplementedError

    def block_moments(self, T, level, block_id, n_blocks) -> np.ndarray:
        """Weighted moment sums per channel and block of OCT values a and MRI values b sampled through T, over points(level):
        float64 [2, n_blocks, 6] = (sum w, sum w a, sum w b, sum w a^2, sum w b^2, sum w a b). block_id: int [N] in 0..n_blocks-1."""
        raise NotImplementedError


def ladder(hypotheses, mri_levels, oct_levels, params: Params = Params(), device="cuda"):
    """The refinement ladder for every hypothesis: for each (level, ((dof, iters), ...), keep) of refine_ladder, every surviving
    pose runs the steps in order and the best `keep` by L survive (compat.v1_retention: v1_keep over the v1 joint list).
    hypotheses: {name: [Hypothesis from FFTSearcher.run]}; mri_levels, oct_levels as for Refiner.
    -> ({name: [Hypothesis at the finest level, best L first, with S, L, log_scales, shears, overlap, level_mm]},
        {name: Refiner used for that hypothesis})."""
    raise NotImplementedError
