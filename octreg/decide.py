"""Decisions and status (spec steps 8-9): selection by penalised loss, paired block jackknife, scale identifiability.

No gate uses absolute NCC, restart counts, raw top1/top2 margins or corpus-fitted volume constants.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def block_partition(points, V_fg_mm3, params: Params = Params()):
    """Jackknife blocks: cubes of edge max(jk_block_min_mm, (V_fg_mm3 / jk_target_blocks)^(1/3)) mm over points (numpy [N, 3],
    OCT world mm, the foreground points at h); a cube holding < jk_merge_frac x the median point count merges into its nearest
    neighbouring cube. -> (block_id int [N] in 0..B-1, B)."""
    raise NotImplementedError


def paired_jackknife(moments_W, moments_C, prior_W, prior_C):
    """Paired block jackknife of delta = L_C - L_W, exact from moment sums (no re-optimisation).
    moments_W, moments_C: [2, B, 6] from Refiner.block_moments on the same blocks; prior_W, prior_C: prior terms (constant under
    the jackknife). delta_(-j) leaves block j out of both NCCs; sigma^2 = (B - 1) / B sum_j (delta_(-j) - mean)^2.
    -> (delta, sigma, z = delta / sigma, B)."""
    raise NotImplementedError


def select(refined, refiners, params: Params = Params()):
    """Winner and its three decisions. W = the lowest L over all refined poses (never raw S). Competitors: 'polarity' = best of
    the other polarity with W's handedness; 'handedness' = best of the other handedness; 'runner_up' = best pose inside W's
    hypothesis whose mean foreground-point displacement from W is >= max(distinct_mm, distinct_rg_frac x R_g). Each is judged
    by paired_jackknife on block_partition of W's foreground points at h: decided iff z >= jk_z and B >= jk_min_blocks.
    compat.v1_retention: the v1 selection rule. refined, refiners: as returned by refine.ladder.
    -> (W: Hypothesis, {'polarity', 'handedness', 'runner_up': Hypothesis | None}, [Decision x 3])."""
    raise NotImplementedError


def scale_profile(W, refiner, params: Params = Params()) -> dict:
    """Scale identifiability per OCT axis k: S at ls_k +- each of scale_offsets with everything else fixed; identifiable iff some
    offset gives a paired jackknife z >= jk_z on the data term against W (priors 0); saturated iff |ls_k| >= scale_saturated.
    -> {k: {'identifiable': bool, 'z': largest z, 'saturated': bool}} for k = 0, 1, 2."""
    raise NotImplementedError


def status(competitors, decisions, flags):
    """Status (spec step 9): 'ambiguous' if any decision is undecided; else 'flagged' if any flag fired; else 'ok'.
    -> (status, alternatives: the competitors of undecided decisions, lowest L first)."""
    raise NotImplementedError
