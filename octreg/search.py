"""Global search (spec step 6): masked NCC over all translations by FFT, for every orientation in both handedness branches,
with both relative polarities read from the sign of one score map."""
from __future__ import annotations

import numpy as np

from .params import Params


class FFTSearcher:
    """Exhaustive search over orientation x translation x handedness x polarity at one pyramid level."""

    def __init__(self, mri_v, mri_mask, A_M, oct_u, oct_w, A_O, level_mm, params: Params = Params(), init=None, device="cuda"):
        """mri_v: MRI channels [2, D, H, W] on the search grid (isotropic level_mm, MRI array axes); mri_mask: MRI foreground
        fraction [D, H, W] on that grid; A_M: its voxel -> world affine (mm). oct_u: OCT channels [2, d, h, w] ('same'
        polarity); oct_w: OCT weight [d, h, w]; A_O: its affine. level_mm: grid spacing (mm). numpy or torch; moved to device.
        init: 4x4 prior pose OCT world -> MRI world, or None. With a prior, translations are restricted to prior_radius_mm of
        the prior block centre and rotations to prior_angle_deg of the prior's rotation in each handedness branch (the
        mirrored branch composes it with the mirror); polarity stays free.
        Rotations: geometry.rotations(search_n_rot, search_seed), each as R and R diag(1, 1, -1)."""
        raise NotImplementedError

    def score_map(self, R):
        """One orientation R (numpy [3, 3], det +-1): the OCT channels and weight sampled on an MRI-axis-aligned template covering
        the block's bounding sphere + search_template_pad_vox voxels, correlated with the zero-padded MRI by FFT; local MRI
        variance floored at search_var_floor x foreground variance.
        -> (S torch [D', H', W'] over translations, overlap sum(w M) / sum(w) same shape, admissible bool same shape)."""
        raise NotImplementedError

    def pose_from(self, R, index) -> np.ndarray:
        """4x4 OCT world -> MRI world (mm) of orientation R at flat translation index of score_map."""
        raise NotImplementedError

    def run(self):
        """Search every orientation: per orientation the admissible argmax of S ('same') and of -S ('inverted'); overlap
        admissible iff >= tau = search_rho min(1, V_M / V_O), raised to search_overlap_floor (flag overlap_floor) when lower.
        Per hypothesis: NMS (same pose iff centres < search_nms_mm AND rotations < search_nms_deg), top search_topk;
        decisiveness d = (s1 - s2) / (s1 - median of the per-orientation maxima), reported and never gated.
        compat.v1_retention, v1_masked_oct_channels, v1_overlap: the v1 behaviour of those steps.
        -> ({hypothesis name: [Hypothesis, best first; S = score, L = 1 - S, overlap, search_top1/top2, decisiveness, level_mm]},
            {'n_orientations', 'seconds', 'tau', 'overlap_floor': bool,
             'per_hypothesis': {name: {'top1', 'top2', 'decisiveness', 'n_admissible'}}})."""
        raise NotImplementedError
