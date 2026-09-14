"""Descriptive QC: boundary agreement for whole specimens and the per-run figure. Nothing here is an accuracy measure."""
from __future__ import annotations

from pathlib import Path

from .params import Params


def boundary_agreement(oct_mask, A_O, mri_mask, A_M, T, h, params: Params = Params()):
    """Median distances between foreground boundaries, only for a whole specimen with a real outline: None unless
    >= boundary_min_interior of the OCT foreground boundary voxels lie >= boundary_face_margin_h x h from the OCT FOV faces.
    oct_mask: bool [d, h, w] at h with affine A_O; mri_mask: bool with affine A_M; T: 4x4 OCT world -> MRI world; h: mm.
    -> {'forward_median_mm' (OCT boundary through T to the MRI boundary), 'reverse_median_mm', 'interior_fraction',
        'n_boundary'} or None."""
    raise NotImplementedError


def qc_png(path, oct_h, mri_in_oct, result, oct_p=None, mri_p=None) -> Path:
    """QC figure: three orthogonal planes through the OCT foreground centroid showing the OCT at h, the MRI through T on the same
    grid, their checkerboard and the two-class contours (drawn only when oct_p / mri_p are given), plus a bar panel of the L of
    the 4 hypotheses with jackknife sigma and decided marks. oct_h, mri_in_oct, oct_p, mri_p: numpy [d, h, w] on the same OCT
    grid; result: Result or the result.json dict. -> path."""
    raise NotImplementedError


def rerender(out) -> Path:
    """`octreg qc OUT`: rebuild OUT/qc.png from OUT/result.json, OUT/mri_in_oct.nii.gz and the OCT at h in OUT/cache.
    Raises FileNotFoundError naming the missing file."""
    raise NotImplementedError
