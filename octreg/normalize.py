"""Intensity normalisation: foreground threshold and mask (both modalities), sectioning gain (OCT), bias flattening."""
from __future__ import annotations

import numpy as np

from .params import Params
from .types import Volume


def foreground_sample(reader, params: Params = Params()) -> np.ndarray:
    """Positive finite values of every fg_sample_stride-th voxel per axis of an array-like [D, H, W] (memmap or plane reader,
    read plane by plane), in native units. -> float32 [N]."""
    raise NotImplementedError


def foreground_threshold(sample, params: Params = Params()):
    """Foreground threshold (spec step 2), the same rule for OCT and MRI.

    sample: intensities (any shape, native units; non-positive and non-finite values ignored), clipped at fg_clip_percentile.
    Histogram of fg_bins bins, running mean over fg_smooth_bins; from the rightmost peak (prominence >= fg_prominence x max,
    distance >= fg_peak_distance_bins) walk left to the first valley whose depth ratio (valley / smaller neighbouring peak) is
    < fg_valley_ratio and threshold there: status 'ok'. No such valley: t = min(p[fg_fallback_percentile],
    fg_fallback_otsu_factor x first cut of a fg_fallback_otsu_classes-class multi-Otsu), status 'no_valley' (the caller sets
    flag foreground_no_valley:<modality>; the run continues).
    -> (t in sample units, {'valley_ratio': float | None, 'status': 'ok' | 'no_valley', 'n_peaks': int, 'peaks': [float]}).
    """
    raise NotImplementedError


def section_gain(volume: Volume, t, params: Params = Params()):
    """Sectioning gain (spec step 3, OCT only), decided by the data; a recorded no-op when the data do not show sectioning.

    volume: the OCT at native resolution (read plane by plane); t: its step-2 threshold (native units).
    Detection on the OCT box-pooled to >= section_detect_mm per axis. Axis: per array axis, median NCC over section_n_pairs
    adjacent-plane pairs (planes minus a Gaussian of section_hp_sigma_vox; outer section_skip_frac skipped; a pair needs
    >= min(section_pair_min_voxels, section_pair_min_frac x plane) voxels > t in both planes); a* = lowest median, decisive
    iff < section_ncc_ratio x the second lowest. Period P: FFT peak of the per-plane median m_k of voxels > t along a*, after
    a section_detrend_mm running-mean detrend, searched in section_period_mm, accepted iff peak >= section_snr x median power.
    Applied iff decisive and accepted: g_k = m_k / (running mean of m over section_window_periods x P), per native plane k;
    planes with < section_min_plane_voxels voxels > t get g = 1.
    compat.v1_sectioning: v1 slab normalisation instead (axis 0, window max(v1_slab_window_min, round(v1_slab_window_um /
    spacing_0 um)), its own raw threshold v1_slab_otsu_factor x Otsu of raw[::v1_slab_stride]; t unused).
    -> (gain float64 [n planes along info['axis']] (divide plane k by gain[k]) or None when not applied,
        {'axis': int | None, 'ncc_by_axis': [3], 'ratio', 'period_mm', 'snr', 'window_planes', 'applied': bool, 'reason': str}).
    """
    raise NotImplementedError


def apply_gain(volume: Volume, gain, axis, params: Params = Params()) -> Volume:
    """Volume whose reader returns float32 planes divided by gain[k] along axis, lazily, for geometry.resample (gain None:
    the input volume unchanged). compat.v1_sectioning: v1 storage as well (x v1_f16_gain / median reference, clip to [0, v1_f16_cap])."""
    raise NotImplementedError


def foreground_mask(img_h, t, h, modality, params: Params = Params()):
    """Foreground mask at h (spec step 4): img_h > t, closing with a ball of radius cleanup_closing_h x h, holes filled, every
    connected component >= cleanup_min_component of the foreground volume kept (multi-piece blocks allowed; no largest-component
    rule). img_h: float array [D, H, W] on the h grid (gained, pooled); t: threshold in img_h units (foreground_threshold on
    img_h); h: mm; modality: 'oct' | 'mri' (recorded; the release rule is the same for both).
    compat.v1_cleanup: OCT = v1_oct_closing_iter closing iterations + fill + largest component; MRI = img_h > t only.
    -> (bool [D, H, W], {'volume_cm3', 'n_components', 'modality'})."""
    raise NotImplementedError


def flatten(img, mask, h, params: Params = Params()):
    """Bias flattening (spec step 5): img / max(G(img mask) / G(mask), flat_floor_factor x p[flat_floor_percentile] of that
    field where the pooled mask fraction > flat_support_min), G a Gaussian of sigma flat_sigma_mm computed on the
    flat_level_h x h grid and trilinearly upsampled. img: float array [D, H, W] at h; mask: bool or [0, 1] same shape; h: mm.
    -> float32 [D, H, W], dimensionless (foreground local mean about 1)."""
    raise NotImplementedError
