"""Label-free vascular channel for sub-mm refinement (real data only, no annotations).

Why: the tissue-class geometry (WM/GM) of a cortical block is nearly invariant to a scale along the block's depth
axis and to a shift along a sulcus, so structural NCC / label Dice cannot pin those down (see scripts/scale_probe.py);
vessels can.  Nearest-vessel distances between *automatic* candidates are ambiguous (dense-vs-dense), but the
correlation of a MRI dark-blob map with the OCT vessel *density* is a proper similarity: where the OCT has many vessels
the MRI is darker.  Used as a third NCC channel next to [WM, GM] in the affine refinement.

    MRI channel : local-median darkness of the 0.15 mm MRI inside tissue (or Frangi dark-tube vesselness)
    OCT channel : Frangi dark-tube vesselness of the OCT at 24-48 um, top-q inside tissue -> binary -> density on the
                  0.15 mm OCT grid (or the density of a provided vessel segmentation, e.g. the DANDI ves_seg)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .common import to_t, apply_affine, sample_at_world, avg_pool_iso, grid_points, DEVICE
from .features import frangi_dark_vesselness
from .refine import Refiner


def mri_dark_channel(mri_reg: np.ndarray, tissue_reg: np.ndarray, size: int | None = None, q: float = 99.5, vox_mm: float = 0.15,
                     window_mm: float = 1.35, device=DEVICE) -> torch.Tensor:
    """Darkness relative to the local median (vessels are dark blobs/tubes in ex-vivo FLASH), inside tissue, scaled to [0,1].
    The median window is a physical size (window_mm, default 1.35 mm = 9 voxels at 0.15 mm)."""
    if size is None:
        size = max(3, int(round(window_mm / vox_mm)) | 1)
    x = mri_reg.astype(np.float32)
    if x.size > 4e7 and size >= 7:
        # the local median is a low-frequency background field: compute it on a half-resolution grid and upsample
        med2 = ndimage.median_filter(x[::2, ::2, ::2], size=max(3, (size // 2) | 1))
        loc = ndimage.zoom(med2, [x.shape[i] / med2.shape[i] for i in range(3)], order=1, mode="nearest")
    else:
        loc = ndimage.median_filter(x, size=size)
    dark = np.clip(loc - x, 0, None) * tissue_reg
    dark = dark / max(float(np.percentile(dark[tissue_reg > 0], q)), 1e-6)
    return to_t(np.clip(dark, 0, 1).astype(np.float32), device=device)


def mri_frangi_channel(mri_reg: np.ndarray, tissue_reg: np.ndarray, q: float = 99.5, sigmas=(1.0, 1.5, 2.2), device=DEVICE) -> torch.Tensor:
    fr = frangi_dark_vesselness(to_t(mri_reg, device=device)[None], sigmas_vox=sigmas)[0] * to_t(tissue_reg.astype(np.float32), device=device)
    t = to_t(tissue_reg.astype(bool), device=device, dtype=torch.bool)
    k = fr[t].flatten().kthvalue(int(q / 100.0 * int(t.sum()))).values.clamp(min=1e-6)
    return (fr / k).clamp(0, 1)


@torch.no_grad()
def frangi_chunked(vol: torch.Tensor, sigmas=(1.0, 1.5, 2.2), zchunk: int = 24, overlap: int = 10, gamma=None) -> torch.Tensor:
    """Frangi dark-tube vesselness of a large [D,H,W] volume, processed in overlapping z-slabs (GPU memory)."""
    D = vol.shape[0]
    out = torch.zeros_like(vol)
    for z0 in range(0, D, zchunk):
        a, b = max(0, z0 - overlap), min(D, z0 + zchunk + overlap)
        v = frangi_dark_vesselness(vol[a:b][None], sigmas_vox=sigmas, gamma=gamma)[0]
        out[z0:min(D, z0 + zchunk)] = v[z0 - a:z0 - a + min(zchunk, D - z0)]
    return out


@torch.no_grad()
def oct_vessel_mask(oct_lvl: np.ndarray, tissue_lvl: np.ndarray, q: float = 99.0, sigmas=(1.0, 1.5, 2.2), device=DEVICE) -> torch.Tensor:
    """Automatic OCT vessel mask (bool [D,H,W]) at the given level: Frangi dark-tube vesselness, top-q % inside tissue."""
    v = frangi_chunked(to_t(oct_lvl, device=device), sigmas)
    t = to_t(tissue_lvl.astype(bool), device=device, dtype=torch.bool)
    thr = v[t].flatten().kthvalue(int(q / 100.0 * int(t.sum()))).values
    return (v > thr) & t


@torch.no_grad()
def density_on_grid(mask: torch.Tensor, A_mask: np.ndarray, A_target: np.ndarray, target_shape, target_mask: np.ndarray,
                    pool: int, q: float = 99.5, device=DEVICE) -> torch.Tensor:
    """Vessel density: pool a binary vessel mask by `pool` (-> roughly the target spacing), sample it on the target grid,
    scale to [0,1] (q-th percentile inside the target tissue mask), zero outside the target mask."""
    dens, A_d = avg_pool_iso(mask.float()[None], A_mask, pool)
    pts = grid_points(to_t(A_target, device=device), target_shape).reshape(-1, 3)
    d = sample_at_world(dens, to_t(A_d, device=device), pts)[0].reshape(tuple(target_shape))
    tm = to_t(target_mask.astype(bool), device=device, dtype=torch.bool)
    k = d[tm].flatten().kthvalue(int(q / 100.0 * int(tm.sum()))).values.clamp(min=1e-6)
    return (d / k).clamp(0, 1) * tm.float()


def vascular_refine(T0: np.ndarray, FM2: torch.Tensor, A_reg: np.ndarray, FO2: torch.Tensor, A_oct: np.ndarray, MO: torch.Tensor,
                    mri_v: torch.Tensor, oct_v: torch.Tensor, w: float = 1.0, reg: float = 0.5, clamp: float = 0.3,
                    iters: int = 300, factors=(2, 1), device=DEVICE):
    """Affine refinement from T0 with channels [WM, GM, vessel] (channel weights 1,1,w), coarse-to-fine over `factors`.
    Returns (T, info) with per-channel NCC at the finest level."""
    FM = torch.cat([FM2, mri_v[None]], 0); FO = torch.cat([FO2, oct_v[None]], 0)
    T = np.asarray(T0).copy(); ref = None
    for f in factors:
        if f == 1:
            ref = Refiner(FM, A_reg, FO, A_oct, MO, device=device, chan_w=[1.0, 1.0, w])
        else:
            a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(MO, A_oct, f)
            ref = Refiner(a1, A1, a2, A2, a3, device=device, chan_w=[1.0, 1.0, w])
        T, loss = ref.refine(T, dof="affine", iters=iters, ls_clamp=clamp, sh_clamp=clamp, reg=reg)
    return T, {"loss": float(loss), "ncc_channels": ref.ncc_channels(T)}
