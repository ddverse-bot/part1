"""Interior evidence of the smooth deformation (§6, octreg.deform): block matching of a contrast-free structure feature
between the MRI and the registered OCT on the MRI base grid (docs/METHOD.md §6). The constants are Params fields (df_*).

Feature. A volume I with mask m is smoothed inside the mask by normalised convolution, s = G(I m) / G(m) (sigma df_sigma_mm), so
the mask edge adds no gradient; g = grad s, n = g / sqrt(|g|^2 + eps^2) with eps the median |g| inside the mask, and the feature
is the trace-free part of n n^T in six channels (xx, yy, zz, sqrt2 xy, sqrt2 xz, sqrt2 yz), whose dot product is the Frobenius
product. It does not change when the intensities are inverted or rescaled and changes little under any monotone remapping, so
it needs neither an intensity mapping nor the polarity.

Matching. Both features live on the MRI base grid, so no tensor has to be transported. On a grid of blocks (edge df_block_mm,
step df_step_mm, at least CORE_FRACTION inside the core: both masks eroded by df_erode_mm) the MRI block is searched in the OCT
feature within +-df_range_mm by zero-mean normalised cross-correlation of the six channels. A match is confident when its peak
stands df_z_min standard deviations above the mean of its score map and does not lie on the border of the range; the peak is
refined by a parabola per axis.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F_

from . import geometry as G
from .search import to_torch

CORE_FRACTION = 0.7        # a block is used when at least this fraction of it lies in the core
VAR_FLOOR = 1e-3           # OCT feature variance per block voxel below which a window counts as empty (typical: 0.1)
BATCH = 32                 # blocks correlated at a time (memory only)
SLAB = 1 << 21             # voxels resampled at a time (memory only)


def structure_feature(arr, mask, h_mm, sigma_mm, device="cuda"):
    """Trace-free gradient-orientation tensor of arr [D, H, W] (numpy or torch) inside mask on an isotropic grid of spacing h_mm
    (module docstring), in array axes; zero outside the mask. -> torch float32 [6, D, H, W]."""
    s = sigma_mm / h_mm
    x, m = to_torch(arr, device), to_torch(mask, device)
    x = (x - (x * m).sum(dtype=torch.float64) / m.sum(dtype=torch.float64).clamp(min=1)).to(torch.float32) * m      # offset-free: precision
    g = G.nc_gradient(x, m, s)
    del x
    mag2 = (g * g).sum(0)
    inside = mag2[m > 0]
    eps2 = inside.median() if inside.numel() else mag2.new_tensor(1.0)      # median |g|^2 = (median |g|)^2
    n = g / torch.sqrt(mag2 + eps2.clamp(min=1e-30))
    del g, mag2
    r2 = math.sqrt(2.0)
    P = torch.stack([n[0] * n[0], n[1] * n[1], n[2] * n[2], r2 * n[0] * n[1], r2 * n[0] * n[2], r2 * n[1] * n[2]])
    P[:3] -= (P[0] + P[1] + P[2]) / 3
    return P


def match(feat_m, feat_o, core, base_mm, *, block_mm, step_mm, range_mm, z_min):
    """Block matching of feat_m in feat_o (torch [6, D, H, W] on one isotropic grid of spacing base_mm) under the mask core (torch
    [D, H, W], 0 / 1): blocks of edge block_mm every step_mm (rounded to voxels) with at least CORE_FRACTION of core, the template
    (the MRI block under core, its mean removed) searched within +-range_mm (feat_o zero beyond the grid); a match is confident
    when its peak is z_min SDs above the mean of its score map and not on the border of the range.
    -> (centres [N, 3] voxel indices, displacements [N, 3] voxels (the OCT content of the block sits at centre + displacement),
    z-score of the peak [N], NCC at zero shift [N], NCC at the peak [N]) of the confident matches (numpy float64), and the number of
    blocks tried."""
    B, R, S = max(2, round(block_mm / base_mm)), max(1, round(range_mm / base_mm)), max(1, round(step_mm / base_mm))
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0), 0)
    if min(core.shape) < B:
        return empty
    core = core.to(torch.float32)
    starts = torch.nonzero(F_.avg_pool3d(core[None, None], B, S)[0, 0] >= CORE_FRACTION) * S
    if not len(starts):
        return empty
    fo, n, m, out = F_.pad(feat_o, (R,) * 6), B + 2 * R, 2 * R + 1, []
    for batch in starts.split(BATCH):
        N, cut = len(batch), lambda x, i, j, k, w: x[..., i:i + w, j:j + w, k:k + w]
        msk = torch.stack([cut(core, *c, B) for c in batch.tolist()])[:, None]
        t = torch.stack([cut(feat_m, *c, B) for c in batch.tolist()])
        f = torch.stack([cut(fo, *c, n) for c in batch.tolist()])
        cnt = msk.sum((1, 2, 3, 4))
        t = (t - (t * msk).sum((2, 3, 4), keepdim=True) / cnt.view(N, 1, 1, 1, 1)) * msk
        num = F_.conv3d(f.reshape(1, N * 6, n, n, n), t, groups=N)[0]
        sf = F_.conv3d(f.reshape(1, N * 6, n, n, n), msk.repeat(1, 6, 1, 1, 1).reshape(N * 6, 1, B, B, B), groups=N * 6)
        sff = F_.conv3d((f * f).sum(1).reshape(1, N, n, n, n), msk, groups=N)[0]
        var = (sff - (sf.reshape(N, 6, m, m, m) ** 2).sum(1) / cnt.view(N, 1, 1, 1)).clamp_min(VAR_FLOOR * cnt.view(N, 1, 1, 1))
        sc = num / ((t * t).sum((1, 2, 3, 4)).clamp_min(1e-12).sqrt().view(N, 1, 1, 1) * var.sqrt())
        flat = sc.reshape(N, -1)
        peak, arg = flat.max(1)
        z = (peak - flat.mean(1)) / flat.std(1).clamp_min(1e-12)
        p = torch.stack([arg // (m * m), arg // m % m, arg % m], 1)
        ok = (z >= z_min) & (p > 0).all(1) & (p < 2 * R).all(1)
        q, sub, rows = p.clamp(1, 2 * R - 1), p.to(torch.float32), torch.arange(N, device=sc.device)
        for ax in range(3):                                            # parabola through the peak and its two neighbours
            e = torch.zeros(3, dtype=q.dtype, device=q.device)
            e[ax] = 1
            a, c = (sc[rows, (q + s * e)[:, 0], (q + s * e)[:, 1], (q + s * e)[:, 2]] for s in (-1, 1))
            dd = a - 2 * peak + c
            sub[:, ax] += torch.where(dd < 0, 0.5 * (a - c) / dd.clamp(max=-1e-12), torch.zeros_like(dd))
        out.append(torch.cat([batch + (B - 1) / 2, sub - R, z[:, None], sc[:, R, R, R, None], peak[:, None]], 1)[ok])
    a = torch.cat(out).double().cpu().numpy()
    return a[:, :3], a[:, 3:6], a[:, 6], a[:, 7], a[:, 8], len(starts)


def _erode(mask, n):
    """Binary erosion of mask (torch [D, H, W], 0 / 1) by n steps of the 6-neighbourhood, zero beyond the grid."""
    x = F_.pad(mask[None, None], (1,) * 6)
    for _ in range(n):
        x = -torch.stack([F_.max_pool3d(-x, k, 1, p) for k, p in (((3, 1, 1), (1, 0, 0)), ((1, 3, 1), (0, 1, 0)),
                                                                  ((1, 1, 3), (0, 0, 1)))]).amax(0)
    return x[0, 0, 1:-1, 1:-1, 1:-1]


def _on_grid(fields, affine_src, shape, affine_dst, T_dst_to_src):
    """fields (torch [C, d, h, w], voxel -> world affine_src) sampled trilinearly on the grid (shape, affine_dst) through the world
    map T_dst_to_src, 0 outside (the values of geometry.resample_to), SLAB voxels at a time. -> torch [C, *shape]."""
    M = np.asarray(T_dst_to_src, float) @ np.asarray(affine_dst, float)
    out, dev = fields.new_empty((fields.shape[0], *shape)), fields.device
    j, k = (torch.arange(s, dtype=torch.float64, device=dev) for s in shape[1:])
    step = max(1, SLAB // (shape[1] * shape[2]))
    for i0 in range(0, shape[0], step):
        i = torch.arange(i0, min(shape[0], i0 + step), dtype=torch.float64, device=dev)
        out[:, i0:i0 + len(i)] = G.sample_world(fields, affine_src, G.apply_affine(M, torch.stack(torch.meshgrid(i, j, k, indexing="ij"), -1)))
    return out
