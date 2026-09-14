"""Preprocessing (docs/METHOD.md §1-2): the MRI foreground, the OCT specimen mask from isotropic texture, two-class maps and
channels. numpy/scipy on the CPU; arrays [D, H, W] on isotropic grids of voxel_mm (mm). Exact zeros in
the OCT are missing data (black tiles): never specimen."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.special import expit
from skimage.filters import threshold_multiotsu, threshold_otsu

from .params import Params

_CHUNK = 64          # planes per chunk along axis 0: memory only, results do not depend on it


def _close(m, r):
    """Binary closing with a ball of radius r voxels on the edge-padded mask (a face that cuts the tissue is not eroded)."""
    if r < 1:
        return m
    o = np.arange(-r, r + 1) ** 2
    ball = o[:, None, None] + o[None, :, None] + o[None, None, :] <= r * r
    return ndimage.binary_closing(np.pad(m, r, mode="edge"), ball)[r:-r, r:-r, r:-r]


def _fill_planes(m):
    """Holes filled in every array plane (union over the three axes): background enclosed by specimen within a plane is specimen,
    also where the hole reaches a cut face of the block and so is not enclosed in 3-D."""
    out = m.copy()
    for ax in range(3):
        s = np.zeros((3, 3, 3), bool)
        s[tuple(1 if d == ax else slice(None) for d in range(3))] = ndimage.generate_binary_structure(2, 1)
        out |= ndimage.binary_fill_holes(m, structure=s)
    return out


def _components(m, min_fraction=None):
    """6-connected components of m -> (mask of the largest component, or of every component holding >= min_fraction of the
    mask voxels; number of components before the rule)."""
    lab, n = ndimage.label(m)
    size = np.bincount(lab.ravel())
    size[0] = 0
    keep = (np.arange(n + 1) == np.argmax(size)) if min_fraction is None else size >= min_fraction * size.sum()
    keep[0] = False
    return keep[lab], int(n)


def _pool(a, k):
    """Sums over k^3 blocks of a [D, H, W] zero-padded to multiples of k (block j covers voxels j k .. j k + k - 1)."""
    a = np.pad(a, [(0, -s % k) for s in a.shape])
    D, H, W = (s // k for s in a.shape)
    return a.reshape(D, k, H, k, W, k).sum(axis=(1, 3, 5))


def _upsample(a, k, shape, rows=None):
    """Linear interpolation of a block grid a (block centres at j k + (k - 1) / 2 voxels, clamped beyond the outer centres) at
    the voxels of a grid of `shape`; rows = (lo, hi) returns only rows lo:hi along axis 0. -> float32."""
    x = np.asarray(a, np.float32)
    for ax, n in enumerate(shape):
        c = np.clip((np.arange(n) - (k - 1) / 2) / k, 0, a.shape[ax] - 1)[slice(*rows) if ax == 0 and rows else slice(None)]
        i = np.floor(c).astype(int)
        f = (c - i).astype(np.float32).reshape([-1 if d == ax else 1 for d in range(3)])
        x = np.take(x, i, axis=ax) * (1 - f) + np.take(x, np.minimum(i + 1, a.shape[ax] - 1), axis=ax) * f
    return x


# ----------------------------------------------------------------------------- foreground (histogram-valley rule)
def foreground(arr, voxel_mm, params: Params = Params()):
    """Foreground mask by the histogram-valley rule (the MRI foreground). Positive finite values below their
    p99.5, 256-bin histogram smoothed over 5 bins, peaks of its square root (so the narrow background peak of a box-averaged
    grid cannot hide the tissue peak) of prominence >= 5 % of the maximum and >= 5 bins apart; walking left
    from the rightmost peak, t = the first valley < valley_ratio x the smaller of its two peaks (status 'ok'), else
    t = min(p1, 0.5 x lower cut of a 3-class multi-Otsu) (status 'no_valley': the whole field of view). Mask = arr > t, closing
    (ball of 2 voxels), holes filled, components >= min_component of the foreground kept. arr: [D, H, W]; voxel_mm: spacing.
    -> (bool [D, H, W], {threshold (arr units), status, valley_ratio, volume_cm3, n_components})."""
    v = np.asarray(arr, np.float32)
    s = v[np.isfinite(v) & (v > 0)]
    s = s[s < np.percentile(s, 99.5)] if s.size else s
    if s.size == 0:
        raise ValueError("foreground: the image has no positive intensity variation")
    h, e = np.histogram(s, bins=256)
    c, hs = 0.5 * (e[1:] + e[:-1]), ndimage.uniform_filter1d(h.astype(float), 5)
    pk = find_peaks(np.sqrt(hs), prominence=0.05 * np.sqrt(hs.max()), distance=5)[0]
    t, status, ratio = None, "no_valley", None
    for j, lo in enumerate(pk[:-1][::-1]):
        hi = pk[len(pk) - 1 - j]
        val = lo + int(np.argmin(hs[lo:hi + 1]))
        r = float(hs[val] / max(min(hs[lo], hs[hi]), 1.0))
        if r < params.valley_ratio:
            t, status, ratio = float(c[val]), "ok", r
            break
    if t is None:
        t = float(min(np.percentile(s, 1), 0.5 * threshold_multiotsu(s, classes=3)[0]))
    m = ndimage.binary_fill_holes(_close(v > t, 2))
    m, n = _components(m, params.min_component)
    return m, {"threshold": t, "status": status, "valley_ratio": ratio, "volume_cm3": float(m.sum()) * voxel_mm ** 3 / 1e3,
               "n_components": n}


# ----------------------------------------------------------------------------- specimen mask (§1)
def specimen_mask(fine, voxel_mm, params: Params = Params()):
    """Specimen mask from isotropic texture: doped agarose has the intensity of tissue, but its artefacts
    (section stripes, tile seams) leave it quiet along at least one array axis while tissue texture varies along all three.
    Measured voxels m = fine > 0; band-pass b = G(fine m) / G(m) (Gaussian sigma texture_bandpass_mm); per array axis the
    running coefficient of variation of b over texture_window_mm (weights m; counted where >= half the window is measured);
    F = minimum over the axes, mean over the measured voxels of blocks of ~texture_grid_mm (blocks >= half measured);
    log F smoothed over those blocks (normalised Gaussian of sigma texture_smooth_mm: single local estimates are too noisy
    to classify), two-class Otsu threshold; closing (ball texture_close_mm), largest component, holes filled in every array
    plane (_fill_planes: uniform tissue such as white matter has little texture, the specimen is solid); linear interpolation to
    the fine grid > 0.5, and m. Chunked along axis 0 (equal to the whole-volume result, ~one fine float32 array of temporaries).
    fine: [D, H, W] OCT intensities; voxel_mm: spacing (mm).
    -> (bool [D, H, W], {threshold (F units), volume_cm3, n_components (before the largest-component rule)})."""
    P, D = params, fine.shape[0]
    sig = P.texture_bandpass_mm / voxel_mm
    win = int(round(P.texture_window_mm / voxel_mm)) // 2 * 2 + 1
    k = max(1, int(round(P.texture_grid_mm / voxel_mm)))
    pad, step = int(4 * sig + 0.5) + win // 2, max(1, _CHUNK // k) * k
    num = np.zeros([-(-s // k) for s in fine.shape])
    cnt = np.zeros_like(num)
    with np.errstate(divide="ignore", invalid="ignore"):
        for z0 in range(0, D, step):
            a, z1 = max(0, z0 - pad), min(D, z0 + step)
            x = np.asarray(fine[a:min(D, z1 + pad)], np.float32)
            m = (x > 0).astype(np.float32)
            x = np.divide(ndimage.gaussian_filter(x * m, sig), ndimage.gaussian_filter(m, sig), out=np.zeros_like(x), where=m > 0)
            F = np.full(x.shape, np.inf, np.float32)
            for ax in range(3):
                n = ndimage.uniform_filter1d(m, win, axis=ax)
                mu = ndimage.uniform_filter1d(x, win, axis=ax) / n
                cv = ndimage.uniform_filter1d(x * x, win, axis=ax) / n - mu * mu
                cv = np.sqrt(np.maximum(cv, 0)) / mu
                np.minimum(F, np.where(n >= 0.5, cv, np.inf), out=F)
                del n, mu, cv
            ok = (m > 0) & np.isfinite(F)
            core = slice(z0 - a, z1 - a)
            pf, pc = _pool(np.where(ok, F, 0)[core], k), _pool(ok[core].astype(np.float32), k)
            num[z0 // k:z0 // k + len(pf)], cnt[z0 // k:z0 // k + len(pc)] = pf, pc
            del x, m, F, ok
        real = np.einsum("i,j,k->ijk", *[np.minimum(k, s - k * np.arange(-(-s // k))) for s in fine.shape])
        F = num / cnt
    valid = (cnt >= 0.5 * real) & (F > 0)
    w, s = valid.astype(np.float64), P.texture_smooth_mm / (k * voxel_mm)
    L = np.divide(ndimage.gaussian_filter(np.log(np.where(valid, F, 1.0)) * w, s), ndimage.gaussian_filter(w, s),
                  out=np.zeros_like(w), where=valid)
    if not (valid.sum() > 1 and L[valid].std() > 0):
        raise ValueError("specimen mask: no texture variation in the measured data; supply an OCT mask")
    lt = float(threshold_otsu(L[valid]))
    mc, n_comp = _components(_close(valid & (L > lt), int(round(P.texture_close_mm / (k * voxel_mm)))))
    mc = _fill_planes(mc).astype(np.float32)
    out = np.empty(fine.shape, bool)
    for z0 in range(0, D, _CHUNK):
        z1 = min(D, z0 + _CHUNK)
        out[z0:z1] = (_upsample(mc, k, fine.shape, (z0, z1)) > 0.5) & (np.asarray(fine[z0:z1]) > 0)
    return out, {"threshold": float(np.exp(lt)), "volume_cm3": float(out.sum()) * voxel_mm ** 3 / 1e3,
                 "n_components": n_comp}


# ----------------------------------------------------------------------------- two-class maps and channels (§2)
def flattened(arr, mask, voxel_mm, params: Params = Params()):
    """arr / its local foreground mean G(arr M) / G(M), Gaussian sigma flatten_sigma_mm on blocks of ~sigma / 4 voxels, linearly
    interpolated (removes slow multiplicative intensity changes). arr: [D, H, W]; mask: bool [D, H, W]. -> float32."""
    x, s = np.asarray(arr, np.float32), params.flatten_sigma_mm / voxel_mm
    k = max(1, int(s // 4))
    gn = ndimage.gaussian_filter(_pool(np.where(mask, x, 0).astype(np.float64), k), s / k)
    gd = ndimage.gaussian_filter(_pool(mask.astype(np.float64), k), s / k)
    return x / _upsample(np.divide(gn, gd, out=np.ones_like(gn), where=gd > 0), k, x.shape)


def two_class(arr, mask, voxel_mm, params: Params = Params(), flatten=True):
    """Soft two-class map, one rule for OCT and MRI: x = flattened(arr) (arr itself when flatten is False), blurred with sigma
    one voxel; foreground values clipped at p99.5 give the Otsu threshold t and std s; p = sigmoid((x - t) / (sigmoid_std s)),
    1 = bright. arr: [D, H, W] positive intensities; mask: bool or fraction [D, H, W] (> 0.5 = foreground); voxel_mm: spacing
    (mm). -> float32 [D, H, W] in [0, 1]."""
    m = np.asarray(mask) > 0.5
    if not m.any():
        raise ValueError("two_class: empty foreground mask")
    x = ndimage.gaussian_filter(flattened(arr, m, voxel_mm, params) if flatten else np.asarray(arr, np.float32), 1.0)
    vals = np.minimum(x[m], np.percentile(x[m], 99.5))
    sd = float(vals.std())
    if not sd > 0:
        raise ValueError("two_class: the foreground has no intensity variation")
    return expit((x - float(threshold_otsu(vals))) / (params.sigmoid_std * sd)).astype(np.float32)


def oct_channels(p, mask):
    """OCT channels u = (p, 1 - p), not masked, and weight w = mask (bool or [0, 1], [D, H, W]): swapping the channels negates
    every weighted NCC against MRI channels, which is how the polarity is read. -> (u float32 [2, D, H, W], w float32 [D, H, W])."""
    p = np.asarray(p, np.float32)
    return np.stack([p, 1 - p]), np.asarray(mask, np.float32)


def mri_channels(p, mask):
    """MRI channels v = (p M, (1 - p) M); M (bool or [0, 1]) carries the specimen outline. -> float32 [2, D, H, W]."""
    p, M = np.asarray(p, np.float32), np.asarray(mask, np.float32)
    return np.stack([p * M, (1 - p) * M])
