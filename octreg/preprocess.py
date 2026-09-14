"""Preprocessing (method steps 2-3): foreground masks, the OCT specimen mask from isotropic texture (innovation 1), the
section-stripe flat field, two-class maps and channels (innovation 2). numpy/scipy on the CPU; arrays [D, H, W] on isotropic
grids of voxel_mm (mm). Exact zeros in the OCT are missing data (black tiles): never specimen, and they stay zero."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.special import expit
from skimage.filters import threshold_multiotsu, threshold_otsu

from .params import Params

_CHUNK = 64          # planes per chunk along the chunk axis: memory only, results do not depend on it
_TINY = np.finfo(np.float32).tiny


def _close(m, r):
    """Binary closing with a ball of radius r voxels on the edge-padded mask (a face that cuts the tissue is not eroded)."""
    if r < 1:
        return m
    o = np.arange(-r, r + 1) ** 2
    ball = o[:, None, None] + o[None, :, None] + o[None, None, :] <= r * r
    return ndimage.binary_closing(np.pad(m, r, mode="edge"), ball)[r:-r, r:-r, r:-r]


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
    """Foreground mask by the histogram-valley rule (MRI; the OCT intensity ablation). Positive finite values below their
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


# ----------------------------------------------------------------------------- specimen mask (innovation 1)
def specimen_mask(fine, voxel_mm, params: Params = Params()):
    """Specimen mask from isotropic texture (innovation 1): doped agarose has the intensity of tissue, but its artefacts
    (section stripes, tile seams) leave it quiet along at least one array axis while tissue texture varies along all three.
    Measured voxels m = fine > 0; band-pass b = G(fine m) / G(m) (Gaussian sigma texture_bandpass_mm); per array axis the
    running coefficient of variation of b over texture_window_mm (weights m; counted where >= half the window is measured);
    F = minimum over the axes, mean over the measured voxels of blocks of ~texture_grid_mm (blocks >= half measured);
    log F smoothed over those blocks (normalised Gaussian of sigma texture_smooth_mm: single local estimates are too noisy
    to classify), two-class Otsu threshold; closing (ball texture_close_mm), largest component, holes filled; linear
    interpolation to the fine grid > 0.5, and m. Chunked along axis 0 (equal to the whole-volume result, ~one fine float32
    array of temporaries). fine: [D, H, W] OCT intensities; voxel_mm: spacing (mm).
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
    mc = ndimage.binary_fill_holes(mc).astype(np.float32)
    out = np.empty(fine.shape, bool)
    for z0 in range(0, D, _CHUNK):
        z1 = min(D, z0 + _CHUNK)
        out[z0:z1] = (_upsample(mc, k, fine.shape, (z0, z1)) > 0.5) & (np.asarray(fine[z0:z1]) > 0)
    return out, {"threshold": float(np.exp(lt)), "volume_cm3": float(out.sum()) * voxel_mm ** 3 / 1e3,
                 "n_components": n_comp}


# ----------------------------------------------------------------------------- section-stripe flat field
def _stripe_axis(fine, mask, hp_sigma):
    """Median NCC between adjacent planes along each array axis: each plane high-passed (plane - 2-D Gaussian of hp_sigma
    voxels), NCC over the voxels in the mask in both planes, median over the pairs weighted by their voxel counts. -> [3]."""
    out = []
    for ax in range(3):
        vals, wts, prev = [], [], None
        for i in range(fine.shape[ax]):
            p = np.take(fine, i, axis=ax).astype(np.float32)
            cur = (p - ndimage.gaussian_filter(p, hp_sigma), np.take(mask, i, axis=ax) if mask is not None else p > 0)
            if prev is not None and (both := cur[1] & prev[1]).sum() > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals.append(np.corrcoef(prev[0][both], cur[0][both])[0, 1])
                wts.append(both.sum())
            prev = cur
        vals, wts = np.asarray(vals), np.asarray(wts, float)
        o = np.argsort(np.where(np.isfinite(vals), vals, np.inf))[:int(np.isfinite(vals).sum())]
        cw = np.cumsum(wts[o])
        out.append(float(vals[o][np.searchsorted(cw, 0.5 * cw[-1])]) if o.size else float("nan"))
    return out


def destripe(fine, mask, voxel_mm, params: Params = Params()):
    """Section-stripe flat field of the OCT fine grid. Sectioning axis a = lowest adjacent-plane NCC (planes high-passed in
    plane with sigma destripe_detect_mm, inside mask, None: fine > 0) if < destripe_ncc_ratio x the second lowest, else a
    recorded no-op that returns fine itself. With valid = fine > 0: A = G_s(fine valid) / G_s(valid), G_s Gaussian of sigma
    destripe_smooth_mm over the two in-sheet axes; B = G_a(A valid) / G_a(valid), G_a Gaussian of sigma destripe_highpass_mm
    along a; output = fine - (A - B) on valid voxels (floored at the smallest positive float32), zeros stay zero. Chunked along
    an in-sheet axis, equal to the whole-volume result; fine is not modified. fine: float32 [D, H, W]; mask: bool [D, H, W]
    or None; voxel_mm: spacing (mm). -> (float32 [D, H, W], {axis, applied, reason 'applied' | 'not_decisive' | 'off',
    ncc_by_axis, ratio})."""
    P = params
    info = {"axis": None, "applied": False, "reason": "off", "ncc_by_axis": None, "ratio": None}
    if not P.destripe:
        return fine, info
    ncc = _stripe_axis(fine, mask, P.destripe_detect_mm / voxel_mm)
    a0, a1 = np.argsort([v if np.isfinite(v) else np.inf for v in ncc])[:2]
    ratio = ncc[a0] / ncc[a1] if np.isfinite(ncc[a0]) and np.isfinite(ncc[a1]) and ncc[a1] > 0 else float("nan")
    info.update(ncc_by_axis=ncc, ratio=ratio, reason="not_decisive")
    if not ratio < P.destripe_ncc_ratio:
        return fine, info
    ca = 1 if a0 == 0 else 0                                   # chunk axis: an in-sheet axis
    out = np.empty(fine.shape, np.float32)
    src, dst, am = np.moveaxis(fine, ca, 0), np.moveaxis(out, ca, 0), [ca] + [d for d in range(3) if d != ca]
    sm = [0.0 if d == a0 else P.destripe_smooth_mm / voxel_mm for d in am]
    hp = [P.destripe_highpass_mm / voxel_mm if d == a0 else 0.0 for d in am]
    pad, n = int(4 * sm[0] + 0.5), src.shape[0]
    for c0 in range(0, n, _CHUNK):
        c1, a = min(n, c0 + _CHUNK), max(0, c0 - pad)
        x = np.asarray(src[a:min(n, c1 + pad)], np.float32)
        valid = x > 0
        vf = valid.astype(np.float32)
        A = np.divide(ndimage.gaussian_filter(x * vf, sm), ndimage.gaussian_filter(vf, sm), out=np.zeros_like(x), where=valid)
        B = np.divide(ndimage.gaussian_filter(A, hp), ndimage.gaussian_filter(vf, hp), out=np.zeros_like(x), where=valid)
        s = slice(c0 - a, c1 - a)
        dst[c0:c1] = np.where(valid[s], np.maximum(x[s] - (A[s] - B[s]), _TINY), 0)
    info.update(axis=int(a0), applied=True, reason="applied")
    return out, info


# ----------------------------------------------------------------------------- two-class maps and channels (innovation 2)
def two_class(arr, mask, voxel_mm, params: Params = Params(), flatten=True):
    """Soft two-class map, one rule for OCT and MRI. x = arr / local foreground mean G(arr M) / G(M) (Gaussian sigma
    flatten_sigma_mm on blocks of ~sigma / 4, linearly interpolated; skipped when flatten is False), blurred with sigma one
    voxel; foreground values clipped at p99.5 give the Otsu threshold t and std s; p = sigmoid((x - t) / (sigmoid_std s)),
    1 = bright. params.features 'intensity' (ablation): z = (x - mean) / s instead.
    arr: [D, H, W] positive intensities; mask: bool or fraction [D, H, W] (> 0.5 = foreground); voxel_mm: spacing (mm).
    -> float32 [D, H, W]: p in [0, 1], or z."""
    x, m = np.asarray(arr, np.float32), np.asarray(mask) > 0.5
    if not m.any():
        raise ValueError("two_class: empty foreground mask")
    if flatten:
        s = params.flatten_sigma_mm / voxel_mm
        k = max(1, int(s // 4))
        gn = ndimage.gaussian_filter(_pool(np.where(m, x, 0).astype(np.float64), k), s / k)
        gd = ndimage.gaussian_filter(_pool(m.astype(np.float64), k), s / k)
        x = x / _upsample(np.divide(gn, gd, out=np.ones_like(gn), where=gd > 0), k, x.shape)
    x = ndimage.gaussian_filter(x, 1.0)
    vals = x[m]
    vals = np.minimum(vals, np.percentile(vals, 99.5))
    sd = float(vals.std())
    if not sd > 0:
        raise ValueError("two_class: the foreground has no intensity variation")
    if params.features == "intensity":
        return ((x - float(vals.mean())) / sd).astype(np.float32)
    return expit((x - float(threshold_otsu(vals))) / (params.sigmoid_std * sd)).astype(np.float32)


def oct_channels(p, mask, features="two_class"):
    """OCT channels u = (p, 1 - p) ('intensity': (z, -z)), not masked; weight w = mask (bool or [0, 1], [D, H, W]).
    features: params.features. -> (u float32 [2, D, H, W], w float32 [D, H, W])."""
    p = np.asarray(p, np.float32)
    return np.stack([p, -p if features == "intensity" else 1 - p]), np.asarray(mask, np.float32)


def mri_channels(p, mask, features="two_class"):
    """MRI channels v = (p M, (1 - p) M) ('intensity': (z M, -z M)); M (bool or [0, 1]) carries the specimen outline.
    -> float32 [2, D, H, W]."""
    p, M = np.asarray(p, np.float32), np.asarray(mask, np.float32)
    return np.stack([p * M, (-p if features == "intensity" else 1 - p) * M])
