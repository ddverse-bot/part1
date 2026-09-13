"""Serial-section stripe removal for the OCT vessel-level volume (P3 recipe, octreg v1.1).

The sectioning artefact of serial-section OCT is a set of intensity sheets of constant array index along ONE array axis
(the physical sectioning axis; I58: axis 2, period 7.5 vox = 0.30 mm = 15 sections at 20 um).  The recipe is an additive
flat field by normalised convolution: smooth over the two in-sheet axes (sigma 0.48 mm), high-pass along the stripe axis
(sigma 0.24 mm), subtract.  Zeros are missing data (black tiles) and stay exactly zero; everything is done chunk-wise in
place on the float16 array with a fork Pool.  All physical sizes are in mm; ``vox_mm`` is the (isotropic) voxel size.
"""
from __future__ import annotations

import math
import os
import time
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import numpy as np
from scipy import ndimage

CAP_F16 = 60000.0            # fp16 cap of the slab-normalised OCT (a local in common.oct_slab_normalize; re-declared here)
TRUNC = 3.0                  # Gaussian truncation (sigma units) of every filter below
DETREND_MM = 2.4             # running-mean window of the per-plane profile detrend (60 planes at 0.04 mm, 16 at 0.15 mm)
MIN_PLANE_COUNT = 2000       # a plane enters the profile only with at least this many mask voxels
_CTX: dict = {}              # per-process context for the fork workers (set by destripe_inplace before the Pool starts)


# ----------------------------------------------------------------------------- per-plane profiles and their statistics
def plane_profiles(vol: np.ndarray, mask: np.ndarray | None = None, chunk: int = 32):
    """One chunked pass over (Z,Y,X): per-plane sum/count inside ``mask`` (``vol > 0`` when None) along all three axes plus
    global validity statistics -> (means [3], counts [3], {n_zero, n_total, min_nonzero, max, n_nonfinite})."""
    n = vol.shape
    s = [np.zeros(n[i], np.float64) for i in range(3)]; c = [np.zeros(n[i], np.float64) for i in range(3)]
    n_zero = 0; n_bad = 0; vmin = np.inf; vmax = -np.inf
    for z0 in range(0, n[0], chunk):
        z1 = min(n[0], z0 + chunk)
        blk = np.asarray(vol[z0:z1]).astype(np.float32)
        fin = np.isfinite(blk); n_bad += int((~fin).sum()); blk = np.where(fin, blk, 0.0)
        zero = blk == 0; n_zero += int(zero.sum())
        nz = blk[~zero]
        if nz.size: vmin = min(vmin, float(nz.min())); vmax = max(vmax, float(nz.max()))
        m = (~zero) if mask is None else (np.asarray(mask[z0:z1]) & ~zero)
        v = np.where(m, blk, 0.0)
        s[0][z0:z1] += v.sum((1, 2)); c[0][z0:z1] += m.sum((1, 2))
        s[1] += v.sum((0, 2)); c[1] += m.sum((0, 2)); s[2] += v.sum((0, 1)); c[2] += m.sum((0, 1))
    means = [np.where(ci >= MIN_PLANE_COUNT, si / np.maximum(ci, 1), np.nan) for si, ci in zip(s, c)]
    g = {"n_zero": n_zero, "n_total": int(np.prod(n)), "min_nonzero": (vmin if np.isfinite(vmin) else None),
         "max": (vmax if np.isfinite(vmax) else None), "n_nonfinite": n_bad}
    return means, c, g


def profile_residual(p: np.ndarray, win: int):
    """Detrended per-plane profile (running mean of ``win`` planes removed over the valid range, gaps interpolated) and its
    Hann-windowed spectrum -> (dict{resid_std, p2p95, fft_period_vox, fft_peak_frac, fft_peak_over_median, n_planes,
    range}, residual array, first valid index).  Frequencies below 1/win are ignored (they are removed by the detrend)."""
    ok = np.isfinite(p); ii = np.where(ok)[0]
    if ii.size < max(8, win // 2):
        return {"resid_std": None, "p2p95": None, "fft_period_vox": None, "fft_peak_frac": None, "fft_peak_over_median": None,
                "n_planes": int(ii.size), "range": None}, np.zeros(0), 0
    i0, i1 = int(ii.min()), int(ii.max()) + 1
    pp = p[i0:i1].astype(np.float64).copy(); okk = ok[i0:i1]
    if (~okk).any(): pp[~okk] = np.interp(np.where(~okk)[0], np.where(okk)[0], pp[okk])
    d = pp - ndimage.uniform_filter1d(pp, int(win), mode="nearest")
    n = d.size; f = np.fft.rfftfreq(n); P = np.abs(np.fft.rfft(d * np.hanning(n))) ** 2; band = f >= 1.0 / win
    P = np.where(band, P, 0.0)
    if band.sum() < 3 or P.max() <= 0:
        per, frac, snr = None, None, None
    else:
        k = int(np.argmax(P)); per = float(1.0 / f[k]); frac = float(P[k] / P.sum())
        med = float(np.median(P[band])); snr = float(P[k] / med) if med > 0 else float("inf")
    return {"resid_std": float(d.std()), "p2p95": float(np.percentile(d, 97.5) - np.percentile(d, 2.5)), "fft_period_vox": per,
            "fft_peak_frac": frac, "fft_peak_over_median": snr, "n_planes": int(ii.size), "range": [i0, i1]}, d, i0


def profile_stats(vol: np.ndarray, mask: np.ndarray | None, vox_mm: float, detrend_mm: float = DETREND_MM):
    """Per-axis detrended profile statistics of ``vol`` inside ``mask`` -> ([3] dicts of profile_residual, global stats, win)."""
    win = max(8, int(round(detrend_mm / vox_mm)))
    means, counts, g = plane_profiles(vol, mask)
    return [profile_residual(means[ax], win)[0] for ax in range(3)], g, win


# ----------------------------------------------------------------------------- stripe axis detection
def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-9))


def detect_stripe_axis(vol: np.ndarray, mask: np.ndarray | None = None, n_planes: int = 40, hp_sigma_vox: float = 3.0, ratio_max: float = 0.8):
    """Which array axis carries the section sheets?  For each axis: ``n_planes`` evenly spaced adjacent-plane pairs (outer 5 %
    skipped), each plane high-passed (plane - G(hp_sigma_vox)), NCC over voxels where both planes are in ``mask`` (``> 0``
    when None); median over pairs.  The stripe axis decorrelates adjacent planes: axis = argmin if the smallest median is
    < ratio_max x the second smallest, else None.  Also the FFT period of the per-plane mean profile along the chosen axis
    (60-plane running-mean detrend, Hann; returned only when the peak is >= 3x the spectrum median).
    -> (axis | None, {ncc_median_per_axis, ncc_p5_per_axis, ratio, period_vox, fft_peak_over_median, n_pairs_per_axis,
    profile_resid_std_per_axis, fft_period_per_axis})"""
    n = vol.shape; med = []; p5 = []; npairs = []
    for ax in range(3):
        L = n[ax]; lo, hi = 0.05 * L, 0.95 * L - 1
        idx = np.unique(np.clip(np.round(np.linspace(lo, hi, n_planes)).astype(int), 0, L - 2))
        vals = []
        for i in idx:
            p0 = np.take(vol, i, axis=ax).astype(np.float32); p1 = np.take(vol, i + 1, axis=ax).astype(np.float32)
            m = (np.take(mask, i, axis=ax) & np.take(mask, i + 1, axis=ax)) if mask is not None else ((p0 > 0) & (p1 > 0))
            if m.sum() < min(5000, 0.05 * m.size): continue
            hp0 = p0 - ndimage.gaussian_filter(p0, hp_sigma_vox, truncate=TRUNC); hp1 = p1 - ndimage.gaussian_filter(p1, hp_sigma_vox, truncate=TRUNC)
            vals.append(_ncc(hp0[m], hp1[m]))
        med.append(float(np.median(vals)) if vals else float("nan")); p5.append(float(np.percentile(vals, 5)) if vals else float("nan")); npairs.append(len(vals))
    order = np.argsort([v if np.isfinite(v) else np.inf for v in med])
    a0, a1 = int(order[0]), int(order[1])
    ratio = float(med[a0] / med[a1]) if (np.isfinite(med[a0]) and np.isfinite(med[a1]) and med[a1] > 1e-6) else float("nan")
    axis = a0 if (np.isfinite(ratio) and ratio < ratio_max) else None
    means, counts, _ = plane_profiles(vol, mask)
    prof = [profile_residual(means[ax], 60)[0] for ax in range(3)]
    period = None; snr = None
    if axis is not None:
        snr = prof[axis]["fft_peak_over_median"]
        if snr is not None and snr >= 3.0: period = prof[axis]["fft_period_vox"]
    info = {"ncc_median_per_axis": med, "ncc_p5_per_axis": p5, "ratio": ratio, "ratio_max": ratio_max, "period_vox": period,
            "fft_peak_over_median": snr, "n_pairs_per_axis": npairs, "hp_sigma_vox": hp_sigma_vox,
            "profile_resid_std_per_axis": [q["resid_std"] for q in prof], "fft_period_per_axis": [q["fft_period_vox"] for q in prof],
            "fft_peak_over_median_per_axis": [q["fft_peak_over_median"] for q in prof]}
    return axis, info


# ----------------------------------------------------------------------------- flat field (chunked, in place)
def _masked_lp(img, m, sig):
    """Normalised convolution: G(sig)(img*m) / G(sig)(m); ok where the denominator > 0.05 (sig entries of 0 skip that axis)."""
    num = ndimage.gaussian_filter(np.where(m, img, 0.0).astype(np.float32), sig, truncate=TRUNC)
    den = ndimage.gaussian_filter(m.astype(np.float32), sig, truncate=TRUNC)
    ok = den > 0.05
    return np.where(ok, num / np.maximum(den, 1e-6), 0.0).astype(np.float32), ok


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"                   # belt: the parent already exported OMP_NUM_THREADS=1 before the fork
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


class _single_thread:
    """Context: export OMP_NUM_THREADS=1 (inherited by the forked workers) and, when threadpoolctl is installed, limit the
    already-initialised BLAS/OpenMP pools of the parent to 1 thread (the children inherit that state at fork); restores both."""
    def __enter__(self):
        self.omp0 = os.environ.get("OMP_NUM_THREADS"); os.environ["OMP_NUM_THREADS"] = "1"; self.tp = None
        try:
            from threadpoolctl import threadpool_limits
            self.tp = threadpool_limits(1)
        except Exception:
            pass
        return self
    def __exit__(self, *exc):
        if self.tp is not None:
            try: self.tp.unregister()
            except Exception: pass
        if self.omp0 is None: os.environ.pop("OMP_NUM_THREADS", None)
        else: os.environ["OMP_NUM_THREADS"] = self.omp0
        return False


def _destripe_chunk(job):
    """One padded chunk along the chunk axis: S = A - B (A: in-sheet smoothing, B: low-pass of A along the stripe axis);
    v' = where(valid, clip(v - S, 1, cap), 0).  Returns the un-padded core in the volume dtype (+ std of S inside valid)."""
    c0, c1 = job; C = _CTX; vol = C["vol"]; ca = C["chunk_axis"]; pad = C["pad"]; n = vol.shape[ca]
    a, b = max(0, c0 - pad), min(n, c1 + pad)
    sl = [slice(None)] * 3; sl[ca] = slice(a, b); sl = tuple(sl)
    blk = np.asarray(vol[sl]).astype(np.float32); valid = blk > 0
    A_, ok = _masked_lp(blk, valid, C["sig"]); B_, _ = _masked_lp(A_, ok, C["sig_hp"])
    S = np.where(ok, A_ - B_, 0.0).astype(np.float32); del A_, B_, ok
    out = np.where(valid, np.clip(blk - S, 1.0, C["cap"]), 0.0)
    core = [slice(None)] * 3; core[ca] = slice(c0 - a, c1 - a); core = tuple(core)
    Sc = S[core]; vc = valid[core]
    s_std = float(Sc[vc].std()) if vc.any() else 0.0
    if C["save_S"] is not None:
        g = [slice(None)] * 3; g[ca] = slice(c0, c1)
        sm = np.load(C["save_S"], mmap_mode="r+"); sm[tuple(g)] = Sc.astype(np.float16); sm.flush(); del sm
    return c0, c1, out[core].astype(vol.dtype), s_std


def destripe_inplace(vol: np.ndarray, axis: int, vox_mm: float, sigma_smooth_mm: float = 0.48, sigma_hp_mm: float = 0.24, cap: float = CAP_F16,
                     zchunk: int = 64, pad: int = 40, n_workers: int = 4, mask: np.ndarray | None = None, save_S: Path | None = None) -> dict:
    """Additive x-flat-field destripe of ``vol`` (Z,Y,X float16) IN PLACE along array ``axis``: per chunk of ``zchunk`` slices
    along array axis 0 (axis 1 when the stripe axis is 0) with ``pad`` slices each side, v = chunk float32, valid = v > 0,
    A = G(sig)(v*valid)/G(sig)(valid) with sig = sigma_smooth_mm/vox_mm on the two non-stripe axes (0 along axis),
    B = G(sigma_hp_mm/vox_mm along axis only)(A*ok)/G(ok), S = where(ok, A - B, 0), v' = where(valid, clip(v - S, 1, cap), 0).
    Zeros stay exactly 0, values are finite in [1, cap] (fp16 safe).  fork Pool(n_workers) (serial when n_workers <= 1 or no
    fork), OMP_NUM_THREADS=1 exported before the fork; cores are written back only after every neighbouring chunk has read
    its padding, so the result is identical for the serial and the pooled path and ``vol`` may be a private ndarray or a shared
    r+ memmap.  ``save_S`` writes the stripe field as float16 (.npy memmap).  ``mask`` (caller-eroded) only enters the
    before/after profile statistics.
    -> {applied, axis, chunk_axis, sigma_vox, sigma_hp_vox, profile_resid_std_before/after [3], fft_peak_before/after,
    zero_fraction_before/after, min_nonzero_after, max_after, n_nonfinite_after, S_std_mean, n_chunks, n_workers, seconds}"""
    t0 = time.time(); axis = int(axis); assert axis in (0, 1, 2), axis
    if pad < int(math.ceil(TRUNC * sigma_smooth_mm / vox_mm)) + 1:
        pad = int(math.ceil(TRUNC * sigma_smooth_mm / vox_mm)) + 2                # the pad must cover the smoothing kernel
    sig = [sigma_smooth_mm / vox_mm] * 3; sig[axis] = 0.0
    sig_hp = [0.0, 0.0, 0.0]; sig_hp[axis] = sigma_hp_mm / vox_mm
    chunk_axis = 0 if axis != 0 else 1
    before, g0, win = profile_stats(vol, mask, vox_mm)
    if save_S is not None:
        save_S = str(save_S); Path(save_S).parent.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(save_S, mode="w+", dtype=np.float16, shape=vol.shape).flush()
    n = vol.shape[chunk_axis]; jobs = [(c0, min(n, c0 + zchunk)) for c0 in range(0, n, zchunk)]
    _CTX.clear(); _CTX.update(vol=vol, axis=axis, chunk_axis=chunk_axis, sig=sig, sig_hp=sig_hp, cap=float(cap), pad=int(pad), save_S=save_S)
    # Every chunk must read its +-pad slices from the ORIGINAL data, so chunk k is written back only once every chunk whose
    # padding overlaps it (k-r .. k+r, r = ceil(pad/zchunk)) has been computed.  Writing back as results arrive was correct
    # only for forked workers on a private ndarray (pre-fork copy-on-write pages); in the serial path and on a shared r+
    # memmap chunk k+1 would read the already-corrected core of chunk k into its padding (double correction).  The lag
    # bounds the held cores to ~(n_workers + r) chunks instead of a full f16 copy of vol.
    r_pad = int(math.ceil(pad / zchunk)); done: dict = {}; s_stds = [0.0] * len(jobs); n_written = 0
    def _put(res):
        nonlocal n_written
        done[res[0] // zchunk] = res
        while n_written < len(jobs) and all(j in done for j in range(n_written, min(len(jobs), n_written + r_pad + 1))):
            c0, c1, core, s_std = done.pop(n_written)
            g = [slice(None)] * 3; g[chunk_axis] = slice(c0, c1); vol[tuple(g)] = core; s_stds[n_written] = s_std; n_written += 1
    if n_workers > 1 and "fork" in get_all_start_methods() and len(jobs) > 1:
        with _single_thread(), get_context("fork").Pool(min(n_workers, len(jobs)), initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_destripe_chunk, jobs): _put(res)
    else:
        for j in jobs: _put(_destripe_chunk(j))
    assert n_written == len(jobs) and not done, (n_written, len(jobs), sorted(done))
    _CTX.clear()
    after, g1, _ = profile_stats(vol, mask, vox_mm)
    fft = lambda q: {"period_vox": q["fft_period_vox"], "peak_over_median": q["fft_peak_over_median"], "peak_frac": q["fft_peak_frac"]}
    return {"applied": True, "axis": axis, "chunk_axis": chunk_axis, "vox_mm": float(vox_mm), "sigma_vox": sig, "sigma_hp_vox": sig_hp, "cap": float(cap),
            "zchunk": int(zchunk), "pad": int(pad), "detrend_planes": int(win),
            "profile_resid_std_before": [q["resid_std"] for q in before], "profile_resid_std_after": [q["resid_std"] for q in after],
            "profile_p2p95_before": [q["p2p95"] for q in before], "profile_p2p95_after": [q["p2p95"] for q in after],
            "fft_peak_before": fft(before[axis]), "fft_peak_after": fft(after[axis]),
            "zero_fraction_before": g0["n_zero"] / g0["n_total"], "zero_fraction_after": g1["n_zero"] / g1["n_total"],
            "min_nonzero_after": g1["min_nonzero"], "max_after": g1["max"], "n_nonfinite_after": g1["n_nonfinite"],
            "S_std_mean": float(np.mean(s_stds)) if s_stds else None, "n_chunks": len(jobs), "n_workers": int(n_workers), "seconds": time.time() - t0}


# ----------------------------------------------------------------------------- vessel-mask section-phase diagnostic
def section_phase_modulation(vessel_mask: np.ndarray, tissue_mask: np.ndarray, axis: int, period_vox: float | None, n_bins: int = 15, chunk: int = 32) -> dict:
    """Is the vessel mask modulated by the section phase?  Per-plane density along ``axis`` = vessel count / tissue count
    (planes with >= MIN_PLANE_COUNT tissue voxels), plane index folded mod ``period_vox`` into ``n_bins``.
    -> {ratio: max/min of the bin means, var_frac: fraction of the plane-to-plane variance explained by the folded waveform,
    waveform: [n_bins] relative density, period_vox, axis, n_planes, density_mean}  (I58 v1 mask: ratio 2.54, var_frac 0.35)."""
    axis = int(axis); n = vessel_mask.shape
    cnt = np.zeros(n[axis], np.float64); tot = np.zeros(n[axis], np.float64)
    for z0 in range(0, n[0], chunk):
        z1 = min(n[0], z0 + chunk); m = np.asarray(tissue_mask[z0:z1]).astype(bool); v = np.asarray(vessel_mask[z0:z1]).astype(bool) & m
        if axis == 0: cnt[z0:z1] = v.sum((1, 2)); tot[z0:z1] = m.sum((1, 2))
        else:
            red = (0, 2) if axis == 1 else (0, 1); cnt += v.sum(red); tot += m.sum(red)
    ok = tot >= MIN_PLANE_COUNT
    out = {"ratio": None, "var_frac": None, "waveform": None, "period_vox": period_vox, "axis": axis, "n_planes": int(ok.sum()), "density_mean": None}
    if period_vox is None or not np.isfinite(period_vox) or period_vox <= 1 or ok.sum() < 2 * n_bins:
        out["reason"] = "no_period" if (period_vox is None or not np.isfinite(period_vox) or period_vox <= 1) else "too_few_planes"; return out
    ii = np.where(ok)[0]; r = cnt[ii] / tot[ii]; base = float(r.mean())
    if base <= 0: out["reason"] = "no_vessels"; return out
    ph = (ii % period_vox) / period_vox; b = np.minimum((ph * n_bins).astype(int), n_bins - 1)
    w = np.array([r[b == k].mean() if (b == k).any() else np.nan for k in range(n_bins)]); w = np.where(np.isfinite(w), w, base)
    fit = np.interp(ph, (np.arange(n_bins) + 0.5) / n_bins, w)
    out.update(ratio=float(w.max() / max(w.min(), 1e-12)), var_frac=float(np.var(fit) / max(np.var(r), 1e-24)), waveform=(w / base).tolist(), density_mean=base)
    return out
