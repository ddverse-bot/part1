#!/usr/bin/env python
"""P3 probe part 2: seam detection, stripe-pattern stability along z, destripe recipes, Frangi orientation test.
CPU only.  Writes under work/probe_xr/stripes/."""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from scipy import ndimage
import torch
torch.set_num_threads(4)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)
BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration")
sys.path.insert(0, os.path.join(BASE, "octreg"))
from octreg.features import frangi_dark_vesselness
W = os.path.join(BASE, "work/xiangrui_I58bs"); OUT = os.path.join(BASE, "work/probe_xr/stripes"); os.makedirs(OUT, exist_ok=True)
R = {}

octv = np.asarray(np.load(f"{W}/octv.npy", mmap_mode="r"))          # float16 in RAM (1.2 GB)
m150e = np.load(f"{OUT}/mask150_eroded2mm.npy"); m150 = np.load(f"{W}/oct150_mask.npy")
A150 = np.load(f"{W}/oct150_affine.npy"); Av = np.load(f"{W}/octv_affine.npy")
M_ = np.linalg.inv(A150) @ Av
idx = [np.clip(np.rint(M_[r, r] * np.arange(octv.shape[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
mve = m150e[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
D, H, Wd = octv.shape
prof = np.load(f"{OUT}/profiles.npz"); cnt0 = prof["cnt_v0"]; med0 = prof["med_v0"]; mean0 = prof["means_v0"]
zvalid = np.where(cnt0 > 2000)[0]; z0v, z1v = int(zvalid.min()), int(zvalid.max()) + 1
say("loaded", octv.shape, "valid z", z0v, z1v)
TIS_STD = 5414.25   # from part 1 (octv, eroded mask)
PREP_GAMMA = [7907.89208984375, 5673.7646484375, 5811.470703125, 5946.33544921875, 6294.06396484375]
SIG = (0.6, 0.9, 1.325, 1.8, 2.65)

# ------------------------------------------------------------------ 1. seam detection: NCC of in-plane high-passed adjacent planes
def ncc(a, b, m):
    a = a[m]; b = b[m]
    if a.size < 500: return np.nan
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

def plane_ncc_series(axis, lo, hi, mask, sig=3.0, lags=(1, 2, 4)):
    out = {l: np.full(hi - lo, np.nan) for l in lags}; prev = {}
    def get(i):
        if i in prev: return prev[i]
        sl = np.take(octv, i, axis=axis).astype(np.float32); mk = np.take(mask, i, axis=axis)
        hp = sl - ndimage.gaussian_filter(sl, sig)
        prev[i] = (hp, mk)
        for k in [k for k in prev if k < i - max(lags)]: del prev[k]
        return prev[i]
    for i in range(lo, hi):
        hp0, m0 = get(i)
        for l in lags:
            if i + l < hi:
                hp1, m1 = get(i + l); out[l][i - lo] = ncc(hp0, hp1, m0 & m1)
    return out

say("--- seam detection along z ---")
nz = plane_ncc_series(0, z0v, z1v, mve)
dmean = np.abs(np.diff(mean0[z0v:z1v]))
say("--- seam detection along x ---")
xvalid = np.where(prof["cnt_v2"] > 2000)[0]; x0v, x1v = int(xvalid.min()), int(xvalid.max()) + 1
nx = plane_ncc_series(2, x0v, x1v, mve)
say("--- seam detection along y ---")
yvalid = np.where(prof["cnt_v1"] > 2000)[0]; y0v, y1v = int(yvalid.min()), int(yvalid.max()) + 1
ny = plane_ncc_series(1, y0v, y1v, mve)
np.savez(f"{OUT}/seam_ncc.npz", z1=nz[1], z2=nz[2], z4=nz[4], x1=nx[1], x2=nx[2], y1=ny[1], y2=ny[2], z0=z0v, x0=x0v, y0=y0v)

def dips(series, lo, prom=0.05):
    from scipy.signal import find_peaks
    s = np.nan_to_num(series, nan=np.nanmedian(series))
    base = ndimage.median_filter(s, 31)
    pk, pr = find_peaks(base - s, prominence=prom, distance=3)
    d = np.diff(pk)
    return {"n_dips": int(len(pk)), "positions": (pk + lo).tolist()[:60], "spacings": d.tolist()[:60], "median_spacing": float(np.median(d)) if d.size else None,
            "prominences": np.round(pr["prominences"], 3).tolist()[:60], "series_median": float(np.nanmedian(s)), "series_p5": float(np.nanpercentile(s, 5))}
R["seams"] = {"z_lag1": dips(nz[1], z0v), "x_lag1": dips(nx[1], x0v), "y_lag1": dips(ny[1], y0v)}
# periodicity of the NCC(z) series
def series_period(s, win):
    s = np.nan_to_num(s, nan=np.nanmedian(s)); d = s - ndimage.uniform_filter1d(s, win, mode="nearest")
    n = d.size; f = np.fft.rfftfreq(n); P = np.abs(np.fft.rfft(d * np.hanning(n))) ** 2; P[f < 1.0 / win] = 0
    k = int(np.argmax(P)); ac = np.correlate(d - d.mean(), d - d.mean(), "full")[n - 1:]; ac /= ac[0]
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ac[1:150], height=0.05, distance=3)
    return {"fft_period": float(1 / f[k]) if f[k] > 0 else None, "fft_top3": [float(1 / f[j]) for j in np.argsort(P)[::-1][:3] if f[j] > 0], "ac_peaks": [(int(p + 1), round(float(ac[p + 1]), 3)) for p in pk[:5]]}
R["seams"]["z_lag1_period"] = series_period(nz[1], 60); R["seams"]["x_lag1_period"] = series_period(nx[1], 60); R["seams"]["y_lag1_period"] = series_period(ny[1], 60)
say("seams", json.dumps(R["seams"])[:2500])
fig, ax = plt.subplots(4, 1, figsize=(15, 14))
zz = np.arange(z0v, z1v)
for l, c in ((1, "C0"), (2, "C1"), (4, "C2")): ax[0].plot(zz, nz[l], lw=0.7, color=c, label=f"lag {l}")
ax[0].set_title("NCC between in-plane high-passed x-y slices z and z+lag (eroded mask): seams = dips"); ax[0].legend(); ax[0].set_xlabel("z (0.04 mm)")
ax[1].plot(zz[:-1], dmean, lw=0.7); ax[1].set_title("|d/dz| of the per-slice mean (eroded mask)"); ax[1].set_xlabel("z")
xx = np.arange(x0v, x1v)
for l, c in ((1, "C0"), (2, "C1")): ax[2].plot(xx, nx[l], lw=0.7, color=c, label=f"lag {l}")
ax[2].set_title("NCC between high-passed z-y planes x and x+lag: B-scan seams / stripe pitch"); ax[2].legend(); ax[2].set_xlabel("x")
yy = np.arange(y0v, y1v)
for l, c in ((1, "C0"), (2, "C1")): ax[3].plot(yy, ny[l], lw=0.7, color=c, label=f"lag {l}")
ax[3].set_title("NCC between high-passed z-x planes y and y+lag"); ax[3].legend(); ax[3].set_xlabel("y")
plt.tight_layout(); plt.savefig(f"{OUT}/seam_ncc.png", dpi=100); plt.close(fig)

# ------------------------------------------------------------------ 2. stripe pattern per z-chunk and its stability along z
say("--- per-chunk stripe maps ---")
CH = 25   # 1 mm chunks
valid_all = None
def masked_lp(img, m, sig):   # normalised convolution (2-D or 3-D, sig per axis)
    num = ndimage.gaussian_filter(np.where(m, img, 0.0).astype(np.float32), sig); den = ndimage.gaussian_filter(m.astype(np.float32), sig)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0), den > 0.05
chunks = list(range(z0v, z1v - CH + 1, CH)); Smaps = []; Vmaps = []
for c0 in chunks:
    blk = octv[c0:c0 + CH].astype(np.float32); mk = blk > 0
    num = (blk * mk).sum(0); den = mk.sum(0); Mc = np.where(den > 0.8 * CH, num / np.maximum(den, 1), 0.0); vm = den > 0.8 * CH
    A_, ok = masked_lp(Mc, vm, (12, 0)); B_, _ = masked_lp(A_, ok, (0, 6)); S = np.where(ok, A_ - B_, 0.0)
    Smaps.append(S.astype(np.float32)); Vmaps.append(ok & mve[c0:c0 + CH].all(0))
Smaps = np.stack(Smaps); Vmaps = np.stack(Vmaps); n = len(chunks)
C = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        m = Vmaps[i] & Vmaps[j]
        if m.sum() > 5000: C[i, j] = ncc(Smaps[i], Smaps[j], m)
R["stripe_chunks"] = {"chunk": CH, "z_starts": chunks, "S_std_in_eroded_per_chunk": [float(Smaps[i][Vmaps[i]].std()) for i in range(n)],
                      "corr_adjacent": [float(C[i, i + 1]) for i in range(n - 1)], "corr_lag4": [float(C[i, i + 4]) for i in range(n - 4)], "corr_lag10": [float(C[i, i + 10]) for i in range(n - 10)],
                      "corr_mean_offdiag": float(np.nanmean(C[~np.eye(n, dtype=bool)]))}
say("chunk stripe corr adjacent", np.round(R["stripe_chunks"]["corr_adjacent"], 2).tolist())
say("chunk stripe corr lag10", np.round(R["stripe_chunks"]["corr_lag10"], 2).tolist())
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
im = ax[0].imshow(C, vmin=-0.2, vmax=1, cmap="viridis"); ax[0].set_title(f"NCC between x-stripe maps of 1-mm z-chunks (chunk {CH} slices)"); plt.colorbar(im, ax=ax[0]); ax[0].set_xlabel("chunk"); ax[0].set_ylabel("chunk")
a = np.percentile(np.abs(Smaps[n // 2][Vmaps[n // 2]]), 99)
ax[1].imshow(np.where(Vmaps[n // 2], Smaps[n // 2], np.nan), cmap="gray", vmin=-a, vmax=a); ax[1].set_title(f"stripe map S (HP_x(LP_y(mean_z))) chunk z={chunks[n//2]}")
ax[2].imshow(np.where(Vmaps[n // 4], Smaps[n // 4], np.nan), cmap="gray", vmin=-a, vmax=a); ax[2].set_title(f"stripe map S chunk z={chunks[n//4]}")
plt.tight_layout(); plt.savefig(f"{OUT}/stripe_chunk_stability.png", dpi=90); plt.close(fig)
np.save(f"{OUT}/stripe_maps_chunks25.npy", Smaps)
# stripe amplitude on the whole depth (mean of all chunks) relative to tissue contrast
Sall = np.nanmean(np.where(Vmaps, Smaps, np.nan), 0); vall = Vmaps.sum(0) > 3
R["stripe_amplitude"] = {"S_mean_chunk_std_in_eroded": float(np.mean(R["stripe_chunks"]["S_std_in_eroded_per_chunk"])), "S_alldepth_std": float(np.nanstd(Sall[vall])),
                         "S_alldepth_p2p95": float(np.nanpercentile(Sall[vall], 97.5) - np.nanpercentile(Sall[vall], 2.5)), "S_alldepth_p2p99": float(np.nanpercentile(Sall[vall], 99.5) - np.nanpercentile(Sall[vall], 0.5)),
                         "tissue_std_ref": TIS_STD, "level_ref": 16841.0}
# x-profile of S (mean over y) -> its FFT gives the stripe pitch
px = np.nanmean(np.where(vall, Sall, np.nan), 0); okx = np.isfinite(px); ii = np.where(okx)[0]; pr = np.nan_to_num(px[ii.min():ii.max() + 1])
f = np.fft.rfftfreq(pr.size); P = np.abs(np.fft.rfft(pr * np.hanning(pr.size))) ** 2; P[f < 1 / 60.] = 0; k = np.argmax(P)
R["stripe_amplitude"]["S_xprofile_std"] = float(np.nanstd(px)); R["stripe_amplitude"]["S_xprofile_fft_period"] = float(1 / f[k])
say("stripe amplitude", R["stripe_amplitude"])
fig, ax = plt.subplots(2, 1, figsize=(15, 8)); ax[0].imshow(np.where(vall, Sall, np.nan), cmap="gray", vmin=-a, vmax=a); ax[0].set_title("x-stripe map averaged over all depth chunks")
ax[1].plot(px, lw=0.7); ax[1].set_title(f"its x-profile: std {R['stripe_amplitude']['S_xprofile_std']:.0f}, FFT period {1/f[k]:.2f} vox"); ax[1].set_xlabel("x")
plt.tight_layout(); plt.savefig(f"{OUT}/stripe_alldepth.png", dpi=90); plt.close(fig)

# ------------------------------------------------------------------ 3. destripe recipes on 200^3 crops
def destripe_x(vol, valid, sy, sz, sx, mode="add"):
    """x-stripe flat-field: A = LP_yz(vol) (masked), B = LP_x(A); S = A - B.  add: vol - S ; mult: vol * B / A."""
    A_, ok = masked_lp(vol, valid, (sz, sy, 0)); B_, _ = masked_lp(A_, ok, (0, 0, sx))
    if mode == "add":
        out = np.where(valid, vol - (A_ - B_), vol)
    else:
        g = np.where(ok & (A_ > 0.2 * B_), B_ / np.maximum(A_, 1e-3), 1.0); g = np.clip(g, 0.5, 2.0)
        out = np.where(valid, vol * g, vol)
    return out.astype(np.float32), (A_ - B_).astype(np.float32)

def per_slice_offset(vol, z0):
    """(a) additive per-slice offset: subtract per-slice median (full-field, eroded mask) and add the global median."""
    m = med0[z0:z0 + vol.shape[0]]; g = np.nanmedian(med0[z0v:z1v]); off = np.nan_to_num(m - g)
    return (vol - off[:, None, None]).astype(np.float32)

def running_median_hp(vol, z0, P):
    """(b) per-slice profile high-pass with a running median of length P along z (removes periodic banding at ~P)."""
    m = np.nan_to_num(med0, nan=np.nanmedian(med0)); trend = ndimage.median_filter(m, size=int(P) | 1, mode="nearest"); off = (m - trend)[z0:z0 + vol.shape[0]]
    return (vol - off[:, None, None]).astype(np.float32)

def hessian_dirs(vol, pts, scales, best_scale):
    """principal (tube) direction at pts for the Hessian at each voxel's best scale: eigenvector of the smallest-|lambda| eigenvalue."""
    dirs = np.zeros((len(pts), 3), np.float32); ev = np.zeros((len(pts), 3), np.float32)
    for si, s in enumerate(scales):
        sel = np.where(best_scale == si)[0]
        if sel.size == 0: continue
        Hc = {}
        for (i, j) in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            order = [0, 0, 0]; order[i] += 1; order[j] += 1
            Hc[(i, j)] = ndimage.gaussian_filter(vol, s, order=order, truncate=3.0)
        p = pts[sel]
        Hm = np.zeros((sel.size, 3, 3), np.float32)
        for (i, j), arr in Hc.items():
            v = arr[p[:, 0], p[:, 1], p[:, 2]]; Hm[:, i, j] = v; Hm[:, j, i] = v
        w, V = np.linalg.eigh(Hm)                     # ascending eigenvalues
        k = np.argmin(np.abs(w), axis=1)
        dirs[sel] = V[np.arange(sel.size), :, k]; ev[sel] = w
    return dirs, ev

def frangi_orient(vol, emask, tag):
    """Frangi (octreg, prep sigmas & gammas) -> top-1% inside eroded mask -> orientation fractions."""
    x = torch.from_numpy(vol)[None]
    best = None; bs = None
    for si, (s, g) in enumerate(zip(SIG, PREP_GAMMA)):
        v = frangi_dark_vesselness(x, sigmas_vox=(s,), gamma=(g,))[0].numpy()
        if best is None: best = v; bs = np.zeros(v.shape, np.int8)
        else:
            upd = v > best; best = np.where(upd, v, best); bs = np.where(upd, si, bs)
    vals = best[emask]; thr = float(np.percentile(vals, 99.0))
    top = (best > thr) & emask
    pts = np.argwhere(top)
    dirs, ev = hessian_dirs(vol, pts, SIG, bs[top])
    c15 = np.cos(np.radians(15.0)); ad = np.abs(dirs)
    fz, fy, fx = float((ad[:, 0] > c15).mean()), float((ad[:, 1] > c15).mean()), float((ad[:, 2] > c15).mean())
    # tube-likeness: |l1| << l2 ~ l3 ; and the mean vesselness of top set
    lab, ncomp = ndimage.label(top); sizes = ndimage.sum(top, lab, np.arange(1, ncomp + 1)) if ncomp else np.array([])
    # extent along z of the largest components (stripes -> long along z)
    ext = []
    if ncomp:
        objs = ndimage.find_objects(lab)
        big = np.argsort(sizes)[::-1][:50]
        ext = [[int(objs[k][0].stop - objs[k][0].start), int(objs[k][1].stop - objs[k][1].start), int(objs[k][2].stop - objs[k][2].start)] for k in big]
        ext = np.array(ext)
    out = {"tag": tag, "thr_top1pct": thr, "n_top": int(top.sum()), "frac_within15deg_z": fz, "frac_within15deg_y": fy, "frac_within15deg_x": fx, "isotropic_expectation": float(1 - c15),
           "mean_abs_cos_z": float(ad[:, 0].mean()), "mean_abs_cos_y": float(ad[:, 1].mean()), "mean_abs_cos_x": float(ad[:, 2].mean()),
           "n_components": int(ncomp), "median_comp_size": float(np.median(sizes)) if ncomp else None, "top50_comp_extent_zyx_median": ext.tolist() if len(ext) == 0 else np.median(ext, 0).tolist(),
           "top50_comp_zextent_over_max_xy": float(np.median(ext[:, 0] / np.maximum(ext[:, 1:].max(1), 1))) if len(ext) else None,
           "best_scale_hist": np.bincount(bs[top], minlength=len(SIG)).tolist(), "vesselness_p99.9": float(np.percentile(vals, 99.9)), "vesselness_p90": float(np.percentile(vals, 90))}
    return out, best, top, dirs

def crop_metrics(vol, emask, tag):
    """anisotropy metrics of a crop: gradient energy per axis, spectrum fraction on the kx line (ky=kz=0) and kz=0 plane, x/y/z-profile residual std."""
    g = [np.diff(vol, axis=i) for i in range(3)]
    ge = {"dz": float(np.mean(g[0][emask[1:] & emask[:-1]] ** 2)), "dy": float(np.mean(g[1][emask[:, 1:] & emask[:, :-1]] ** 2)), "dx": float(np.mean(g[2][emask[:, :, 1:] & emask[:, :, :-1]] ** 2))}
    bw = (vol - ndimage.gaussian_filter(vol, 8)) * emask
    n = vol.shape[0]; w = np.hanning(n); Fq = np.fft.fftshift(np.abs(np.fft.fftn(bw * w[:, None, None] * w[None, :, None] * w[None, None, :])) ** 2); c = n // 2; tot = Fq.sum()
    kxline = Fq[c - 1:c + 2, c - 1:c + 2, :].sum() / tot; kyline = Fq[c - 1:c + 2, :, c - 1:c + 2].sum() / tot; kzline = Fq[:, c - 1:c + 2, c - 1:c + 2].sum() / tot
    kz0 = Fq[c - 1:c + 2].sum() / tot
    # x profile inside mask (mean over y,z), detrended, std
    def prof_std(ax_):
        axes = tuple(i for i in range(3) if i != ax_); num = (vol * emask).sum(axes); den = emask.sum(axes); p = np.where(den > 200, num / np.maximum(den, 1), np.nan)
        ok = np.isfinite(p)
        if ok.sum() < 30: return None
        pp = p.copy(); pp[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], p[ok]); d = pp - ndimage.uniform_filter1d(pp, 40, mode="nearest"); return float(d[ok].std())
    return {"tag": tag, "grad_energy": ge, "grad_dx_over_dz": ge["dx"] / ge["dz"], "spec_frac_kx_line": float(kxline), "spec_frac_ky_line": float(kyline), "spec_frac_kz_line": float(kzline),
            "spec_frac_kz0_plane": float(kz0), "iid_line": 9.0 / n ** 2, "iid_plane": 3.0 / n, "xprof_resid_std": prof_std(2), "yprof_resid_std": prof_std(1), "zprof_resid_std": prof_std(0),
            "tissue_std": float(vol[emask].std()), "mean": float(vol[emask].mean())}

CROPS = {"A_centroid": (370, 509, 398), "B_striped": (400, 620, 230)}
c = 100; MARG = 130
R["crops"] = {}
for cname, (cz, cy, cx) in CROPS.items():
    say(f"=== crop {cname} centre {(cz, cy, cx)} ===")
    lo = np.array([cz - c - MARG, cy - c - MARG, cx - c - MARG]); hi = lo + 2 * (c + MARG)
    lo = np.maximum(lo, 0); hi = np.minimum(hi, octv.shape)
    big = octv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32); valid = big > 0
    off = np.array([cz - c, cy - c, cx - c]) - lo
    sl = tuple(slice(int(off[i]), int(off[i]) + 2 * c) for i in range(3))
    emask = mve[cz - c:cz + c, cy - c:cy + c, cx - c:cx + c]
    say("crop eroded-mask fraction", emask.mean())
    variants = {}
    variants["orig"] = big[sl].copy()
    variants["a_slice_offset"] = per_slice_offset(big, lo[0])[sl]
    variants["b_runmed_z25"] = running_median_hp(big, lo[0], 25)[sl]
    v, S = destripe_x(big, valid, sy=12, sz=12, sx=6, mode="add"); variants["c_xff_add_s12"] = v[sl]; S_short = S[sl]
    v, _ = destripe_x(big, valid, sy=12, sz=12, sx=6, mode="mult"); variants["c_xff_mult_s12"] = v[sl]
    v, S = destripe_x(big, valid, sy=12, sz=40, sx=6, mode="add"); variants["c_xff_add_sz40"] = v[sl]; S_long = S[sl]
    v, _ = destripe_x(big, valid, sy=30, sz=40, sx=6, mode="add"); variants["c_xff_add_sy30sz40"] = v[sl]
    v, _ = destripe_x(big, valid, sy=12, sz=12, sx=3, mode="add"); variants["c_xff_add_sx3"] = v[sl]
    v2 = per_slice_offset(big, lo[0]); v, _ = destripe_x(v2, valid, sy=12, sz=40, sx=6, mode="add"); variants["ac_combo"] = v[sl]
    del big
    R["crops"][cname] = {"centre": [cz, cy, cx], "emask_frac": float(emask.mean()), "S_short_std": float(S_short[emask].std()), "S_long_std": float(S_long[emask].std()),
                         "S_short_vs_long_ncc": ncc(S_short, S_long, emask), "variants": {}}
    fr = {}
    for vn, vol in variants.items():
        met = crop_metrics(vol, emask, vn)
        fo, best, top, dirs = frangi_orient(vol, emask, vn)
        met["frangi"] = fo; R["crops"][cname]["variants"][vn] = met; fr[vn] = (best, top, dirs)
        say(cname, vn, "grad dx/dz %.2f  kx-line %.4f  kz0-plane %.4f  xprof %.0f | frangi z15 %.3f y15 %.3f x15 %.3f  zext/xy %.2f" % (
            met["grad_dx_over_dz"], met["spec_frac_kx_line"], met["spec_frac_kz0_plane"], met["xprof_resid_std"] or -1, fo["frac_within15deg_z"], fo["frac_within15deg_y"], fo["frac_within15deg_x"], fo["top50_comp_zextent_over_max_xy"] or -1))
        # overlap with original top set
        R["crops"][cname]["variants"][vn]["frangi"]["top_overlap_with_orig"] = float((top & fr["orig"][1]).sum() / max(top.sum(), 1))
    # figures: before/after slices + vesselness MIPs for orig, a, c_add_s12, c_add_sz40
    show = ["orig", "a_slice_offset", "c_xff_add_s12", "c_xff_add_sz40", "c_xff_mult_s12"]
    v0, v1 = np.percentile(variants["orig"][emask], [1, 99])
    fig, ax = plt.subplots(4, len(show), figsize=(5 * len(show), 20))
    for j, vn in enumerate(show):
        vol = variants[vn]; best, top, dirs = fr[vn]
        ax[0, j].imshow(vol[c], cmap="gray", vmin=v0, vmax=v1); ax[0, j].set_title(f"{cname} {vn}\nx-y slice (rows y, cols x)")
        ax[1, j].imshow(vol[:, c, :], cmap="gray", vmin=v0, vmax=v1); ax[1, j].set_title("z-x slice (rows z, cols x)")
        ax[2, j].imshow(best.max(0), cmap="magma", vmin=0, vmax=max(1e-3, float(np.percentile(best[emask], 99.9)))); ax[2, j].set_title("vesselness MIP over z (top-1%% aligned z: %.2f)" % R["crops"][cname]["variants"][vn]["frangi"]["frac_within15deg_z"])
        ax[3, j].imshow(top.max(1), cmap="gray"); ax[3, j].set_title("top-1%% mask MIP over y (rows z, cols x); y15 %.2f x15 %.2f" % (R["crops"][cname]["variants"][vn]["frangi"]["frac_within15deg_y"], R["crops"][cname]["variants"][vn]["frangi"]["frac_within15deg_x"]))
    plt.tight_layout(); plt.savefig(f"{OUT}/destripe_crop_{cname}.png", dpi=70); plt.close(fig)
    np.savez_compressed(f"{OUT}/crop_{cname}_slices.npz", orig_xy=variants["orig"][c], orig_zx=variants["orig"][:, c, :], c12_xy=variants["c_xff_add_s12"][c], c12_zx=variants["c_xff_add_s12"][:, c, :],
                        c40_xy=variants["c_xff_add_sz40"][c], c40_zx=variants["c_xff_add_sz40"][:, c, :], emask_xy=emask[c])
    # orientation histogram figure
    fig, ax = plt.subplots(1, len(show), figsize=(5 * len(show), 4))
    for j, vn in enumerate(show):
        dirs = fr[vn][2]; ad = np.abs(dirs)
        ax[j].hist(np.degrees(np.arccos(np.clip(ad[:, 0], 0, 1))), bins=45, range=(0, 90), alpha=0.6, label="angle to z")
        ax[j].hist(np.degrees(np.arccos(np.clip(ad[:, 2], 0, 1))), bins=45, range=(0, 90), alpha=0.6, label="angle to x")
        ax[j].hist(np.degrees(np.arccos(np.clip(ad[:, 1], 0, 1))), bins=45, range=(0, 90), alpha=0.4, label="angle to y")
        th = np.linspace(0, 90, 46); iso = np.diff(1 - np.cos(np.radians(th))) * len(dirs); ax[j].plot(0.5 * (th[1:] + th[:-1]), iso, "k--", lw=0.8, label="isotropic")
        ax[j].set_title(f"{cname} {vn}: tube-direction angles"); ax[j].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/orientation_hist_{cname}.png", dpi=80); plt.close(fig)
    json.dump(R, open(f"{OUT}/destripe_results.json", "w"), indent=1, default=float)

json.dump(R, open(f"{OUT}/destripe_results.json", "w"), indent=1, default=float)
say("done")
