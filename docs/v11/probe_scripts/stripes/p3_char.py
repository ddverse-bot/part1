#!/usr/bin/env python
"""P3 probe part 1: characterise stripe / seam artefacts in the OCT (octv 0.04 mm, oct150 0.15 mm).
Writes under work/probe_xr/stripes/.  CPU only."""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)

BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration")
W = os.path.join(BASE, "work/xiangrui_I58bs")
OUT = os.path.join(BASE, "work/probe_xr/stripes")
os.makedirs(OUT, exist_ok=True)
R = {}

# ------------------------------------------------------------------ load
o150 = np.load(f"{W}/oct150.npy", mmap_mode="r")
m150 = np.load(f"{W}/oct150_mask.npy")
A150 = np.load(f"{W}/oct150_affine.npy")
octv = np.load(f"{W}/octv.npy", mmap_mode="r")
Av = np.load(f"{W}/octv_affine.npy")
mv = np.load(f"{W}/octv_mask.npy", mmap_mode="r")
say("shapes", o150.shape, octv.shape, octv.dtype)
say("A150", np.round(A150, 4).tolist())
say("Av", np.round(Av, 4).tolist())

# ------------------------------------------------------------------ eroded mask (2 mm) at 0.15 mm, mapped to octv grid
er_vox = int(round(2.0 / 0.15))
m150e = ndimage.binary_erosion(m150, iterations=er_vox)
say("eroded mask frac150", m150e.mean(), "n", m150e.sum())
M_ = np.linalg.inv(A150) @ Av
idx = [np.clip(np.rint(M_[r, r] * np.arange(octv.shape[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
mve = m150e[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
say("eroded mask on octv grid frac", mve.mean())
np.save(f"{OUT}/mask150_eroded2mm.npy", m150e)

# ------------------------------------------------------------------ black blocks / zeros
say("--- zeros / black blocks ---")
o150a = np.asarray(o150)
z150 = (o150a == 0)
R["oct150_exact_zero_frac"] = float(z150.mean())
R["oct150_min_nonzero"] = float(o150a[o150a > 0].min())
R["oct150_zero_in_mask_frac"] = float((z150 & m150).sum() / max(m150.sum(), 1))
R["oct150_zero_in_eroded_mask_frac"] = float((z150 & m150e).sum() / max(m150e.sum(), 1))
lab, n = ndimage.label(z150)
sizes = ndimage.sum(z150, lab, index=np.arange(1, n + 1))
order = np.argsort(sizes)[::-1]
blocks = []
for k in order[:12]:
    sl = ndimage.find_objects(lab == (k + 1))[0]
    bb = [[int(s.start), int(s.stop)] for s in sl]
    vol = int(sizes[k]); bbvol = int(np.prod([s.stop - s.start for s in sl]))
    inmask = int((m150 & (lab == (k + 1))).sum())
    blocks.append({"bbox_zyx": bb, "n_vox": vol, "fill": round(vol / bbvol, 3), "n_in_oldmask": inmask})
R["oct150_zero_components"] = int(n)
R["oct150_zero_blocks_top12"] = blocks
say("zero components", n, "top blocks", json.dumps(blocks, indent=0)[:1500])
# octv zeros, chunked
nz_v = 0; nz_in_mask = 0; nz_in_er = 0; zmin = np.inf; small_pos = 0
zero_per_slice = np.zeros(octv.shape[0], np.int64)
for z0 in range(0, octv.shape[0], 32):
    blk = np.asarray(octv[z0:z0 + 32]).astype(np.float32)
    zz = blk == 0
    zero_per_slice[z0:z0 + blk.shape[0]] = zz.sum(axis=(1, 2))
    nz_v += zz.sum(); nz_in_mask += (zz & np.asarray(mv[z0:z0 + 32])).sum(); nz_in_er += (zz & mve[z0:z0 + 32]).sum()
    pos = blk[blk > 0]
    if pos.size: zmin = min(zmin, float(pos.min()))
    small_pos += (blk > 0).sum() - (blk > 50).sum()
R["octv_exact_zero_frac"] = float(nz_v / octv.size)
R["octv_zero_in_oldmask_frac"] = float(nz_in_mask / np.asarray(mv).sum())
R["octv_zero_in_eroded_mask_frac"] = float(nz_in_er / mve.sum())
R["octv_min_nonzero"] = float(zmin)
R["octv_frac_0<v<=50"] = float(small_pos / octv.size)
say("octv zero frac", R["octv_exact_zero_frac"], "in old mask", R["octv_zero_in_oldmask_frac"], "in eroded", R["octv_zero_in_eroded_mask_frac"], "min nonzero", zmin)

# ------------------------------------------------------------------ depth profiles (axis 0) and in-plane profiles (axes 1,2)
def axis_profiles(vol, mask, chunk=32, name=""):
    """per-index mean, median (approx via percentile on subsample), count inside mask along each of the 3 axes."""
    D, H, Wd = vol.shape
    sums = [np.zeros(D), np.zeros(H), np.zeros(Wd)]
    sq = [np.zeros(D), np.zeros(H), np.zeros(Wd)]
    cnt = [np.zeros(D), np.zeros(H), np.zeros(Wd)]
    med0 = np.full(D, np.nan)
    for z0 in range(0, D, chunk):
        blk = np.asarray(vol[z0:z0 + chunk]).astype(np.float32)
        mk = np.asarray(mask[z0:z0 + chunk])
        v = np.where(mk, blk, 0.0)
        sums[0][z0:z0 + blk.shape[0]] += v.sum(axis=(1, 2)); sums[1] += v.sum(axis=(0, 2)); sums[2] += v.sum(axis=(0, 1))
        sq[0][z0:z0 + blk.shape[0]] += (v * v).sum(axis=(1, 2)); sq[1] += (v * v).sum(axis=(0, 2)); sq[2] += (v * v).sum(axis=(0, 1))
        cnt[0][z0:z0 + blk.shape[0]] += mk.sum(axis=(1, 2)); cnt[1] += mk.sum(axis=(0, 2)); cnt[2] += mk.sum(axis=(0, 1))
        for i in range(blk.shape[0]):
            if mk[i].sum() > 500:
                med0[z0 + i] = np.median(blk[i][mk[i]])
    means = [np.where(c > 0, s / np.maximum(c, 1), np.nan) for s, c in zip(sums, cnt)]
    stds = [np.where(c > 1, np.sqrt(np.maximum(q / np.maximum(c, 1) - m ** 2, 0)), np.nan) for q, c, m in zip(sq, cnt, means)]
    return means, stds, cnt, med0

say("--- profiles octv ---")
means_v, stds_v, cnt_v, med_v = axis_profiles(octv, mve, name="octv")
say("--- profiles oct150 ---")
means_1, stds_1, cnt_1, med_1 = axis_profiles(o150, m150e, name="oct150")
np.savez(f"{OUT}/profiles.npz", means_v0=means_v[0], means_v1=means_v[1], means_v2=means_v[2], med_v0=med_v, cnt_v0=cnt_v[0], cnt_v1=cnt_v[1], cnt_v2=cnt_v[2],
         stds_v0=stds_v[0], means_10=means_1[0], means_11=means_1[1], means_12=means_1[2], med_10=med_1, cnt_10=cnt_1[0], cnt_11=cnt_1[1], cnt_12=cnt_1[2], stds_10=stds_1[0])

def analyse_profile(p, cnt, min_cnt, detrend_win, tag, vox_mm):
    """detrended profile (running-mean removed), FFT peak, autocorrelation; returns dict."""
    ok = (cnt >= min_cnt) & np.isfinite(p)
    idx = np.where(ok)[0]
    if idx.size < 20:
        return {"tag": tag, "n_valid": int(idx.size)}
    i0, i1 = idx.min(), idx.max() + 1
    pp = p[i0:i1].copy(); okk = ok[i0:i1]
    pp[~okk] = np.interp(np.where(~okk)[0], np.where(okk)[0], pp[okk])
    trend = ndimage.uniform_filter1d(pp, size=detrend_win, mode="nearest")
    d = pp - trend
    # fft with hann window
    n = d.size; w = np.hanning(n); f = np.fft.rfftfreq(n, d=1.0); P = np.abs(np.fft.rfft(d * w)) ** 2
    P[0] = 0; P[f < 1.0 / detrend_win] = 0    # ignore below the detrend cut
    k = int(np.argmax(P)); period = 1.0 / f[k] if f[k] > 0 else np.inf
    # peak significance: peak / median of spectrum in a band
    band = P[(f > 0.02) & (f < 0.5)]; snr = float(P[k] / np.median(band[band > 0])) if band.size and (band > 0).any() else 0.0
    # autocorrelation of detrended profile
    dd = d - d.mean(); ac = np.correlate(dd, dd, mode="full")[n - 1:]; ac = ac / max(ac[0], 1e-12)
    lagmax = min(200, n // 2)
    from scipy.signal import find_peaks
    pk, props = find_peaks(ac[1:lagmax], height=0.05, distance=3)
    ac_peaks = [(int(pi + 1), float(ac[pi + 1])) for pi in pk[:6]]
    out = {"tag": tag, "n_valid": int(idx.size), "range": [int(i0), int(i1)], "mean_level": float(np.nanmean(pp)),
           "detrend_win": int(detrend_win), "resid_std": float(d.std()), "resid_p2p_95": float(np.percentile(d, 97.5) - np.percentile(d, 2.5)),
           "fft_peak_period_vox": float(period), "fft_peak_period_mm": float(period * vox_mm), "fft_peak_snr_vs_median": snr,
           "fft_top5": [(float(1.0 / f[j]), float(P[j] / P.max())) for j in np.argsort(P)[::-1][:5] if f[j] > 0],
           "ac_peaks_lag_val": ac_peaks, "ac_lag1": float(ac[1]) if n > 1 else None}
    return out, d, f, P, ac, (i0, i1)

def plot_profile(p, cnt, min_cnt, med, res, fname, title, vox_mm):
    out, d, f, P, ac, (i0, i1) = res
    fig, ax = plt.subplots(4, 1, figsize=(14, 13))
    x = np.arange(p.size)
    ax[0].plot(x, p, lw=0.7, label="mean in eroded mask")
    if med is not None: ax[0].plot(x, med, lw=0.7, label="median in eroded mask", alpha=0.8)
    ax[0].set_title(title + f"  (valid {i0}-{i1}, vox {vox_mm} mm)"); ax[0].legend(); ax[0].set_ylabel("intensity")
    ax2 = ax[0].twinx(); ax2.plot(x, cnt, color="gray", lw=0.5, alpha=0.5); ax2.set_ylabel("n voxels", color="gray")
    ax[1].plot(np.arange(i0, i1), d, lw=0.7); ax[1].set_title(f"detrended (running mean {out['detrend_win']} removed): std {out['resid_std']:.0f}, p2p95 {out['resid_p2p_95']:.0f}")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[2].semilogy(f[1:], P[1:] / P.max() + 1e-9, lw=0.8); ax[2].set_xlabel("cycles / voxel"); ax[2].set_title(f"FFT (Hann) peak period {out['fft_peak_period_vox']:.1f} vox = {out['fft_peak_period_mm']:.2f} mm, snr {out['fft_peak_snr_vs_median']:.1f}")
    ax[2].set_ylim(1e-4, 2)
    for per in (12.5, 25, 6.25, 50):
        pass
    L = min(150, ac.size)
    ax[3].plot(np.arange(L), ac[:L], lw=0.8); ax[3].axhline(0, color="k", lw=0.5); ax[3].set_xlabel("lag (voxels)"); ax[3].set_title(f"autocorrelation, peaks {out['ac_peaks_lag_val'][:4]}")
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close(fig)

R["profiles"] = {}
for ax_i, nm in enumerate(["z(depth)", "y", "x"]):
    res = analyse_profile(means_v[ax_i], cnt_v[ax_i], 2000, 60, f"octv_axis{ax_i}_{nm}", 0.04)
    if isinstance(res, tuple):
        R["profiles"][f"octv_axis{ax_i}"] = res[0]
        plot_profile(means_v[ax_i], cnt_v[ax_i], 2000, med_v if ax_i == 0 else None, res, f"{OUT}/profile_octv_axis{ax_i}.png", f"octv 0.04 mm: mean along axis {ax_i} ({nm})", 0.04)
        say(nm, json.dumps(res[0]))
    res = analyse_profile(means_1[ax_i], cnt_1[ax_i], 300, 16, f"oct150_axis{ax_i}_{nm}", 0.15)
    if isinstance(res, tuple):
        R["profiles"][f"oct150_axis{ax_i}"] = res[0]
        plot_profile(means_1[ax_i], cnt_1[ax_i], 300, med_1 if ax_i == 0 else None, res, f"{OUT}/profile_oct150_axis{ax_i}.png", f"oct150 0.15 mm: mean along axis {ax_i} ({nm})", 0.15)
        say(nm, json.dumps(res[0]))
# also depth profile detrended with a shorter window (to catch ~12 vox period) and median-based
res = analyse_profile(med_v, cnt_v[0], 2000, 30, "octv_axis0_median_win30", 0.04)
if isinstance(res, tuple): R["profiles"]["octv_axis0_median_win30"] = res[0]; say("median win30", json.dumps(res[0]))
res = analyse_profile(means_v[0], cnt_v[0], 2000, 200, "octv_axis0_mean_win200", 0.04)
if isinstance(res, tuple): R["profiles"]["octv_axis0_mean_win200"] = res[0]; say("mean win200", json.dumps(res[0]))

# tissue contrast reference: std of intensities inside eroded mask (octv, subsample) + tissue-vs-agarose
sub = np.asarray(octv[::4, ::4, ::4]).astype(np.float32); sube = mve[::4, ::4, ::4]; subm = np.asarray(mv[::4, ::4, ::4])
tis = sub[sube]
R["octv_tissue_stats"] = {"mean": float(tis.mean()), "std": float(tis.std()), "p5": float(np.percentile(tis, 5)), "p50": float(np.percentile(tis, 50)), "p95": float(np.percentile(tis, 95))}
outside = sub[(~subm) & (sub > 0)]
R["octv_outside_oldmask_stats"] = {"mean": float(outside.mean()), "p50": float(np.percentile(outside, 50)), "n": int(outside.size)}
say("tissue stats", R["octv_tissue_stats"], "outside", R["octv_outside_oldmask_stats"])

# ------------------------------------------------------------------ z-averaged stripe map (stripes along z at fixed x,y)
say("--- z-averaged stripe maps ---")
D, H, Wd = octv.shape
# pick a z range where the eroded mask has many voxels
zc = np.where(cnt_v[0] > 0.5 * np.nanmax(cnt_v[0]))[0]
zmid = int(np.median(zc))
sm = {}
for L in (12, 50, 200):
    z0, z1 = max(0, zmid - L // 2), min(D, zmid + L // 2)
    blk = np.asarray(octv[z0:z1]).astype(np.float32); mk = mve[z0:z1]
    num = np.where(mk, blk, 0).sum(0); den = mk.sum(0)
    img = np.where(den > 0.8 * (z1 - z0), num / np.maximum(den, 1), np.nan)
    good = np.isfinite(img)
    # high-pass in-plane: subtract a large gaussian smooth of the z-mean image
    filled = np.where(good, img, np.nanmean(img))
    lo = ndimage.gaussian_filter(filled, 15); hp = img - lo
    # expected residual if voxels were independent: tissue std / sqrt(L)
    sm[L] = {"z_range": [int(z0), int(z1)], "hp_std": float(np.nanstd(hp[good])), "expected_iid": float(R["octv_tissue_stats"]["std"] / np.sqrt(z1 - z0)),
             "hp_p2p95": float(np.nanpercentile(hp[good], 97.5) - np.nanpercentile(hp[good], 2.5))}
    # profiles of the hp image along x and y (tile seams would be lines)
    px = np.nanmean(np.where(good, hp, np.nan), axis=0); py = np.nanmean(np.where(good, hp, np.nan), axis=1)
    for nm, pr in (("x", px), ("y", py)):
        okp = np.isfinite(pr)
        if okp.sum() > 50:
            ii = np.where(okp)[0]; prr = pr[ii.min():ii.max() + 1]; prr = np.where(np.isfinite(prr), prr, 0)
            n = prr.size; f = np.fft.rfftfreq(n); P = np.abs(np.fft.rfft(prr * np.hanning(n))) ** 2; P[f < 1 / 80.] = 0
            k = int(np.argmax(P)); sm[L][f"hp_line_{nm}_std"] = float(np.nanstd(pr)); sm[L][f"hp_line_{nm}_fft_period"] = float(1 / f[k]) if f[k] > 0 else None
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    v0, v1 = np.nanpercentile(img, [1, 99])
    ax[0].imshow(img, cmap="gray", vmin=v0, vmax=v1); ax[0].set_title(f"octv mean over z {z0}-{z1} ({L} slices), eroded-mask columns")
    a = np.nanpercentile(np.abs(hp), 99)
    ax[1].imshow(hp, cmap="gray", vmin=-a, vmax=a); ax[1].set_title(f"in-plane high-pass (sigma 15 vox): std {sm[L]['hp_std']:.0f} (iid expect {sm[L]['expected_iid']:.0f})")
    sl = np.asarray(octv[zmid]).astype(np.float32); ax[2].imshow(np.where(mve[zmid], sl, np.nan), cmap="gray", vmin=v0 - 3 * (v1 - v0), vmax=v1 + 1 * (v1 - v0)); ax[2].set_title(f"single slice z={zmid}")
    for a_ in ax: a_.set_xlabel("x"); a_.set_ylabel("y")
    plt.tight_layout(); plt.savefig(f"{OUT}/zmean_stripemap_L{L}.png", dpi=90); plt.close(fig)
    say("zmean L", L, sm[L])
R["zmean_stripe_maps"] = sm
np.save(f"{OUT}/zmean_hp_L50.npy", hp.astype(np.float32))

# ------------------------------------------------------------------ orthogonal planes (octv full res) through the mask centroid
say("--- planes ---")
cz, cy, cx = [int(v) for v in ndimage.center_of_mass(mve)]
fig, ax = plt.subplots(2, 3, figsize=(24, 15))
def show(a_, img, title, vmin, vmax):
    a_.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"); a_.set_title(title)
v0, v1 = R["octv_tissue_stats"]["p5"] * 0.3, R["octv_tissue_stats"]["p95"] * 1.3
zx = np.asarray(octv[:, cy, :]).astype(np.float32); zy = np.asarray(octv[:, :, cx]).astype(np.float32); xy = np.asarray(octv[cz]).astype(np.float32)
show(ax[0, 0], zx, f"octv z-x plane (y={cy}); rows=z(depth), cols=x", v0, v1)
show(ax[0, 1], zy, f"octv z-y plane (x={cx}); rows=z, cols=y", v0, v1)
show(ax[0, 2], xy, f"octv x-y plane (z={cz}); rows=y, cols=x", v0, v1)
# crops 300x300 at full res around the centroid
c = 150
show(ax[1, 0], zx[cz - c:cz + c, cx - c:cx + c], f"z-x crop 300 (z {cz-c}-{cz+c}, x {cx-c}-{cx+c})", v0, v1)
show(ax[1, 1], zy[cz - c:cz + c, cy - c:cy + c], f"z-y crop 300", v0, v1)
show(ax[1, 2], xy[cy - c:cy + c, cx - c:cx + c], f"x-y crop 300", v0, v1)
plt.tight_layout(); plt.savefig(f"{OUT}/planes_octv.png", dpi=80); plt.close(fig)
np.savez(f"{OUT}/planes_octv.npz", zx=zx, zy=zy, xy=xy, c=(cz, cy, cx))
# oct150 planes
cz1, cy1, cx1 = [int(v) for v in ndimage.center_of_mass(m150e)]
fig, ax = plt.subplots(1, 3, figsize=(24, 8))
o = o150a
show(ax[0], o[:, cy1, :], f"oct150 z-x (y={cy1})", v0, v1); show(ax[1], o[:, :, cx1], f"oct150 z-y (x={cx1})", v0, v1); show(ax[2], o[cz1], f"oct150 x-y (z={cz1})", v0, v1)
plt.tight_layout(); plt.savefig(f"{OUT}/planes_oct150.png", dpi=80); plt.close(fig)
# mask overlay + zero blocks on oct150
fig, ax = plt.subplots(1, 3, figsize=(24, 8))
for a_, img, mk, zz, t in ((ax[0], o[:, cy1, :], m150[:, cy1, :], z150[:, cy1, :], "z-x"), (ax[1], o[:, :, cx1], m150[:, :, cx1], z150[:, :, cx1], "z-y"), (ax[2], o[cz1], m150[cz1], z150[cz1], "x-y")):
    a_.imshow(img, cmap="gray", vmin=0, vmax=v1); a_.contour(mk, levels=[0.5], colors="r", linewidths=0.6); a_.contour(m150e[:, cy1, :] if t == "z-x" else (m150e[:, :, cx1] if t == "z-y" else m150e[cz1]), levels=[0.5], colors="y", linewidths=0.6)
    a_.imshow(np.ma.masked_where(~zz, zz), cmap="cool", alpha=0.6); a_.set_title(f"oct150 {t}: red=old mask, yellow=eroded 2mm, cyan=exact zeros")
plt.tight_layout(); plt.savefig(f"{OUT}/mask_zeros_oct150.png", dpi=80); plt.close(fig)

# ------------------------------------------------------------------ directional anisotropy of fine structure: gradient energy along each axis in the eroded tissue
say("--- gradient anisotropy ---")
c = 100; blk = np.asarray(octv[cz - c:cz + c, cy - c:cy + c, cx - c:cx + c]).astype(np.float32); mk = mve[cz - c:cz + c, cy - c:cy + c, cx - c:cx + c]
g = [np.diff(blk, axis=i) for i in range(3)]
R["grad_energy_crop200"] = {"dz": float(np.mean(g[0][mk[1:] & mk[:-1]] ** 2)), "dy": float(np.mean(g[1][mk[:, 1:] & mk[:, :-1]] ** 2)), "dx": float(np.mean(g[2][mk[:, :, 1:] & mk[:, :, :-1]] ** 2))}
# 3D power spectrum of the crop: energy on the axes
bw = blk - ndimage.gaussian_filter(blk, 8)
Fq = np.abs(np.fft.fftn(bw * np.hanning(2 * c)[:, None, None] * np.hanning(2 * c)[None, :, None] * np.hanning(2 * c)[None, None, :])) ** 2
Fq = np.fft.fftshift(Fq); cc = c
# energy concentrated on the kz axis (structures constant in x,y -> planes), on kx-ky plane (structures constant in z -> lines along z)
tot = Fq.sum()
onz = Fq[:, cc - 1:cc + 2, cc - 1:cc + 2].sum() / tot; onxy = Fq[cc - 1:cc + 2].sum() / tot
R["spectrum_crop200"] = {"frac_on_kz_axis(planes_perp_z)": float(onz), "frac_on_kxky_plane(lines_along_z)": float(onxy), "note": "a 3-voxel-wide axis/plane of a 200^3 spectrum; iid expectation ~ 9/40000 and 3/200"}
say(R["grad_energy_crop200"], R["spectrum_crop200"])
# radial-averaged 2D power spectrum of kx-ky plane at kz=0 -> stripe pitch
pxy = Fq[cc]; kx = np.fft.fftshift(np.fft.fftfreq(2 * c)); KX, KY = np.meshgrid(kx, kx); kr = np.sqrt(KX ** 2 + KY ** 2)
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax[0].imshow(np.log10(pxy + 1), cmap="magma", extent=[kx.min(), kx.max(), kx.min(), kx.max()]); ax[0].set_title("log power, kz=0 plane (kx,ky): lines along z -> energy here")
pzx = Fq[:, cc, :]; ax[1].imshow(np.log10(pzx + 1), cmap="magma", extent=[kx.min(), kx.max(), kx.min(), kx.max()]); ax[1].set_title("log power, ky=0 plane (kz rows, kx cols)")
pzy = Fq[:, :, cc]; ax[2].imshow(np.log10(pzy + 1), cmap="magma", extent=[kx.min(), kx.max(), kx.min(), kx.max()]); ax[2].set_title("log power, kx=0 plane (kz rows, ky cols)")
plt.tight_layout(); plt.savefig(f"{OUT}/spectrum_crop200.png", dpi=90); plt.close(fig)
# 1-D spectrum along kz (mean over kx,ky within a small radius) to find the depth period in the texture itself
Pz = Fq[:, cc - 3:cc + 4, cc - 3:cc + 4].mean(axis=(1, 2)); fz = kx
sel = fz > 1 / 80.
k = np.argmax(Pz * sel); R["spectrum_crop200"]["kz_axis_peak_period_vox"] = float(1 / fz[k]) if fz[k] > 0 else None
say("kz axis peak period vox", R["spectrum_crop200"]["kz_axis_peak_period_vox"])

json.dump(R, open(f"{OUT}/characterisation.json", "w"), indent=1, default=float)
say("done; wrote", OUT)
