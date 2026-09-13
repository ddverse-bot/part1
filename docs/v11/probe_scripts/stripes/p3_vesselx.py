#!/usr/bin/env python
"""P3 part 4: section-phase modulation of the vessel mask (global, existing octv_vessels.npy) and before/after destripe
on the crops; component-based vessel-likeness.  CPU only."""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from scipy import ndimage
import torch; torch.set_num_threads(4)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0 = time.time()
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)
BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration"); sys.path.insert(0, os.path.join(BASE, "octreg"))
from octreg.features import frangi_dark_vesselness
W = os.path.join(BASE, "work/xiangrui_I58bs"); OUT = os.path.join(BASE, "work/probe_xr/stripes")
PX = 7.518; PT = 70.5
SIG = (0.6, 0.9, 1.325, 1.8, 2.65); PREP_GAMMA = [7907.89208984375, 5673.7646484375, 5811.470703125, 5946.33544921875, 6294.06396484375]
R = {}
m150e = np.load(f"{OUT}/mask150_eroded2mm.npy"); m150 = np.load(f"{W}/oct150_mask.npy"); A150 = np.load(f"{W}/oct150_affine.npy"); Av = np.load(f"{W}/octv_affine.npy")
shape_v = (728, 1006, 797)
M_ = np.linalg.inv(A150) @ Av
idx = [np.clip(np.rint(M_[r, r] * np.arange(shape_v[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
mve = m150e[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]

def fold_counts(cnt, ok, P, nb=15, i0=0):
    """phase-fold a per-plane count (normalised by the per-plane mask size) -> modulation depth."""
    ii = np.where(ok)[0]; r = cnt[ii]; ph = ((ii + i0) % P) / P; b = np.minimum((ph * nb).astype(int), nb - 1)
    w = np.array([r[b == k].mean() for k in range(nb)]); base = r.mean()
    return {"folded_rel": (w / base).round(3).tolist(), "max_over_min": float(w.max() / max(w.min(), 1e-9)), "p2p_rel": float((w.max() - w.min()) / base),
            "var_frac_explained": float(np.var(np.interp(ph, (np.arange(nb) + 0.5) / nb, w)) / max(np.var(r), 1e-12))}

# ------------------------------------------------------------------ global: existing prep vessel mask
say("--- global octv_vessels per-plane density ---")
ves = np.load(f"{W}/octv_vessels.npy", mmap_mode="r")
cx = np.zeros(shape_v[2]); cy = np.zeros(shape_v[1]); cz = np.zeros(shape_v[0]); mx = np.zeros(shape_v[2]); my = np.zeros(shape_v[1]); mz = np.zeros(shape_v[0])
for z0 in range(0, shape_v[0], 32):
    v = np.asarray(ves[z0:z0 + 32]) & mve[z0:z0 + 32]; m = mve[z0:z0 + 32]
    cz[z0:z0 + v.shape[0]] = v.sum((1, 2)); mz[z0:z0 + v.shape[0]] = m.sum((1, 2)); cy += v.sum((0, 2)); my += m.sum((0, 2)); cx += v.sum((0, 1)); mx += m.sum((0, 1))
dens = lambda c, m: np.where(m > 2000, c / np.maximum(m, 1), np.nan)
dx_, dy_, dz_ = dens(cx, mx), dens(cy, my), dens(cz, mz)
R["global_vessel_mask"] = {"n_in_eroded": int(cx.sum()), "frac_in_eroded": float(cx.sum() / mx.sum()),
                           "x_fold_P7.518": fold_counts(dx_, np.isfinite(dx_), PX), "z_fold_P70.5": fold_counts(dz_, np.isfinite(dz_), PT, 20), "y_fold_P70.5": fold_counts(dy_, np.isfinite(dy_), PT, 20),
                           "x_fold_random_P7.9(control)": fold_counts(dx_, np.isfinite(dx_), 7.9)}
say(json.dumps(R["global_vessel_mask"]))
fig, ax = plt.subplots(3, 1, figsize=(15, 10))
ax[0].plot(dx_, lw=0.7); ax[0].set_title("existing octv_vessels: fraction of eroded-mask voxels flagged, per x-plane (period 7.518 = section)"); ax[0].set_xlabel("x")
ax[1].plot(np.arange(15) / 15 * PX, R["global_vessel_mask"]["x_fold_P7.518"]["folded_rel"], marker="o"); ax[1].set_title("phase-folded at 7.518 vox (relative density): max/min %.2f" % R["global_vessel_mask"]["x_fold_P7.518"]["max_over_min"])
ax[2].plot(dz_, lw=0.7, label="per z-plane"); ax[2].plot(dy_, lw=0.7, label="per y-plane"); ax[2].legend(); ax[2].set_title("per z / y plane density (tile period 70.5)")
plt.tight_layout(); plt.savefig(f"{OUT}/vessel_density_per_plane_global.png", dpi=90); plt.close(fig)

# ------------------------------------------------------------------ crops: Frangi before/after, per-x-plane fold + component elongation
octv = np.load(f"{W}/octv.npy", mmap_mode="r"); dst = np.load(f"{OUT}/octv_destriped.npy", mmap_mode="r")
def masked_lp(img, m, sig):
    num = ndimage.gaussian_filter(np.where(m, img, 0.0).astype(np.float32), sig, truncate=3.0); den = ndimage.gaussian_filter(m.astype(np.float32), sig, truncate=3.0)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0), den > 0.05
def destripe_x(vol, valid, sy, sz, sx):
    A_, ok = masked_lp(vol, valid, (sz, sy, 0)); B_, _ = masked_lp(A_, ok, (0, 0, sx)); return np.where(valid, vol - (A_ - B_), vol).astype(np.float32)
def frangi(vol):
    x = torch.from_numpy(np.ascontiguousarray(vol))[None]; best = None
    for s, g in zip(SIG, PREP_GAMMA):
        v = frangi_dark_vesselness(x, sigmas_vox=(s,), gamma=(g,))[0].numpy(); best = v if best is None else np.maximum(best, v)
    return best
def comp_stats(top):
    lab, n = ndimage.label(top)
    if n == 0: return {}
    sizes = ndimage.sum(top, lab, np.arange(1, n + 1)); big = np.where(sizes >= 30)[0]
    elong = []; axes = []
    for k in big[:4000]:
        p = np.argwhere(lab == (k + 1)).astype(np.float32); p -= p.mean(0); C = p.T @ p / len(p); w, V = np.linalg.eigh(C)
        elong.append(np.sqrt(w[2] / max(w[1], 1e-6))); axes.append(np.abs(V[:, 2]))
    elong = np.array(elong); axes = np.array(axes) if len(axes) else np.zeros((0, 3))
    e3 = elong >= 3
    c15 = np.cos(np.radians(15))
    return {"n_components": int(n), "frac_voxels_in_comp_ge30": float(sizes[big].sum() / top.sum()), "n_comp_ge30": int(len(big)), "median_size_ge30": float(np.median(sizes[big])) if len(big) else None,
            "frac_comp_ge30_elongated_ge3": float(e3.mean()) if len(elong) else None, "elongated_axis_within15_z": float((axes[e3, 0] > c15).mean()) if e3.any() else None,
            "elongated_axis_within15_y": float((axes[e3, 1] > c15).mean()) if e3.any() else None, "elongated_axis_within15_x": float((axes[e3, 2] > c15).mean()) if e3.any() else None,
            "elongated_mean_abs_cos_x": float(axes[e3, 2].mean()) if e3.any() else None}
CROPS = {"A_centroid": (370, 509, 398), "B_striped": (400, 620, 230)}; c = 100; MARG = 40
R["crops"] = {}
for cname, (cz_, cy_, cx_) in CROPS.items():
    lo = np.array([cz_ - c - MARG, cy_ - c - MARG, cx_ - c - MARG]); hi = lo + 2 * (c + MARG); lo = np.maximum(lo, 0); hi = np.minimum(hi, shape_v)
    off = np.array([cz_ - c, cy_ - c, cx_ - c]) - lo; sl = tuple(slice(int(off[i]), int(off[i]) + 2 * c) for i in range(3))
    big = np.asarray(octv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); valid = big > 0
    emask = mve[cz_ - c:cz_ + c, cy_ - c:cy_ + c, cx_ - c:cx_ + c]
    variants = {"orig": big[sl].copy(), "xff_s12_sx6 (full-volume file)": np.asarray(dst[cz_ - c:cz_ + c, cy_ - c:cy_ + c, cx_ - c:cx_ + c]).astype(np.float32),
                "xff_s12_sx12": destripe_x(big, valid, 12, 12, 12)[sl], "xff_s12_sx20": destripe_x(big, valid, 12, 12, 20)[sl]}
    R["crops"][cname] = {}
    fig, ax = plt.subplots(3, len(variants), figsize=(5 * len(variants), 14))
    for j, (vn, vol) in enumerate(variants.items()):
        best = frangi(vol); vals = best[emask]; thr = float(np.percentile(vals, 99)); top = (best > thr) & emask
        cnt = top.sum((0, 1)).astype(float); mk = emask.sum((0, 1)); d = np.where(mk > 500, cnt / np.maximum(mk, 1), np.nan)
        fx = fold_counts(d, np.isfinite(d), PX, i0=cx_ - c)
        vm = best.mean((0, 1)); vmf = fold_counts(vm, np.ones_like(vm, bool), PX, i0=cx_ - c)
        cs = comp_stats(top)
        # x-profile residual of the volume itself (stripe energy)
        num = (vol * emask).sum((0, 1)); den = emask.sum((0, 1)); p = num / np.maximum(den, 1); dpp = p - ndimage.uniform_filter1d(p, 40, mode="nearest"); pf = fold_counts(dpp + p.mean(), np.ones_like(p, bool), PX, i0=cx_ - c)
        R["crops"][cname][vn] = {"top1_thr": thr, "n_top": int(top.sum()), "top_density_x_fold": fx, "mean_vesselness_x_fold": vmf, "xprofile_resid_std": float(dpp.std()), "xprofile_fold_p2p": float(pf["p2p_rel"] * p.mean()), "components": cs}
        say(cname, vn, "top1 x-fold max/min %.2f varfrac %.2f | vesselness x-fold max/min %.2f | xprof std %.0f | comp: elong>=3 frac %s, axis x15 %s z15 %s y15 %s" % (
            fx["max_over_min"], fx["var_frac_explained"], vmf["max_over_min"], dpp.std(), cs.get("frac_comp_ge30_elongated_ge3"), cs.get("elongated_axis_within15_x"), cs.get("elongated_axis_within15_z"), cs.get("elongated_axis_within15_y")))
        v0, v1 = np.percentile(variants["orig"][emask], [1, 99])
        ax[0, j].imshow(vol[:, c, :], cmap="gray", vmin=v0, vmax=v1); ax[0, j].set_title(f"{cname} {vn}\nz-x slice")
        ax[1, j].imshow(best[:, c, :], cmap="magma", vmin=0, vmax=float(np.percentile(vals, 99.9))); ax[1, j].set_title("vesselness z-x slice")
        ax[2, j].plot(np.arange(15) / 15 * PX, fx["folded_rel"], marker="o", label="top-1% density"); ax[2, j].plot(np.arange(15) / 15 * PX, vmf["folded_rel"], marker="s", label="mean vesselness"); ax[2, j].axhline(1, color="k", lw=0.5); ax[2, j].legend(); ax[2, j].set_title("phase-folded at section period 7.518 vox (rel.)"); ax[2, j].set_xlabel("phase (vox)")
    plt.tight_layout(); plt.savefig(f"{OUT}/vessel_sectionphase_{cname}.png", dpi=75); plt.close(fig)
    json.dump(R, open(f"{OUT}/vesselx_results.json", "w"), indent=1, default=float)
json.dump(R, open(f"{OUT}/vesselx_results.json", "w"), indent=1, default=float)
say("done")
