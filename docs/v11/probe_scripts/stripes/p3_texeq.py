#!/usr/bin/env python
"""P3 part 5: can the vessel channel be salvaged?  per-x-plane texture equalisation + large-scale-only Frangi on the crops."""
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
PX = 7.518
SIG = (0.6, 0.9, 1.325, 1.8, 2.65); PREP_GAMMA = [7907.89208984375, 5673.7646484375, 5811.470703125, 5946.33544921875, 6294.06396484375]
m150e = np.load(f"{OUT}/mask150_eroded2mm.npy"); m150 = np.load(f"{W}/oct150_mask.npy"); A150 = np.load(f"{W}/oct150_affine.npy"); Av = np.load(f"{W}/octv_affine.npy")
shape_v = (728, 1006, 797); M_ = np.linalg.inv(A150) @ Av
idx = [np.clip(np.rint(M_[r, r] * np.arange(shape_v[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
mve = m150e[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
octv = np.load(f"{W}/octv.npy", mmap_mode="r"); dst = np.load(f"{OUT}/octv_destriped.npy", mmap_mode="r")

def fold_counts(cnt, ok, P, nb=15, i0=0):
    ii = np.where(ok)[0]; r = cnt[ii]; ph = ((ii + i0) % P) / P; b = np.minimum((ph * nb).astype(int), nb - 1)
    w = np.array([r[b == k].mean() for k in range(nb)]); base = r.mean()
    return {"folded_rel": (w / base).round(3).tolist(), "max_over_min": float(w.max() / max(w.min(), 1e-9)), "var_frac_explained": float(np.var(np.interp(ph, (np.arange(nb) + 0.5) / nb, w)) / max(np.var(r), 1e-12))}
def frangi(vol, sig, gam):
    x = torch.from_numpy(np.ascontiguousarray(vol))[None]; best = None
    for s, g in zip(sig, gam):
        v = frangi_dark_vesselness(x, sigmas_vox=(s,), gamma=(g,))[0].numpy(); best = v if best is None else np.maximum(best, v)
    return best
def comp_stats(top):
    lab, n = ndimage.label(top)
    if n == 0: return {}
    sizes = ndimage.sum(top, lab, np.arange(1, n + 1)); big = np.where(sizes >= 30)[0]; elong = []; axes = []
    for k in big[:4000]:
        p = np.argwhere(lab == (k + 1)).astype(np.float32); p -= p.mean(0); C = p.T @ p / len(p); w, V = np.linalg.eigh(C); elong.append(np.sqrt(w[2] / max(w[1], 1e-6))); axes.append(np.abs(V[:, 2]))
    elong = np.array(elong); axes = np.array(axes) if len(axes) else np.zeros((0, 3)); e3 = elong >= 3; c15 = np.cos(np.radians(15))
    return {"n_components": int(n), "median_size": float(np.median(sizes)), "frac_voxels_in_comp_ge30": float(sizes[big].sum() / top.sum()), "n_comp_ge30": int(len(big)),
            "frac_comp_ge30_elongated_ge3": float(e3.mean()) if len(elong) else None, "elong_axis_within15_x": float((axes[e3, 2] > c15).mean()) if e3.any() else None,
            "elong_axis_within15_z": float((axes[e3, 0] > c15).mean()) if e3.any() else None, "elong_axis_within15_y": float((axes[e3, 1] > c15).mean()) if e3.any() else None,
            "elong_mean_abs_cos_x": float(axes[e3, 2].mean()) if e3.any() else None}
def texeq(vol, emask, sig_inplane=2.0, ref_win=15):
    """per-x-plane texture equalisation: in-plane (z,y) high-pass, rescale each plane's HP std to the running median over ref_win planes."""
    lp = ndimage.gaussian_filter(vol, (sig_inplane, sig_inplane, 0)); hp = vol - lp
    s = np.array([hp[:, :, i][emask[:, :, i]].std() if emask[:, :, i].sum() > 500 else np.nan for i in range(vol.shape[2])])
    ok = np.isfinite(s); s[~ok] = np.nanmedian(s); ref = ndimage.median_filter(s, size=ref_win, mode="nearest"); g = np.clip(ref / np.maximum(s, 1e-6), 0.3, 3.0)
    return (lp + hp * g[None, None, :]).astype(np.float32), s, ref
def evaluate(vol, emask, x0, sig, gam, tag):
    best = frangi(vol, sig, gam); vals = best[emask]; thr = float(np.percentile(vals, 99)); top = (best > thr) & emask
    cnt = top.sum((0, 1)).astype(float); mk = emask.sum((0, 1)); d = np.where(mk > 500, cnt / np.maximum(mk, 1), np.nan); fx = fold_counts(d, np.isfinite(d), PX, i0=x0)
    vm = best.mean((0, 1)); vmf = fold_counts(vm, np.ones_like(vm, bool), PX, i0=x0); cs = comp_stats(top)
    out = {"tag": tag, "top1_thr": thr, "top_density_x_fold_max_over_min": fx["max_over_min"], "top_density_x_fold_varfrac": fx["var_frac_explained"], "vesselness_x_fold_max_over_min": vmf["max_over_min"], "components": cs}
    say(tag, "top1 x-fold max/min %.2f varfrac %.2f | vess x-fold %.2f | comps: n %d med %.0f elong>=3 %.2f axis x15 %s z15 %s y15 %s mean|cos x| %s" % (
        fx["max_over_min"], fx["var_frac_explained"], vmf["max_over_min"], cs["n_components"], cs["median_size"], cs["frac_comp_ge30_elongated_ge3"] or -1, cs["elong_axis_within15_x"], cs["elong_axis_within15_z"], cs["elong_axis_within15_y"], cs["elong_mean_abs_cos_x"]))
    return out, best, top
CROPS = {"A_centroid": (370, 509, 398), "B_striped": (400, 620, 230)}; c = 100
R = {}
for cname, (cz_, cy_, cx_) in CROPS.items():
    sl = (slice(cz_ - c, cz_ + c), slice(cy_ - c, cy_ + c), slice(cx_ - c, cx_ + c)); emask = mve[sl]
    v_orig = np.asarray(octv[sl]).astype(np.float32); v_xff = np.asarray(dst[sl]).astype(np.float32)
    v_teq, s_pl, ref_pl = texeq(v_xff, emask)
    R[cname] = {"plane_hp_std_fold": fold_counts(s_pl, np.isfinite(s_pl), PX, i0=cx_ - c), "plane_hp_std_median": float(np.median(s_pl))}
    say(cname, "per-plane HP std section-phase fold (before texeq):", R[cname]["plane_hp_std_fold"])
    res = {}
    res["xff_all5"] = evaluate(v_xff, emask, cx_ - c, SIG, PREP_GAMMA, f"{cname} xff all5")[0]
    res["xff+texeq_all5"] = evaluate(v_teq, emask, cx_ - c, SIG, PREP_GAMMA, f"{cname} xff+texeq all5")[0]
    res["xff_large3"] = evaluate(v_xff, emask, cx_ - c, SIG[2:], PREP_GAMMA[2:], f"{cname} xff large3 (1.325,1.8,2.65)")[0]
    r4, best4, top4 = evaluate(v_teq, emask, cx_ - c, SIG[2:], PREP_GAMMA[2:], f"{cname} xff+texeq large3"); res["xff+texeq_large3"] = r4
    res["orig_large3"] = evaluate(v_orig, emask, cx_ - c, SIG[2:], PREP_GAMMA[2:], f"{cname} orig large3")[0]
    R[cname]["variants"] = res
    v0, v1 = np.percentile(v_orig[emask], [1, 99])
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax[0, 0].imshow(v_xff[:, c, :], cmap="gray", vmin=v0, vmax=v1); ax[0, 0].set_title(f"{cname} xff z-x slice")
    ax[0, 1].imshow(v_teq[:, c, :], cmap="gray", vmin=v0, vmax=v1); ax[0, 1].set_title("xff + per-x-plane texture equalisation")
    ax[0, 2].plot(s_pl, lw=0.7, label="per-plane HP std"); ax[0, 2].plot(ref_pl, lw=0.7, label="running median ref"); ax[0, 2].legend(); ax[0, 2].set_title("in-plane high-pass std per x-plane (section-phase modulated)")
    ax[1, 0].imshow(best4[:, c, :], cmap="magma", vmin=0, vmax=float(np.percentile(best4[emask], 99.9))); ax[1, 0].set_title("vesselness (texeq, large3) z-x slice")
    ax[1, 1].imshow(top4.max(1), cmap="gray"); ax[1, 1].set_title("top-1% (texeq, large3) MIP over y (rows z, cols x)")
    ax[1, 2].imshow(top4.max(0), cmap="gray"); ax[1, 2].set_title("top-1% (texeq, large3) MIP over z (rows y, cols x)")
    plt.tight_layout(); plt.savefig(f"{OUT}/vessel_texeq_{cname}.png", dpi=75); plt.close(fig)
    json.dump(R, open(f"{OUT}/texeq_results.json", "w"), indent=1, default=float)
say("done")
