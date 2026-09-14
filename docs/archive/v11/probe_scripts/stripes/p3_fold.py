#!/usr/bin/env python
"""P3 part 2b: phase-folded waveforms of the x-periodicity and the z/y periodic bumps (from saved profiles)."""
import os, json, numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration"); OUT = os.path.join(BASE, "work/probe_xr/stripes")
pr = np.load(f"{OUT}/profiles.npz"); sn = np.load(f"{OUT}/seam_ncc.npz")
R = {}
def detr(p, cnt, mn, win):
    ok = (cnt >= mn) & np.isfinite(p); ii = np.where(ok)[0]; i0, i1 = ii.min(), ii.max() + 1
    pp = p[i0:i1].copy(); okk = ok[i0:i1]; pp[~okk] = np.interp(np.where(~okk)[0], np.where(okk)[0], pp[okk])
    return pp - ndimage.uniform_filter1d(pp, win, mode="nearest"), i0
def fold(d, i0, P, nb=15):
    ph = ((np.arange(d.size) + i0) % P) / P; b = np.minimum((ph * nb).astype(int), nb - 1)
    return np.array([d[b == k].mean() for k in range(nb)]), np.array([d[b == k].std() / np.sqrt(max((b == k).sum(), 1)) for k in range(nb)])
def best_period(d, i0, lo, hi):
    Ps = np.linspace(lo, hi, 401); v = [fold(d, i0, P)[0].std() for P in Ps]; return float(Ps[int(np.argmax(v))]), float(max(v))
# x: mean profile (octv) and adjacent-plane NCC
dx, ix0 = detr(pr["means_v2"], pr["cnt_v2"], 2000, 60)
Px, vx = best_period(dx, ix0, 7.2, 7.8); wx, ex = fold(dx, ix0, Px)
nx = np.nan_to_num(sn["x1"], nan=np.nanmedian(sn["x1"])); dnx = nx - ndimage.uniform_filter1d(nx, 60, mode="nearest"); Pn, vn = best_period(dnx, int(sn["x0"]), 7.2, 7.8); wn, en = fold(dnx, int(sn["x0"]), Pn)
R["x"] = {"best_period_profile": Px, "folded_profile_std": vx, "folded_profile_p2p": float(wx.max() - wx.min()), "folded_profile": wx.round(1).tolist(), "residual_after_fold_std": float((dx - np.interp(((np.arange(dx.size) + ix0) % Px) / Px, (np.arange(15) + 0.5) / 15, wx)).std()),
          "detrended_std": float(dx.std()), "best_period_ncc": Pn, "folded_ncc": wn.round(3).tolist(), "period_mm_at_0.04": Px * 0.04, "period_slices_at_20um": Px * 2}
# dropout planes: strongly negative isolated dips in the x profile
pk, pp = find_peaks(-dx, prominence=250)
R["x"]["dropout_planes"] = {"n": int(len(pk)), "positions_octv": (pk + ix0).tolist(), "depths": (-dx[pk]).round(0).tolist()}
# z: bumps in the adjacent-slice NCC
for nm, key, off in (("z", "z1", "z0"), ("y", "y1", "y0")):
    s = np.nan_to_num(sn[key], nan=np.nanmedian(sn[key])); ds = s - ndimage.uniform_filter1d(s, 60, mode="nearest")
    pk, pp = find_peaks(ds, prominence=0.02, distance=10); spac = np.diff(pk)
    dips, dp = find_peaks(-ds, prominence=0.02, distance=10)
    ac = np.correlate(ds - ds.mean(), ds - ds.mean(), "full")[ds.size - 1:]; ac /= ac[0]; apk, _ = find_peaks(ac[1:200], height=0.1, distance=5)
    R[nm] = {"ncc_bump_positions": (pk + int(sn[off])).tolist(), "bump_spacings": spac.tolist(), "median_bump_spacing": float(np.median(spac)) if spac.size else None, "bump_prominences": pp["prominences"].round(3).tolist(),
             "dip_positions": (dips + int(sn[off])).tolist(), "dip_prominences": dp["prominences"].round(3).tolist(), "ac_peaks": [(int(p + 1), round(float(ac[p + 1]), 3)) for p in apk[:6]]}
    if nm == "z":
        Pz, vz = best_period(ds, int(sn[off]), 60, 80); R[nm]["best_period_ncc_60_80"] = Pz; R[nm]["folded_ncc"] = fold(ds, int(sn[off]), Pz)[0].round(3).tolist()
        dz, iz0 = detr(pr["means_v0"], pr["cnt_v0"], 2000, 200); wz, _ = fold(dz, iz0, Pz); R[nm]["folded_mean_profile_at_P"] = wz.round(0).tolist(); R[nm]["folded_mean_profile_p2p"] = float(wz.max() - wz.min())
    else:
        dy, iy0 = detr(pr["means_v1"], pr["cnt_v1"], 2000, 200); Py, vy = best_period(dy, iy0, 65, 75); wy, _ = fold(dy, iy0, Py, 20); R[nm]["best_period_profile_65_75"] = Py; R[nm]["folded_mean_profile"] = wy.round(0).tolist(); R[nm]["folded_p2p"] = float(wy.max() - wy.min())
        Py2, vy2 = best_period(dy, iy0, 33, 38); R[nm]["best_period_profile_33_38"] = Py2; R[nm]["fold_std_70_vs_35"] = [vy, vy2]
print(json.dumps(R, indent=1))
json.dump(R, open(f"{OUT}/fold_results.json", "w"), indent=1)
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].errorbar((np.arange(15) + 0.5) / 15 * Px, wx, ex, marker="o"); ax[0, 0].set_title(f"x: phase-folded detrended mean profile, P={Px:.3f} vox ({Px*0.04:.3f} mm = {Px*2:.1f} slices @20um)"); ax[0, 0].set_xlabel("phase (vox)")
ax[0, 1].plot((np.arange(15) + 0.5) / 15 * Pn, wn, marker="o"); ax[0, 1].set_title(f"x: phase-folded NCC(x,x+1) of high-passed z-y planes, P={Pn:.3f}"); ax[0, 1].set_xlabel("phase (vox)")
ax[1, 0].plot(np.arange(sn["z1"].size) + int(sn["z0"]), sn["z1"], lw=0.7); [ax[1, 0].axvline(p, color="r", lw=0.5) for p in R["z"]["ncc_bump_positions"]]; ax[1, 0].set_title(f"z: NCC(z,z+1) with detected bumps; median spacing {R['z']['median_bump_spacing']} vox")
ax[1, 1].plot(np.arange(20) / 20 * R["y"]["best_period_profile_65_75"], R["y"]["folded_mean_profile"], marker="o"); ax[1, 1].set_title(f"y: phase-folded mean profile at P={R['y']['best_period_profile_65_75']:.1f} vox")
plt.tight_layout(); plt.savefig(f"{OUT}/fold_waveforms.png", dpi=90)
