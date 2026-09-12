#!/usr/bin/env python3
"""Diagnostic: is the true pose among the search candidates?  Run the 0.6 mm search with a large top-K, refine
each candidate rigidly (0.6 -> 0.3 mm), and score every candidate with the held-out vessel test.  Also report the
MRI vessel-label centroid (the labels are concentrated around the block, so it is a weak location hint)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, avg_pool_iso, world_bbox_to_voxel
from octreg.features import build_features
from octreg.refine import Refiner
from octreg.evaluate import vessel_distance_stats
from octreg.search import FFTSearcher
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--features", default="parser"); ap.add_argument("--parser-channels", default="wm_gm"); ap.add_argument("--topk", type=int, default=48)
ap.add_argument("--n-rot", type=int, default=3000); ap.add_argument("--crop-half-mm", type=float, default=30.0); ap.add_argument("--min-overlap", type=float, default=0.85)
ap.add_argument("--sulcus-weight", action="store_true", help="include sulcal background (closing - tissue) in the OCT weight mask")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
ref_T = np.array(json.load(open(a.work / "ref_author_T.json")))
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves))
ves_w = (A_mri @ np.c_[ves_ijk, np.ones(len(ves_ijk))].T).T[:, :3]
print("MRI vessel labels: centroid mm", np.round(ves_w.mean(0), 1).tolist(), "std", np.round(ves_w.std(0), 1).tolist(), "n", len(ves_w))
centre = (ref_T @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - a.crop_half_mm, centre + a.crop_half_mm)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
mri_tis = to_t(load_region("tissue") > 0.5)[None].float()
if a.features.startswith("parser"):
    FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
    if a.parser_channels == "tissue_wm_gm":
        FM = torch.cat([to_t(load_region("tissue")), FM], 0)
    FO = build_features("parser", None, to_t(oct_mask)[None].float(), prob=to_t(oct_prob), parser_channels=a.parser_channels)
    if a.features == "parser_vessel":
        FM = torch.cat([FM, build_features("vessel", to_t(mri_reg)[None], mri_tis)], 0); FO = torch.cat([FO, build_features("vessel", to_t(oct150)[None], to_t(oct_mask)[None].float())], 0)
elif a.features == "mixed":   # MRI: parser WM/GM probabilities; OCT: Otsu soft WM/GM (GM bright in this OCT)
    FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
    FO = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=False)
else:
    FM = build_features(a.features, to_t(mri_reg)[None], mri_tis, wm_bright=False); FO = build_features(a.features, to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=False)
W = oct_mask.astype(np.float32)
if a.sulcus_weight:
    from scipy import ndimage
    ball = np.zeros((17, 17, 17), bool); zz, yy, xx = np.indices(ball.shape) - 8; ball[(zz**2 + yy**2 + xx**2) <= 64] = True
    closed = ndimage.binary_closing(oct_mask, structure=ball, iterations=1)
    sulcus = closed & ~oct_mask
    W = (oct_mask | sulcus).astype(np.float32) * 1.0
    print("sulcal background voxels added to weight:", int(sulcus.sum()), "tissue:", int(oct_mask.sum()))
MO = to_t(W)[None]
lv = {}
for f in (4, 2, 1):
    if f == 1: lv[f] = (FM, A_reg, FO, A_oct, MO, mri_tis)
    else:
        a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(MO, A_oct, f); a4, _ = avg_pool_iso(mri_tis, A_reg, f); lv[f] = (a1, A1, a2, A2, a3, a4)
srch = FFTSearcher(lv[4][0], lv[4][1], lv[4][5], lv[4][2], lv[4][3], lv[4][4], spacing=0.6, min_overlap=a.min_overlap)
t0 = time.time(); cands, info = srch.run(n_rot=a.n_rot, topk=a.topk, seed=0, log_every=0, mirror=True); print(f"search {time.time()-t0:.0f}s, top1 {info['top1']:.3f}")
refs = {f: Refiner(lv[f][0], lv[f][1], lv[f][2], lv[f][3], lv[f][4]) for f in (4, 2)}
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); labels4 = np.asarray(labels4)
from octreg.features import otsu_two_class
from octreg.evaluate import gm_wm_overlap
otsu_f, _ = otsu_two_class(to_t(oct150)[None], to_t(oct_mask)[None].bool(), wm_bright=False)
oct_class_otsu = np.where(oct_mask, np.where(otsu_f[0].cpu().numpy() > 0.5, 1, 2), 0)
rows = []
for i, c in enumerate(cands):
    T = c["T"]
    for f, dof, it in ((4, "rigid", 120), (2, "rigid", 120), (2, "affine", 120)):
        T, l = refs[f].refine(T, dof=dof, iters=it, ls_clamp=0.2, sh_clamp=0.2)
    v = vessel_distance_stats(T, ves_ijk, A_mri, d0, A0, n_ctrl=0)["registered"]
    gw = gm_wm_overlap(T, oct_class_otsu, oct_mask, A_oct, labels4, A_mri)
    cen = (T @ np.r_[c_o, 1.0])[:3]
    rows.append({"rank": i, "search_score": c["score"], "mirror": c.get("mirror"), "ncc03": 1 - l, "centre": cen.tolist(), "dist_to_vessel_centroid_mm": float(np.linalg.norm(cen - ves_w.mean(0))),
                 "n_ves_inside": v.get("n_inside", 0), "ves_median_um": v.get("median_um"), "ves_p75_um": v.get("p75_um"), "ves_f300": v.get("frac_within_300um"),
                 "otsu_dice_WM": gw["dice_WM"], "otsu_dice_GM": gw["dice_GM"], "scales": np.linalg.norm(T[:3, :3], axis=0).tolist(), "T": T.tolist()})
    print(f"cand {i:2d} score {c['score']:.3f} m={int(bool(c.get('mirror')))} ncc@0.3 {1-l:.3f} centre {np.round(cen,1).tolist()} dcent {rows[-1]['dist_to_vessel_centroid_mm']:.1f}mm | vessels n={v.get('n_inside',0)} med={v.get('median_um',-1) if v.get('median_um') else -1:.0f} f300={v.get('frac_within_300um',-1) if v.get('frac_within_300um') else -1:.2f} | otsu Dice WM {gw['dice_WM']:.2f} GM {gw['dice_GM']:.2f} | scales {np.round(rows[-1]['scales'],2).tolist()}", flush=True)
json.dump({"vessel_centroid": ves_w.mean(0).tolist(), "rows": rows}, open(a.out / "diag_candidates.json", "w"), indent=1)
