#!/usr/bin/env python3
"""Diagnostics: NCC and held-out vessel distance for (a) the coarse author placement, (b) the author placement
locally refined (rigid / similarity / affine), (c) the top search candidates refined rigidly. Helps decide whether
the objective or the optimizer is the problem."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, avg_pool_iso, world_bbox_to_voxel, rotvec_to_matrix
from octreg.features import build_features
from octreg.refine import Refiner
from octreg.evaluate import vessel_distance_stats, transform_diff
from octreg.search import FFTSearcher

ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--run", type=Path, required=True); ap.add_argument("--features", default="parser"); ap.add_argument("--parser-channels", default="wm_gm")
ap.add_argument("--ls-clamp", type=float, default=0.15); ap.add_argument("--sh-clamp", type=float, default=0.15)
a = ap.parse_args()
res = json.load(open(a.run / "result.json"))
lo, hi = np.array(res["mri_region_ijk"][0]), np.array(res["mri_region_ijk"][1])
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
ref_T = np.array(json.load(open(a.work / "ref_author_T.json")))
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
mri_tis = to_t(load_region("tissue") > 0.5)[None].float()
if a.features == "parser":
    FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
    FO = build_features("parser", None, to_t(oct_mask)[None].float(), prob=to_t(oct_prob), parser_channels=a.parser_channels)
else:
    FM = build_features(a.features, to_t(mri_reg)[None], mri_tis, wm_bright=False)
    FO = build_features(a.features, to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=False)
MO = to_t(oct_mask)[None].float()
lv = {}
for f in (4, 2, 1):
    if f == 1: lv[f] = (FM, A_reg, FO, A_oct, MO)
    else:
        a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(MO, A_oct, f); lv[f] = (a1, A1, a2, A2, a3)
refs = {f: Refiner(*lv[f]) for f in (4, 2, 1)}
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves))
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
d66 = np.load(a.work / "oct_ves_dist48_z66.npy"); A66 = np.load(a.work / "oct_ves_dist48_z66_affine.npy")
def report(name, T):
    v0 = vessel_distance_stats(T, ves_ijk, A_mri, d0, A0, n_ctrl=10)["registered"]; v66 = vessel_distance_stats(T, ves_ijk, A_mri, d66, A66, n_ctrl=5)["registered"]
    print(f"{name:34s} NCC@0.6={1-refs[4].evaluate(T):.3f} @0.3={1-refs[2].evaluate(T):.3f} @0.15={1-refs[1].evaluate(T):.3f} | vessels z0 med={v0.get('median_um',-1):.0f} p75={v0.get('p75_um',-1):.0f} f300={v0.get('frac_within_300um',-1):.2f} n={v0.get('n_inside')} | z66 med={v66.get('median_um',-1):.0f} | scales={np.round(np.linalg.norm(T[:3,:3],axis=0),3).tolist()} centre={np.round((T@np.r_[c_o,1])[:3],1).tolist()}", flush=True)
    return T
# monkeypatch clamps
import octreg.refine as R
orig = R.Refiner.refine
def refine_clamped(self, T0, **kw):
    return orig(self, T0, **kw)
report("author coarse placement", ref_T)
T = ref_T
for f, dof, it in ((4, "rigid", 200), (2, "rigid", 200), (1, "rigid", 200)):
    T, l = refs[f].refine(T, dof=dof, iters=it)
report("author -> rigid refined (0.6/0.3/0.15)", T)
Ts = T
for f, dof, it in ((2, "affine", 200), (1, "affine", 200)):
    Ts, l = refs[f].refine(Ts, dof=dof, iters=it)
report("author -> rigid -> affine (clamp 0.35)", Ts)
Tf = np.load(a.run / "T_oct2mri.npy")
report("run final (affine, scale collapsed)", Tf)
# top search candidates: rebuild from result.json (need R; we stored centres only) -> rerun a light search? instead refine from stored T? not stored.
# so re-run the 0.6mm search quickly with fewer rotations to get candidate T's
srch = FFTSearcher(lv[4][0], lv[4][1], avg_pool_iso(mri_tis, A_reg, 4)[0], lv[4][2], lv[4][3], lv[4][4], spacing=0.6)
cands, info = srch.run(n_rot=3000, topk=8, seed=0, log_every=0)
print("search top scores", [round(c["score"], 3) for c in cands])
for i, c in enumerate(cands[:5]):
    T = c["T"]
    for f, it in ((4, 150), (2, 150), (1, 150)):
        T, l = refs[f].refine(T, dof="rigid", iters=it)
    report(f"cand{i} (score {c['score']:.3f}) rigid-refined", T)
    d = transform_diff(T, ref_T, c_o); print(f"    vs author: centre {d['centre_mm']:.1f} mm rot {d['rotation_deg']:.1f} deg")
