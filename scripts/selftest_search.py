#!/usr/bin/env python3
"""Self-test of the global search + refinement on a SYNTHETIC block cut from the MRI features with a known
random pose (rotation, optional mirror, translation).  Validates conventions end-to-end."""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, avg_pool_iso, world_bbox_to_voxel, random_rotations, sample_at_world, apply_affine, grid_points, rotation_geodesic_deg, polar_rotation
from octreg.search import FFTSearcher
from octreg.refine import Refiner
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mirror", action="store_true"); ap.add_argument("--noise", type=float, default=0.1); ap.add_argument("--n-rot", type=int, default=3000)
a = ap.parse_args()
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
centroid = np.array([-6.8, 23.9, 16.0])
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centroid - 30, centroid + 30)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0); TIS = to_t(load_region("tissue") > 0.5)[None].float()
# synthetic OCT block: grid 0.15 mm, 100x100x50 voxels, own world frame (corner origin), true pose T_true (block world -> MRI world)
rng = np.random.default_rng(a.seed)
R = random_rotations(2, seed=a.seed + 7)[1]
if a.mirror: R = R @ np.diag([1, 1, -1.0])
shape = (50, 100, 100); A_blk = np.diag([0.15, 0.15, 0.15, 1.0])
c_blk = (A_blk @ np.r_[(np.array(shape) - 1) / 2, 1.0])[:3]
centre_true = centroid + rng.uniform(-6, 6, 3)
T_true = np.eye(4); T_true[:3, :3] = R; T_true[:3, 3] = centre_true - R @ c_blk
pts = grid_points(to_t(A_blk), shape).reshape(-1, 3)
pm = apply_affine(to_t(T_true), pts)
FO = sample_at_world(FM, to_t(A_reg), pm).reshape(2, *shape); MO = sample_at_world(TIS, to_t(A_reg), pm).reshape(1, *shape)
FO = (FO + a.noise * torch.randn_like(FO)).clamp(0, 1)
print(f"synthetic block at {np.round(centre_true,1).tolist()}, mirror={a.mirror}, tissue frac {MO.mean().item():.2f}")
lv = {}
for f in (4, 2, 1):
    if f == 1: lv[f] = (FM, A_reg, FO, A_blk, MO, TIS)
    else:
        a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_blk, f); a3, _ = avg_pool_iso(MO, A_blk, f); a4, _ = avg_pool_iso(TIS, A_reg, f); lv[f] = (a1, A1, a2, A2, a3, a4)
srch = FFTSearcher(lv[4][0], lv[4][1], lv[4][5], lv[4][2], lv[4][3], lv[4][4], spacing=0.6)
t0 = time.time(); cands, info = srch.run(n_rot=a.n_rot, topk=10, seed=1, log_every=0, mirror=True); print(f"search {time.time()-t0:.0f}s top scores {[round(c['score'],3) for c in cands[:5]]}")
c_o = srch.c_o.cpu().numpy()
for i, c in enumerate(cands[:5]):
    T = c["T"]; dc = np.linalg.norm((T @ np.r_[c_o, 1])[:3] - (T_true @ np.r_[c_o, 1])[:3]); rot = rotation_geodesic_deg(polar_rotation(T[:3, :3]) if not c["mirror"] else T[:3, :3] @ np.diag([1, 1, -1.0]), R if not a.mirror else R @ np.diag([1, 1, -1.0]))
    print(f"  cand {i}: score {c['score']:.3f} mirror={c['mirror']} centre err {dc:.1f} mm, rot err {rot:.1f} deg")
refs = {f: Refiner(lv[f][0], lv[f][1], lv[f][2], lv[f][3], lv[f][4]) for f in (4, 2, 1)}
best = None
for c in cands[:5]:
    T = c["T"]
    for f, dof, it in ((4, "rigid", 120), (2, "rigid", 120), (1, "affine", 120)):
        T, l = refs[f].refine(T, dof=dof, iters=it)
    dc = np.linalg.norm((T @ np.r_[c_o, 1])[:3] - (T_true @ np.r_[c_o, 1])[:3])
    corners = np.array([[i, j, k] for i in (0, 49) for j in (0, 99) for k in (0, 99)], float); cw = (A_blk @ np.c_[corners, np.ones(8)].T).T[:, :3]
    err = np.linalg.norm((T @ np.c_[cw, np.ones(8)].T).T[:, :3] - (T_true @ np.c_[cw, np.ones(8)].T).T[:, :3], axis=1)
    print(f"  refined: loss {l:.4f} centre err {dc:.2f} mm, corner err mean {err.mean():.2f} max {err.max():.2f} mm")
    if best is None or l < best[0]: best = (l, dc, err.mean())
print("BEST refined:", {"loss": round(best[0], 4), "centre_err_mm": round(best[1], 2), "corner_err_mm": round(best[2], 2)})
