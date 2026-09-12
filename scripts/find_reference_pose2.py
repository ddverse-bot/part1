#!/usr/bin/env python3
"""Reference pose from the MANUAL vessel annotations with a proper global search:
FFT search (rotations x all translations, both handedness) of the OCT vessel indicator against an MRI
vessel-proximity map exp(-EDT/0.5mm), then symmetric vessel-distance refinement (rigid -> affine).
Evaluation reference only (uses annotations); not part of the label-free method."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, avg_pool_iso
from octreg.refine import compose, params_from_matrix
from octreg.search import FFTSearcher
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-rot", type=int, default=6000); ap.add_argument("--topk", type=int, default=40); ap.add_argument("--half-mm", type=float, default=32.0)
ap.add_argument("--scales", type=str, default="1.0"); ap.add_argument("--ls-clamp", type=float, default=0.2)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; c_o_t = to_t(c_o)
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves))
ves_w = (A_mri @ np.c_[ves_ijk, np.ones(len(ves_ijk))].T).T[:, :3]; centroid = ves_w.mean(0)
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centroid - a.half_mm, centroid + a.half_mm)
sub = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
edt_m = ndimage.distance_transform_edt(~sub, sampling=(0.15,) * 3).astype(np.float32)
A_sub = A_mri.copy(); A_sub[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
prox_m = np.exp(-edt_m / 0.5).astype(np.float32)
tis = np.load(a.work / "mri_tissue.npy", mmap_mode="r"); tis_sub = np.asarray(tis[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
# OCT vessel indicator on the oct150 grid: fraction of vessel voxels per 0.15 mm cell (from the 48 um EDT==0 grid)
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
vox = np.argwhere(d0 <= 0.0); pts48 = (A0 @ np.c_[vox, np.ones(len(vox))].T).T[:, :3]
ijk150 = np.rint((np.linalg.inv(A_oct) @ np.c_[pts48, np.ones(len(pts48))].T).T[:, :3]).astype(int)
ok = np.all((ijk150 >= 0) & (ijk150 < np.array(oct150.shape)), 1)
ind = np.zeros(oct150.shape, np.float32); np.add.at(ind, (ijk150[ok, 0], ijk150[ok, 1], ijk150[ok, 2]), 1.0)
ind = np.minimum(ind / 8.0, 1.0)      # ~fraction of the 4x4x4... crude but fine
prox_o = ndimage.gaussian_filter(ind, 0.8)
print(f"MRI region {(hi-lo).tolist()}, prox_m mean {prox_m.mean():.4f}; OCT vessel indicator mean {ind.mean():.4f}", flush=True)
FM = to_t(prox_m)[None]; FO = to_t(prox_o)[None]; MO = to_t(oct_mask.astype(np.float32))[None]; TIS = to_t(tis_sub.astype(np.float32))[None]
lv = {}
for f in (4, 2, 1):
    if f == 1: lv[f] = (FM, A_sub, FO, A_oct, MO, TIS)
    else:
        a1, A1 = avg_pool_iso(FM, A_sub, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(MO, A_oct, f); a4, _ = avg_pool_iso(TIS, A_sub, f); lv[f] = (a1, A1, a2, A2, a3, a4)
srch = FFTSearcher(lv[4][0], lv[4][1], lv[4][5], lv[4][2], lv[4][3], lv[4][4], spacing=0.6, min_overlap=0.8)
scales = tuple(float(x) for x in a.scales.split(","))
t0 = time.time(); cands, info = srch.run(n_rot=a.n_rot, scales=scales, topk=a.topk, seed=2, log_every=0, mirror=True)
print(f"vessel-proximity search {time.time()-t0:.0f}s: top scores {[round(c['score'],3) for c in cands[:8]]}", flush=True)
# symmetric vessel loss refinement (fine)
EDT_M = to_t(edt_m)[None]; A_SUB = to_t(A_sub); EDT_O = to_t(d0 / 1000.0)[None]; A_O48 = to_t(A0)
rng = np.random.default_rng(0); sel = vox[rng.choice(len(vox), size=min(60000, len(vox)), replace=False)]
oct_pts = to_t((A0 @ np.c_[sel, np.ones(len(sel))].T).T[:, :3]); mri_pts = to_t(ves_w); oct_shape_w = to_t(np.array(d0.shape) - 1)
def sym_loss(T):
    Tinv = torch.linalg.inv(T); p_o = apply_affine(Tinv, mri_pts); v = apply_affine(torch.linalg.inv(A_O48), p_o)
    inside = ((v >= 0) & (v <= oct_shape_w)).all(1); d_mo = sample_at_world(EDT_O, A_O48, p_o)[0]; n_in = inside.float().sum()
    l1 = (d_mo * inside).sum() / n_in.clamp(min=1); p_m = apply_affine(T, oct_pts); l2 = sample_at_world(EDT_M, A_SUB, p_m, padding=8.0)[0].mean()
    return l1 + l2, l1, l2, n_in
def refine(T0, dof, iters):
    r0, t0_, ls0, sh0, mirror = params_from_matrix(T0, c_o)
    r = to_t(r0).requires_grad_(True); t = to_t(t0_).requires_grad_(True); ls = to_t(ls0).requires_grad_(dof == "affine"); sh = to_t(sh0).requires_grad_(dof == "affine")
    groups = [{"params": [r], "lr": 0.02}, {"params": [t], "lr": 0.3}] + ([{"params": [ls], "lr": 0.01}, {"params": [sh], "lr": 0.01}] if dof == "affine" else [])
    opt = torch.optim.Adam(groups); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 0.0); best = (1e9, None)
    for it in range(iters):
        opt.zero_grad(); T = compose(r, t, ls, sh, c_o_t, mirror); L, l1, l2, n = sym_loss(T); (L + 2.0 * ((ls ** 2).sum() + (sh ** 2).sum())).backward(); opt.step(); sched.step()
        with torch.no_grad(): ls.clamp_(-a.ls_clamp, a.ls_clamp); sh.clamp_(-0.2, 0.2)
        if L.item() < best[0]: best = (L.item(), T.detach().cpu().numpy().copy())
    return best[1], best[0]
results = []
for i, c in enumerate(cands):
    T, L = refine(c["T"], "rigid", 200); T, L = refine(T, "affine", 200)
    with torch.no_grad(): _, l1, l2, n = sym_loss(to_t(T))
    results.append({"rank": i, "search_score": c["score"], "mirror": c["mirror"], "scale0": c["scale"], "loss": L, "l1": float(l1), "l2": float(l2), "n_in": int(n), "T": T, "centre": (T @ np.r_[c_o, 1.0])[:3].tolist(), "scales": np.linalg.norm(T[:3, :3], axis=0).tolist()})
    print(f"cand {i:2d} score {c['score']:.3f} mirror={c['mirror']} s0={c['scale']} -> sym loss {L:.3f} (l1 {float(l1):.3f} l2 {float(l2):.3f} n_in {int(n)}) centre {np.round(results[-1]['centre'],1).tolist()} scales {np.round(results[-1]['scales'],2).tolist()}", flush=True)
results.sort(key=lambda d: d["loss"]); best = results[0]; T = best["T"]
ev = vessel_distance_stats(T, ves_ijk, A_mri, d0, A0)
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
oct_class = np.where(oct_mask, np.where(oct_prob.argmax(0) == 1, 1, np.where(oct_prob.argmax(0) >= 2, 2, 0)), 0)
gw = gm_wm_overlap(T, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
out = {"reference_T_oct2mri": T.tolist(), "block_centre_mm": best["centre"], "mirror": best["mirror"], "column_scales": np.linalg.norm(T[:3, :3], axis=0).tolist(),
       "sym_vessel_loss_mm": best["loss"], "l1": best["l1"], "l2": best["l2"], "n_mri_ves_inside": best["n_in"], "vessel_eval": ev, "gmwm_parser_classes": gw,
       "top": [{k: v for k, v in r.items() if k != "T"} for r in results[:8]]}
json.dump(out, open(a.out / "reference_pose.json", "w"), indent=1, default=float); np.save(a.out / "T_reference.npy", T)
print(json.dumps({k: v for k, v in out.items() if k not in ("reference_T_oct2mri", "top")}, indent=1, default=lambda o: round(float(o), 3)))
