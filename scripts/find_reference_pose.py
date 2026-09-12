#!/usr/bin/env python3
"""Reference (label-driven, Costantini-style) OCT->MRI pose using the MANUAL vessel annotations only:
symmetric vessel-to-vessel distance (MRI vessel labels -> OCT vessel EDT, OCT vessel voxels -> MRI vessel EDT),
dense rotation scan around the MRI vessel-label centroid, then rigid -> affine refinement.  This is used as an
evaluation reference for the label-free method; it is NOT part of the method."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, random_rotations, world_bbox_to_voxel, avg_pool_iso, rotvec_to_matrix
from octreg.refine import compose, params_from_matrix, Refiner
from octreg.features import build_features
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-rot", type=int, default=6000); ap.add_argument("--topk", type=int, default=24)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves))
ves_w = (A_mri @ np.c_[ves_ijk, np.ones(len(ves_ijk))].T).T[:, :3]
centroid = ves_w.mean(0)
# MRI vessel EDT in a crop around the labels (mm)
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, ves_w.min(0) - 8, ves_w.max(0) + 8)
sub = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
edt_m = ndimage.distance_transform_edt(~sub, sampling=(0.15,) * 3).astype(np.float32)
A_sub = A_mri.copy(); A_sub[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
EDT_M = to_t(edt_m)[None]; A_SUB = to_t(A_sub)
# OCT vessel EDT (um -> mm) at 48 um, and OCT vessel points (pooled 4x grid centres where vessels present)
d0 = np.load(a.work / "oct_ves_dist48_z0.npy") / 1000.0; A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0)[None]; A_O48 = to_t(A0)
vox = np.argwhere(d0 <= 0.0)          # vessel voxels on the 48um grid
rng = np.random.default_rng(0)
vox = vox[rng.choice(len(vox), size=min(60000, len(vox)), replace=False)]
oct_pts = to_t((A0 @ np.c_[vox, np.ones(len(vox))].T).T[:, :3])       # OCT world
mri_pts = to_t(ves_w)                                                    # MRI world
print(f"MRI vessel points {len(mri_pts)}, OCT vessel points {len(oct_pts)}, centroid {np.round(centroid,1).tolist()}", flush=True)
c_o_t = to_t(c_o)
oct_shape_w = to_t(np.array(d0.shape) - 1)

def sym_loss(T, hard=False):
    """symmetric mean vessel distance (mm) with out-of-FOV handling."""
    Tinv = torch.linalg.inv(T)
    # MRI vessel points -> OCT: distance from OCT EDT; points outside the OCT block get 3 mm (constant, no gradient)
    p_o = apply_affine(Tinv, mri_pts)
    v = apply_affine(torch.linalg.inv(A_O48), p_o)
    inside = ((v >= 0) & (v <= oct_shape_w)).all(1)
    d_mo = sample_at_world(EDT_O, A_O48, p_o)[0]
    n_in = inside.float().sum()
    l1 = (d_mo * inside).sum() / n_in.clamp(min=1)
    # OCT vessel points -> MRI: distance from MRI vessel EDT (all OCT vessel voxels are inside the labelled region if pose is right)
    p_m = apply_affine(T, oct_pts)
    d_om = sample_at_world(EDT_M, A_SUB, p_m, padding=8.0)[0]
    l2 = d_om.mean()
    # coverage: fraction of MRI vessel points inside the block (want it high but it depends on the true overlap); use as a soft term
    return l1 + l2, l1, l2, n_in

@torch.no_grad()
def scan():
    Rs = random_rotations(a.n_rot, seed=1)
    Rs = np.concatenate([Rs, Rs @ np.diag([1.0, 1.0, -1.0])], 0)      # both handedness
    out = []
    for i, R in enumerate(Rs):
        T = torch.eye(4, device=dev); T[:3, :3] = to_t(R); T[:3, 3] = to_t(centroid) - T[:3, :3] @ c_o_t
        L, l1, l2, n = sym_loss(T)
        out.append((float(L), i))
    out.sort()
    return [(s, Rs[i]) for s, i in out[:a.topk]]

def refine(T0, dof, iters, lr_rot=0.02, lr_t=0.3):
    r0, t0, ls0, sh0, mirror = params_from_matrix(T0, c_o)
    r = to_t(r0).requires_grad_(True); t = to_t(t0).requires_grad_(True); ls = to_t(ls0).requires_grad_(dof == "affine"); sh = to_t(sh0).requires_grad_(dof == "affine")
    groups = [{"params": [r], "lr": lr_rot}, {"params": [t], "lr": lr_t}]
    if dof == "affine": groups += [{"params": [ls], "lr": 0.01}, {"params": [sh], "lr": 0.01}]
    opt = torch.optim.Adam(groups); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 0.0)
    best = (1e9, None)
    for it in range(iters):
        opt.zero_grad(); T = compose(r, t, ls, sh, c_o_t, mirror); L, l1, l2, n = sym_loss(T); (L + 2.0 * ((ls ** 2).sum() + (sh ** 2).sum())).backward(); opt.step(); sched.step()
        with torch.no_grad(): ls.clamp_(-0.2, 0.2); sh.clamp_(-0.2, 0.2)
        if L.item() < best[0]: best = (L.item(), T.detach().cpu().numpy().copy())
    return best[1], best[0]

t0 = time.time(); top = scan(); print(f"scan done {time.time()-t0:.0f}s; best scan losses {[round(s,3) for s,_ in top[:5]]}", flush=True)
results = []
for s, R in top:
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = centroid - R @ c_o
    T, L = refine(T, "rigid", 200); T, L = refine(T, "affine", 200)
    with torch.no_grad(): _, l1, l2, n = sym_loss(to_t(T))
    results.append({"loss": L, "l_mri2oct_mm": float(l1), "l_oct2mri_mm": float(l2), "n_mri_ves_inside": int(n), "T": T, "mirror": bool(np.linalg.det(T[:3, :3]) < 0)})
results.sort(key=lambda d: d["loss"])
best = results[0]
T = best["T"]
d0um = np.load(a.work / "oct_ves_dist48_z0.npy")
ev = vessel_distance_stats(T, ves_ijk, A_mri, d0um, A0)
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
oct_class = np.where(oct_mask, np.where(oct_prob.argmax(0) == 1, 1, np.where(oct_prob.argmax(0) >= 2, 2, 0)), 0)
gw = gm_wm_overlap(T, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
# parser-feature NCC at the reference pose (for the diagnosis of the label-free objective)
lo2, hi2 = world_bbox_to_voxel(A_mri, mri.shape, centroid - 30, centroid + 30)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo2[0]:hi2[0], lo2[1]:hi2[1], lo2[2]:hi2[2]]).astype(np.float32)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo2, 1.0])[:3]
FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
FO = build_features("parser", None, to_t(oct_mask)[None].float(), prob=to_t(oct_prob))
ncc = {}
for f in (4, 2, 1):
    if f == 1: ref = Refiner(FM, A_reg, FO, A_oct, to_t(oct_mask)[None].float())
    else:
        a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(to_t(oct_mask)[None].float(), A_oct, f); ref = Refiner(a1, A1, a2, A2, a3)
    ncc[f"parser_ncc_at_{0.15*f:.2f}mm"] = 1 - ref.evaluate(T)
    ncc[f"parser_ncc_channels_{0.15*f:.2f}mm"] = ref.ncc_channels(T)
out = {"reference_T_oct2mri": T.tolist(), "block_centre_mm": (T @ np.r_[c_o, 1.0])[:3].tolist(), "column_scales": np.linalg.norm(T[:3, :3], axis=0).tolist(),
       "sym_vessel_loss_mm": best["loss"], "l_mri2oct_mm": best["l_mri2oct_mm"], "l_oct2mri_mm": best["l_oct2mri_mm"], "n_mri_ves_inside": best["n_mri_ves_inside"],
       "mirror": best["mirror"],
       "top_solutions": [{"loss": r["loss"], "centre": (r["T"] @ np.r_[c_o, 1.0])[:3].tolist(), "n_inside": r["n_mri_ves_inside"], "mirror": r["mirror"]} for r in results[:6]],
       "vessel_eval": ev, "gmwm_parser_classes": gw, "parser_ncc": ncc}
json.dump(out, open(a.out / "reference_pose.json", "w"), indent=1, default=float)
np.save(a.out / "T_reference.npy", T)
print(json.dumps({k: v for k, v in out.items() if k not in ("reference_T_oct2mri",)}, indent=1, default=lambda o: round(float(o), 3) if isinstance(o, (np.floating, float)) else str(o))[:3000])
