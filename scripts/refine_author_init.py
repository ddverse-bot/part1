#!/usr/bin/env python3
"""Start from the authors' affine.lta placed under each plausible OCT frame convention (SRP/SPR/IRP, corner
origin, fs_raw MRI frame) and refine with the manual-vessel symmetric distance loss (rigid -> affine).  If one of
them converges to a coherent solution (small MRI->OCT vessel distance, OCT GM mostly on MRI GM), use it as the
evaluation reference."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, layout_affine
from octreg.refine import compose, params_from_matrix
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--code", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy"); A12 = np.load(a.work / "oct12_affine.npy")
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; c_o_t = to_t(c_o)
def read_lta(p):
    lines = [l.strip() for l in Path(p).read_text().splitlines()]; i = lines.index("1 4 4"); return np.array([[float(x) for x in lines[i + k].split()] for k in range(1, 5)])
lta = read_lta(a.code / "results/mri2oct/affine.lta")
vs = 0.15; Mfs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]) * vs; cras = np.array([5.128326416015625, 9.682868957519531, 14.58111572265625]); N = np.array([1280.0, 1040.0, 576.0])
vox2ras_fs = np.eye(4); vox2ras_fs[:3, :3] = Mfs; vox2ras_fs[:3, 3] = cras - Mfs @ (N / 2.0)
fs_to_nifti = A_mri @ np.linalg.inv(vox2ras_fs)
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves)); ves_w = (A_mri @ np.c_[ves_ijk, np.ones(len(ves_ijk))].T).T[:, :3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, ves_w.min(0) - 8, ves_w.max(0) + 8)
sub = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); edt_m = ndimage.distance_transform_edt(~sub, sampling=(0.15,) * 3).astype(np.float32)
A_sub = A_mri.copy(); A_sub[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]; EDT_M = to_t(edt_m)[None]; A_SUB = to_t(A_sub)
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy"); EDT_O = to_t(d0 / 1000.0)[None]; A_O48 = to_t(A0)
vox = np.argwhere(d0 <= 0.0); rng = np.random.default_rng(0); vox = vox[rng.choice(len(vox), size=min(60000, len(vox)), replace=False)]
oct_pts = to_t((A0 @ np.c_[vox, np.ones(len(vox))].T).T[:, :3]); mri_pts = to_t(ves_w); oct_shape_w = to_t(np.array(d0.shape) - 1)
def sym_loss(T):
    Tinv = torch.linalg.inv(T); p_o = apply_affine(Tinv, mri_pts); v = apply_affine(torch.linalg.inv(A_O48), p_o)
    inside = ((v >= 0) & (v <= oct_shape_w)).all(1); d_mo = sample_at_world(EDT_O, A_O48, p_o)[0]; n_in = inside.float().sum()
    l1 = (d_mo * inside).sum() / n_in.clamp(min=1); p_m = apply_affine(T, oct_pts); l2 = sample_at_world(EDT_M, A_SUB, p_m, padding=8.0)[0].mean()
    return l1 + l2, l1, l2, n_in
def refine(T0, dof, iters, ls_clamp=0.25):
    r0, t0_, ls0, sh0, mirror = params_from_matrix(T0, c_o)
    r = to_t(r0).requires_grad_(True); t = to_t(t0_).requires_grad_(True); ls = to_t(ls0).requires_grad_(dof == "affine"); sh = to_t(sh0).requires_grad_(dof == "affine")
    groups = [{"params": [r], "lr": 0.01}, {"params": [t], "lr": 0.2}] + ([{"params": [ls], "lr": 0.01}, {"params": [sh], "lr": 0.01}] if dof == "affine" else [])
    opt = torch.optim.Adam(groups); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 0.0); best = (1e9, None)
    for it in range(iters):
        opt.zero_grad(); T = compose(r, t, ls, sh, c_o_t, mirror); L, l1, l2, n = sym_loss(T); (L + 1.0 * ((ls ** 2).sum() + (sh ** 2).sum())).backward(); opt.step(); sched.step()
        with torch.no_grad(): ls.clamp_(-ls_clamp, ls_clamp); sh.clamp_(-0.25, 0.25)
        if L.item() < best[0]: best = (L.item(), T.detach().cpu().numpy().copy())
    return best[1], best[0]
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
oct_class = np.where(oct_mask, np.where(oct_prob.argmax(0) == 1, 1, np.where(oct_prob.argmax(0) >= 2, 2, 0)), 0)
from octreg.features import otsu_two_class
otsu_f, _ = otsu_two_class(to_t(oct150)[None], to_t(oct_mask)[None].bool(), wm_bright=False)
oct_class_otsu = np.where(oct_mask, np.where(otsu_f[0].cpu().numpy() > 0.5, 1, 2), 0)
results = {}
for layout in ("SRP", "SPR", "IRP", "SPL", "IPR", "SRA"):
    for center in (False, True):
        A_l = layout_affine((627, 1271, 1230), 0.012, layout, center)          # OCT zyx voxel -> authors' OCT world
        # OCT voxel -> MRI nifti world under this hypothesis: fs_to_nifti @ lta @ A_l ; our OCT world = A12 @ voxel
        T0 = fs_to_nifti @ lta @ A_l @ np.linalg.inv(A12)
        with torch.no_grad(): L0, l1_0, l2_0, n0 = sym_loss(to_t(T0))
        T, L = refine(T0, "rigid", 250); T, L = refine(T, "affine", 250)
        with torch.no_grad(): _, l1, l2, n = sym_loss(to_t(T))
        ev = vessel_distance_stats(T, ves_ijk, A_mri, d0, A0, n_ctrl=5)["registered"]
        gw = gm_wm_overlap(T, oct_class_otsu, oct_mask, A_oct, np.asarray(labels4), A_mri)
        key = f"{layout}_{'centre' if center else 'corner'}"
        results[key] = {"init_loss": float(L0), "init_n_in": int(n0), "refined_loss": L, "l1": float(l1), "l2": float(l2), "n_in": int(n), "vessel_median_um": ev.get("median_um"), "vessel_f300": ev.get("frac_within_300um"),
                        "otsu_dice_WM": gw["dice_WM"], "otsu_dice_GM": gw["dice_GM"], "conf": gw["confusion_oct_rows_mri_cols"], "centre": (T @ np.r_[c_o, 1.0])[:3].tolist(), "scales": np.linalg.norm(T[:3, :3], axis=0).tolist(), "T": T.tolist()}
        r = results[key]
        print(f"{key:12s} init loss {L0:.3f} (n_in {int(n0)}) -> {L:.3f} l1 {float(l1):.3f} l2 {float(l2):.3f} n_in {int(n)} | ves med {ev.get('median_um', -1):.0f} f300 {ev.get('frac_within_300um', -1):.2f} | otsu Dice WM {gw['dice_WM']:.2f} GM {gw['dice_GM']:.2f} | centre {np.round(r['centre'],1).tolist()} scales {np.round(r['scales'],2).tolist()}", flush=True)
json.dump(results, open(a.out / "author_init_refined.json", "w"), indent=1)
