#!/usr/bin/env python3
"""Self-contained vascular channel + robustness/control tests (real data only).

  1. OCT vessel density from OUR OWN automatic Frangi segmentation of the OCT (24 um / 48 um levels, top-q in tissue)
     vs the density of the DANDI ves_seg (authors' Frangi @12 um).
  2. Robustness: vascular refinement from perturbed starts (rotation / shift / depth-scale) -> same solution?
  3. Control: OCT density flipped along the depth axis inside the block -> the gain must vanish.
Every variant scored with the manual-vessel metric (held out from the method), manual-label Dice, stretch, distance to oracle.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, oct_slab_normalize, oct_tissue_mask, pool_mean_np, random_rotations
from octreg.features import build_features
from octreg.evaluate import transform_diff, gm_wm_overlap
from octreg.vascular import mri_dark_channel, mri_frangi_channel, oct_vessel_mask, density_on_grid, vascular_refine
from octreg.refine import params_from_matrix, compose

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T0", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--Tref", type=Path, default=None); ap.add_argument("--dandi", type=Path, required=True)
ap.add_argument("--w", type=float, default=1.0); ap.add_argument("--reg", type=float, default=0.5); ap.add_argument("--clamp", type=float, default=0.3)
ap.add_argument("--n-restart", type=int, default=12)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
def say(*x): print(*x, flush=True)

A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy"); A12 = np.load(a.work / "oct12_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32); T0 = np.load(a.T0)
sh = np.array(oct150.shape); c_o = (A_oct @ np.r_[(sh - 1) / 2.0, 1.0])[:3]; centre = (T0 @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); tis_reg = np.asarray(tissue[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
FM2 = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO2 = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright); MO = to_t(oct_mask)[None].float()
oct_class = np.where(oct_mask, np.where(FO2[0].cpu().numpy() > 0.5, 1, 2), 0)
MRI_V = {"dark": mri_dark_channel(mri_reg, tis_reg), "frangi": mri_frangi_channel(mri_reg, tis_reg)}

# evaluation
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
man_w = to_t((A_reg @ np.c_[np.argwhere(man), np.ones(int(man.sum()))].T).T[:, :3])
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)
R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0); Tref = np.load(a.Tref) if a.Tref else None
def vessel_score(T):
    with torch.no_grad():
        p_o = apply_affine(torch.linalg.inv(to_t(T)), man_w); v = apply_affine(torch.linalg.inv(A_O48), p_o)
        inside = ((v >= 0) & (v <= oct_shape48)).all(1); d = sample_at_world(EDT_O, A_O48, p_o)[0][inside] * 1000
        return float(d.median()), float((d <= 150).float().mean())
rows = []
def report(name, T, extra=""):
    vm, vf = vessel_score(T); gw = gm_wm_overlap(T, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
    st = [round(float(np.linalg.norm(T[:3, :3] @ R[:, q])), 3) for q in range(3)]
    d0_ = transform_diff(T, T0, c_o)["corner_mean_mm"]; dr = transform_diff(T, Tref, c_o)["corner_mean_mm"] if Tref is not None else float("nan")
    say(f"{name:52s} vessels med {vm:4.0f} f150 {vf:.2f} | Dice WM {gw['dice_WM']:.3f} GM {gw['dice_GM']:.3f} | stretch {st} | corners vs T0 {d0_:.2f} vs oracle {dr:.2f} mm {extra}")
    rows.append({"name": name, "vessel_median_um": vm, "vessel_f150": vf, "dice_WM": gw["dice_WM"], "dice_GM": gw["dice_GM"], "stretch_ijk": st, "corner_vs_T0": d0_, "corner_vs_oracle": dr, "T": T.tolist()})
    return T
report("T0", T0)
if Tref is not None: report("oracle affine (manual, reference)", Tref)

# ---- 1. OCT vessel densities
t0 = time.time()
vb48 = to_t((d0um <= 0).astype(np.float32)) > 0.5
DENS = {"ves_seg(DANDI Frangi@12um)": density_on_grid(vb48, A0, A_oct, oct150.shape, oct_mask, pool=3)}
import tifffile
oct12 = tifffile.imread(str(a.dandi / "sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff"))
octn, _ = oct_slab_normalize(oct12, tissue_thresh=30.0, window=100); del oct12
say(f"OCT loaded + slab-normalised ({time.time()-t0:.0f}s)")
levels = {}
for k in (2, 4):
    p = pool_mean_np(octn, k); m, _ = oct_tissue_mask(p, thresh=None, closing_iter=2)
    A = A12.copy(); A[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * (k - 1) / 2.0); A[:3, :3] *= k
    levels[k] = (p.astype(np.float32), m, A)
    say(f"level {k*12} um: {p.shape}, tissue {m.mean():.2f}")
del octn
for k, (p, m, A) in levels.items():
    for q in (99.0, 99.5):
        msk = oct_vessel_mask(p, m, q=q)
        say(f"own Frangi @{k*12}um q={q}: vessel voxels {int(msk.sum())} ({time.time()-t0:.0f}s)")
        DENS[f"own Frangi@{k*12}um q{q}"] = density_on_grid(msk, A, A_oct, oct150.shape, oct_mask, pool=(12 // k) if k == 2 else 3)
        del msk
    torch.cuda.empty_cache()
# correlation between own densities and ves_seg density (inside tissue)
ref_d = DENS["ves_seg(DANDI Frangi@12um)"]; OM = to_t(oct_mask, dtype=torch.bool)
for k_, d in DENS.items():
    c = torch.corrcoef(torch.stack([ref_d[OM], d[OM]]))[0, 1].item(); say(f"  corr(density, ves_seg density) {k_}: {c:.3f}")
sols = {}
for k_, d in DENS.items():
    for mv in ("dark", "frangi"):
        T, info = vascular_refine(T0, FM2, A_reg, FO2, A_oct, MO, MRI_V[mv], d, w=a.w, reg=a.reg, clamp=a.clamp)
        sols[(k_, mv)] = T
        report(f"vascular: oct {k_} | mri {mv}", T, extra=f"| ncc/ch {np.round(info['ncc_channels'],3).tolist()}")

# ---- 3. control: OCT density flipped along the depth axis (same statistics, wrong geometry)
d_flip = torch.flip(DENS["ves_seg(DANDI Frangi@12um)"], dims=[0])
T, info = vascular_refine(T0, FM2, A_reg, FO2, A_oct, MO, MRI_V["dark"], d_flip, w=a.w, reg=a.reg, clamp=a.clamp)
report("CONTROL: ves_seg density flipped along depth | mri dark", T, extra=f"| ncc/ch {np.round(info['ncc_channels'],3).tolist()}")
d_perm = DENS["ves_seg(DANDI Frangi@12um)"][:, :, torch.randperm(oct150.shape[2], generator=torch.Generator().manual_seed(0)).to(ref_d.device)] * MO[0]
T, info = vascular_refine(T0, FM2, A_reg, FO2, A_oct, MO, MRI_V["dark"], d_perm, w=a.w, reg=a.reg, clamp=a.clamp)
report("CONTROL: ves_seg density k-slices permuted | mri dark", T, extra=f"| ncc/ch {np.round(info['ncc_channels'],3).tolist()}")

# ---- 2. robustness from perturbed starts (best label-free config: ves_seg density + dark; and own Frangi@24um q99 + dark)
rng = np.random.default_rng(0)
for key in (("ves_seg(DANDI Frangi@12um)", "dark"), ("own Frangi@24um q99.0", "dark")):
    if key[0] not in DENS: continue
    Tsol = sols[key]; res = []
    r0, t0_, ls0, sh0, mirror = params_from_matrix(T0, c_o)
    for i in range(a.n_restart):
        ang = np.radians(rng.uniform(3, 8)); axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        dr = axis * ang; dt = rng.uniform(-2.5, 2.5, 3); dls = np.array([rng.uniform(-0.1, 0.1), 0.0, 0.0])
        Tp = compose(to_t(r0 + dr), to_t(t0_ + dt), to_t(ls0 + dls), to_t(sh0), to_t(c_o), mirror).cpu().numpy()
        T, info = vascular_refine(Tp, FM2, A_reg, FO2, A_oct, MO, MRI_V[key[1]], DENS[key[0]], w=a.w, reg=a.reg, clamp=a.clamp, factors=(4, 2, 1))
        dsol = transform_diff(T, Tsol, c_o)["corner_mean_mm"]; vm, vf = vessel_score(T)
        res.append({"perturb_deg": float(np.degrees(ang)), "perturb_mm": float(np.linalg.norm(dt)), "corner_vs_solution_mm": dsol, "vessel_median_um": vm, "vessel_f150": vf})
        say(f"  restart {i:2d} ({np.degrees(ang):.1f} deg, {np.linalg.norm(dt):.1f} mm, dls_i {dls[0]:+.2f}) -> corners vs solution {dsol:.2f} mm, vessels {vm:.0f}/{vf:.2f}")
    ok = sum(r["corner_vs_solution_mm"] < 0.3 for r in res)
    say(f"robustness [{key[0]} | {key[1]}]: {ok}/{len(res)} restarts within 0.3 mm (corner mean) of the unperturbed solution")
    rows.append({"name": f"robustness {key[0]} | {key[1]}", "restarts": res, "n_within_0.3mm": ok})
json.dump({"rows": rows}, open(a.out / "test_own_vessels.json", "w"), indent=1, default=float)
say("saved", a.out / "test_own_vessels.json")
