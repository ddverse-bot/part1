#!/usr/bin/env python3
"""OCT block -> MRI registration, label-free and without a position prior (octreg v1).

    python register.py --work work/I38 --out work/runs/I38_otsu [--features otsu|parser --parser-dir work/parsers/I38]
                       [--crop-centre x,y,z --crop-half-mm 30]   # optional: register against a crop of the MRI

Stages
  1. structural representation on both sides: tissue mask + two-class intensity split (Otsu) -> [P(WM), P(GM)];
     'parser' = a trained 3D U-Net gives the MRI side (ablation / optional)
  2. global FFT search at 0.6 mm: masked NCC over all translations x N rotations x {proper, mirrored} x {both class polarities}
  3. structural refinement: rigid -> similarity -> affine at 0.6/0.3/0.15 mm (scale prior), + perturbed restarts
  4. vascular refinement: third channel = MRI local darkness vs the density of our own OCT vessel segmentation; affine
  5. evaluation with whatever annotations exist in --work (manual MRI vessels, manual labels) — never used by the method.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import (DEVICE, avg_pool_iso, lta_text, to_t, world_bbox_to_voxel, write_json, save_nifti, resample_to_grid, rotvec_to_matrix)
from octreg.features import otsu_two_class, otsu_two_class_lowmem
from octreg.search import FFTSearcher
from octreg.refine import Refiner, params_from_matrix, compose
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap, transform_diff, qc_figure
from octreg.vascular import mri_dark_channel, density_on_grid, vascular_refine

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--features", choices=["otsu", "parser"], default="otsu"); ap.add_argument("--parser-dir", type=Path, default=None, help="dir with mri_prob_{wm,gm,tissue}.npy (features=parser)")
ap.add_argument("--crop-centre", type=str, default=None, help="x,y,z mm: register against a crop of the MRI around this point"); ap.add_argument("--crop-half-mm", type=float, default=30.0)
ap.add_argument("--n-rot", type=int, default=8000); ap.add_argument("--topk", type=int, default=24); ap.add_argument("--min-overlap", type=float, default=0.85)
ap.add_argument("--no-mirror", action="store_true"); ap.add_argument("--oct-wm-bright", choices=["auto", "yes", "no"], default="no", help="is the brighter OCT class WM? (imaging-protocol property; serial-sectioning OCT pooled to 0.15 mm: GM brighter -> 'no'); auto = run both, keep the better refined NCC")
ap.add_argument("--mri-wm-bright", action="store_true", help="MRI has WM brighter than GM (default: GM bright, as ex-vivo FLASH ~20 deg)")
ap.add_argument("--ls-clamp", type=float, default=0.15); ap.add_argument("--sh-clamp", type=float, default=0.15); ap.add_argument("--reg", type=float, default=2.0)
ap.add_argument("--vascular", choices=["on", "off"], default="on"); ap.add_argument("--vascular-w", type=float, default=2.0); ap.add_argument("--vascular-reg", type=float, default=0.5); ap.add_argument("--vascular-clamp", type=float, default=0.3)
ap.add_argument("--n-restarts", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ref-transform", type=Path, default=None, help="optional 4x4 (json/npy) OCT->MRI reference, reported only")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t_start = time.time()
log = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
def say(*x): print(*x, flush=True)

# ------------------------------------------------------------------ load
prep = json.load(open(a.work / "prep.json"))
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); mri_tissue_all = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r") if (a.work / "labels4.npy").exists() else None
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]
ref_T = None
if a.ref_transform: ref_T = np.load(a.ref_transform) if a.ref_transform.suffix == ".npy" else np.array(json.load(open(a.ref_transform)))

if a.crop_centre:
    cc = np.array([float(x) for x in a.crop_centre.split(",")]); lo, hi = world_bbox_to_voxel(A_mri, mri.shape, cc - a.crop_half_mm, cc + a.crop_half_mm)
else:
    lo, hi = np.zeros(3, int), np.array(mri.shape)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
tis_reg = np.asarray(mri_tissue_all[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
say(f"MRI region voxels {lo.tolist()}..{hi.tolist()} ({(hi - lo).tolist()}), voxel {np.linalg.norm(A_mri[:3,:3],axis=0).round(3).tolist()} mm; OCT block {oct150.shape} @ {np.linalg.norm(A_oct[:3,:3],axis=0).round(3).tolist()} mm")
log["mri_region_ijk"] = [lo.tolist(), hi.tolist()]

# ------------------------------------------------------------------ structural representation; MRI features stay on the CPU (float16),
# only pooled levels and block-sized crops are moved to the GPU (whole-hemisphere MRIs at 0.12 mm are 1.4 G voxels)
VOX_M = float(np.linalg.norm(A_mri[:3, :3], axis=0).mean()); VOX_O = float(np.linalg.norm(A_oct[:3, :3], axis=0).mean())
if a.features == "parser":
    FM_cpu = [np.load(a.parser_dir / f"mri_prob_{n}.npy", mmap_mode="r") for n in ("wm", "gm")]          # whole-MRI maps (mmap)
    FM_cpu = [x[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] for x in FM_cpu]
else:
    _fm, thr_m = otsu_two_class_lowmem(mri_reg, tis_reg, wm_bright=a.mri_wm_bright, vox_mm=VOX_M)       # bias-flattened two-class split [P(WM), P(GM)]
    FM_cpu = [_fm[0], _fm[1]]; log["mri_otsu_thresh"] = thr_m
del mri_reg
TIS_cpu = tis_reg

class MRIFeatures:
    """CPU-backed MRI feature maps [C,D,H,W] with the region affine; pooled whole-region levels and cropped regions on the GPU."""
    def __init__(self, chans, tissue, A, vox):
        self.chans, self.tissue, self.A, self.vox = chans, tissue, np.asarray(A), vox; self.shape = np.array(chans[0].shape)
    def factor(self, level_mm): return max(1, int(round(level_mm / self.vox)))
    def pooled(self, level_mm, what="feat"):
        f = self.factor(level_mm); D, H, W = self.shape; srcs = self.chans if what == "feat" else [self.tissue]
        Dp, Hp, Wp = D // f, H // f, W // f; out = torch.empty((len(srcs), Dp, Hp, Wp), device=DEVICE); step = max(f, (64 // f) * f)
        for z0 in range(0, Dp * f, step):
            z1 = min(Dp * f, z0 + step)
            blk = torch.stack([to_t(np.asarray(x[z0:z1, :Hp * f, :Wp * f]).astype(np.float32)) for x in srcs])
            out[:, z0 // f:z1 // f] = torch.nn.functional.avg_pool3d(blk[None], f, f)[0]; del blk
        A = self.A.copy(); A[:3, 3] = A[:3, 3] + A[:3, :3] @ (np.ones(3) * (f - 1) / 2.0); A[:3, :3] *= f
        return out, A
    def region(self, centre_mm, half_mm, level_mm):
        """crop (GPU float32) of the feature maps around a world point, pooled to the level."""
        f = self.factor(level_mm); lo_, hi_ = world_bbox_to_voxel(self.A, tuple(self.shape), centre_mm - half_mm, centre_mm + half_mm)
        lo_ = (lo_ // f) * f; hi_ = lo_ + ((hi_ - lo_) // f) * f
        blk = torch.stack([to_t(np.asarray(x[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]).astype(np.float32)) for x in self.chans])
        A = self.A.copy(); A[:3, 3] = (self.A @ np.r_[lo_, 1.0])[:3]
        if f > 1: blk = torch.nn.functional.avg_pool3d(blk[None], f, f)[0]; A[:3, 3] = A[:3, 3] + A[:3, :3] @ (np.ones(3) * (f - 1) / 2.0); A[:3, :3] *= f
        return blk, A
MF = MRIFeatures(FM_cpu, TIS_cpu, A_reg, VOX_M)
MO = to_t(oct_mask)[None].float()
FO_bright_first, thr_o = otsu_two_class(to_t(oct150)[None], MO.bool(), wm_bright=True)          # [bright, dark] classes; polarity resolved by the search
log["oct_otsu_thresh"] = thr_o
LEVELS = (0.6, 0.3, 0.15)                                   # search at 0.6 mm; refinement 0.6 -> 0.3 -> finest (mm, rounded to the voxel grids)
def pyramid(F_, A_, vox):
    out, done = {}, {}
    for L in LEVELS:
        f = max(1, int(round(L / vox)))
        out[L] = done[f] if f in done else ((F_, np.asarray(A_)) if f == 1 else avg_pool_iso(F_, A_, f)); done[f] = out[L]
    return out
PMask = pyramid(MO, A_oct, VOX_O)
# overlap gate adapted to the data: a cropped MRI can cover only part of the OCT box (or of an over-inclusive OCT
# mask, e.g. agarose embedding that OCT cannot separate from tissue) — the best possible pose then has
# overlap ~= V(MRI tissue)/V(OCT mask), and a fixed 0.85 gate would reject everything.
V_m = float(np.asarray(TIS_cpu).sum()) * VOX_M ** 3; V_o = float(oct_mask.sum()) * VOX_O ** 3
MIN_OV = float(min(a.min_overlap, max(0.15, 0.8 * V_m / max(V_o, 1e-6))))
log["overlap_gate"] = {"requested": a.min_overlap, "effective": MIN_OV, "V_mri_tissue_cm3": V_m / 1000, "V_oct_mask_cm3": V_o / 1000}
say(f"overlap gate: MRI tissue {V_m/1000:.1f} cm3, OCT mask {V_o/1000:.1f} cm3 -> effective min_overlap {MIN_OV:.2f} (requested {a.min_overlap})")
PM06, A06 = MF.pooled(0.6); PT06, _ = MF.pooled(0.6, "tissue"); SEARCH_MM = float(np.linalg.norm(A06[:3, :3], axis=0).mean())
_corn = np.array([[i, j, k] for i in (0, oct150.shape[0] - 1) for j in (0, oct150.shape[1] - 1) for k in (0, oct150.shape[2] - 1)], float)
BLOCK_R = float(np.linalg.norm((A_oct @ np.c_[_corn, np.ones(8)].T).T[:, :3] - c_o, axis=1).max())   # bounding-sphere radius of the block (mm)
BOX = BLOCK_R + 8.0                                                                                   # half-size of the MRI box around the block (vascular stage / outputs)
say(f"features ready: MRI level 0.6 grid {tuple(PM06.shape[1:])} @ {SEARCH_MM:.2f} mm; block radius {BLOCK_R:.1f} mm")

# ------------------------------------------------------------------ 1+2. global search at 0.6 mm (both handedness) + structural refinement, per class polarity
# The relative class polarity (is the OCT 'bright' class WM or GM?) is a property of the imaging protocol; give it with
# --oct-wm-bright yes|no.  'auto' runs the whole structural stage for both and keeps the one with the better refined NCC
# (the 0.6 mm search scores alone do not separate the two).
RK = dict(ls_clamp=a.ls_clamp, sh_clamp=a.sh_clamp, reg=a.reg)
def refiner_at(T, level, PO, half_mm=None):
    """Refiner on a block-sized MRI crop around the pose T (level 0.6 uses the whole pooled region)."""
    if level == 0.6: return Refiner(PM06, A06, PO[level][0], PO[level][1], PMask[level][0])
    ctr = (T @ np.r_[c_o, 1.0])[:3]; FMr, Ar = MF.region(ctr, half_mm or (BLOCK_R + 6.0), level)
    return Refiner(FMr, Ar, PO[level][0], PO[level][1], PMask[level][0])
def refine_level(cand_list, level, dofs, iters, keep, PO):
    out = []; ref = None
    for c in cand_list:
        T = c["T"]; ref = refiner_at(T, level, PO)
        for dof in dofs: T, loss = ref.refine(T, dof=dof, iters=iters, subsample=1, **RK)
        out.append({"T": T, "loss": loss, "ncc_channels": ref.ncc_channels(T)})
    out.sort(key=lambda d: d["loss"]); return out[:keep], ref
polarities = {"auto": [False, True], "yes": [True], "no": [False]}[a.oct_wm_bright]
trials = {}
for wm_bright in polarities:
    t0 = time.time()
    FO = FO_bright_first if wm_bright else FO_bright_first[[1, 0]]; PO = pyramid(FO, A_oct, VOX_O)
    s = FFTSearcher(PM06, A06, PT06, PO[0.6][0], PO[0.6][1], PMask[0.6][0], spacing=SEARCH_MM, min_overlap=MIN_OV)
    cands, sinfo = s.run(n_rot=a.n_rot, scales=(1.0,), topk=a.topk, seed=a.seed, log_every=4000, mirror=not a.no_mirror)
    del s; torch.cuda.empty_cache()
    lvl06, ref06 = refine_level(cands, 0.6, ("rigid", "similarity"), 120, keep=8, PO=PO)
    lvl03, _ = refine_level(lvl06, 0.3, ("rigid", "affine"), 200, keep=3, PO=PO)
    lvl015, _ = refine_level(lvl03, 0.15, ("affine",), 200, keep=3, PO=PO)
    best = lvl015[0]
    # refiners for the restarts: crops around the best pose, wide enough for +-30 deg / 5 mm perturbations
    ref03 = refiner_at(best["T"], 0.3, PO, half_mm=BLOCK_R + 12.0); ref015 = refiner_at(best["T"], 0.15, PO, half_mm=BLOCK_R + 12.0)
    say(f"polarity OCT wm_bright={wm_bright}: search top1/top2 {sinfo['top1']:.3f}/{sinfo['top2']:.3f}; refined NCC {1 - best['loss']:.3f} at centre {np.round((best['T'] @ np.r_[c_o, 1.0])[:3], 1).tolist()} mirror {bool(np.linalg.det(best['T'][:3,:3]) < 0)}  ({time.time()-t0:.0f}s)")
    trials[wm_bright] = dict(FO=FO, PO=PO, cands=cands, sinfo=sinfo, lvl015=lvl015, ref03=ref03, ref015=ref015, best=best, seconds=time.time() - t0)
    del ref06, lvl06, lvl03; torch.cuda.empty_cache()
    log.setdefault("polarity_trials", {})[str(wm_bright)] = {"search_top1": sinfo["top1"], "search_top2": sinfo["top2"], "refined_ncc": 1 - best["loss"], "centre": (best["T"] @ np.r_[c_o, 1.0])[:3].tolist(), "mirror": bool(np.linalg.det(best["T"][:3, :3]) < 0), "seconds": time.time() - t0}
OCT_WM_BRIGHT = min(trials, key=lambda k: trials[k]["best"]["loss"])
tr = trials[OCT_WM_BRIGHT]; FO, PO, cands, sinfo, lvl015, ref03, ref015, best = tr["FO"], tr["PO"], tr["cands"], tr["sinfo"], tr["lvl015"], tr["ref03"], tr["ref015"], tr["best"]
for k in list(trials):
    if k != OCT_WM_BRIGHT: del trials[k]
torch.cuda.empty_cache()
log["oct_wm_bright"] = OCT_WM_BRIGHT
log["search"] = sinfo | {"candidates": [{"score": c["score"], "overlap": c["overlap"], "mirror": c["mirror"], "centre": c["centre"].tolist()} for c in cands]}
T_struct = best["T"].copy()
runner = None
for d in lvl015[1:]:
    diff = transform_diff(T_struct, d["T"], c_o)
    if diff["centre_mm"] > 2.0 or diff["rotation_deg"] > 5.0: runner = {"loss": d["loss"], **diff}; break
log["refine"] = {"final_ncc": 1 - best["loss"], "final_ncc_channels": best["ncc_channels"], "distinct_runner_up": runner,
                 "level015_candidates": [{"loss": d["loss"], "centre": (d["T"] @ np.r_[c_o, 1.0])[:3].tolist()} for d in lvl015]}
say(f"structural stage: OCT wm_bright={OCT_WM_BRIGHT}; search top1/top2 {sinfo['top1']:.3f}/{sinfo['top2']:.3f}; NCC {1 - best['loss']:.3f} (channels {np.round(best['ncc_channels'], 3).tolist()}), centre {np.round((T_struct @ np.r_[c_o, 1.0])[:3], 1).tolist()}, mirror {bool(np.linalg.det(T_struct[:3,:3]) < 0)}")
# perturbed restarts (5-30 deg, <=5 mm) at 0.3/0.15 mm
rng = np.random.default_rng(a.seed + 1); restarts = []
for i in range(a.n_restarts):
    ang = np.deg2rad(rng.uniform(5, 30)); axis = rng.normal(size=3); axis /= np.linalg.norm(axis); Rp = rotvec_to_matrix(to_t(axis * ang)).cpu().numpy(); tp = rng.uniform(-5, 5, 3)
    Tp = T_struct.copy(); Tp[:3, :3] = Rp @ T_struct[:3, :3]; Tp[:3, 3] = (T_struct @ np.r_[c_o, 1.0])[:3] + tp - Tp[:3, :3] @ c_o
    Tr, _ = ref03.refine(Tp, dof="rigid", iters=150, **RK); Tr, _ = ref03.refine(Tr, dof="affine", iters=150, **RK); Tr, l2 = ref015.refine(Tr, dof="affine", iters=150, **RK)
    restarts.append({"perturb_deg": float(np.rad2deg(ang)), "perturb_mm": float(np.linalg.norm(tp)), "final_loss": l2, **transform_diff(Tr, T_struct, c_o)})
succ = [r for r in restarts if r["corner_mean_mm"] < 0.5]                    # block corners within 0.5 mm = same solution
say(f"structural restarts converged to the solution: {len(succ)}/{len(restarts)}")
del ref03, ref015, lvl015, PO, PMask, FO, trials, tr, PM06, PT06; torch.cuda.empty_cache()

# ------------------------------------------------------------------ 3. label-free vascular refinement (own OCT vessels vs MRI darkness)
R_axes = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
_corn = np.array([[i, j, k] for i in (0, oct150.shape[0] - 1) for j in (0, oct150.shape[1] - 1) for k in (0, oct150.shape[2] - 1)], float)
BLOCK_R = float(np.linalg.norm((A_oct @ np.c_[_corn, np.ones(8)].T).T[:, :3] - c_o, axis=1).max())   # bounding-sphere radius of the block (mm)
BOX = BLOCK_R + 8.0                                                                                   # half-size of the MRI box around the block for the vascular stage / outputs
def stretch_ijk(T): return [round(float(np.linalg.norm(T[:3, :3] @ R_axes[:, q])), 3) for q in range(3)]
T_final = T_struct.copy(); vasc = a.vascular == "on" and (a.work / "octv_vessels.npy").exists()
if vasc:
    t2 = time.time(); bc = (T_struct @ np.r_[c_o, 1.0])[:3]
    vlo, vhi = world_bbox_to_voxel(A_mri, mri.shape, bc - BOX, bc + BOX); A_v = A_mri.copy()
    # box clipped to the MRI region used for the features (region coords = MRI coords - lo)
    vlo = np.maximum(vlo, lo); vhi = np.minimum(vhi, hi); A_v[:3, 3] = (A_mri @ np.r_[vlo, 1.0])[:3]
    mri_v_reg = np.asarray(mri[vlo[0]:vhi[0], vlo[1]:vhi[1], vlo[2]:vhi[2]]).astype(np.float32); tis_v = np.asarray(mri_tissue_all[vlo[0]:vhi[0], vlo[1]:vhi[1], vlo[2]:vhi[2]]).astype(bool)
    rl, rh = vlo - lo, vhi - lo
    FM2v = torch.stack([to_t(np.asarray(x[rl[0]:rh[0], rl[1]:rh[1], rl[2]:rh[2]]).astype(np.float32)) for x in FM_cpu])   # same structural maps as the search/refinement
    FO2v = FO_bright_first if OCT_WM_BRIGHT else FO_bright_first[[1, 0]]
    mri_v = mri_dark_channel(mri_v_reg, tis_v, vox_mm=VOX_M)
    vmask = to_t(np.load(a.work / "octv_vessels.npy"), dtype=torch.bool); A_vv = np.load(a.work / "octv_affine.npy")
    sp_v = np.linalg.norm(A_vv[:3, :3], axis=0) * 1000
    oct_v = density_on_grid(vmask, A_vv, A_oct, oct150.shape, oct_mask, pool=int(max(1, round(VOX_O * 1000 / sp_v.mean())))); del vmask; torch.cuda.empty_cache()
    T_final, vinfo = vascular_refine(T_struct, FM2v, A_v, FO2v, A_oct, MO, mri_v, oct_v, w=a.vascular_w, reg=a.vascular_reg, clamp=a.vascular_clamp)
    log["vascular"] = {"ncc_channels": vinfo["ncc_channels"], "vs_structural": transform_diff(T_final, T_struct, c_o), "stretch_ijk_structural": stretch_ijk(T_struct), "stretch_ijk_vascular": stretch_ijk(T_final), "seconds": time.time() - t2}
    say(f"vascular refinement: ncc/channel {np.round(vinfo['ncc_channels'], 3).tolist()}; moved {log['vascular']['vs_structural']['corner_mean_mm']:.2f} mm (corner mean); stretch i,j,k {stretch_ijk(T_struct)} -> {stretch_ijk(T_final)}  ({time.time()-t2:.0f}s)")
    # restarts of the vascular stage around the structural solution (3-8 deg, <=2.5 mm, +-10% depth scale)
    r0_, t0_, ls0_, sh0_, mir_ = params_from_matrix(T_struct, c_o); vres = []
    for i in range(8):
        ang = np.deg2rad(rng.uniform(3, 8)); axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        Tp = compose(to_t(r0_ + axis * ang), to_t(t0_ + rng.uniform(-2.5, 2.5, 3)), to_t(ls0_ + rng.uniform(-0.1, 0.1, 3)), to_t(sh0_), to_t(c_o), mir_).cpu().numpy()
        Tr, _ = vascular_refine(Tp, FM2v, A_v, FO2v, A_oct, MO, mri_v, oct_v, w=a.vascular_w, reg=a.vascular_reg, clamp=a.vascular_clamp, factors=(4, 2, 1))
        vres.append({"perturb_deg": float(np.rad2deg(ang)), **transform_diff(Tr, T_final, c_o)})
    log["vascular"]["restarts"] = {"n": len(vres), "n_within_0.3mm": sum(r["corner_mean_mm"] < 0.3 for r in vres), "detail": vres}
    say(f"vascular restarts within 0.3 mm of the solution: {log['vascular']['restarts']['n_within_0.3mm']}/{len(vres)}")
else:
    log["vascular"] = {"mode": "off"}
if vasc:
    del FM2v, FO2v, mri_v, oct_v
torch.cuda.empty_cache()

# ------------------------------------------------------------------ 4. evaluation with whatever annotations exist (never used above)
oct_class = np.where(oct_mask, np.where(FO_bright_first[0].cpu().numpy() > 0.5, 1 if OCT_WM_BRIGHT else 2, 2 if OCT_WM_BRIGHT else 1), 0)
ves_ijk = np.argwhere(np.asarray(np.load(a.work / "mri_vessels.npy", mmap_mode="r"))) if (a.work / "mri_vessels.npy").exists() else None
def evaluate_T(T):
    e = {"stretch_ijk": stretch_ijk(T), "block_centre_mri_mm": (T @ np.r_[c_o, 1.0])[:3].tolist()}
    if ves_ijk is not None and len(ves_ijk) > 50:
        for tag in ("own", "vesseg"):
            f = a.work / f"oct_ves_dist48_{tag}.npy"
            if f.exists(): e[f"vessels_{tag}"] = vessel_distance_stats(T, ves_ijk, A_mri, np.load(f), np.load(a.work / f"oct_ves_dist48_{tag}_affine.npy"))
    if labels4 is not None: e["gmwm"] = gm_wm_overlap(T, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
    if ref_T is not None: e["vs_reference"] = transform_diff(T, ref_T, c_o)
    return e
ev = {"final": evaluate_T(T_final), "structural": evaluate_T(T_struct) if vasc else None, "restarts": {"n": len(restarts), "n_converged": len(succ), "detail": restarts}}
log["evaluation"] = ev
def _v(e, tag):
    r = e.get(f"vessels_{tag}", {}).get("registered", {}); c = e.get(f"vessels_{tag}", {}).get("random_shift_control", {})
    return f"{r.get('median_um', float('nan')):.0f} um / f150 {r.get('frac_within_150um', float('nan')):.2f} (ctrl median {c.get('median_um_mean') or float('nan'):.0f})" if r else "n/a"
for tag in ("own", "vesseg"):
    if f"vessels_{tag}" in ev["final"]:
        say(f"manual MRI vessels -> nearest OCT vessel [{tag}]: structural {_v(ev['structural'], tag) if vasc else '-'} -> final {_v(ev['final'], tag)}")
if "gmwm" in ev["final"]:
    g0 = ev["structural"]["gmwm"] if vasc else ev["final"]["gmwm"]; g1 = ev["final"]["gmwm"]
    say(f"Dice OCT classes vs manual labels: structural WM {g0['dice_WM']:.3f} GM {g0['dice_GM']:.3f} -> final WM {g1['dice_WM']:.3f} GM {g1['dice_GM']:.3f} (on-label frac {g1['frac_on_mri_labels']:.2f})")

# ------------------------------------------------------------------ 5. outputs
sc = np.linalg.norm(T_final[:3, :3], axis=0)
log["final_transform"] = {"T_oct2mri_world": T_final.tolist(), "T_structural": T_struct.tolist(), "block_centre_mri_mm": (T_final @ np.r_[c_o, 1.0])[:3].tolist(),
                          "column_scales": sc.tolist(), "det": float(np.linalg.det(T_final[:3, :3])), "mirror": bool(np.linalg.det(T_final[:3, :3]) < 0), "stretch_ijk": stretch_ijk(T_final)}
np.save(a.out / "T_oct2mri.npy", T_final); np.save(a.out / "T_oct2mri_structural.npy", T_struct)
(a.out / "T_oct2mri.lta").write_text(lta_text(T_final, "OCT world (oct150 affine)", "MRI NIfTI world"))
np.save(a.out / "oct_class150.npy", oct_class.astype(np.uint8))
log["total_seconds"] = time.time() - t_start
write_json(log, a.out / "result.json")           # write the result BEFORE the heavy QC/NIfTI outputs (they must not lose it)
bc = (T_final @ np.r_[c_o, 1.0])[:3]
qlo, qhi = world_bbox_to_voxel(A_mri, mri.shape, bc - BOX + 4, bc + BOX - 4); A_q = A_mri.copy(); A_q[:3, 3] = (A_mri @ np.r_[qlo, 1.0])[:3]
qc_figure(a.out / "qc_oct_space.png", T_final, oct150, A_oct, np.asarray(mri[qlo[0]:qhi[0], qlo[1]:qhi[1], qlo[2]:qhi[2]]).astype(np.float32), A_q, oct_mask, oct_class=oct_class,
          mri_labels=np.asarray(labels4[qlo[0]:qhi[0], qlo[1]:qhi[1], qlo[2]:qhi[2]]) if labels4 is not None else None, title=f"{a.work.name} {a.features}: NCC={1 - best['loss']:.3f}")
olo, ohi = world_bbox_to_voxel(A_mri, mri.shape, bc - BOX, bc + BOX); A_out = A_mri.copy(); A_out[:3, 3] = (A_mri @ np.r_[olo, 1.0])[:3]
mri_out = np.asarray(mri[olo[0]:ohi[0], olo[1]:ohi[1], olo[2]:ohi[2]]).astype(np.float32)
# chunk the output resampling over z-slabs (a whole-block box at 0.08-0.12 mm is hundreds of millions of voxels)
oct_in_mri = np.empty(mri_out.shape, np.float32); _O = to_t(oct150)[None]; _Ainv = to_t(np.linalg.inv(T_final))
for z0 in range(0, mri_out.shape[0], 64):
    z1 = min(mri_out.shape[0], z0 + 64)
    A_sub = A_out.copy(); A_sub[:3, 3] = (A_out @ np.r_[z0, 0, 0, 1.0])[:3]
    oct_in_mri[z0:z1] = resample_to_grid(_O, to_t(A_oct), to_t(A_sub), (z1 - z0, *mri_out.shape[1:]), T_grid_to_vol=_Ainv)[0].cpu().numpy()
del _O
save_nifti(oct_in_mri, A_out, a.out / "oct_in_mri_region.nii.gz", dtype=np.float32); save_nifti(mri_out, A_out, a.out / "mri_region.nii.gz", dtype=np.float32)
if labels4 is not None: save_nifti(np.asarray(labels4[olo[0]:ohi[0], olo[1]:ohi[1], olo[2]:ohi[2]]), A_out, a.out / "mri_labels_region.nii.gz", dtype=np.uint8)
log["total_seconds"] = time.time() - t_start
write_json(log, a.out / "result.json")          # (second write: identical content, after the heavy outputs succeeded)
say(json.dumps({"subject": a.work.name, "features": a.features, "search_top1_top2": [round(sinfo["top1"], 3), round(sinfo["top2"], 3)], "oct_wm_bright": OCT_WM_BRIGHT, "mirror": log["final_transform"]["mirror"],
                "centre": np.round(bc, 1).tolist(), "stretch_ijk": stretch_ijk(T_final), "final_ncc": round(1 - best["loss"], 3), "restarts": f"{len(succ)}/{len(restarts)}",
                "vascular_restarts": log["vascular"].get("restarts", {}).get("n_within_0.3mm"), "total_s": round(time.time() - t_start)}))
