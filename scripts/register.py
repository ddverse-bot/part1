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
from octreg.fine import FineContext, FineConfig, fine_stage, mri_box, block_corners, pose_move, restart_tol_mm, weight_fixed

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--features", choices=["otsu", "parser"], default="otsu"); ap.add_argument("--parser-dir", type=Path, default=None, help="dir with mri_prob_{wm,gm,tissue}.npy (features=parser)")
ap.add_argument("--crop-centre", type=str, default=None, help="x,y,z mm: register against a crop of the MRI around this point (a negative x must be written --crop-centre=-x,y,z: argparse reads '-5,..' as an option)"); ap.add_argument("--crop-half-mm", type=float, default=30.0)
ap.add_argument("--n-rot", type=int, default=8000); ap.add_argument("--topk", type=int, default=24); ap.add_argument("--min-overlap", type=float, default=0.85)
ap.add_argument("--no-mirror", action="store_true"); ap.add_argument("--oct-wm-bright", choices=["auto", "yes", "no"], default="no", help="is the brighter OCT class WM? (imaging-protocol property; serial-sectioning OCT pooled to 0.15 mm: GM brighter -> 'no'); auto = run both, keep the better refined NCC")
ap.add_argument("--mri-wm-bright", action="store_true", help="MRI has WM brighter than GM (default: GM bright, as ex-vivo FLASH ~20 deg)")
ap.add_argument("--ls-clamp", type=float, default=0.15); ap.add_argument("--sh-clamp", type=float, default=0.15); ap.add_argument("--reg", type=float, default=2.0)
ap.add_argument("--vascular-w", type=float, default=2.0); ap.add_argument("--vascular-reg", type=float, default=0.5); ap.add_argument("--vascular-clamp", type=float, default=0.3)
ap.add_argument("--n-restarts", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ref-transform", type=Path, default=None, help="optional 4x4 (json/npy) OCT->MRI reference, reported only")
# v1.1 (spec 2.1): position prior, vascular gate, fine stage.  Defaults reproduce v1 (the vascular gate only evaluates two NCCs).
ap.add_argument("--init-transform", type=Path, default=None, help="4x4 .npy/.json OCT world (oct150 affine) -> MRI world: skips the FFT search and the structural refinement (polarity evaluated, not optimised)")
ap.add_argument("--vascular", choices=["auto", "on", "off"], default="auto", help="auto = on unless prep.json oct.vessel_section_modulation.ratio > --vessel-modulation-max (missing key / no octv_vessels.npy -> as v1)")
ap.add_argument("--vessel-modulation-max", type=float, default=1.6)
ap.add_argument("--vascular-min-ncc", type=float, default=0.02, help="vessel-channel NCC required to accept the vascular pose (I58 v1 0.0055, I46 0.296, I55 0.083)")
ap.add_argument("--vascular-struct-drop", type=float, default=0.05, help="max drop of the mean [WM,GM] NCC at T_vasc vs T_struct (v1's accepted DANDI solutions drop 0.027 on I46 and 0.004 on I55; the spec's 0.03 left I46 a 0.003 margin)")
ap.add_argument("--vascular-max-move", type=float, default=3.0, help="mm, mean displacement of the 8 OCT block corners (vs_structural keeps the 7 mm transform_diff for comparability)")
ap.add_argument("--fine", choices=["off", "on"], default="off"); ap.add_argument("--fine-sim", choices=["auto", "ncc", "lcc", "lcc2"], default="auto", help="auto = ncc when the polarity guard is decisive, lcc2 when mixed")
ap.add_argument("--fine-levels", type=str, default="0.3,0.15", help="mm"); ap.add_argument("--fine-flatten-mm", type=float, default=3.0); ap.add_argument("--fine-erode-mm", type=float, default=1.3)
ap.add_argument("--fine-lcc-mm", type=float, default=4.5); ap.add_argument("--fine-clamp", type=float, default=0.15, help="RELATIVE ls/sh clamp per step"); ap.add_argument("--fine-reg", type=float, default=0.5)
ap.add_argument("--fine-iters", type=int, default=200); ap.add_argument("--fine-restarts", type=int, default=8, help="0 = diagnostic only: gate (d) then rejects the stage (unverified)")
ap.add_argument("--fine-max-move", type=float, default=3.0, help="mm, mean displacement of the 8 OCT block corners vs the start pose; also caps U_mm"); ap.add_argument("--fine-max-rot", type=float, default=5.0, help="deg, rotation of the fine delta vs the start pose")
ap.add_argument("--fine-subsample", type=int, default=1)
ap.add_argument("--fine-verify", choices=["basic", "full"], default="basic", help="basic = restarts + multi-similarity + split-half + landscapes; full adds inverse consistency")
# v1.1 fine-stage options after the P7/P8 probes; defaults = v1.1 behaviour (see octreg/fine.py header)
ap.add_argument("--fine-fixed-mask", choices=["on", "off"], default="off", help="on = P7's configuration: m_spec chosen once per level at the start pose and kept for every chain, AND (unless --fine-fixed-weight off) the ncc loss's MRI-tissue weight frozen there too (v1.1's 'mask follows pose' rewards drifting)")
ap.add_argument("--fine-fixed-weight", choices=["auto", "on", "off"], default="auto", help="the ncc loss's detached MRI-tissue weight: auto = frozen iff --fine-fixed-mask on (P7's pair); on = frozen at the mask pose even without a fixed mask; off = re-sampled through the current pose at every evaluation (v1.1; with --fine-fixed-mask on = the fixed-mask-only I58 runs (ii)/(iii), where points that drift off the MRI tissue silently drop out)")
ap.add_argument("--fine-restart-mm", type=float, default=3.0, help="restart translation size (U(-2/3,2/3) x this per axis; v1.1 = 3.0)")
ap.add_argument("--fine-restart-tol-mm", type=float, default=0.3, help="a restart counts as converged within this block-corner distance (mm) of T_fine (v1.1 gate 0.3; NOT scaled with --fine-restart-mm: pass e.g. 0.3 x that explicitly for larger restarts)")
ap.add_argument("--fine-restart-deg", type=float, default=5.0, help="restart rotation size (angle U(0.4,1.2) x this; v1.1 = 5.0)"); ap.add_argument("--fine-restart-logscale", type=float, default=0.03, help="restart log-scale U(-1,1) x this per axis")
ap.add_argument("--fine-dof", choices=["rigid", "similarity", "affine"], default="affine", help="degrees of freedom of the fine polish after the first level's rigid step (v1.1 = affine; R10: the affine polish over-stretched the I46 depth axis 1.219 -> 1.31, so a conservative polish is rigid)")
ap.add_argument("--fine-exclude-fov-mm", type=float, default=0.0, help="mm; > 0 drops MRI tissue this close to a field-of-view face from the fine stage's tissue.  FOV faces = the faces of the MRI region the stage sees: the mri.npy array faces, or the --crop-centre box faces when that is set (which flagged faces are array faces is recorded in result.json fine.fov_faces; P8: the I58 MRI is a whole-brain crop whose tissue touches 3 array faces)")
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
_po = prep.get("oct", {}); _vsm = _po.get("vessel_section_modulation") or None                       # v1.1 prep keys; absent = legacy prep -> v1 behaviour (F5)
if V_o > 1.5 * V_m and _po.get("mask_mode") in (None, "intensity"): say("WARNING: OCT mask over-inclusive: re-prep with --oct-mask auto/texture")
log["prep_summary"] = {"mask_mode": _po.get("mask_mode"), "mask_reason": _po.get("mask_reason"), "destripe_applied": (_po.get("destripe") or {}).get("applied"),
                       "V_oct_mask_cm3": _po.get("V_oct_cm3_tex") if _po.get("mask_mode") == "texture" else _po.get("V_oct_cm3_int"), "V_mri_tissue_cm3": _po.get("V_mri_cm3"),
                       "vessel_section_modulation": _vsm, "vessels_skipped": _po.get("vessels_skipped")}
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
fine_on = a.fine == "on"; keep_ref015 = fine_on or a.init_transform is not None          # the fine stage's structural gate needs ref015
trials = {}
if a.init_transform is not None:
    # ---- position prior (spec 2.2): the FFT search and the structural refinement are skipped; the prior is honoured as given.
    # The class polarity is EVALUATED (not optimised) at T_init with the v1 ref015 construction for each candidate polarity.
    T_init = np.load(a.init_transform) if a.init_transform.suffix == ".npy" else np.array(json.load(open(a.init_transform)), float)
    for wm_bright in polarities:
        t0 = time.time()
        FO = FO_bright_first if wm_bright else FO_bright_first[[1, 0]]; PO = pyramid(FO, A_oct, VOX_O)
        ref015 = refiner_at(T_init, 0.15, PO, half_mm=BLOCK_R + 12.0); l = ref015.evaluate(T_init)
        best = {"T": T_init.copy(), "loss": l, "ncc_channels": ref015.ncc_channels(T_init)}
        say(f"polarity OCT wm_bright={wm_bright}: NCC at the init transform {1 - l:.3f} (channels {np.round(best['ncc_channels'], 3).tolist()})  ({time.time()-t0:.0f}s)")
        trials[wm_bright] = dict(FO=FO, PO=PO, ref015=ref015, best=best, seconds=time.time() - t0)
        log.setdefault("polarity_trials", {})[str(wm_bright)] = {"search_top1": None, "search_top2": None, "refined_ncc": 1 - l, "evaluated_only": True, "centre": (T_init @ np.r_[c_o, 1.0])[:3].tolist(), "mirror": bool(np.linalg.det(T_init[:3, :3]) < 0), "seconds": time.time() - t0}
    OCT_WM_BRIGHT = min(trials, key=lambda k: trials[k]["best"]["loss"])
    tr = trials[OCT_WM_BRIGHT]; FO, PO, ref015, best = tr["FO"], tr["PO"], tr["ref015"], tr["best"]
    for k in list(trials):
        if k != OCT_WM_BRIGHT: del trials[k]
    torch.cuda.empty_cache()
    cands = []; sinfo = {"skipped": "init_transform", "top1": None, "top2": None}
    log["oct_wm_bright"] = OCT_WM_BRIGHT; log["search"] = dict(sinfo)
    T_struct = T_init.copy()
    log["refine"] = {"final_ncc": 1 - best["loss"], "final_ncc_channels": best["ncc_channels"], "init_transform": str(a.init_transform)}
    say(f"structural stage skipped (--init-transform {a.init_transform}): OCT wm_bright={OCT_WM_BRIGHT}; NCC {1 - best['loss']:.3f} (channels {np.round(best['ncc_channels'], 3).tolist()}), centre {np.round((T_struct @ np.r_[c_o, 1.0])[:3], 1).tolist()}, mirror {bool(np.linalg.det(T_struct[:3,:3]) < 0)}")
    rng = np.random.default_rng(a.seed + 1); restarts = []; succ = []
    del PO, PMask, FO, trials, tr, PM06, PT06; torch.cuda.empty_cache()
else:
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
    del ref03, lvl015, PO, PMask, FO, trials, tr, PM06, PT06
    if not keep_ref015: del ref015                                                # kept alive for the fine stage's structural gate (deleted after it)
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ 3. label-free vascular refinement (own OCT vessels vs MRI darkness)
R_axes = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
_corn = np.array([[i, j, k] for i in (0, oct150.shape[0] - 1) for j in (0, oct150.shape[1] - 1) for k in (0, oct150.shape[2] - 1)], float)
BLOCK_R = float(np.linalg.norm((A_oct @ np.c_[_corn, np.ones(8)].T).T[:, :3] - c_o, axis=1).max())   # bounding-sphere radius of the block (mm)
BOX = BLOCK_R + 8.0                                                                                   # half-size of the MRI box around the block for the vascular stage / outputs
def stretch_ijk(T): return [round(float(np.linalg.norm(T[:3, :3] @ R_axes[:, q])), 3) for q in range(3)]
T_final = T_struct.copy(); _ratio = (_vsm or {}).get("ratio"); vasc_file = (a.work / "octv_vessels.npy").exists()
# v1.1 (spec 2.3): auto = as v1 unless the prep flags a section-phase vessel artefact; the vascular pose is then GATED, never accepted blindly
vasc = vasc_file and (a.vascular == "on" or (a.vascular == "auto" and (_ratio is None or _ratio <= a.vessel_modulation_max)))
if vasc:
    t2 = time.time(); bc = (T_struct @ np.r_[c_o, 1.0])[:3]
    # box clipped to the MRI region used for the features (region coords = MRI coords - lo); shared helper with the fine stage
    vlo, vhi, A_v, mri_v_reg, tis_v = mri_box(mri, mri_tissue_all, A_mri, lo, hi, bc, BOX)
    rl, rh = vlo - lo, vhi - lo
    FM2v = torch.stack([to_t(np.asarray(x[rl[0]:rh[0], rl[1]:rh[1], rl[2]:rh[2]]).astype(np.float32)) for x in FM_cpu])   # same structural maps as the search/refinement
    FO2v = FO_bright_first if OCT_WM_BRIGHT else FO_bright_first[[1, 0]]
    mri_v = mri_dark_channel(mri_v_reg, tis_v, vox_mm=VOX_M)
    vmask = to_t(np.load(a.work / "octv_vessels.npy"), dtype=torch.bool); A_vv = np.load(a.work / "octv_affine.npy")
    sp_v = np.linalg.norm(A_vv[:3, :3], axis=0) * 1000
    oct_v = density_on_grid(vmask, A_vv, A_oct, oct150.shape, oct_mask, pool=int(max(1, round(VOX_O * 1000 / sp_v.mean())))); del vmask; torch.cuda.empty_cache()
    T_vasc, vinfo = vascular_refine(T_struct, FM2v, A_v, FO2v, A_oct, MO, mri_v, oct_v, w=a.vascular_w, reg=a.vascular_reg, clamp=a.vascular_clamp)
    # acceptance gate: vessel-channel NCC, no loss of structural [WM,GM] agreement (same refiner at both poses), bounded move
    ref_v = Refiner(FM2v, A_v, FO2v, A_oct, MO); s_before = 1 - ref_v.evaluate(T_struct); s_after = 1 - ref_v.evaluate(T_vasc); del ref_v
    vdiff = transform_diff(T_vasc, T_struct, c_o); vmove = pose_move(T_vasc, T_struct, c_o, block_corners(A_oct, oct150.shape)); vreasons = []; vdrop = s_before - s_after
    if vinfo["ncc_channels"][2] < a.vascular_min_ncc: vreasons.append(f"vessel NCC {vinfo['ncc_channels'][2]:.4f} < {a.vascular_min_ncc}")
    if vdrop > a.vascular_struct_drop: vreasons.append(f"structural NCC {s_before:.4f} -> {s_after:.4f} (drop {vdrop:.4f} > {a.vascular_struct_drop})")
    if vmove["block_corner_mean_mm"] > a.vascular_max_move: vreasons.append(f"moved {vmove['block_corner_mean_mm']:.2f} mm at the block corners > {a.vascular_max_move}")
    vasc_ok = not vreasons; T_final = T_vasc.copy() if vasc_ok else T_struct.copy()
    log["vascular"] = {"ncc_channels": vinfo["ncc_channels"], "vs_structural": vdiff, "vs_structural_block": vmove, "stretch_ijk_structural": stretch_ijk(T_struct), "stretch_ijk_vascular": stretch_ijk(T_vasc),
                       "accepted": vasc_ok, "reasons": vreasons, "struct_ncc_before": s_before, "struct_ncc_after": s_after, "struct_drop": vdrop, "struct_drop_margin": a.vascular_struct_drop - vdrop,
                       "T_candidate": T_vasc.tolist(), "seconds": time.time() - t2}
    say(f"vascular refinement: ncc/channel {np.round(vinfo['ncc_channels'], 3).tolist()}; moved {vdiff['corner_mean_mm']:.2f} mm (7 mm corner mean) / {vmove['block_corner_mean_mm']:.2f} mm (block corners), rotated {vmove['delta_rotation_deg']:.2f} deg; "
        f"stretch i,j,k {stretch_ijk(T_struct)} -> {stretch_ijk(T_vasc)}; structural NCC {s_before:.4f} -> {s_after:.4f} (drop {vdrop:.4f}, margin {a.vascular_struct_drop - vdrop:+.4f} to {a.vascular_struct_drop}); accepted={vasc_ok} {vreasons}  ({time.time()-t2:.0f}s)")
    # restarts of the vascular stage around the structural solution (3-8 deg, <=2.5 mm, +-10% depth scale); diagnostic, vs the vascular solution
    r0_, t0_, ls0_, sh0_, mir_ = params_from_matrix(T_struct, c_o); vres = []
    for i in range(8):
        ang = np.deg2rad(rng.uniform(3, 8)); axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        Tp = compose(to_t(r0_ + axis * ang), to_t(t0_ + rng.uniform(-2.5, 2.5, 3)), to_t(ls0_ + rng.uniform(-0.1, 0.1, 3)), to_t(sh0_), to_t(c_o), mir_).cpu().numpy()
        Tr, _ = vascular_refine(Tp, FM2v, A_v, FO2v, A_oct, MO, mri_v, oct_v, w=a.vascular_w, reg=a.vascular_reg, clamp=a.vascular_clamp, factors=(4, 2, 1))
        vres.append({"perturb_deg": float(np.rad2deg(ang)), **transform_diff(Tr, T_vasc, c_o)})
    log["vascular"]["restarts"] = {"n": len(vres), "n_within_0.3mm": sum(r["corner_mean_mm"] < 0.3 for r in vres), "detail": vres}
    say(f"vascular restarts within 0.3 mm of the solution: {log['vascular']['restarts']['n_within_0.3mm']}/{len(vres)}")
elif vasc_file and a.vascular == "auto":
    log["vascular"] = {"mode": "skipped_section_artefact", "ratio": _ratio}; say(f"vascular stage skipped: vessel section-phase modulation ratio {_ratio} > {a.vessel_modulation_max}")
else:
    log["vascular"] = {"mode": "off"}
if vasc:
    del FM2v, FO2v, mri_v, oct_v
torch.cuda.empty_cache()

# ------------------------------------------------------------------ 3b. fine stage (v1.1, spec 2.4): single sharp loss, delta about the start pose, gated
if fine_on:
    t3 = time.time(); T_prefine = T_final.copy(); np.save(a.out / "T_oct2mri_prefine.npy", T_prefine)
    ctx = FineContext(mri=mri, mri_tissue=mri_tissue_all, A_mri=A_mri, lo=lo, hi=hi, oct150=oct150, oct_mask=oct_mask, A_oct=A_oct, c_o=c_o,
                      BLOCK_R=BLOCK_R, VOX_M=VOX_M, VOX_O=VOX_O, ref015=ref015, device=DEVICE)
    cfg = FineConfig(sim=a.fine_sim, levels=tuple(float(x) for x in a.fine_levels.split(",")), flatten_mm=a.fine_flatten_mm, erode_mm=a.fine_erode_mm, lcc_mm=a.fine_lcc_mm,
                     clamp=a.fine_clamp, reg=a.fine_reg, iters=a.fine_iters, n_restarts=a.fine_restarts, max_move=a.fine_max_move, max_rot=a.fine_max_rot, subsample=a.fine_subsample, verify=a.fine_verify,
                     fixed_mask=a.fine_fixed_mask == "on", fixed_weight=a.fine_fixed_weight, restart_mm=a.fine_restart_mm, restart_deg=a.fine_restart_deg, restart_tol_mm=a.fine_restart_tol_mm, restart_logscale=a.fine_restart_logscale, exclude_fov_mm=a.fine_exclude_fov_mm, dof=a.fine_dof)
    say(f"fine options: dof={cfg.dof}, fixed_mask={cfg.fixed_mask}, fixed_weight={cfg.fixed_weight}, restarts {cfg.restart_mm} mm / {cfg.restart_deg} deg / logscale {cfg.restart_logscale} (converged < {restart_tol_mm(cfg):g} mm), exclude_fov {cfg.exclude_fov_mm} mm; "
        f"ncc weight {'frozen at the mask pose' if weight_fixed(cfg) else 'follows the pose (v1.1)'}")
    T_final, T_fine_cand, finfo = fine_stage(T_prefine, ctx, cfg, rng)
    np.save(a.out / "T_oct2mri_fine_candidate.npy", T_fine_cand); log["fine"] = finfo; _vs = finfo.get("vs_start") or {}
    log["fine"]["options"] = {"dof": cfg.dof, "fixed_mask": cfg.fixed_mask, "fixed_weight": weight_fixed(cfg), "fixed_weight_option": cfg.fixed_weight, "restart_mm": cfg.restart_mm, "restart_deg": cfg.restart_deg, "restart_tol_mm": cfg.restart_tol_mm,
                              "restart_logscale": cfg.restart_logscale, "exclude_fov_mm": cfg.exclude_fov_mm}
    say(f"fine stage: sign/level {finfo.get('sign_per_level')} sim/level {finfo.get('sim_per_level')}; guard {[(g['level'], g.get('frac_neg'), g.get('mean')) for g in finfo.get('guard', [])]}; "
        f"loss {finfo.get('loss_start_finest')} -> {finfo.get('loss_fine_finest')}; structural NCC {finfo.get('struct_ncc_before')} -> {finfo.get('struct_ncc_after')}; MI {finfo.get('mi_before')} -> {finfo.get('mi_after')}; "
        f"moved {_vs.get('corner_mean_mm')} mm (7 mm cube) / {_vs.get('block_corner_mean_mm')} mm (block corners), rotated {_vs.get('delta_rotation_deg')} deg (OCT-axis shift {_vs.get('centre_shift_oct_axes_mm')}); "
        f"restarts {finfo.get('restarts', {}).get('n_converged')}/{finfo.get('restarts', {}).get('n')}; U_mm {finfo.get('U_mm')}; accepted={finfo['accepted']} {finfo['reasons']}  ({time.time()-t3:.0f}s)")
    del ctx
if keep_ref015:
    del ref015
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
ev = {"final": evaluate_T(T_final), "structural": evaluate_T(T_struct) if (vasc or fine_on) else None, "restarts": {"n": len(restarts), "n_converged": len(succ), "detail": restarts}}
log["evaluation"] = ev
def _v(e, tag):
    r = e.get(f"vessels_{tag}", {}).get("registered", {}); c = e.get(f"vessels_{tag}", {}).get("random_shift_control", {})
    return f"{r.get('median_um', float('nan')):.0f} um / f150 {r.get('frac_within_150um', float('nan')):.2f} (ctrl median {c.get('median_um_mean') or float('nan'):.0f})" if r else "n/a"
for tag in ("own", "vesseg"):
    if f"vessels_{tag}" in ev["final"]:
        say(f"manual MRI vessels -> nearest OCT vessel [{tag}]: structural {_v(ev['structural'], tag) if (vasc or fine_on) else '-'} -> final {_v(ev['final'], tag)}")
if "gmwm" in ev["final"]:
    g0 = ev["structural"]["gmwm"] if (vasc or fine_on) else ev["final"]["gmwm"]; g1 = ev["final"]["gmwm"]
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
def _r3(x): return None if x is None else round(x, 3)
_fi = log.get("fine", {})
say(json.dumps({"subject": a.work.name, "features": a.features, "search_top1_top2": [_r3(sinfo["top1"]), _r3(sinfo["top2"])], "oct_wm_bright": OCT_WM_BRIGHT, "mirror": log["final_transform"]["mirror"],
                "centre": np.round(bc, 1).tolist(), "stretch_ijk": stretch_ijk(T_final), "final_ncc": round(1 - best["loss"], 3), "restarts": f"{len(succ)}/{len(restarts)}",
                "vascular_restarts": log["vascular"].get("restarts", {}).get("n_within_0.3mm"), "vascular_accepted": log["vascular"].get("accepted"),
                "fine_accepted": _fi.get("accepted"), "fine_moved_mm": _r3((_fi.get("vs_start") or {}).get("corner_mean_mm")), "fine_moved_block_mm": _r3((_fi.get("vs_start") or {}).get("block_corner_mean_mm")),
                "fine_rot_deg": _r3((_fi.get("vs_start") or {}).get("delta_rotation_deg")), "U_mm": _r3(_fi.get("U_mm")), "total_s": round(time.time() - t_start)}))
