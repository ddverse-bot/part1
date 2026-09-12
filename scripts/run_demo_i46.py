#!/usr/bin/env python3
"""End-to-end OCT-block -> MRI registration on Costantini sub-I46, without any location prior.

    python run_demo_i46.py --work work/i46 --parser work/parser_v1 --out work/runs/demo1 \
        --scenario crop|whole --features parser|mind|otsu|intensity --n-rot 3000

Pipeline
  1. features on both sides (parser probabilities by default; MIND-SSC / Otsu / intensity as baselines)
  2. global FFT search (masked NCC over all translations x random SO(3) rotations) at 0.6 mm
  3. multi-resolution refinement of the top-K candidates: rigid -> similarity (0.6 mm), affine (0.3, 0.15 mm)
  4. held-out evaluation: MRI vessel labels vs OCT vessel segmentation, MRI GM/WM labels vs OCT classes,
     agreement with the coarse author-affine placement, robustness restarts; QC figures.

Scenario 'crop' mimics Xiangrui's data (a cropped MRI around the block; here: 60 mm cube around the coarse
author placement); 'whole' searches the entire hemisphere.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import (DEVICE, avg_pool_iso, crop_volume, lta_text, polar_rotation, to_t, world_bbox_to_voxel, write_json,
                           rotation_geodesic_deg, save_nifti, apply_affine, sample_at_world)
from octreg.features import build_features, mind_ssc, otsu_two_class
from octreg.search import FFTSearcher
from octreg.refine import Refiner
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap, transform_diff, qc_figure
from octreg.vascular import mri_dark_channel, mri_frangi_channel, oct_vessel_mask, density_on_grid, vascular_refine
from octreg.nonrigid import nonrigid_refine, inverse_points

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--parser", type=Path, required=True, help="dir with parser.pt, oct150_prob.npy, mri_prob_*.npy")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--scenario", choices=["crop", "whole"], default="crop")
ap.add_argument("--features", choices=["mixed", "parser", "mind", "otsu", "intensity", "parser_vessel", "mind_vessel", "vessel"], default="mixed",
                help="features for the global search: 'mixed' = MRI parser probabilities + OCT intensity-derived WM/GM (polarity resolved by the parser)")
ap.add_argument("--no-mirror", action="store_true", help="search proper rotations only (default: both handedness)")
ap.add_argument("--refine-features", default=None, help="features for refinement (default: same as --features)")
ap.add_argument("--ls-clamp", type=float, default=0.15); ap.add_argument("--sh-clamp", type=float, default=0.15); ap.add_argument("--reg", type=float, default=2.0)
ap.add_argument("--parser-channels", default="wm_gm")
ap.add_argument("--crop-half-mm", type=float, default=30.0)
ap.add_argument("--n-rot", type=int, default=3000)
ap.add_argument("--scales", type=str, default="1.0")
ap.add_argument("--topk", type=int, default=24)
ap.add_argument("--min-overlap", type=float, default=0.85)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-restarts", type=int, default=12, help="robustness restarts around the solution")
ap.add_argument("--ref-transform", type=Path, default=None, help="optional 4x4 (json list) coarse reference OCT->MRI world")
ap.add_argument("--vascular", choices=["off", "auto", "own", "ves_seg"], default="auto",
                help="label-free vascular refinement after the structural affine: OCT vessel density from our own Frangi on the fine OCT level "
                     "(work/oct24.npy, 'own') or from a provided vessel segmentation (work/oct_ves_dist48_z0.npy, 'ves_seg'); auto = own if available else ves_seg")
ap.add_argument("--vascular-mri", choices=["dark", "frangi"], default="dark", help="MRI vessel channel: local-median darkness or Frangi dark-tube vesselness")
ap.add_argument("--vascular-w", type=float, default=2.0); ap.add_argument("--vascular-reg", type=float, default=0.5); ap.add_argument("--vascular-clamp", type=float, default=0.3)
ap.add_argument("--vascular-q", type=float, default=99.0, help="top-q%% vesselness inside tissue kept as OCT vessels (own Frangi)")
ap.add_argument("--oct-vesdens", type=Path, default=None, help="cached OCT vessel-density map on the oct150 grid (prep_oct_vessels.py); overrides the on-the-fly 24 um Frangi")
ap.add_argument("--nonrigid", choices=["off", "on"], default="on", help="label-free free-form stage after the vascular affine (3-channel NCC, control grid over the block)")
ap.add_argument("--nonrigid-grid", type=str, default="5,8,8"); ap.add_argument("--nonrigid-smooth", type=float, default=1.0); ap.add_argument("--nonrigid-mag", type=float, default=0.5)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
t_start = time.time()
log = {"args": vars(a) | {k: str(v) for k, v in vars(a).items() if isinstance(v, Path)}}
def say(*x):
    print(*x, flush=True)

# ------------------------------------------------------------------ load
prep = json.load(open(a.work / "prep.json"))
A_mri = np.load(a.work / "mri_affine.npy")
mri = np.load(a.work / "mri.npy", mmap_mode="r")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)          # [4,d,h,w]
ref_T = np.array(json.load(open(a.ref_transform))) if a.ref_transform else None
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]

# ------------------------------------------------------------------ MRI region
if a.scenario == "crop":
    centre = (ref_T @ np.r_[c_o, 1.0])[:3] if ref_T is not None else np.array([-19.0, 24.0, 20.0])
    lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - a.crop_half_mm, centre + a.crop_half_mm)
else:
    lo, hi = np.zeros(3, int), np.array(mri.shape)
say(f"MRI region voxels {lo.tolist()}..{hi.tolist()} ({(hi - lo).tolist()})")
log["mri_region_ijk"] = [lo.tolist(), hi.tolist()]

def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r")
    return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)

A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
mri_tis = to_t(load_region("tissue") > 0.5)[None].float()

# ------------------------------------------------------------------ features (0.15 mm), then pyramids
def mri_features(kind):
    if kind in ("parser", "mixed"):
        f = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
        if a.parser_channels == "tissue_wm_gm":
            f = torch.cat([mri_tis, f], 0)
        return f
    vol = to_t(mri_reg)[None]
    if kind == "parser_vessel":
        return torch.cat([mri_features("parser"), build_features("vessel", vol, mri_tis)], 0)
    return build_features(kind, vol, mri_tis, wm_bright=False)      # MRI: GM brighter than WM in this contrast

# OCT WM/GM intensity polarity resolved automatically from the parser: mean intensity of parser-WM vs parser-GM voxels
_pred = oct_prob.argmax(0)
_wm_i = oct150[(_pred == 1) & oct_mask]; _gm_i = oct150[(_pred >= 2) & oct_mask]
OCT_WM_BRIGHT = bool(_wm_i.mean() > _gm_i.mean()) if (_wm_i.size > 100 and _gm_i.size > 100) else False
say(f"OCT polarity from parser: mean(WM)={_wm_i.mean():.1f} mean(GM)={_gm_i.mean():.1f} -> wm_bright={OCT_WM_BRIGHT}")
log["oct_wm_bright"] = OCT_WM_BRIGHT

def oct_features(kind):
    prob = to_t(oct_prob); mask = to_t(oct_mask)[None].float()
    if kind == "parser":
        return build_features("parser", None, mask, prob=prob, parser_channels=a.parser_channels)
    vol = to_t(oct150)[None]
    if kind == "mixed":
        return build_features("otsu", vol, mask, wm_bright=OCT_WM_BRIGHT)
    if kind == "parser_vessel":
        return torch.cat([oct_features("parser"), build_features("vessel", vol, mask)], 0)
    return build_features(kind, vol, mask, wm_bright=OCT_WM_BRIGHT)

rf = a.refine_features or a.features
FM = mri_features(a.features); FO = oct_features(a.features)
FMr = FM if rf == a.features else mri_features(rf); FOr = FO if rf == a.features else oct_features(rf)
MO = to_t(oct_mask)[None].float()
say(f"search features '{a.features}': MRI {tuple(FM.shape)} OCT {tuple(FO.shape)}; refine features '{rf}': {tuple(FMr.shape)} / {tuple(FOr.shape)}")

def pyramid(F_, A_, factors=(4, 2, 1)):
    out = {}
    for f in factors:
        if f == 1:
            out[f] = (F_, np.asarray(A_))
        else:
            v, A2 = avg_pool_iso(F_, A_, f)
            out[f] = (v, A2)
    return out

PM = pyramid(FM, A_reg); PO = pyramid(FO, A_oct); PMask = pyramid(MO, A_oct); PTis = pyramid(mri_tis, A_reg)
PMr = pyramid(FMr, A_reg); POr = pyramid(FOr, A_oct)

# ------------------------------------------------------------------ 1. global search at 0.6 mm
scales = tuple(float(s) for s in a.scales.split(","))
searcher = FFTSearcher(PM[4][0], PM[4][1], PTis[4][0], PO[4][0], PO[4][1], PMask[4][0], spacing=0.6, min_overlap=a.min_overlap)
say(f"search: template {searcher.n}^3 vox @0.6mm over MRI grid {searcher.shape}, {a.n_rot} rotations x scales {scales}")
cands, sinfo = searcher.run(n_rot=a.n_rot, scales=scales, topk=a.topk, seed=a.seed, log_every=2000, mirror=not a.no_mirror)
log["search"] = sinfo | {"candidates": [{"score": c["score"], "overlap": c["overlap"], "scale": c["scale"], "mirror": c["mirror"], "centre": c["centre"].tolist()} for c in cands]}
say(f"search done in {sinfo['seconds']:.0f}s; top scores {[round(c['score'], 3) for c in cands[:6]]}")
del searcher; torch.cuda.empty_cache()

# ------------------------------------------------------------------ 2. refinement
RK = dict(ls_clamp=a.ls_clamp, sh_clamp=a.sh_clamp, reg=a.reg)
def refine_level(cand_list, level, dofs, iters, keep):
    FMl, AMl = PMr[level]; FOl, AOl = POr[level]; Ml = PMask[level][0]
    ref = Refiner(FMl, AMl, FOl, AOl, Ml)
    out = []
    for c in cand_list:
        T = c["T"]
        for dof in dofs:
            T, loss = ref.refine(T, dof=dof, iters=iters, subsample=1, **RK)
        out.append({"T": T, "loss": loss, "src": c.get("src", c.get("rot_index")), "ncc_channels": ref.ncc_channels(T)})
    out.sort(key=lambda d: d["loss"])
    return out[:keep], ref

t1 = time.time()
lvl06, ref06 = refine_level(cands, 4, ("rigid", "similarity"), 120, keep=8)
say(f"level 0.6mm: best losses {[round(d['loss'], 4) for d in lvl06[:5]]}  ({time.time() - t1:.0f}s)")
lvl03, ref03 = refine_level(lvl06, 2, ("rigid", "affine"), 200, keep=3)
say(f"level 0.3mm: best losses {[round(d['loss'], 4) for d in lvl03]}  ncc/channel {[[round(x, 3) for x in d['ncc_channels']] for d in lvl03]}")
lvl015, ref015 = refine_level(lvl03, 1, ("affine",), 200, keep=3)
say(f"level 0.15mm: best losses {[round(d['loss'], 4) for d in lvl015]}  ncc/channel {[[round(x, 3) for x in d['ncc_channels']] for d in lvl015]}")
best = lvl015[0]
T_final = best["T"]
log["refine"] = {"level06_losses": [d["loss"] for d in lvl06], "level03_losses": [d["loss"] for d in lvl03], "level015_losses": [d["loss"] for d in lvl015],
                 "final_ncc": 1 - best["loss"], "final_ncc_channels": best["ncc_channels"], "seconds": time.time() - t1,
                 "level015_candidates": [{"loss": d["loss"], "ncc_channels": d["ncc_channels"], "centre": (d["T"] @ np.r_[c_o, 1.0])[:3].tolist()} for d in lvl015]}
# margin: best vs runner-up that is a distinct pose
runner = None
for d in lvl015[1:]:
    diff = transform_diff(T_final, d["T"], c_o)
    if diff["centre_mm"] > 2.0 or diff["rotation_deg"] > 5.0:
        runner = {"loss": d["loss"], **diff}; break
log["refine"]["distinct_runner_up"] = runner

# ------------------------------------------------------------------ 2a. robustness of the structural stage: perturbed restarts at 0.3/0.15 mm
T_struct = T_final.copy()
rng = np.random.default_rng(a.seed + 1)
restarts = []
for i in range(a.n_restarts):
    ang = np.deg2rad(rng.uniform(5, 30)); axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    from octreg.common import rotvec_to_matrix
    Rp = rotvec_to_matrix(to_t(axis * ang)).cpu().numpy()
    tp = rng.uniform(-5, 5, 3)
    Tp = T_struct.copy(); Tp[:3, :3] = Rp @ T_struct[:3, :3]
    Tp[:3, 3] = (T_struct @ np.r_[c_o, 1.0])[:3] + tp - Tp[:3, :3] @ c_o
    Tr, l1 = ref03.refine(Tp, dof="rigid", iters=150, **RK)
    Tr, l1 = ref03.refine(Tr, dof="affine", iters=150, **RK)
    Tr, l2 = ref015.refine(Tr, dof="affine", iters=150, **RK)
    diff = transform_diff(Tr, T_struct, c_o)
    restarts.append({"perturb_deg": float(np.rad2deg(ang)), "perturb_mm": float(np.linalg.norm(tp)), "final_loss": l2, **diff})
succ = [r for r in restarts if r["centre_mm"] < 1.0 and r["rotation_deg"] < 2.0]
say(f"structural restarts converged to the solution: {len(succ)}/{len(restarts)}")
# free the (possibly whole-hemisphere) pyramids and refiners before the vascular stage
del ref06, ref03, ref015, lvl06, lvl03, lvl015, PM, PMr, PO, POr, PTis, PMask, FM, FMr, FO, FOr, mri_tis; torch.cuda.empty_cache()

# ------------------------------------------------------------------ 2b. label-free vascular refinement (sub-mm; depth-axis scale)
R_axes = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
def stretch_ijk(T): return [round(float(np.linalg.norm(T[:3, :3] @ R_axes[:, q])), 3) for q in range(3)]
vasc_mode = a.vascular
if vasc_mode == "auto":
    vasc_mode = "own" if (a.work / "oct24.npy").exists() else ("ves_seg" if (a.work / "oct_ves_dist48_z0.npy").exists() else "off")
if vasc_mode != "off":
    t2 = time.time()
    bc_s = (T_struct @ np.r_[c_o, 1.0])[:3]
    vlo, vhi = world_bbox_to_voxel(A_mri, mri.shape, bc_s - 22, bc_s + 22)
    A_v = A_mri.copy(); A_v[:3, 3] = (A_mri @ np.r_[vlo, 1.0])[:3]
    def _reg(name):
        arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[vlo[0]:vhi[0], vlo[1]:vhi[1], vlo[2]:vhi[2]]).astype(np.float32)
    FM2v = torch.stack([to_t(_reg("wm")), to_t(_reg("gm"))], 0)
    mri_v_reg = np.asarray(mri[vlo[0]:vhi[0], vlo[1]:vhi[1], vlo[2]:vhi[2]]).astype(np.float32); tis_v = _reg("tissue") > 0.5
    mri_v = mri_dark_channel(mri_v_reg, tis_v) if a.vascular_mri == "dark" else mri_frangi_channel(mri_v_reg, tis_v)
    FO2v = build_features("otsu", to_t(oct150)[None], MO, wm_bright=OCT_WM_BRIGHT)
    if vasc_mode == "own" and a.oct_vesdens is not None:
        oct_v = to_t(np.load(a.oct_vesdens).astype(np.float32)) * MO[0]; log["vascular_oct_density"] = str(a.oct_vesdens)
    elif vasc_mode == "own":
        oct24 = np.load(a.work / "oct24.npy").astype(np.float32); m24 = np.load(a.work / "oct24_mask.npy"); A24 = np.load(a.work / "oct24_affine.npy")
        vmask = oct_vessel_mask(oct24, m24, q=a.vascular_q, sigmas=(1.0, 1.5, 2.2, 3.0, 4.4)); del oct24     # 24-106 um dark tubes
        oct_v = density_on_grid(vmask, A24, A_oct, oct150.shape, oct_mask, pool=6); del vmask
    else:
        d48 = np.load(a.work / "oct_ves_dist48_z0.npy"); A48 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
        oct_v = density_on_grid(to_t((d48 <= 0).astype(np.float32)) > 0.5, A48, A_oct, oct150.shape, oct_mask, pool=3)
    torch.cuda.empty_cache()
    T_final, vinfo = vascular_refine(T_struct, FM2v, A_v, FO2v, A_oct, MO, mri_v, oct_v, w=a.vascular_w, reg=a.vascular_reg, clamp=a.vascular_clamp)
    log["vascular"] = {"mode": vasc_mode, "mri_channel": a.vascular_mri, "w": a.vascular_w, "reg": a.vascular_reg, "clamp": a.vascular_clamp,
                       "ncc_channels": vinfo["ncc_channels"], "vs_structural": transform_diff(T_final, T_struct, c_o),
                       "stretch_ijk_structural": stretch_ijk(T_struct), "stretch_ijk_vascular": stretch_ijk(T_final), "seconds": time.time() - t2}
    say(f"vascular refinement ({vasc_mode}, mri {a.vascular_mri}): ncc/channel {np.round(vinfo['ncc_channels'], 3).tolist()}; moved {log['vascular']['vs_structural']['corner_mean_mm']:.2f} mm (corner mean); "
        f"stretch i,j,k {stretch_ijk(T_struct)} -> {stretch_ijk(T_final)}  ({time.time() - t2:.0f}s)")
else:
    log["vascular"] = {"mode": "off"}

# ------------------------------------------------------------------ 3. evaluation (held-out structures)
say("evaluating ...")
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r")
ves_ijk = np.argwhere(np.asarray(mri_ves))
oct_class_parser = np.where(oct_mask, np.where(oct_prob.argmax(0) == 1, 1, np.where(oct_prob.argmax(0) >= 2, 2, 0)), 0)
otsu_f, _ = otsu_two_class(to_t(oct150)[None], to_t(oct_mask)[None].bool(), wm_bright=OCT_WM_BRIGHT)
oct_class_otsu = np.where(oct_mask, np.where(otsu_f[0].cpu().numpy() > 0.5, 1, 2), 0)
def evaluate_T(T):
    e = {}
    for zoff in (0, 66):
        dfile = a.work / f"oct_ves_dist48_z{zoff}.npy"
        if not dfile.exists():
            continue
        d = np.load(dfile); Av = np.load(a.work / f"oct_ves_dist48_z{zoff}_affine.npy")
        e[f"vessels_zoff{zoff}"] = vessel_distance_stats(T, ves_ijk, A_mri, d, Av)
    e["gmwm_parser_classes"] = gm_wm_overlap(T, oct_class_parser, oct_mask, A_oct, np.asarray(labels4), A_mri)
    e["gmwm_otsu_classes"] = gm_wm_overlap(T, oct_class_otsu, oct_mask, A_oct, np.asarray(labels4), A_mri)
    if ref_T is not None:
        e["vs_author_coarse_placement"] = transform_diff(T, ref_T, c_o)
    e["stretch_ijk"] = stretch_ijk(T)
    return e
ev = evaluate_T(T_final)
if vasc_mode != "off":
    ev["structural_stage"] = evaluate_T(T_struct)
    v0 = ev["structural_stage"].get("vessels_zoff0", {}).get("registered", {}); v1 = ev.get("vessels_zoff0", {}).get("registered", {})
    say(f"held-out manual MRI vessels -> OCT vessels: structural median {v0.get('median_um', -1):.0f} um f150 {v0.get('frac_within_150um', -1):.2f} -> vascular median {v1.get('median_um', -1):.0f} um f150 {v1.get('frac_within_150um', -1):.2f}")
ev["restarts"] = {"n": len(restarts), "n_converged_to_solution": len(succ), "detail": restarts}
if vasc_mode != "off":
    # vascular-stage restarts: perturb the structural solution (3-8 deg, <=2.5 mm, +-10% depth scale) and re-run the vascular refinement
    from octreg.refine import params_from_matrix, compose
    r0_, t0_, ls0_, sh0_, mir_ = params_from_matrix(T_struct, c_o); vres = []
    for i in range(min(a.n_restarts, 8)):
        ang = np.deg2rad(rng.uniform(3, 8)); axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        Tp = compose(to_t(r0_ + axis * ang), to_t(t0_ + rng.uniform(-2.5, 2.5, 3)), to_t(ls0_ + np.array([rng.uniform(-0.1, 0.1), 0, 0])), to_t(sh0_), to_t(c_o), mir_).cpu().numpy()
        Tr, _ = vascular_refine(Tp, FM2v, A_v, FO2v, A_oct, MO, mri_v, oct_v, w=a.vascular_w, reg=a.vascular_reg, clamp=a.vascular_clamp, factors=(4, 2, 1))
        vres.append({"perturb_deg": float(np.rad2deg(ang)), **transform_diff(Tr, T_final, c_o)})
    ev["vascular_restarts"] = {"n": len(vres), "n_within_0.3mm": sum(r["corner_mean_mm"] < 0.3 for r in vres), "detail": vres}
    say(f"vascular restarts within 0.3 mm of the solution: {ev['vascular_restarts']['n_within_0.3mm']}/{len(vres)}")


# ------------------------------------------------------------------ 3b. label-free non-rigid stage (small residual deformation), evaluated like the affine
disp = None
if a.nonrigid == "on" and vasc_mode != "off":
    try:
        t3 = time.time()
        FM3 = torch.cat([FM2v, mri_v[None]], 0); FO3 = torch.cat([FO2v, oct_v[None]], 0)
        disp, ninfo = nonrigid_refine(T_final, FM3, A_v, FO3, A_oct, MO, grid=tuple(int(x) for x in a.nonrigid_grid.split(",")), iters=300,
                                      lam_smooth=a.nonrigid_smooth, lam_mag=a.nonrigid_mag, chan_w=(1.0, 1.0, a.vascular_w))
        # evaluation through the non-rigid map: manual MRI vessels -> OCT via the inverse; OCT classes -> MRI labels via the forward map
        d0f = a.work / "oct_ves_dist48_z0.npy"
        nr = {"disp_mean_mm": ninfo["disp_mean_mm"], "disp_p95_mm": ninfo["disp_p95_mm"], "disp_max_mm": ninfo["disp_max_mm"], "ncc_channels": ninfo["ncc_channels"], "seconds": time.time() - t3}
        if d0f.exists():
            d48 = np.load(d0f); A48 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
            with torch.no_grad():
                p_o = inverse_points(T_final, disp, to_t((A_mri @ np.c_[ves_ijk, np.ones(len(ves_ijk))].T).T[:, :3]))
                v = apply_affine(torch.linalg.inv(to_t(A48)), p_o); inside = ((v >= 0) & (v <= to_t(np.array(d48.shape) - 1))).all(1)
                dd = sample_at_world(to_t(d48)[None], to_t(A48), p_o)[0][inside]
                nr["vessels_zoff0"] = {"n_inside": int(inside.sum()), "median_um": float(dd.median()), "frac_within_150um": float((dd <= 150).float().mean()), "frac_within_300um": float((dd <= 300).float().mean())}
        with torch.no_grad():
            idx = np.argwhere(oct_mask); p_o = to_t((A_oct @ np.c_[idx, np.ones(len(idx))].T).T[:, :3])
            pm = apply_affine(to_t(T_final), p_o) + disp(p_o)
            ijk = torch.round(apply_affine(torch.linalg.inv(to_t(A_mri)), pm)).long().cpu().numpy()
            ok = np.all((ijk >= 0) & (ijk < np.array(labels4.shape)), axis=1); lab = np.zeros(len(idx), dtype=np.int64)
            lab[ok] = np.asarray(labels4)[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]; lab_c = np.where(lab == 1, 1, np.where(np.isin(lab, (2, 3)), 2, 0))
            oc = oct_class_otsu[idx[:, 0], idx[:, 1], idx[:, 2]]
            nr["gmwm_otsu_classes"] = {f"dice_{n}": float(2 * ((oc == c) & (lab_c == c)).sum() / max((oc == c).sum() + (lab_c == c).sum(), 1)) for c, n in ((1, "WM"), (2, "GM"))}
        ev["nonrigid"] = nr; np.save(a.out / "nonrigid_disp_grid.npy", disp.state())
        vv = nr.get("vessels_zoff0", {})
        say(f"non-rigid stage: disp mean {nr['disp_mean_mm']:.3f} p95 {nr['disp_p95_mm']:.3f} max {nr['disp_max_mm']:.3f} mm | vessels median {vv.get('median_um', -1):.0f} um f150 {vv.get('frac_within_150um', -1):.2f} | Dice WM {nr['gmwm_otsu_classes']['dice_WM']:.3f} GM {nr['gmwm_otsu_classes']['dice_GM']:.3f} ({time.time()-t3:.0f}s)")
    except Exception as e:                       # never lose the affine result because of the optional stage
        say(f"non-rigid stage failed: {e!r}"); ev["nonrigid"] = {"error": repr(e)}
log["evaluation"] = ev

# ------------------------------------------------------------------ 4. outputs
sc = np.linalg.norm(T_final[:3, :3], axis=0)
log["final_transform"] = {"T_oct2mri_world": T_final.tolist(), "block_centre_mri_mm": (T_final @ np.r_[c_o, 1.0])[:3].tolist(),
                          "column_scales": sc.tolist(), "det": float(np.linalg.det(T_final[:3, :3])), "mirror": bool(np.linalg.det(T_final[:3, :3]) < 0)}
(a.out / "T_oct2mri.lta").write_text(lta_text(T_final, "OCT world (oct150 affine)", "MRI NIfTI world"))
np.save(a.out / "T_oct2mri.npy", T_final); np.save(a.out / "T_oct2mri_structural.npy", T_struct)
# QC figure + resampled volumes
bcq = (T_final @ np.r_[c_o, 1.0])[:3]
qlo, qhi = world_bbox_to_voxel(A_mri, mri.shape, bcq - 16, bcq + 16)
A_q = A_mri.copy(); A_q[:3, 3] = (A_mri @ np.r_[qlo, 1.0])[:3]
qc_figure(a.out / "qc_oct_space.png", T_final, oct150, A_oct, np.asarray(mri[qlo[0]:qhi[0], qlo[1]:qhi[1], qlo[2]:qhi[2]]).astype(np.float32), A_q, oct_mask, oct_class=oct_class_parser,
          mri_labels=np.asarray(labels4[qlo[0]:qhi[0], qlo[1]:qhi[1], qlo[2]:qhi[2]]), title=f"{a.scenario}/{a.features}: NCC={1 - best['loss']:.3f}")
# OCT warped into an MRI sub-grid (0.15 mm, +-20 mm around the block) as NIfTI for Freeview/ITK-SNAP
from octreg.common import resample_to_grid
torch.cuda.empty_cache()
bc = (T_final @ np.r_[c_o, 1.0])[:3]
olo, ohi = world_bbox_to_voxel(A_mri, mri.shape, bc - 20, bc + 20)
A_out = A_mri.copy(); A_out[:3, 3] = (A_mri @ np.r_[olo, 1.0])[:3]
mri_out = np.asarray(mri[olo[0]:ohi[0], olo[1]:ohi[1], olo[2]:ohi[2]]).astype(np.float32)
Tinv = np.linalg.inv(T_final)
oct_in_mri = resample_to_grid(to_t(oct150)[None], to_t(A_oct), to_t(A_out), tuple(mri_out.shape), T_grid_to_vol=to_t(Tinv))[0].cpu().numpy()
save_nifti(oct_in_mri, A_out, a.out / "oct_in_mri_region.nii.gz", dtype=np.float32)
save_nifti(mri_out, A_out, a.out / "mri_region.nii.gz", dtype=np.float32)
save_nifti(np.asarray(labels4[olo[0]:ohi[0], olo[1]:ohi[1], olo[2]:ohi[2]]), A_out, a.out / "mri_labels_region.nii.gz", dtype=np.uint8)
log["total_seconds"] = time.time() - t_start
write_json(log, a.out / "result.json")
say(json.dumps({"final_ncc": round(1 - best["loss"], 4), "mirror": bool(np.linalg.det(T_final[:3, :3]) < 0), "centre": np.round((T_final @ np.r_[c_o, 1.0])[:3], 1).tolist(), "scales": np.round(sc, 3).tolist(), "search_top1_top2": [sinfo["top1"], sinfo["top2"]],
                "vessels": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v["registered"].items()} for k, v in ev.items() if k.startswith("vessels")},
                "gmwm_parser": {k: round(v, 3) for k, v in ev["gmwm_parser_classes"].items() if isinstance(v, float)},
                "gmwm_otsu": {k: round(v, 3) for k, v in ev["gmwm_otsu_classes"].items() if isinstance(v, float)},
                "vs_author": ev.get("vs_author_coarse_placement"), "restarts_converged": f"{len(succ)}/{len(restarts)}", "stretch_ijk": ev["stretch_ijk"],
                "vascular": {k: v for k, v in log["vascular"].items() if k in ("mode", "stretch_ijk_structural", "stretch_ijk_vascular")} | ({"moved_corner_mm": round(log["vascular"]["vs_structural"]["corner_mean_mm"], 2)} if vasc_mode != "off" else {}),
                "total_s": round(time.time() - t_start)}, indent=1))
