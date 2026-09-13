#!/usr/bin/env python3
"""Standalone fine stage on cached prep arrays (CPU or GPU): builds FineContext exactly as register.py does (ref015 from
otsu_two_class_lowmem + the 0.15 mm region crop) and runs fine_stage from a given pose; never writes into --work.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python scripts/test_fine.py --work work/xiangrui_I58bs \
        --T work/runs/xiangrui_I58bs_novasc/T_oct2mri.npy --out work/v11/tests/fine_cpu --levels 0.3 --iters 5 --restarts 0 --verify basic
    PASS (P2 at 0.3 mm): guard frac_neg 0.6-0.8 with mean <= -0.05 (26/37, -0.15); sign-corrected NCC at T_start 0.1295 +- 0.01.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import DEVICE, to_t, world_bbox_to_voxel, write_json
from octreg.features import otsu_two_class, otsu_two_class_lowmem
from octreg.refine import Refiner
from octreg.evaluate import transform_diff
from octreg.fine import FineContext, FineConfig, fine_stage

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--T", type=Path, required=True, help="4x4 .npy start pose (OCT world -> MRI world)"); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--levels", type=str, default="0.3,0.15"); ap.add_argument("--iters", type=int, default=200); ap.add_argument("--rigid-iters", type=int, default=150); ap.add_argument("--restarts", type=int, default=8)
ap.add_argument("--verify", choices=["basic", "full"], default="basic"); ap.add_argument("--init-only", action="store_true", help="guard + loss at T per level, no optimisation")
ap.add_argument("--sim", choices=["auto", "ncc", "lcc", "lcc2"], default="auto"); ap.add_argument("--subsample", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--oct-wm-bright", choices=["auto", "yes", "no", "run"], default="run", help="class polarity for ref015; run = read from result.json next to --T (fallback auto)")
ap.add_argument("--mri-wm-bright", action="store_true"); ap.add_argument("--no-ref015", action="store_true", help="skip the structural gate refiner (faster)")
ap.add_argument("--flatten-mm", type=float, default=3.0); ap.add_argument("--erode-mm", type=float, default=1.3); ap.add_argument("--lcc-mm", type=float, default=4.5)
ap.add_argument("--clamp", type=float, default=0.15); ap.add_argument("--reg", type=float, default=0.5); ap.add_argument("--max-move", type=float, default=3.0, help="mm at the block corners"); ap.add_argument("--max-rot", type=float, default=5.0, help="deg")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t_all = time.time()
def say(*x): print(*x, flush=True)
say(f"device {DEVICE}, torch threads {torch.get_num_threads()}")

# ---- arrays as register.py loads them (whole MRI region)
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); mri_tissue_all = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; lo, hi = np.zeros(3, int), np.array(mri.shape)
VOX_M = float(np.linalg.norm(A_mri[:3, :3], axis=0).mean()); VOX_O = float(np.linalg.norm(A_oct[:3, :3], axis=0).mean())
_corn = np.array([[i, j, k] for i in (0, oct150.shape[0] - 1) for j in (0, oct150.shape[1] - 1) for k in (0, oct150.shape[2] - 1)], float)
BLOCK_R = float(np.linalg.norm((A_oct @ np.c_[_corn, np.ones(8)].T).T[:, :3] - c_o, axis=1).max())
T0 = np.load(a.T) if a.T.suffix == ".npy" else np.array(json.load(open(a.T)), float)
say(f"OCT {oct150.shape} @ {VOX_O:.3f} mm, MRI {mri.shape} @ {VOX_M:.3f} mm, block radius {BLOCK_R:.1f} mm; start centre {np.round((T0 @ np.r_[c_o, 1.0])[:3], 2).tolist()}")

# ---- ref015 = the structural gate, rebuilt as register.py: [P(WM),P(GM)] of the MRI (otsu_two_class_lowmem) vs the OCT two-class split,
#      Refiner on the 0.15 mm crop of half-size BLOCK_R + 12 around the pose (MRIFeatures.region pattern)
ref015 = None
if not a.no_ref015:
    t0 = time.time(); mri_reg = np.asarray(mri).astype(np.float32); tis_reg = np.asarray(mri_tissue_all).astype(bool)
    FM_cpu, thr_m = otsu_two_class_lowmem(mri_reg, tis_reg, wm_bright=a.mri_wm_bright, vox_mm=VOX_M); del mri_reg
    MO = to_t(oct_mask)[None].float(); FO_bright_first, thr_o = otsu_two_class(to_t(oct150)[None], MO.bool(), wm_bright=True)
    f = max(1, int(round(0.15 / VOX_M))); ctr = (T0 @ np.r_[c_o, 1.0])[:3]; half = BLOCK_R + 12.0
    lo_, hi_ = world_bbox_to_voxel(A_mri, mri.shape, ctr - half, ctr + half); lo_ = (lo_ // f) * f; hi_ = lo_ + ((hi_ - lo_) // f) * f
    blk = torch.stack([to_t(np.asarray(FM_cpu[c][lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]).astype(np.float32)) for c in range(2)])
    Ar = A_mri.copy(); Ar[:3, 3] = (A_mri @ np.r_[lo_, 1.0])[:3]
    if f > 1: blk = torch.nn.functional.avg_pool3d(blk[None], f, f)[0]; Ar[:3, 3] = Ar[:3, 3] + Ar[:3, :3] @ (np.ones(3) * (f - 1) / 2.0); Ar[:3, :3] *= f
    pol = a.oct_wm_bright
    if pol == "run":
        rj = a.T.parent / "result.json"; pol = {True: "yes", False: "no"}.get(json.load(open(rj)).get("oct_wm_bright")) if rj.exists() else None; pol = pol or "auto"
    refs = {}
    for wm_bright in {"auto": [False, True], "yes": [True], "no": [False]}[pol]:
        FO = FO_bright_first if wm_bright else FO_bright_first[[1, 0]]; r = Refiner(blk, Ar, FO, A_oct, MO); refs[wm_bright] = (r, r.evaluate(T0))
        say(f"ref015 polarity wm_bright={wm_bright}: structural NCC at T {1 - refs[wm_bright][1]:.4f} (channels {np.round(r.ncc_channels(T0), 4).tolist()})")
    wmb = min(refs, key=lambda k: refs[k][1]); ref015 = refs[wmb][0]; del refs, blk, FM_cpu
    say(f"ref015 ready (OCT wm_bright={wmb}, MRI otsu thr {thr_m:.4f}, OCT otsu thr {thr_o:.1f}) in {time.time() - t0:.0f}s")

ctx = FineContext(mri=mri, mri_tissue=mri_tissue_all, A_mri=A_mri, lo=lo, hi=hi, oct150=oct150, oct_mask=oct_mask, A_oct=A_oct, c_o=c_o, BLOCK_R=BLOCK_R, VOX_M=VOX_M, VOX_O=VOX_O, ref015=ref015, device=DEVICE)
cfg = FineConfig(sim=a.sim, levels=tuple(float(x) for x in a.levels.split(",")), flatten_mm=a.flatten_mm, erode_mm=a.erode_mm, lcc_mm=a.lcc_mm, clamp=a.clamp, reg=a.reg, iters=a.iters,
                 n_restarts=a.restarts, max_move=a.max_move, max_rot=a.max_rot, subsample=a.subsample, verify=a.verify, rigid_iters=a.rigid_iters)
rng = np.random.default_rng(a.seed + 1)
t0 = time.time(); T_acc, T_cand, info = fine_stage(T0, ctx, cfg, rng, probe_only=a.init_only); info["test_args"] = {k: str(v) for k, v in vars(a).items()}
write_json(info, a.out / "fine_info.json"); np.save(a.out / "T_fine_candidate.npy", T_cand); np.save(a.out / "T_fine_accepted.npy", T_acc); np.save(a.out / "T_start.npy", T0)

# ---- report
for g in info["guard"]:
    say(f"guard level {g['level']}: blocks {g['n_blocks']}, negative {g['n_neg']} (frac {g['frac_neg']}), vol-weighted mean {g['mean']}, sign {g.get('sign')}, n_vox {g['n_vox']}")
for e in info["levels"]:
    say(f"level {e['level']} sim {e['sim']} sign {e['sign']} m_spec {e['n_spec']} vox = {e.get('spec_cm3', 0):.2f} cm3: "
        + (f"skipped {e['skipped']}" if e["skipped"] else f"NCC (sign-corrected) at level start {e.get('ncc_before'):.4f}" + (f" -> after {e.get('ncc_after'):.4f} (loss {e['loss_before']:.4f} -> {e['loss_after']:.4f}), moved {e['moved_mm']:.3f} mm, {e['seconds']:.0f}s" if "loss_after" in e else "")))
if not a.init_only and info.get("vs_start"):
    vs = info["vs_start"]
    say(f"move vs start: corner mean {vs['corner_mean_mm']:.3f} mm (7 mm cube) / {vs['block_corner_mean_mm']:.3f} mm (block corners, max {vs['block_corner_max_mm']:.3f}), centre {vs['centre_mm']:.3f} mm, "
        f"delta rotation {vs['delta_rotation_deg']:.3f} deg (transform_diff {vs['rotation_deg']:.3f}); centre shift along OCT axes (z,y,x) {np.round(vs['centre_shift_oct_axes_mm'], 3).tolist()} mm; "
        f"stretch {info['stretch_ijk_before']} -> {info['stretch_ijk_after']}, cum log-scale {np.round(info['cum_log_scale'], 4).tolist()}")
    say(f"finest {info['finest_level']} mm {info['finest_sim']}: loss at T_start {info['loss_start_finest']:.4f} -> at T_fine {info['loss_fine_finest']:.4f}; structural NCC {info['struct_ncc_before']} -> {info['struct_ncc_after']}; MI32 {info['mi_before']:.4f} -> {info['mi_after']:.4f}")
    rs = info["restarts"]; say(f"restarts converged {rs['n_converged']}/{rs['n']} (block corners < 0.3 mm; 7 mm cube {rs.get('n_converged_7mm')}), p95 disp of the converged {rs['p95_disp_converged_mm']} mm ({rs.get('seconds', 0):.0f}s)")
    ms = info["multi_sim"]; _p = lambda d: "n/a (degenerate LCC)" if d is None else f"{d['mean_mm']:.3f}/{d['p95_mm']:.3f}"
    say(f"multi-sim disp mean/p95 mm: fine-lcc {_p(ms['fine_lcc'])}, fine-mi {_p(ms['fine_mi'])}, lcc-mi {_p(ms['lcc_mi'])} {ms['lcc_notes'] or ''} ({ms['seconds']:.0f}s)")
    say(f"split-half per axis {info['split_half']['per_axis_mm']} mm, loho_max {info['split_half']['loho_max_mm']} mm")
    for nm in ("fine", "mi"):
        pa = info["landscape"][nm]["per_axis"]
        say(f"landscape {nm}: " + ", ".join(f"{k} argmax {v['argmax_offset']:+.2f}{'' if v['unimodal'] else ' MULTIMODAL'}{' END' if v['at_end'] else ''}" for k, v in pa.items()))
    if info.get("ice"): say(f"inverse consistency: mean {info['ice']['mean_mm']} mm, p95 {info['ice']['p95_mm']} mm {info['ice'].get('notes') or ''}")
    say(f"U_mm {info['U_mm']}  accepted={info['accepted']} {info['reasons']}")
say(f"fine_stage {time.time() - t0:.0f}s, total {time.time() - t_all:.0f}s; wrote {a.out}/fine_info.json")
