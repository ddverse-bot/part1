#!/usr/bin/env python3
"""Standalone CPU check of the texture specimen mask (octreg.specimen.oct_specimen_mask) on a prepped --work dir.

    python scripts/test_specimen_mask.py --work work/xiangrui_I58bs --out work/v11/tests/mask [--ref work/probe_xr/mask/final_mask150.npy] [--n-workers 4]

Reads W/octv.npy (memmap), octv_affine.npy, oct150.npy, oct150_affine.npy (+ prep.json for the MRI tissue volume and
oct150_mask.npy for the red intensity contour); writes ONLY into --out: mask150_texture.npy, octv_mask_texture.npy,
mask_qc.png, result.json.  Prints volumes (0.15 / W / octv grids), ratio to the MRI tissue volume, n_cc, seconds, peak RSS
and the Dice vs --ref.  PASS on I58 (spec R0): volume 17.5-18.6 cm3, Dice >= 0.98 vs the P1 mask, 1 CC, < 10 min."""
import argparse, json, os, resource, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.specimen import oct_specimen_mask, mask_qc_png
from scipy.ndimage import label

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--ref", type=Path, default=None, help="reference 0.15 mm bool mask (.npy) for the Dice")
ap.add_argument("--n-workers", type=int, default=4)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)

octv = np.load(a.work / "octv.npy", mmap_mode="r"); A_v = np.load(a.work / "octv_affine.npy")
o150 = np.load(a.work / "oct150.npy"); A150 = np.load(a.work / "oct150_affine.npy")
vox150 = float(np.linalg.norm(A150[:3, :3], axis=0).mean())
say("octv", octv.shape, octv.dtype, "@", np.linalg.norm(A_v[:3, :3], axis=0).round(4).tolist(), "mm; oct150", o150.shape, "@", round(vox150, 4), "mm")
mask150, maskv_path, info = oct_specimen_mask(octv, A_v, o150, A150, a.out, n_workers=a.n_workers)
np.save(a.out / "mask150_texture.npy", mask150)
n_cc = int(label(mask150)[1])
V_mri = None
if (a.work / "prep.json").exists():
    d = json.load(open(a.work / "prep.json")).get("mri", {})
    if all(k in d for k in ("tissue_fraction", "shape", "voxel_mm")): V_mri = float(d["tissue_fraction"] * np.prod(d["shape"]) * np.prod(d["voxel_mm"]) / 1000.0)
mask_int = np.load(a.work / "oct150_mask.npy") if (a.work / "oct150_mask.npy").exists() else None
ru_s, ru_c = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
kb = 1024.0 if sys.platform == "darwin" else 1.0                                     # ru_maxrss: bytes on macOS, KB on Linux
res = {"work": str(a.work), "out": str(a.out), "shape_v": list(octv.shape), "shape150": list(o150.shape),
       "V_cm3_150": info["V_cm3_150"], "V_cm3_W": info["V_cm3_W"], "V_cm3_v": info["V_cm3_v"], "voxels150": info["voxels150"], "frac150": info["frac150"],
       "V_mri_cm3": V_mri, "ratio_to_mri": (info["V_cm3_150"] / V_mri if V_mri else None),
       "V_cm3_intensity_mask": (float(mask_int.sum()) * vox150 ** 3 / 1000.0 if mask_int is not None else None),
       "n_cc_150": n_cc, "n_cc_W_before_largest": info["n_cc_before_largest"], "n_cc_150_before_largest": info["n_cc_150_before_largest"],
       "thr": info["thr"], "gmm": info["gmm"], "params": info["params"],
       "seconds": {k: round(info[k], 1) for k in ("seconds_field", "seconds_rim", "seconds_ws", "seconds_resample", "seconds_total")},
       "peak_rss_gb": {"self": ru_s / kb / 1e6, "child_max": ru_c / kb / 1e6, "upper_bound": (ru_s + a.n_workers * ru_c) / kb / 1e6},
       "octv_mask_texture": str(maskv_path)}
if a.ref is not None:
    ref = np.load(a.ref).astype(bool)
    if ref.shape != mask150.shape: say("WARNING: --ref shape", ref.shape, "!= mask", mask150.shape, "-> Dice skipped"); res["dice_vs_ref"] = None
    else:
        inter = float((ref & mask150).sum()); res["dice_vs_ref"] = 2 * inter / max(1.0, float(ref.sum()) + float(mask150.sum()))
        res["ref_cm3"] = float(ref.sum()) * vox150 ** 3 / 1000.0; res["ref"] = str(a.ref)
        res["mask_minus_ref_voxels"] = int((mask150 & ~ref).sum()); res["ref_minus_mask_voxels"] = int((ref & ~mask150).sum())
mask_qc_png(o150, mask_int, mask150, a.out / "mask_qc.png",
            title=f"{a.work.name}: texture mask {info['V_cm3_150']:.2f} cm3 (frac {info['frac150']:.3f}, {n_cc} CC), MRI tissue {V_mri if V_mri is None else round(V_mri, 2)} cm3"
                  + (f", Dice vs ref {res['dice_vs_ref']:.4f}" if res.get("dice_vs_ref") is not None else ""))
json.dump(res, open(a.out / "result.json", "w"), indent=1)
say(json.dumps({k: v for k, v in res.items() if k not in ("params", "gmm")}, indent=1))
say(f"texture mask: {info['V_cm3_150']:.3f} cm3 at {vox150:.3f} mm ({info['voxels150']} voxels, frac {info['frac150']:.3f}), W {info['V_cm3_W']:.3f} cm3, octv {info['V_cm3_v']:.3f} cm3; "
    f"ratio to MRI tissue {res['ratio_to_mri'] if res['ratio_to_mri'] is None else round(res['ratio_to_mri'], 3)}; n_cc {n_cc} (W before largest {info['n_cc_before_largest']}); "
    f"thr {info['thr']:.4g}; seconds field/rim/ws/resample = {res['seconds']['seconds_field']}/{res['seconds']['seconds_rim']}/{res['seconds']['seconds_ws']}/{res['seconds']['seconds_resample']}, "
    f"total {res['seconds']['seconds_total']}; peak RSS self {res['peak_rss_gb']['self']:.1f} GB, worker max {res['peak_rss_gb']['child_max']:.1f} GB"
    + (f"; Dice vs ref {res['dice_vs_ref']:.4f}" if res.get("dice_vs_ref") is not None else ""))
say("done")
