#!/usr/bin/env python3
"""Standalone CPU check of the texture specimen mask (octreg.specimen.oct_specimen_mask) on a prepped --work dir.

    python scripts/test_specimen_mask.py --work work/v11/xiangrui_I58bs --out work/v11/tests/mask \\
        [--ref work/probe_xr/mask/final_mask150.npy] [--T work/runs/v11/xiangrui_I58bs/T_oct2mri_structural.npy] \\
        [--configs all | default,pockets_off,holes_off,both_off,both_off_rim,both_off_erode2] \\
        [--fill-pockets on|off] [--fill-holes on|off] [--rim-required on|off] [--erode-mm 0] [--mri-clean on|off] [--n-workers 4] \\
        [--core-mm 0] [--eval-mask work/v11/xiangrui_I58bs/oct150_mask.npy,<prep>/oct150_mask_core.npy]

Reads W/octv.npy (memmap), octv_affine.npy, oct150.npy, oct150_affine.npy (+ prep.json for the MRI tissue volume,
oct150_mask.npy for the red contour, and with --T mri_tissue.npy / mri_affine.npy).  Without --configs one mask is built
with the four option flags (the v1.1 single-run check); with --configs the named option sets are built after ONE texture-
field pass (steps 1-3 are cached), each into --out/<config>/: mask150_texture.npy, octv_mask_texture.npy, mask_qc.png
(3 planes through the mask centroid), result.json; plus --out/summary.json and a printed table.  Per configuration:
volumes (0.15 / W / octv grids, the W volume after every stage, the un-eroded 0.15 mm volume the prep's band rule judges),
n_cc, seconds, Dice vs --ref, and with --T the agreement with the MRI tissue mapped into OCT space through T (OCT world ->
MRI world, probe P8's regions): A = mask & MRI, B = mask & ~MRI inside the MRI FOV ('mask-but-not-MRI'), C = MRI & ~mask,
D = mask outside the MRI FOV, Dice_FOV = 2 A / (|mask in FOV| + |MRI tissue in the OCT box|).  P8 baseline for the I58 prep
mask at P_R5: Dice_FOV 0.627, B 7.32 cm3.  --core-mm R > 0 also measures each configuration's CORE = the mask eroded to R
from the un-eroded specimen boundary (as prep_subject.py's oct150_mask_core.npy: a configuration with boundary_erode_mm e
is eroded by max(0, R - e) more), saved as <config>/mask150_core.npy with its own volume / n_cc / Dice / MRI agreement
(result.json 'core', table columns coreV, cDice, cB, cC).  --eval-mask evaluates existing 0.15 mm bool masks on the
oct150 grid (e.g. a prep's oct150_mask.npy and oct150_mask_core.npy) with the same metrics, as extra rows 'eval:<dir>/<stem>'
(summary.json 'eval'); no texture pass is spent on them.
PASS on I58 (spec R0, default configuration on the un-destriped octv): volume 17.5-18.6 cm3, Dice >= 0.98 vs the P1 mask,
1 CC, < 10 min.  NOTE: work/v11's octv.npy is DESTRIPED while the prep computed the texture mask on the raw octv, so the
default rebuilt there is ~19.5 cm3 with Dice ~0.94 vs the prep mask (P8 caveat 2); compare configurations to each other."""
import argparse, json, os, resource, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.specimen import oct_specimen_mask, mask_qc_png, ball, erode_mm

CONFIGS = {"default": {}, "pockets_off": {"fill_pockets": False}, "holes_off": {"fill_holes": False}, "both_off": {"fill_pockets": False, "fill_holes": False},
           "both_off_rim": {"fill_pockets": False, "fill_holes": False, "rim_required": True}, "both_off_erode2": {"fill_pockets": False, "fill_holes": False, "boundary_erode_mm": 2.0},
           "rim": {"rim_required": True}, "erode2": {"boundary_erode_mm": 2.0}}
SWEEP = ["default", "pockets_off", "holes_off", "both_off", "both_off_rim"]

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--ref", type=Path, default=None, help="reference 0.15 mm bool mask (.npy) for the Dice")
ap.add_argument("--T", type=Path, default=None, help="4x4 .npy OCT world -> MRI world; maps W/mri_tissue.npy onto the 0.15 mm OCT grid for the MRI-agreement numbers")
ap.add_argument("--mri-tissue", type=Path, default=None, help="MRI tissue mask (.npy, MRI grid; default W/mri_tissue.npy)")
ap.add_argument("--mri-clean", choices=["on", "off"], default="on", help="clean the MRI tissue as qc_fine (closing 2 vox + fill holes + components >= 1 mm3) before mapping")
ap.add_argument("--configs", type=str, default=None, help="comma-separated configuration names or 'all' (= " + ",".join(SWEEP) + ")")
ap.add_argument("--fill-pockets", choices=["on", "off"], default="on"); ap.add_argument("--fill-holes", choices=["on", "off"], default="on")
ap.add_argument("--rim-required", choices=["on", "off"], default="off"); ap.add_argument("--erode-mm", type=float, default=0.0)
ap.add_argument("--n-workers", type=int, default=4)
ap.add_argument("--core-mm", type=float, default=0.0, help="also evaluate each configuration's core: the mask eroded to this distance (mm) from the un-eroded specimen boundary (0 = off)")
ap.add_argument("--eval-mask", type=str, default=None, help="comma-separated 0.15 mm bool masks (.npy on the oct150 grid) to evaluate with the same metrics as extra rows (no texture pass)")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)

octv = np.load(a.work / "octv.npy", mmap_mode="r"); A_v = np.load(a.work / "octv_affine.npy")
o150 = np.load(a.work / "oct150.npy"); A150 = np.load(a.work / "oct150_affine.npy")
vox150 = float(np.linalg.norm(A150[:3, :3], axis=0).mean()); cm3 = vox150 ** 3 / 1000.0
say("octv", octv.shape, octv.dtype, "@", np.linalg.norm(A_v[:3, :3], axis=0).round(4).tolist(), "mm; oct150", o150.shape, "@", round(vox150, 4), "mm")
V_mri = None
if (a.work / "prep.json").exists():
    d = json.load(open(a.work / "prep.json")).get("mri", {})
    if all(k in d for k in ("tissue_fraction", "shape", "voxel_mm")): V_mri = float(d["tissue_fraction"] * np.prod(d["shape"]) * np.prod(d["voxel_mm"]) / 1000.0)
mask_int = np.load(a.work / "oct150_mask.npy") if (a.work / "oct150_mask.npy").exists() else None
ref = None
if a.ref is not None:
    ref = np.load(a.ref).astype(bool)
    if ref.shape != o150.shape: say("WARNING: --ref shape", ref.shape, "!= oct150", o150.shape, "-> Dice vs ref skipped"); ref = None

# ------------------------------------------------------------------ MRI tissue seen through T on the 0.15 mm OCT grid (P8 convention)
Mm = inMRI = None; mri_meta = None
if a.T is not None:
    import torch; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from octreg.common import resample_to_grid, to_t
    dev = torch.device("cpu"); T = np.load(a.T); A_mri = np.load(a.work / "mri_affine.npy")
    tis = np.load(a.mri_tissue if a.mri_tissue is not None else a.work / "mri_tissue.npy").astype(bool); vox_m = np.linalg.norm(A_mri[:3, :3], axis=0); cm3_m = float(np.prod(vox_m)) / 1000.0
    V_raw = float(tis.sum()) * cm3_m
    if a.mri_clean == "on":       # qc_fine.clean_mask: closing 2 vox + fill holes + drop components < 1 mm3
        c = ndimage.binary_fill_holes(ndimage.binary_closing(tis, structure=ball(2))); lab, n = ndimage.label(c); sz = np.bincount(lab.ravel())
        keep = np.where(sz * cm3_m >= 1e-3)[0]; tis = np.isin(lab, keep[keep > 0]) if n > 1 else c; del lab, c
    Mm = (resample_to_grid(to_t(tis.astype(np.float32), device=dev)[None], to_t(A_mri, device=dev), to_t(A150, device=dev), tuple(o150.shape), T_grid_to_vol=to_t(T, device=dev))[0].numpy() > 0.5)
    M = np.linalg.inv(A_mri) @ T @ A150; sh = np.asarray(tis.shape) - 1                       # OCT voxel -> MRI voxel; inMRI = the OCT voxel centre falls inside the MRI array
    zz = np.arange(o150.shape[0], dtype=np.float64)[:, None, None]; yy = np.arange(o150.shape[1], dtype=np.float64)[None, :, None]; xx = np.arange(o150.shape[2], dtype=np.float64)[None, None, :]
    inMRI = np.ones(o150.shape, bool)
    for r in range(3):
        q = M[r, 0] * zz + M[r, 1] * yy + M[r, 2] * xx + M[r, 3]; inMRI &= (q >= 0) & (q <= sh[r]); del q
    mri_meta = {"T": str(a.T), "mri_tissue": str(a.mri_tissue or a.work / "mri_tissue.npy"), "mri_clean": a.mri_clean, "V_mri_raw_cm3": V_raw, "V_mri_used_cm3": float(tis.sum()) * cm3_m,
                "MRI_tissue_in_OCT_box_cm3": float(Mm.sum()) * cm3, "OCT_box_in_MRI_FOV_frac": float(inMRI.mean())}
    say(f"MRI tissue through T: raw {V_raw:.3f} cm3, used {mri_meta['V_mri_used_cm3']:.3f} ({a.mri_clean}), in the OCT box {mri_meta['MRI_tissue_in_OCT_box_cm3']:.3f} cm3; OCT box inside the MRI FOV {inMRI.mean():.3f}"); del tis

def mri_agreement(m):
    """P8 regions of a 0.15 mm mask vs the mapped MRI tissue -> dict (cm3) or None."""
    if Mm is None: return None
    A_ = float((m & Mm).sum()); B_ = float((m & ~Mm & inMRI).sum()); C_ = float((~m & Mm).sum()); D_ = float((m & ~inMRI).sum()); n_fov = float((m & inMRI).sum()); n_mm = float(Mm.sum())
    return {"A_cm3": A_ * cm3, "B_cm3": B_ * cm3, "C_cm3": C_ * cm3, "D_cm3": D_ * cm3, "mask_not_MRI_total_cm3": (B_ + D_) * cm3, "mask_in_FOV_cm3": n_fov * cm3,
            "dice_FOV": 2 * A_ / max(1.0, n_fov + n_mm), "dice_all": 2 * A_ / max(1.0, float(m.sum()) + n_mm), "frac_mask_in_FOV_agreeing": A_ / max(1.0, n_fov)}

def metrics(m):
    """Volume, components, Dice vs --ref and the MRI agreement of any 0.15 mm bool mask (configuration mask, its core, or an --eval-mask file)."""
    r = {"V_cm3": float(m.sum()) * cm3, "n_cc": int(ndimage.label(m)[1])}
    if ref is not None:
        inter = float((ref & m).sum())
        r.update(dice_vs_ref=2 * inter / max(1.0, float(ref.sum()) + float(m.sum())), ref_cm3=float(ref.sum()) * cm3, mask_minus_ref_cm3=float((m & ~ref).sum()) * cm3, ref_minus_mask_cm3=float((ref & ~m).sum()) * cm3)
    r["mri"] = mri_agreement(m); return r

def fig3(m, path, title):
    """3 planes through the mask centroid: oct150 gray; B (mask & ~MRI, in FOV) red, C (MRI & ~mask) blue; contours mask lime, --ref orange, MRI tissue cyan, MRI FOV white dashed."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pos = o150[o150 > 0]; vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    cen = np.rint(ndimage.center_of_mass(m)).astype(int) if m.any() else np.asarray(m.shape) // 2
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    for r in range(3):
        i = int(np.clip(cen[r], 0, m.shape[r] - 1)); sl = lambda x: np.take(np.asarray(x), i, axis=r)
        ax[r].imshow(sl(o150), cmap="gray", vmin=0, vmax=vmax)
        if Mm is not None:
            rgba = np.zeros(sl(m).shape + (4,), np.float32); B_ = sl(m) & ~sl(Mm) & sl(inMRI); C_ = ~sl(m) & sl(Mm)
            rgba[B_] = (1, 0, 0, 0.35); rgba[C_] = (0.2, 0.4, 1, 0.35); ax[r].imshow(rgba)
        for arr, col, ls in ((m, "lime", "-"), (ref, "orange", "-"), (Mm, "cyan", "-"), (inMRI, "white", "--")):
            if arr is not None and 0 < sl(arr).sum() < sl(arr).size: ax[r].contour(sl(arr).astype(float), levels=[0.5], colors=col, linewidths=0.8, linestyles=ls)
        ax[r].set_title(f"axis {r} idx {i}", fontsize=9); ax[r].axis("off")
    fig.suptitle(title + "\nlime = mask, orange = --ref, cyan = MRI tissue via T, white dashed = MRI FOV; red = mask & ~MRI (B), blue = MRI & ~mask (C)", fontsize=9)
    plt.tight_layout(); plt.savefig(str(path), dpi=80); plt.close(fig)

# ------------------------------------------------------------------ configurations
if a.configs is None:
    names = ["custom"]; CONFIGS["custom"] = {"fill_pockets": a.fill_pockets == "on", "fill_holes": a.fill_holes == "on", "rim_required": a.rim_required == "on", "boundary_erode_mm": a.erode_mm}
else:
    names = SWEEP if a.configs.strip() == "all" else [n.strip() for n in a.configs.split(",") if n.strip()]
    bad = [n for n in names if n not in CONFIGS]
    if bad: raise SystemExit(f"unknown configuration(s) {bad}; known: {sorted(CONFIGS)}")
    if a.erode_mm > 0: names = list(names); CONFIGS.update({n: {**CONFIGS[n], "boundary_erode_mm": a.erode_mm} for n in names})   # a global extra erosion applies to every named configuration
cache = {}; summary = {"work": str(a.work), "out": str(a.out), "shape_v": list(octv.shape), "shape150": list(o150.shape), "V_mri_cm3": V_mri, "ref": (str(a.ref) if ref is not None else None),
                       "V_cm3_intensity_mask": (float(mask_int.sum()) * cm3 if mask_int is not None else None), "mri": mri_meta, "configs": {}}
for name in names:
    kw = CONFIGS[name]; out = a.out / name if a.configs is not None else a.out; out.mkdir(parents=True, exist_ok=True)
    say(f"=== {name}: {kw}")
    mask150, maskv_path, info = oct_specimen_mask(octv, A_v, o150, A150, out, n_workers=a.n_workers, cache=cache, **kw)
    np.save(out / "mask150_texture.npy", mask150); mt = metrics(mask150); n_cc = mt.pop("n_cc")
    res = {"config": name, "options": info["options"], "V_cm3_150": info["V_cm3_150"], "V_cm3_W": info["V_cm3_W"], "V_cm3_W_eroded": info["V_cm3_W_eroded"], "V_cm3_150_uneroded": info["V_cm3_150_uneroded"],
           "V_cm3_v": info["V_cm3_v"], "voxels150": info["voxels150"], "frac150": info["frac150"], "ratio_to_mri": (info["V_cm3_150"] / V_mri if V_mri else None),
           "ratio_to_mri_uneroded": (info["V_cm3_150_uneroded"] / V_mri if V_mri else None), "volumes_cm3": info["volumes_cm3"], "rim_gate": info["rim_gate"],
           "n_cc_150": n_cc, "n_cc_W_before_largest": info["n_cc_before_largest"], "n_cc_150_before_largest": info["n_cc_150_before_largest"], "thr": info["thr"], "gmm": info["gmm"], "params": info["params"],
           "seconds": {k: round(info[k], 1) for k in ("seconds_field", "seconds_rim", "seconds_ws", "seconds_resample", "seconds_total")}, "octv_mask_texture": str(maskv_path), **{k: v for k, v in mt.items() if k != "V_cm3"}}
    res["core"] = None
    if a.core_mm > 0:     # core as prep_subject.py writes it: erode_mm(mask, core - already eroded on W) -> core_mm from the un-eroded boundary
        extra = max(0.0, a.core_mm - float(kw.get("boundary_erode_mm", 0.0))); core = erode_mm(mask150, extra, vox150); np.save(out / "mask150_core.npy", core)
        res["core"] = {"erode_total_mm": a.core_mm, "extra_mm": extra, **metrics(core), "ratio_to_mri": (float(core.sum()) * cm3 / V_mri if V_mri else None), "file": str(out / "mask150_core.npy")}; del core
    title = f"{a.work.name} [{name}]: texture mask {info['V_cm3_150']:.2f} cm3 ({n_cc} CC; W basin {info['volumes_cm3']['W_specimen_basin']:.2f} + pockets {info['volumes_cm3']['W_pockets']:.2f} + holes {info['volumes_cm3']['W_holes']:.2f})" \
            + (f", Dice vs ref {res['dice_vs_ref']:.3f}" if res.get("dice_vs_ref") is not None else "") \
            + (f", MRI: Dice_FOV {res['mri']['dice_FOV']:.3f}, A {res['mri']['A_cm3']:.2f} B {res['mri']['B_cm3']:.2f} C {res['mri']['C_cm3']:.2f} D {res['mri']['D_cm3']:.2f} cm3" if res["mri"] else "")
    try:
        if Mm is not None or ref is not None: fig3(mask150, out / "mask_qc.png", title)
        else: mask_qc_png(o150, mask_int, mask150, out / "mask_qc.png", title=title)
    except Exception as e: say(f"WARNING: mask_qc.png not written ({type(e).__name__}: {e})")
    json.dump(res, open(out / "result.json", "w"), indent=1); summary["configs"][name] = {k: v for k, v in res.items() if k not in ("params", "gmm")}
    say(json.dumps({k: v for k, v in res.items() if k not in ("params", "gmm", "rim_gate")}, indent=1))
    if res["rim_gate"]: say("rim_gate:", json.dumps({k: v for k, v in res["rim_gate"].items() if k != "basins"}), "\n  basins:", json.dumps(res["rim_gate"]["basins"][:8]))
# ------------------------------------------------------------------ existing masks (e.g. a prep's oct150_mask.npy / oct150_mask_core.npy): same metrics, no texture pass
summary["eval"] = {}
for p in ([Path(s.strip()) for s in a.eval_mask.split(",") if s.strip()] if a.eval_mask else []):
    m = np.load(p, mmap_mode="r"); nm = f"eval:{p.parent.name}/{p.stem}"
    if m.shape != o150.shape or m.dtype != bool: say(f"WARNING: {p} shape {m.shape} dtype {m.dtype} != oct150 {o150.shape} bool -> skipped"); continue
    m = np.asarray(m); summary["eval"][nm] = {"file": str(p), **metrics(m), "ratio_to_mri": (float(m.sum()) * cm3 / V_mri if V_mri else None)}; say(nm, json.dumps(summary["eval"][nm])); del m
ru_s, ru_c = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
kb = 1024.0 if sys.platform == "darwin" else 1.0                                     # ru_maxrss: bytes on macOS, KB on Linux
summary["peak_rss_gb"] = {"self": ru_s / kb / 1e6, "child_max": ru_c / kb / 1e6, "upper_bound": (ru_s + a.n_workers * ru_c) / kb / 1e6}; summary["seconds_total"] = time.time() - t0
if a.configs is not None: json.dump(summary, open(a.out / "summary.json", "w"), indent=1)          # single-run mode: --out/result.json (above) is the result, as in v1.1
# ------------------------------------------------------------------ table
CORE = a.core_mm > 0
hdr = f"{'config':16s} {'V150':>7s} {'V_W':>7s} {'basin':>7s} {'+pock':>6s} {'+holes':>6s} {'n_cc':>4s} {'Dice_ref':>8s}" + (f" {'Dice_FOV':>8s} {'A':>6s} {'B':>6s} {'C':>6s} {'D':>6s} {'B+D':>6s}" if Mm is not None else "") + f" {'s_ws':>5s}" \
      + ((f" | {'coreV':>6s} {'cDice':>6s}" + (f" {'cB':>6s} {'cC':>6s}" if Mm is not None else "")) if CORE else "")
def dice_s(r): return f"{(r.get('dice_vs_ref') if r.get('dice_vs_ref') is not None else float('nan')):8.4f}"
def mri_s(m): return f" {m['dice_FOV']:8.4f} {m['A_cm3']:6.2f} {m['B_cm3']:6.2f} {m['C_cm3']:6.2f} {m['D_cm3']:6.2f} {m['mask_not_MRI_total_cm3']:6.2f}" if Mm is not None else ""
say("\n" + hdr)
for name in names:
    r = summary["configs"][name]; v = r["volumes_cm3"]
    line = f"{name:16s} {r['V_cm3_150']:7.2f} {r['V_cm3_W']:7.2f} {v['W_specimen_basin']:7.2f} {v['W_pockets']:6.2f} {v['W_holes'] + v['150_holes']:6.2f} {r['n_cc_150']:4d} {dice_s(r)}" + mri_s(r["mri"]) + f" {r['seconds']['seconds_ws']:5.0f}"
    if CORE: c = r["core"]; line += f" | {c['V_cm3']:6.2f} {dice_s(c)[2:]}" + (f" {c['mri']['B_cm3']:6.2f} {c['mri']['C_cm3']:6.2f}" if Mm is not None else "")
    print(line, flush=True)
for name, r in summary["eval"].items():            # existing mask files: no W-grid stages, no timing
    print(f"{name[:16]:16s} {r['V_cm3']:7.2f} {'-':>7s} {'-':>7s} {'-':>6s} {'-':>6s} {r['n_cc']:4d} {dice_s(r)}" + mri_s(r["mri"]) + f" {'-':>5s}   {name}", flush=True)
if Mm is not None: print("P8 baseline (prep mask, cleaned MRI tissue, P_R5): Dice_FOV 0.627, A 8.60, B 7.32, C 2.93, D 2.13 cm3" + ("; core = mask eroded to --core-mm from the un-eroded boundary (cB / cC = its B / C)" if CORE else ""), flush=True)
say(f"peak RSS self {summary['peak_rss_gb']['self']:.1f} GB, worker max {summary['peak_rss_gb']['child_max']:.1f} GB; done")
