#!/usr/bin/env python3
"""Ablations of octreg 1.0 on Xiangrui's I58 brainstem pair.

    python bench/ablate.py --out ABL [--main RUN] [--previous OLD/ablations.json ...] [--only A1,A4] [--device cuda] [--force]

Params holds method constants only, so every variant is the register steps run here with the package's own functions and one
explicit change: another OCT mask (A0 the histogram valley of the OCT, A0b the stored v1.1 mask through the --oct-mask path,
A0c the texture mask with the 3-D hole filling of the first release), two_class(..., flatten=False) for one modality (A1, A2),
standardised intensity channels built in this file (A4),
align(..., polarity=+1 / -1) (A5), Params lam 0 and clamp 1 (A6), the OCT world mirrored so that the search and refinement see
the other handedness (A8), or no outline term (A9). 'base' is the method through this driver; its distance to the CLI run
(--main) is the driver check. The OCT is streamed once, each OCT mask is computed once and the MRI is prepared once.

Removed steps (REMOVED) keep their rows from the ablation run that measured them (--previous, one ablations.json per group):
the section-stripe flat field (A3) and the rigid -> similarity -> affine ladder (A7) from the first run. Each was measured
against the base of its run, which still had the step, and the pose change from that base to the present base is reported
next to them.

Metrics from bench/evaluate.py: pose to base and to R5, boundary agreement with the base masks for every variant (so it reflects
the pose only), OCT mask volume and Dice against the v1.1 mask per mask source.

OUT/prep/<mask>/  oct_h.npz, oct_mask / oct_valid / mri_mask .nii.gz (evaluate.py --masks), prep.json; prep/v11_mask.nii.gz is
                  A0b's mask in the OCT header frame (the same variant through the CLI: --oct-mask prep/v11_mask.nii.gz)
OUT/variants/<name>/  T_oct2mri.txt, result.json (reused unless --force)
OUT/ablations.json    the table read by bench/report.py
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E                                                    # noqa: E402  (bench/evaluate.py)
from octreg import geometry as G, io, preprocess as pp, search as S     # noqa: E402
from octreg.params import Params                                        # noqa: E402
from octreg.register import align, fine_mask                            # noqa: E402

VARIANTS = {    # name: (what changes, the explicit change: mask source, solve() keywords, Params overrides)
    "base": ("the method", {}),
    "A0": ("OCT intensity foreground (histogram valley) instead of the texture specimen mask", {"mask": "intensity"}),
    "A0b": ("v1.1 rim-watershed specimen mask given as the OCT mask", {"mask": "v11mask"}),
    "A0c": ("texture specimen mask with holes filled in 3-D only (method: in every array plane)", {"mask": "texture3d"}),
    "A1": ("MRI flattening off", {"mri_flatten": False}),
    "A2": ("OCT flattening off", {"oct_flatten": False}),
    "A4": ("standardised intensity channels (z, -z) instead of two-class maps", {"features": "intensity"}),
    "A5+1": ("polarity forced +1", {"polarity": 1}),
    "A5-1": ("polarity forced -1", {"polarity": -1}),
    "A6": ("no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15)", {"params": {"lam": 0.0, "clamp": 1.0}}),
    "A8": ("the other handedness: OCT world mirrored (z negated) before the search", {"mirror": True}),
    "A9": ("no outline term: S = 2 S_class / 3 in the search and the refinement", {"outline": False}),
}
MIRROR = np.diag([1.0, 1.0, -1.0, 1.0])
REMOVED = [    # groups of removed steps, each measured in one earlier ablation run
    {"step": {"A3": "section-stripe flat field", "A7": "rigid -> similarity -> affine ladder"},
     "note": "Copied from the first ablation run, whose base still had the section-stripe flat field and the ladder (and the "
             "earlier score, two-class maps under an overlap gate), so pose changes in these rows are against that base. Removing "
             "either step moved the pose by less than the 0.5 mm deletion threshold and both were deleted."},
]


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)   # bytes | kB


def mask_source(name):
    return VARIANTS[name][1].get("mask", "texture")


def prepare(out, sources):
    """Write prep/mri and every prep/<source> not on disk yet (the OCT fine grid is streamed at most once)."""
    P, root = Params(), out / "prep"
    h, mdir = P.base_mm, root / "mri"
    for d in [mdir] + [root / s for s in sources]:
        if (d / "prep.json").exists() and json.loads((d / "prep.json").read_text()).get("params_hash") != P.hash():
            raise ValueError(f"{d} was prepared with other Params: delete {root} to prepare again")
    if not (mdir / "mri_h.npz").exists():
        t0 = time.time()
        arr, A = G.resample_iso(io.load_volume(E.MRI), h)
        mask, info = pp.foreground(arr, h, P)
        mdir.mkdir(parents=True, exist_ok=True)
        np.savez(mdir / "mri_h.npz", arr=arr, mask=mask, affine=A)
        io.save_nifti(mask.astype(np.uint8), A, mdir / "mri_mask.nii.gz")
        io.write_json({"foreground": info, "shape": arr.shape, "seconds": time.time() - t0, "peak_rss_gb": peak_rss_gb(),
                       "params_hash": P.hash()}, mdir / "prep.json")
    todo = [s for s in sources if not (root / s / "oct_h.npz").exists()]
    if not todo:
        return
    t0 = time.time()
    fine, A_f = G.resample_iso(io.load_volume(E.OCT), P.fine_mm)
    arr_h, A_h = G.pool_iso(fine, A_f, P.fine_mm, h)
    valid_h = G.pool_iso(fine > 0, A_f, P.fine_mm, h)[0] > 0.5
    grid = {"fine_shape": fine.shape, "seconds": time.time() - t0, "peak_rss_gb": peak_rss_gb()}
    print(f"OCT fine grid {fine.shape} in {grid['seconds']:.0f} s", flush=True)
    for s in todo:
        t1 = time.time()
        if s == "texture":                                                     # the method
            m, info = fine_mask(fine, A_f, P)
        elif s == "texture3d":                                                 # A0c: the 3-D hole filling of the first release
            fill_planes, pp._fill_planes = pp._fill_planes, ndimage.binary_fill_holes
            try:
                m, info = fine_mask(fine, A_f, P)
            finally:
                pp._fill_planes = fill_planes
        elif s == "v11mask":                                                   # A0b, through the CLI --oct-mask path
            ref = E.reference(E.OCT, E.V11, E.R5)
            io.save_nifti(ref["mask"].astype(np.uint8), ref["A_mask"], root / "v11_mask.nii.gz")
            m, info = fine_mask(fine, A_f, P, io.load_volume(root / "v11_mask.nii.gz"))
        else:                                                                  # A0: histogram valley of the OCT on the base grid
            m_h, info = pp.foreground(arr_h, h, P)
            m = G.resample_to(m_h, A_h, fine.shape, A_f, np.eye(4), order=0)
            np.logical_and(m, fine > 0, out=m)
            info["source"] = "intensity"
        mask_h = G.pool_iso(m, A_f, P.fine_mm, h)[0] > 0.5
        del m
        d = root / s
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "oct_h.npz", arr=arr_h, mask=mask_h, valid=valid_h, affine=A_h)
        io.save_nifti(mask_h.astype(np.uint8), A_h, d / "oct_mask.nii.gz")
        io.save_nifti(valid_h.astype(np.uint8), A_h, d / "oct_valid.nii.gz")
        shutil.copy(mdir / "mri_mask.nii.gz", d / "mri_mask.nii.gz")
        io.write_json({"source": s, "grid": grid, "mask": info, "mask_seconds": time.time() - t1, "peak_rss_gb": peak_rss_gb(),
                       "params_hash": P.hash()}, d / "prep.json")
        print(f"OCT mask '{s}' in {time.time() - t1:.0f} s", flush=True)


def standardised(arr, mask, h, P):
    """A4 map: z = (x - mean) / std over the foreground values clipped at p99.5, x = pp.flattened(arr) blurred with sigma one
    voxel, i.e. pp.two_class without its Otsu threshold and sigmoid. -> float32 [D, H, W]."""
    x = ndimage.gaussian_filter(pp.flattened(arr, mask, h, P), 1.0)
    vals = np.minimum(x[mask], np.percentile(x[mask], 99.5))
    return ((x - float(vals.mean())) / float(vals.std())).astype(np.float32)


def solve(o, m, P, device, mri_flatten=True, oct_flatten=True, features="two_class", polarity=0, mirror=False, outline=True):
    """Register steps 3-5 on prepared base-grid arrays; with the default keywords these are register.register's own calls.
    mirror: search and refine against the OCT world mirrored by MIRROR, poses returned in the OCT header world (det < 0).
    outline False: S = 2 S_class / 3. The search's combined score is patched to leave S_outline out (on the pooled search grid
    the mask edge is fractional, so a zero outline weight alone would not remove it); in the refinement the outline weight is
    the specimen mask itself, on which the mask is constant, so S_outline is 0."""
    h = P.base_mm
    if features == "intensity":
        z_o, z_m = standardised(o["arr"], o["mask"], h, P), standardised(m["arr"], m["mask"], h, P)
        u, w, v = np.stack([z_o, -z_o]), o["mask"].astype(np.float32), np.stack([z_m, -z_m]) * m["mask"]
    else:
        u, w = pp.oct_channels(pp.two_class(o["arr"], o["mask"], h, P, flatten=oct_flatten), o["mask"])
        v = pp.mri_channels(pp.two_class(m["arr"], m["mask"], h, P, flatten=mri_flatten), m["mask"])
    F, q = MIRROR if mirror else np.eye(4), o["valid"] if outline else o["mask"]
    combined = S.combined
    if not outline:
        S.combined = lambda S_class, S_outline, pol: combined(S_class, torch.zeros_like(S_outline), pol)
    try:
        poses, info = align((u, w, q, F @ o["affine"]), (v, m["mask"], m["affine"]), P, device, polarity)
    finally:
        S.combined = combined
    return [{**p, "T": p["T"] @ F} for p in poses], info


def run_variant(name, out, device, force):
    change, spec = VARIANTS[name]
    d, src = out / "variants" / name, mask_source(name)
    P = Params.from_dict(spec.get("params", {}))
    kw = {k: v for k, v in spec.items() if k not in ("mask", "params")}
    old = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    if not force and "error" not in old and old.get("params_hash") == P.hash() and old.get("spec") == spec:
        return old
    d.mkdir(parents=True, exist_ok=True)
    with np.load(out / "prep" / src / "oct_h.npz") as z, np.load(out / "prep" / "mri" / "mri_h.npz") as y:
        o, m = {k: z[k] for k in z.files}, {k: y[k] for k in y.files}
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    head = {"name": name, "change": change, "prep": src, "spec": spec, "params_hash": P.hash()}
    try:
        poses, info = solve(o, m, P, device, **kw)
    except ValueError as e:                     # a variant may break the method: recorded, not hidden
        if name == "base":
            raise
        io.write_json({**head, "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}, d / "result.json")
        print(f"{name}: failed: {e}", flush=True)
        return json.loads((d / "result.json").read_text())
    keys = ("S", "S_class", "S_outline", "L", "polarity", "log_scales", "shears", "search_rank")
    best = poses[0]
    r = {**head, "T": best["T"], "best": {k: best[k] for k in keys}, "poses": [{k: p[k] for k in keys} for p in poses], **info,
         "seconds": time.time() - t0, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None,
         "peak_rss_gb": peak_rss_gb()}
    io.write_transform_txt(best["T"], d / "T_oct2mri.txt")
    io.write_json(r, d / "result.json")
    if cuda:
        torch.cuda.empty_cache()
    print(f"{name}: S {best['S']:.4f} polarity {best['polarity']} in {r['seconds']:.0f} s", flush=True)
    return json.loads((d / "result.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--main", type=Path, default=None, help="CLI run dir (T_oct2mri.txt) for the driver check")
    ap.add_argument("--previous", type=Path, nargs="*", action="extend", default=[],
                    help="ablations.json of the runs that measured the removed steps (repeatable)")
    ap.add_argument("--only", default=None, help=f"comma-separated subset of {','.join(VARIANTS)} (base is always run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="recompute variants that already have a result (delete OUT/prep to redo "
                    "the preprocessing)")
    a = ap.parse_args()
    names = ["base"] + [n for n in (a.only.split(",") if a.only else VARIANTS) if n != "base"]
    if set(names) - set(VARIANTS):
        ap.error(f"unknown variants {sorted(set(names) - set(VARIANTS))}")
    for path, need in ((a.main, "T_oct2mri.txt"), *((p, "") for p in a.previous)):
        if path and not (path / need if need else path).exists():
            ap.error(f"{path} has no {need or 'file'}")
    t0 = time.time()
    sources = sorted({mask_source(n) for n in names})
    prepare(a.out, sources)
    ref = E.reference(E.OCT, E.V11, E.R5)
    pts, cor = E.grid_points(ref["mask"], ref["A_mask"]), E.corners(ref["shape"], ref["A_hdr"])
    b = a.out / "prep" / "texture"
    bnd = E.Boundary(*E.load_mask(b / "oct_mask.nii.gz"), E.load_mask(b / "oct_valid.nii.gz")[0], *E.load_mask(b / "mri_mask.nii.gz"),
                     ref["A_hdr"])
    masks = {s: E.mask_agreement(*E.load_mask(a.out / "prep" / s / "oct_mask.nii.gz"), ref["mask"], ref["A_mask"]) for s in sources}
    rows = {n: run_variant(n, a.out, a.device, a.force) for n in names}
    T_base = np.array(rows["base"]["T"])
    table = {}
    for n, r in rows.items():
        common = {"change": r["change"], "prep": r["prep"], "spec": r["spec"], "mask": masks[r["prep"]], "seconds": r["seconds"]}
        if "error" in r:
            table[n] = {**common, "error": r["error"]}
            continue
        T, best = np.array(r["T"]), r["best"]
        to_base = E.pose(T, T_base, pts, cor)
        table[n] = {**common, "S": best["S"], "S_class": best["S_class"], "S_outline": best["S_outline"], "L": best["L"],
                    "polarity": best["polarity"], "scales": np.exp(best["log_scales"]).tolist(), "shears": best["shears"],
                    "search": {k: r["search"].get(k) for k in ("top1", "top2", "n_orientations")},
                    "refine": r["refine"], "pose_to_base": to_base, "within_0.5mm_of_base": to_base["mean_mm"] <= 0.5,
                    "pose_to_R5": E.pose(T, ref["T_ref"], pts, cor), "boundary": bnd(T), "search_seconds": r["search"]["seconds"],
                    "refine_seconds": r["refine"]["seconds"], "gpu_peak_gb": r["gpu_peak_gb"], "T_oct2mri": r["T"]}
    prep = {s: json.loads((a.out / "prep" / s / "prep.json").read_text()) for s in sources + ["mri"]}
    grid = next((p["grid"] for p in prep.values() if "grid" in p), {})
    work = [grid.get("seconds", 0)] + [p.get("mask_seconds", p.get("seconds", 0)) for p in prep.values()] + [r["seconds"] for r in rows.values()]
    res = {"pair": "Xiangrui I58 brainstem", "oct": E.OCT, "mri": E.MRI, "params_hash": Params().hash(), "prep": prep,
           "variants": table, "seconds": sum(work),          # the computation in the table, also when variants come from the cache
           "peak_rss_gb": max([p.get("peak_rss_gb") or 0 for p in prep.values()] + [r.get("peak_rss_gb") or 0 for r in rows.values()]),
           "driver_seconds": time.time() - t0}
    res["removed_steps"] = []
    for path in a.previous:
        prev = json.loads(path.read_text())
        for group in REMOVED:
            rows = {n: {**prev["variants"][n], "pose_to_present_base": E.pose(np.array(prev["variants"][n]["T_oct2mri"]), T_base, pts,
                                                                              cor)}
                    for n in group["step"] if n in prev["variants"] and "error" not in prev["variants"][n]}
            if rows:
                T_prev = np.array(prev["variants"]["base"]["T_oct2mri"])
                res["removed_steps"].append({**group, "source": path, "rows": rows,
                                             "present_base_vs_previous_base": E.pose(T_base, T_prev, pts, cor)})
    if a.main:
        T_main = E.load_T(a.main / "T_oct2mri.txt")
        res["driver_check"] = {"main_run": a.main, "base_vs_main": E.pose(T_base, T_main, pts, cor),
                               "main_to_R5": E.pose(T_main, ref["T_ref"], pts, cor), "main_boundary": bnd(T_main)}
    io.write_json(res, a.out / "ablations.json")
    print(f"wrote {a.out / 'ablations.json'} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
