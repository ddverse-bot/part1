#!/usr/bin/env python3
"""Ablations of octreg 1.0 on Xiangrui's I58 brainstem pair (docs/design/spec_v1.0_lean.md: claims I1-I3 and the destripe).

    python bench/ablate.py --out /root/autodl-tmp/bench_runs/xiangrui_I58/ablate [--main RUN] [--only A1,A3] [--device cuda]

The OCT is streamed once, each OCT foreground source (texture, intensity, the stored v1.1 mask) is computed once, and each
preprocessing key (source x destripe on/off) is written once; the MRI is prepared once. Variants that change only the two-class
maps, the search or the ladder reuse a key. The driver runs the register steps with the package's own functions: fine grid ->
register.fine_mask -> destripe -> finest level (levels[-1]); register.pyramids; search at the coarsest level; ladder. 'base' is
the default Params through this driver; its distance to the CLI run (--main) is reported as the driver check. Metrics from
bench/evaluate.py: pose to base and to R5, boundary agreement with the base masks for every variant (so it reflects the pose
only), OCT mask volume and Dice against the v1.1 mask per key.

OUT/prep/<key>/  oct_h.npz, oct_mask / oct_valid / mri_mask .nii.gz (evaluate.py --masks), prep.json; prep/v11_mask.nii.gz is
                 A0b's mask in the OCT header frame (the same variant through the CLI: --oct-mask prep/v11_mask.nii.gz)
OUT/variants/<name>/  T_oct2mri.txt, result.json (reused unless --force)
OUT/ablations.json    the table read by bench/report.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E                                                    # noqa: E402  (bench/evaluate.py)
from octreg import geometry as G, io, preprocess as pp                  # noqa: E402
from octreg.params import Params                                        # noqa: E402
from octreg.refine import refine                                        # noqa: E402
from octreg.register import fine_mask, pyramids                         # noqa: E402
from octreg.search import search                                        # noqa: E402

VARIANTS = {    # name: (what changes, Params changes, OCT mask = the stored v1.1 mask)
    "base": ("the method", {}, False),
    "A0": ("OCT intensity foreground (histogram valley) instead of the texture specimen mask", {"oct_foreground": "intensity"}, False),
    "A0b": ("v1.1 rim-watershed specimen mask given as the OCT mask", {}, True),
    "A1": ("MRI flattening off", {"mri_flatten": False}, False),
    "A2": ("OCT flattening off", {"oct_flatten": False}, False),
    "A3": ("section-stripe flat field off", {"destripe": False}, False),
    "A4": ("intensity channels instead of two-class maps", {"features": "intensity"}, False),
    "A5+1": ("polarity forced +1", {"polarity": "+1"}, False),
    "A5-1": ("polarity forced -1", {"polarity": "-1"}, False),
    "A6": ("no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15)", {"lam": 0.0, "clamp": 1.0}, False),
    "A7": ("affine directly at the finest level from every search pose, no ladder", {"ladder": False}, False),
}
BASE = {}       # Params changes shared by every variant and the preprocessing (empty for the benchmark; synthetic smoke tests only)


def params(name):
    return Params.from_dict({**BASE, **VARIANTS[name][1]})


def prep_key(name):
    P = params(name)
    return ("v11mask" if VARIANTS[name][2] else P.oct_foreground) + ("" if P.destripe else "_nodestripe")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)   # bytes | kB


def prepare(out, keys):
    """Write prep/mri and every prep/<key> not on disk yet (the OCT fine grid is streamed at most once)."""
    P, root = Params.from_dict(BASE), out / "prep"
    h, mdir = P.levels[-1], root / "mri"
    for d in [mdir] + [root / k for k in keys]:
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
    todo = [k for k in keys if not (root / k / "oct_h.npz").exists()]
    if not todo:
        return
    t0 = time.time()
    fine, A_f = G.resample_iso(io.load_volume(E.OCT), P.fine_mm)
    raw_h, A_h = G.pool_iso(fine, A_f, P.fine_mm, h)
    valid_h = G.pool_iso((fine > 0).astype(np.float32), A_f, P.fine_mm, h)[0] > 0.5
    grid = {"fine_shape": fine.shape, "seconds": time.time() - t0, "peak_rss_gb": peak_rss_gb()}
    print(f"OCT fine grid {fine.shape} in {grid['seconds']:.0f} s", flush=True)
    masks = {}
    for src in sorted({k.split("_")[0] for k in todo}):
        t1 = time.time()
        if src == "v11mask":
            ref = E.reference(E.OCT, E.V11, E.R5)
            io.save_nifti(ref["mask"].astype(np.uint8), ref["A_mask"], root / "v11_mask.nii.gz")
            m, info = fine_mask(fine, A_f, P, io.load_volume(root / "v11_mask.nii.gz"))    # the CLI --oct-mask path
        else:
            m, info = fine_mask(fine, A_f, dataclasses.replace(P, oct_foreground=src))
        masks[src] = (m, info, time.time() - t1)
        print(f"OCT mask '{src}' in {time.time() - t1:.0f} s", flush=True)
    for k in todo:
        t1, (m, minfo, t_mask) = time.time(), masks[k.split("_")[0]]
        if k.endswith("_nodestripe"):
            arr_h, dinfo = raw_h, {"applied": False, "reason": "off"}
        else:
            f, dinfo = pp.destripe(fine, m, P.fine_mm, P)                      # a new array; fine stays raw
            arr_h = G.pool_iso(f, A_f, P.fine_mm, h)[0]
            del f
        mask_h = G.pool_iso(m, A_f, P.fine_mm, h)[0] > 0.5
        d = root / k
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "oct_h.npz", arr=arr_h, mask=mask_h, valid=valid_h, affine=A_h)
        io.save_nifti(mask_h.astype(np.uint8), A_h, d / "oct_mask.nii.gz")
        io.save_nifti(valid_h.astype(np.uint8), A_h, d / "oct_valid.nii.gz")
        shutil.copy(mdir / "mri_mask.nii.gz", d / "mri_mask.nii.gz")
        io.write_json({"key": k, "grid": grid, "mask": minfo, "mask_seconds": t_mask, "destripe": dinfo,
                       "seconds": time.time() - t1, "peak_rss_gb": peak_rss_gb(), "params_hash": P.hash()}, d / "prep.json")
        print(f"prep {k}: destripe {dinfo.get('reason')} in {time.time() - t1:.0f} s", flush=True)


def solve(P, o, m, device):
    """Register steps 3-5 on prepared finest-level arrays: register.pyramids, search, ladder."""
    oct_pyr, mri_pyr = pyramids(o["arr"], o["mask"], o["affine"], m["arr"], m["mask"], m["affine"], P)
    t0 = time.time()
    cands, sinfo = search(*mri_pyr[P.levels[0]], *oct_pyr[P.levels[0]], P, device)
    t1 = time.time()
    best, finalists = refine(cands, mri_pyr, oct_pyr, P, device, sinfo["tau"])
    return best, finalists, cands, sinfo, {"search_seconds": t1 - t0, "refine_seconds": time.time() - t1}


def run_variant(name, out, device, force):
    d, P, key = out / "variants" / name, params(name), prep_key(name)
    old = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    if not force and "error" not in old and old.get("params_hash") == P.hash():
        return old
    d.mkdir(parents=True, exist_ok=True)
    with np.load(out / "prep" / key / "oct_h.npz") as z, np.load(out / "prep" / "mri" / "mri_h.npz") as y:
        o, m = {k: z[k] for k in z.files}, {k: y[k] for k in y.files}
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        best, finalists, cands, sinfo, times = solve(P, o, m, device)
    except ValueError as e:                     # a variant may break the method (e.g. no admissible pose): recorded, not hidden
        if name == "base":
            raise
        io.write_json({"name": name, "change": VARIANTS[name][0], "prep": key, "params": P.to_dict(), "error": f"{type(e).__name__}: {e}",
                       "seconds": time.time() - t0}, d / "result.json")
        print(f"{name}: failed: {e}", flush=True)
        return json.loads((d / "result.json").read_text())
    keys = ("S", "L", "polarity", "mirror", "log_scales", "shears", "overlap", "search_rank")
    r = {"name": name, "change": VARIANTS[name][0], "prep": key, "params": P.to_dict(), "params_hash": P.hash(), "T": best["T"],
         "best": {k: best[k] for k in keys}, "finalists": [{k: f[k] for k in keys} for f in finalists], "search": sinfo,
         "candidates": [{k: c[k] for k in ("S", "polarity", "mirror", "overlap")} for c in cands], "seconds": time.time() - t0,
         **times, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None, "peak_rss_gb": peak_rss_gb()}
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
    ap.add_argument("--only", default=None, help=f"comma-separated subset of {','.join(VARIANTS)} (base is always run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="recompute variants that already have a result (delete OUT/prep to redo "
                    "the preprocessing)")
    a = ap.parse_args()
    names = ["base"] + [n for n in (a.only.split(",") if a.only else VARIANTS) if n != "base"]
    if set(names) - set(VARIANTS):
        ap.error(f"unknown variants {sorted(set(names) - set(VARIANTS))}")
    if a.main and not (a.main / "T_oct2mri.txt").exists():
        ap.error(f"--main {a.main} has no T_oct2mri.txt")
    t0 = time.time()
    keys = sorted({prep_key(n) for n in names})
    prepare(a.out, keys)
    ref = E.reference(E.OCT, E.V11, E.R5)
    pts, cor = E.grid_points(ref["mask"], ref["A_mask"]), E.corners(ref["shape"], ref["A_hdr"])
    b = a.out / "prep" / prep_key("base")
    bnd = E.Boundary(*E.load_mask(b / "oct_mask.nii.gz"), E.load_mask(b / "oct_valid.nii.gz")[0], *E.load_mask(b / "mri_mask.nii.gz"),
                     ref["A_hdr"])
    masks = {k: E.mask_agreement(*E.load_mask(a.out / "prep" / k / "oct_mask.nii.gz"), ref["mask"], ref["A_mask"]) for k in keys}
    rows = {n: run_variant(n, a.out, a.device, a.force) for n in names}
    T_base = np.array(rows["base"]["T"])
    table = {}
    for n, r in rows.items():
        if "error" in r:
            table[n] = {"change": r["change"], "prep": r["prep"], "params_changed": VARIANTS[n][1], "error": r["error"],
                        "mask": masks[r["prep"]], "seconds": r["seconds"]}
            continue
        T = np.array(r["T"])
        to_base = E.pose(T, T_base, pts, cor)
        table[n] = {"change": r["change"], "prep": r["prep"], "params_changed": VARIANTS[n][1], "S": r["best"]["S"], "L": r["best"]["L"],
                    "polarity": r["best"]["polarity"], "mirror": r["best"]["mirror"], "scales": np.exp(r["best"]["log_scales"]).tolist(),
                    "shears": r["best"]["shears"], "overlap": r["best"]["overlap"],
                    "search": {k: r["search"].get(k) for k in ("top1", "top2", "n_admissible", "tau", "overlap_floor")},
                    "pose_to_base": to_base, "within_0.5mm_of_base": to_base["mean_mm"] <= 0.5,
                    "pose_to_R5": E.pose(T, ref["T_ref"], pts, cor),
                    "boundary": bnd(T), "mask": masks[r["prep"]], "seconds": r["seconds"], "search_seconds": r["search_seconds"],
                    "refine_seconds": r["refine_seconds"], "gpu_peak_gb": r["gpu_peak_gb"], "T_oct2mri": r["T"]}
    res = {"pair": "Xiangrui I58 brainstem", "oct": E.OCT, "mri": E.MRI, "params_hash": Params.from_dict(BASE).hash(),
           "prep": {k: json.loads((a.out / "prep" / k / "prep.json").read_text()) for k in keys + ["mri"]},
           "variants": table, "seconds": time.time() - t0, "peak_rss_gb": peak_rss_gb()}
    if a.main:
        T_main = E.load_T(a.main / "T_oct2mri.txt")
        res["driver_check"] = {"main_run": a.main, "base_vs_main": E.pose(T_base, T_main, pts, cor),
                               "main_to_R5": E.pose(T_main, ref["T_ref"], pts, cor), "main_boundary": bnd(T_main)}
    io.write_json(res, a.out / "ablations.json")
    print(f"wrote {a.out / 'ablations.json'} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
