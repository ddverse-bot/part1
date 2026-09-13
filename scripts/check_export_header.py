#!/usr/bin/env python3
"""Independent check of an export_registration.py export against the ORIGINAL OCT NIfTI and its header affine.

    python scripts/check_export_header.py --export RUN/export --oct-nifti OCT.nii.gz --mri-nifti MRI.nii.gz [--n 40000]

export_registration.py's own check compares its OCT-in-MRI image with register.py's, and the OCT header affine cancels out of that
comparison.  Here the header is used directly: random MRI voxels inside the exported OCT support are mapped through
inv(T_octnii_to_mrinii) and inv(A_header) to raw OCT voxel indices, the raw 20 um mu_s values there, averaged over a box^3 neighbourhood (default 7 = 0.14 mm, matching the 0.15 mm export), are read from the .nii.gz by
streaming it plane by plane along the last axis (NIfTI Fortran order; about 1 GB resident, no full copy), and compared (Spearman) with
the exported oct_in_mri.nii.gz values at the same MRI voxels.  Controls: the same points through the transform composed with a flip of
each OCT header axis (a wrong frame must decorrelate) and through a 2 mm shift.  Writes check_header.json into the export dir."""
from __future__ import annotations
import argparse, gzip, json, time
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("--export", type=Path, required=True); ap.add_argument("--oct-nifti", type=Path, required=True)
ap.add_argument("--mri-nifti", type=Path, required=True); ap.add_argument("--n", type=int, default=40000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--box", type=int, default=7, help="odd; raw values are averaged over box^3 voxels (7 at 20 um = 0.14 mm, the scale of the exported 0.15 mm image)")
a = ap.parse_args(); t0 = time.time()
def say(*x): print(*x, f"[{time.time() - t0:.0f}s]", flush=True)
T = np.load(a.export / "T_octnii_to_mrinii.npy"); oim = nib.load(str(a.export / "oct_in_mri.nii.gz")); O = np.asarray(oim.dataobj, np.float32)
mimg = nib.load(str(a.mri_nifti)); A_m = np.asarray(mimg.affine); assert np.allclose(A_m, oim.affine, atol=1e-6)
oimg = nib.load(str(a.oct_nifti)); A_o = np.asarray(oimg.affine); shp = np.array(oimg.shape[:3]); hdr = oimg.header
rng = np.random.default_rng(a.seed); cand = np.argwhere(O > 0); pick = cand[rng.choice(len(cand), size=min(a.n, len(cand)), replace=False)]
vals_export = O[pick[:, 0], pick[:, 1], pick[:, 2]]
P_m = (A_m @ np.c_[pick, np.ones(len(pick))].T).T                          # MRI world
def to_oct_vox(Tw):
    return np.rint((np.linalg.inv(A_o) @ np.linalg.inv(Tw) @ P_m.T).T[:, :3]).astype(np.int64)
c = (shp - 1) / 2.0
variants = {"export": T}
for k in range(3):                                                         # flip OCT header voxel axis k about the block centre
    F = np.eye(4); F[k, k] = -1; F[k, 3] = 2 * c[k]; variants[f"flip_axis{k}"] = T @ A_o @ F @ np.linalg.inv(A_o)
S = np.eye(4); S[:3, 3] = [2.0, 0.0, 0.0]; variants["shift_2mm_x"] = S @ T
idx = {k: to_oct_vox(v) for k, v in variants.items()}
h = a.box // 2
need = {}                                                                  # centre plane index (last axis) -> list of (variant, row)
for k, v in idx.items():
    ok = np.all((v >= h) & (v < shp - h), axis=1); idx[k] = (v, ok)
    for r in np.nonzero(ok)[0]: need.setdefault(int(v[r, 2]), []).append((k, int(r)))
out = {k: np.full(len(pick), np.nan, np.float32) for k in variants}
# nibabel resets the header vox_offset to 0 after loading; the array proxy keeps the real data offset
dt = hdr.get_data_dtype(); off = int(oimg.dataobj.offset); slope, inter = hdr.get_slope_inter(); slope = 1.0 if slope is None or not np.isfinite(slope) else slope; inter = 0.0 if inter is None or not np.isfinite(inter) else inter
plane = int(shp[0] * shp[1]) * dt.itemsize
say(f"{len(pick)} points, centre planes needed {len(need)} of {shp[2]}, dtype {dt}, offset {off}, box {a.box}")
from collections import deque
buf = deque(maxlen=a.box)
with (gzip.open(a.oct_nifti, "rb") if str(a.oct_nifti).endswith(".gz") else open(a.oct_nifti, "rb")) as f:
    f.read(off)
    for z in range(int(shp[2])):
        buf.append(np.frombuffer(f.read(plane), dtype=dt).reshape((shp[0], shp[1]), order="F"))
        zc = z - h                                                         # the buffer now holds planes zc-h .. zc+h
        if len(buf) == a.box and zc in need:
            for k, r in need[zc]:
                v = idx[k][0][r]; out[k][r] = float(np.mean([p_[v[0] - h:v[0] + h + 1, v[1] - h:v[1] + h + 1].mean() for p_ in buf])) * slope + inter
        if z % 400 == 0: say("plane", z)
res = {}
for k in variants:
    m = np.isfinite(out[k]) & (out[k] > 0)
    rho = float(spearmanr(out[k][m], vals_export[m]).correlation) if m.sum() > 100 else None
    res[k] = {"n": int(m.sum()), "spearman_raw_vs_export": rho}
    say(k, res[k])
verdict = bool(res["export"]["spearman_raw_vs_export"] is not None and all(res["export"]["spearman_raw_vs_export"] > (res[k]["spearman_raw_vs_export"] or 0) + 0.2 for k in variants if k.startswith("flip")))
json.dump({"results": res, "box": a.box, "header_composition_ok": verdict, "seconds": time.time() - t0}, open(a.export / "check_header.json", "w"), indent=1)
say("header composition ok:", verdict)
