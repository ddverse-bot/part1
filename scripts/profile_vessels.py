#!/usr/bin/env python3
"""Residual-error probe: shift the final transform along each OCT axis (± few mm) and record the held-out
MRI-vessel -> OCT-vessel median distance.  A minimum away from 0 reveals a residual translation error along that axis."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.evaluate import vessel_distance_stats
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--T", type=Path, required=True); ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
T = np.load(a.T); A_mri = np.load(a.work / "mri_affine.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(ves))
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
res = {}
for ax, name in enumerate(["z(depth)", "y", "x"]):
    dirv = A_oct[:3, ax] / np.linalg.norm(A_oct[:3, ax])          # OCT axis direction in world
    prof = []
    for dz in np.arange(-2.4, 2.41, 0.3):
        Ts = T.copy(); Ts[:3, 3] = T[:3, 3] + T[:3, :3] @ (dirv * dz)   # shift block in its own frame by dz mm along axis
        v = vessel_distance_stats(Ts, ves_ijk, A_mri, d0, A0, n_ctrl=0)["registered"]
        prof.append((round(float(dz), 2), round(v.get("median_um", -1), 0), round(v.get("frac_within_150um", -1), 2), v.get("n_inside")))
    res[name] = prof
    print(name, "  dz(mm): median_um / f150 / n_in")
    print("   ", [(p[0], p[1], p[2]) for p in prof])
if a.out: json.dump(res, open(a.out, "w"), indent=1)
