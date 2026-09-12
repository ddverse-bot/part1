#!/usr/bin/env python3
"""Cache OCT vessel distance maps (48 um grid, um units) for both plausible z-alignments of ves_seg
(561 slices) inside the OCT stack (627 slices), plus the coarse author-affine reference transform in
OCT-world -> MRI-world form (only used as a sanity reference, never by the registration)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser(); ap.add_argument("--dandi", type=Path, required=True); ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--author-vox2world", type=Path, default=None, help="json with key oct_zyx_voxel_to_mri_nifti_world_mm")
a = ap.parse_args()
A12 = np.load(a.work / "oct12_affine.npy")
seg = tifffile.imread(str(a.dandi / "derivatives/sub-I46/ses-OCT/vessel/ves_seg.tif")) > 0
Z = 627
for z0 in (0, Z - seg.shape[0]):
    full = np.zeros((Z, seg.shape[1], seg.shape[2]), dtype=bool)
    full[z0:z0 + seg.shape[0]] = seg
    k = 4
    z, y, x = (s // k * k for s in full.shape)
    pooled = full[:z, :y, :x].reshape(z // k, k, y // k, k, x // k, k).max(axis=(1, 3, 5))
    dist = ndimage.distance_transform_edt(~pooled, sampling=(12.0 * k,) * 3).astype(np.float32)   # um
    A48 = A12.copy(); A48[:3, :3] *= k; A48[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * (k - 1) / 2.0)
    np.save(a.work / f"oct_ves_dist48_z{z0}.npy", dist); np.save(a.work / f"oct_ves_dist48_z{z0}_affine.npy", A48)
    print("saved dist for z-offset", z0, dist.shape, "vessel frac", pooled.mean().round(5))
if a.author_vox2world:
    M = np.array(json.load(open(a.author_vox2world))["oct_zyx_voxel_to_mri_nifti_world_mm"])
    T = M @ np.linalg.inv(A12)
    json.dump(T.tolist(), open(a.work / "ref_author_T.json", "w"))
    print("saved ref_author_T.json (OCT world -> MRI world, coarse)")
