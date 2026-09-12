#!/usr/bin/env python3
"""Large-vessel subset of the OCT vessel segmentation (radius >= r_min) and its distance map, so the vessel
metric compares MRI-visible vessels (>150 um) with OCT vessels of similar calibre only."""
import argparse, sys
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument("--dandi", type=Path, required=True); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--r-min-um", type=float, default=60.0)
a = ap.parse_args()
seg = tifffile.imread(str(a.dandi / "derivatives/sub-I46/ses-OCT/vessel/ves_seg.tif")) > 0
A12 = np.load(a.work / "oct12_affine.npy")
k = 2   # 24 um working grid
z, y, x = (s // k * k for s in seg.shape)
seg2 = seg[:z, :y, :x].reshape(z // k, k, y // k, k, x // k, k).max(axis=(1, 3, 5))
inside = ndimage.distance_transform_edt(seg2, sampling=(12.0 * k,) * 3)     # um to nearest non-vessel (radius proxy)
large = inside >= a.r_min_um
print("vessel voxels(24um):", int(seg2.sum()), "large-vessel voxels:", int(large.sum()), f"({large.sum()/max(seg2.sum(),1):.3f})")
# dilate the large-vessel cores back to their full radius: keep seg2 voxels within r_min of a large core
core_dist = ndimage.distance_transform_edt(~large, sampling=(12.0 * k,) * 3)
large_full = seg2 & (core_dist <= a.r_min_um)
print("large vessels incl. walls:", int(large_full.sum()))
full = np.zeros((627 // k, y // k, x // k), bool); full[:large_full.shape[0]] = large_full   # z-offset 0
dist = ndimage.distance_transform_edt(~full, sampling=(12.0 * k,) * 3).astype(np.float32)   # um
A24 = A12.copy(); A24[:3, :3] *= k; A24[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * (k - 1) / 2.0)
np.save(a.work / "oct_largeves_dist24_z0.npy", dist); np.save(a.work / "oct_largeves_dist24_z0_affine.npy", A24); np.save(a.work / "oct_largeves_mask24.npy", full)
print("saved", dist.shape, "large-vessel fraction", full.mean())
