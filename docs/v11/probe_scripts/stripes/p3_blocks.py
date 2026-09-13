#!/usr/bin/env python
"""P3: black missing-tile blocks on oct150: per-plane rectangles, z-caps per x, distances to the specimen."""
import os, json, numpy as np
from scipy import ndimage
BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration"); W = os.path.join(BASE, "work/xiangrui_I58bs"); OUT = os.path.join(BASE, "work/probe_xr/stripes")
o = np.load(f"{W}/oct150.npy"); m = np.load(f"{W}/oct150_mask.npy"); me = np.load(f"{OUT}/mask150_eroded2mm.npy")
z = o == 0; nz = ~z
R = {"shape": list(o.shape), "vox_mm": 0.15, "zero_frac": float(z.mean()), "min_nonzero": float(o[nz].min())}
# a tissue-only proxy: inside the old mask, local mean (sigma 1 mm) above the agarose level -> largest component
sm = ndimage.gaussian_filter(np.where(m, o, 0).astype(np.float32), 7) / np.maximum(ndimage.gaussian_filter(m.astype(np.float32), 7), 1e-3)
# columns of scanned data: (y,x) columns having any nonzero
col = nz.any(0); R["scanned_columns_frac"] = float(col.mean())
# 2-D zero rectangles at three z levels
R["xy_rectangles"] = {}
for zc in (40, 97, 150):
    lab, n = ndimage.label(z[zc]); objs = ndimage.find_objects(lab); rects = []
    dt_er = ndimage.distance_transform_edt(~me[zc]) * 0.15; dt_m = ndimage.distance_transform_edt(~m[zc]) * 0.15
    for k in range(n):
        comp = lab == (k + 1); sl = objs[k]
        rects.append({"y_range": [int(sl[0].start), int(sl[0].stop)], "x_range": [int(sl[1].start), int(sl[1].stop)], "n_vox": int(comp.sum()), "fill": round(float(comp.sum() / ((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))), 3),
                      "min_dist_to_eroded2mm_mm": round(float(dt_er[comp].min()), 2), "min_dist_to_oldmask_mm": round(float(dt_m[comp].min()), 2), "n_in_oldmask": int((comp & m[zc]).sum())})
    R["xy_rectangles"][f"z={zc}"] = rects
# z-caps: first / last nonzero z per x (at the central y) and per y (at the central x)
cy, cx = 135, 105
def caps(plane_nz):  # plane_nz: (z, n) -> first,last nonzero z per column
    first = np.where(plane_nz.any(0), plane_nz.argmax(0), -1); last = np.where(plane_nz.any(0), plane_nz.shape[0] - 1 - plane_nz[::-1].argmax(0), -1); return first, last
fx0, fx1 = caps(nz[:, cy, :]); fy0, fy1 = caps(nz[:, :, cx])
def runs(v):
    out = []; s = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[s]: out.append([int(s), int(i), int(v[s])]); s = i
    return out
R["z_caps_along_x_at_cy"] = {"first_nonzero_z_runs[x0,x1,z]": runs(fx0), "last_nonzero_z_runs[x0,x1,z]": runs(fx1)}
R["z_caps_along_y_at_cx"] = {"first_nonzero_z_runs[y0,y1,z]": runs(fy0), "last_nonzero_z_runs[y0,y1,z]": runs(fy1)}
# does the tissue (eroded mask) ever touch a zero? and how much of the old-mask boundary touches zeros
R["eroded_mask_voxels_adjacent_to_zero"] = int((ndimage.binary_dilation(z) & me).sum())
bnd = m & ~ndimage.binary_erosion(m); R["oldmask_boundary_frac_adjacent_to_zero"] = float((ndimage.binary_dilation(z) & bnd).sum() / bnd.sum())
R["oldmask_voxels_that_are_zero"] = int((m & z).sum())
# how many old-mask voxels within 2 voxels (0.3 mm) of a zero (these get contaminated by any trilinear resampling)
R["oldmask_voxels_within_0.3mm_of_zero"] = int((ndimage.binary_dilation(z, iterations=2) & m).sum())
print(json.dumps(R, indent=1)); json.dump(R, open(f"{OUT}/black_blocks.json", "w"), indent=1)
