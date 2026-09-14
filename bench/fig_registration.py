#!/usr/bin/env python3
"""README registration figure read from the original files: for the three QC planes through the specimen centre, the OCT
plane of the 20 um file sampled every STEP voxels; the MRI interpolated from its native 0.08 mm grid at the same points before
registration (orientation from the file headers, the OCT specimen centre moved onto the MRI foreground centre) and after it (the run's transform), each in original contrast and as a 2 mm
checkerboard with the OCT (MRI contrast inverted inside the run's MRI foreground when the polarity is -1, as in qc.png).

    python bench/fig_registration.py --run RUN --mask MASK --mri-mask MRI_MASK --oct OCT --mri MRI -o FIG.png

RUN: octreg register output (result.json). MASK, MRI_MASK: the run's OCT specimen mask and MRI foreground on the base grid
(ablate prep/texture/oct_mask.nii.gz, prep/mri/mri_mask.nii.gz); MASK fixes the planes exactly as qc.png does. The OCT is
streamed once; one plane in memory at a time.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import ndimage

from octreg import io

STEP = 2              # OCT voxels per plotted sample (40 um); the 150 dpi panels show about 70-80 um per pixel
TILE_MM = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--run", "--mask", "--mri-mask", "--oct", "--mri", "-o"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    res = json.loads(open(f"{a.run}/result.json").read())
    T, pol = np.array(res["T_oct2mri"]), res["pose"]["polarity"]
    thr = res["foreground"]["mri"]["threshold"]
    vo = io.load_volume(a.oct)
    A_f, (ni, nj, nk) = vo.affine, vo.shape
    sp = float(np.linalg.norm(A_f[:3, 0]))
    m_img = nib.load(a.mask)
    mask, A_h = np.asarray(m_img.dataobj) > 0, m_img.affine
    c = np.rint(ndimage.center_of_mass(mask))
    idx = np.rint((np.linalg.inv(A_f) @ A_h @ np.append(c, 1.0))[:3]).astype(int)           # file voxel of each plane
    planes = [np.zeros((nj, nk), np.float32), np.zeros((ni, nk), np.float32), None]
    for k, pl in io.iter_planes(vo):
        planes[0][:, k], planes[1][:, k] = pl[idx[0], :], pl[:, idx[1]]
        if k == idx[2]:
            planes[2] = pl.copy()
    mri_img = nib.load(a.mri)
    mri, A_m = np.asarray(mri_img.dataobj, np.float32), mri_img.affine
    fm_img = nib.load(a.mri_mask)
    fg_m, A_fm = np.asarray(fm_img.dataobj, np.float32), fm_img.affine
    hi_m = float(np.percentile(mri[mri > 0], 99.5))
    tissue_vals = mri[mri > thr]
    lo_t, hi_t = np.percentile(tissue_vals, [1, 99])
    mask_to_file = np.linalg.inv(A_h) @ A_f
    T0 = np.eye(4)                                                                         # before registration
    T0[:3, 3] = (A_fm @ np.append(ndimage.center_of_mass(fg_m > 0.5), 1.0))[:3] - (A_h @ np.append(ndimage.center_of_mass(mask), 1.0))[:3]

    rows = []
    for ax in range(3):
        pl = planes[ax][::STEP, ::STEP]                                                     # [u, v] on the two other axes
        others = [d for d in range(3) if d != ax]
        u, v = np.meshgrid(np.arange(pl.shape[0]) * STEP, np.arange(pl.shape[1]) * STEP, indexing="ij")
        ijk = np.zeros((4,) + u.shape)
        ijk[others[0]], ijk[others[1]], ijk[ax], ijk[3] = u, v, idx[ax], 1.0
        pts = ijk.reshape(4, -1)
        mri_vox = (np.linalg.inv(A_m) @ T @ A_f @ pts)[:3]
        mp = ndimage.map_coordinates(mri, mri_vox, order=1, cval=0.0).reshape(u.shape)
        mp0 = ndimage.map_coordinates(mri, (np.linalg.inv(A_m) @ T0 @ A_f @ pts)[:3], order=1, cval=0.0).reshape(u.shape)
        fg, fg0 = (ndimage.map_coordinates(fg_m, (np.linalg.inv(A_fm) @ X @ A_f @ pts)[:3], order=1, cval=0.0).reshape(u.shape) > 0.5
                   for X in (T, T0))
        inside = ndimage.map_coordinates(mask.astype(np.float32), (mask_to_file @ pts)[:3], order=0, cval=0).reshape(u.shape) > 0
        meas = pl > 0
        lo_o, hi_o = np.percentile(pl[meas], [0.5, 99.5])
        oct_raw = np.clip((pl - lo_o) / (hi_o - lo_o), 0, 1) * meas
        lo_s, hi_s = np.percentile(pl[inside & meas], [1, 99]) if (inside & meas).any() else (lo_o, hi_o)
        oct_s = np.clip((pl - lo_s) / (hi_s - lo_s), 0, 1) * meas
        tile = int(round(TILE_MM / (sp * STEP)))
        board = ((np.arange(pl.shape[0])[:, None] // tile + np.arange(pl.shape[1])[None, :] // tile) % 2) == 0

        def checker(x, f):
            y = np.clip((x - lo_t) / (hi_t - lo_t), 0, 1)
            return np.where(board, oct_s, np.where(f, 1 - y if pol < 0 else y, 0.0))

        pos = idx[ax] * sp
        rows.append((ax, pos, [x.T for x in (oct_raw, np.clip(mp0 / hi_m, 0, 1), checker(mp0, fg0), np.clip(mp / hi_m, 0, 1),
                                             checker(mp, fg))]))

    cell, head, ncol, gap = 3.3, 0.6, 5, 0.2
    W = 0.35 + ncol * cell + 2 * gap
    x_of = lambda j: 0.35 + j * cell + gap * (j >= 1) + gap * (j >= 3)
    fig = plt.figure(figsize=(W, head + 3 * cell), dpi=150, facecolor="white")
    board_head = "checkerboard (MRI inverted)" if pol < 0 else "checkerboard"
    heads = ("OCT", "MRI", board_head, "MRI", board_head)
    for (j0, j1), t in (((1, 2), "before registration"), ((3, 4), "after registration")):
        fig.text((x_of(j0) + x_of(j1) + cell) / 2 / W, 1 - 0.14 / (head + 3 * cell), t, ha="center", va="center", fontsize=10,
                 weight="bold")
    for r, (ax, pos, imgs) in enumerate(rows):
        for j, img in enumerate(imgs):
            axh = fig.add_axes(((x_of(j) + 0.03) / W, (3 * cell - (r + 1) * cell + 0.03) / (head + 3 * cell),
                                (cell - 0.06) / W, (cell - 0.06) / (head + 3 * cell)))
            axh.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="antialiased")
            axh.set_axis_off()
            if r == 0:
                axh.set_title(heads[j], fontsize=9)
            if j == 0:
                px_mm = sp * STEP
                bar = 10 if img.shape[1] * px_mm > 35 else 5
                axh.add_artist(AnchoredSizeBar(axh.transData, bar / px_mm, f"{bar} mm", "lower left", color="w", frameon=False,
                                               size_vertical=img.shape[0] / 120, fontproperties={"size": 7}))
        fig.text(0.2 / W, (3 * cell - (r + 0.5) * cell) / (head + 3 * cell), f"axis {ax} at {pos:.2f} mm",
                 rotation=90, ha="center", va="center", fontsize=8)
    fig.savefig(a.o, dpi=150, facecolor="white")
    from PIL import Image
    Image.open(a.o).convert("L").save(a.o, optimize=True)                                   # grey only: smaller file
    print("wrote", a.o, "planes", idx.tolist())


if __name__ == "__main__":
    main()
