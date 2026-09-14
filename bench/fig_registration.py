#!/usr/bin/env python3
"""README registration figure in the MRI frame, read from the original files: three MRI array planes through the registered
OCT specimen centre, the native MRI slice (the reference), and the OCT placed onto it before registration (orientation from the
file headers, the OCT specimen centre moved onto the MRI foreground centre) and after it (the run's transform), each also as a
2 mm checkerboard with the MRI (MRI contrast inverted inside the run's MRI foreground when the polarity is -1, as in qc.png).
The OCT is box-averaged to the MRI voxel size (0.08 mm) before it is sampled.

    python bench/fig_registration.py --run RUN --mask MASK --mri-mask MRI_MASK --oct OCT --mri MRI -o FIG.png

RUN: octreg register output (result.json). MASK, MRI_MASK: the run's OCT specimen mask and MRI foreground on the base grid
(ablate prep/texture/oct_mask.nii.gz, prep/mri/mri_mask.nii.gz).
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

from octreg import geometry as G, io

TILE_MM = 2.0


def sample(vol, affine, world, order=1):
    """Values of vol (voxel -> world affine) at world points [4, N]."""
    return ndimage.map_coordinates(vol, (np.linalg.inv(affine) @ world)[:3], order=order, cval=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--run", "--mask", "--mri-mask", "--oct", "--mri", "-o"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    res = json.loads(open(f"{a.run}/result.json").read())
    T, pol = np.array(res["T_oct2mri"]), res["pose"]["polarity"]
    mri_img = nib.load(a.mri)
    mri, A_m = np.asarray(mri_img.dataobj, np.float32), mri_img.affine
    sp_m = float(np.linalg.norm(A_m[:3, 0]))
    m_img, f_img = nib.load(a.mask), nib.load(a.mri_mask)
    spec, A_h = np.asarray(m_img.dataobj, np.float32), m_img.affine
    fg_m, A_fm = np.asarray(f_img.dataobj, np.float32), f_img.affine
    oct_, A_o = G.resample_iso(io.load_volume(a.oct), sp_m)                                   # OCT at the MRI voxel size

    c_oct = A_h @ np.append(ndimage.center_of_mass(spec > 0.5), 1.0)                         # OCT specimen centre, OCT world
    T0 = np.eye(4)                                                                             # before registration
    T0[:3, 3] = (A_fm @ np.append(ndimage.center_of_mass(fg_m > 0.5), 1.0))[:3] - c_oct[:3]
    idx = np.rint((np.linalg.inv(A_m) @ T @ c_oct)[:3]).astype(int)                           # MRI planes
    sub = oct_[::3, ::3, ::3]
    lo_o, hi_o = np.percentile(sub[sub > 0], [0.5, 99.5])
    hi_m = float(np.percentile(mri[mri > 0], 99.5))
    lo_t, hi_t = np.percentile(mri[mri > res["foreground"]["mri"]["threshold"]], [1, 99])

    rows = []
    for ax in range(3):
        sl = np.take(mri, idx[ax], axis=ax)                                                    # native MRI slice
        others = [d for d in range(3) if d != ax]
        u, v = np.meshgrid(np.arange(sl.shape[0]), np.arange(sl.shape[1]), indexing="ij")
        ijk = np.zeros((4,) + u.shape)
        ijk[others[0]], ijk[others[1]], ijk[ax], ijk[3] = u, v, idx[ax], 1.0
        world = A_m @ ijk.reshape(4, -1)
        fg = sample(fg_m, A_fm, world).reshape(u.shape) > 0.5
        m = np.clip((sl - lo_t) / (hi_t - lo_t), 0, 1)
        m = np.where(fg, 1 - m if pol < 0 else m, 0.0)
        tile = int(round(TILE_MM / sp_m))
        board = ((np.arange(sl.shape[0])[:, None] // tile + np.arange(sl.shape[1])[None, :] // tile) % 2) == 0
        imgs = [np.clip(sl / hi_m, 0, 1)]
        for X in (T0, T):
            back = np.linalg.inv(X) @ world                                                    # MRI world -> OCT world
            o = sample(oct_, A_o, back).reshape(u.shape)
            inside = (sample(spec, A_h, back, order=0).reshape(u.shape) > 0.5) & (o > 0)
            lo_s, hi_s = np.percentile(o[inside], [1, 99]) if inside.any() else (lo_o, hi_o)
            imgs += [np.clip((o - lo_o) / (hi_o - lo_o), 0, 1) * (o > 0),
                     np.where(board, np.clip((o - lo_s) / (hi_s - lo_s), 0, 1) * (o > 0), m)]
        rows.append((ax, idx[ax] * sp_m, [x.T for x in imgs]))

    cell, head, gap = 3.3, 0.6, 0.2
    x_of = lambda j: 0.35 + j * cell + gap * (j >= 1) + gap * (j >= 3)
    W, H = x_of(4) + cell + 0.05, head + 3 * cell
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor="white")
    board_head = "checkerboard (MRI inverted)" if pol < 0 else "checkerboard"
    heads = ("MRI", "OCT", board_head, "OCT", board_head)
    for (j0, j1), t in (((1, 2), "before registration"), ((3, 4), "after registration")):
        fig.text((x_of(j0) + x_of(j1) + cell) / 2 / W, 1 - 0.14 / H, t, ha="center", va="center", fontsize=10, weight="bold")
    for r, (ax, pos, imgs) in enumerate(rows):
        for j, img in enumerate(imgs):
            axh = fig.add_axes(((x_of(j) + 0.03) / W, (3 * cell - (r + 1) * cell + 0.03) / H, (cell - 0.06) / W, (cell - 0.06) / H))
            axh.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="antialiased")
            axh.set_axis_off()
            if r == 0:
                axh.set_title(heads[j], fontsize=9)
            if j == 0:
                bar = 10 if img.shape[1] * sp_m > 35 else 5
                axh.add_artist(AnchoredSizeBar(axh.transData, bar / sp_m, f"{bar} mm", "lower left", color="w", frameon=False,
                                               size_vertical=img.shape[0] / 120, fontproperties={"size": 7}))
        fig.text(0.2 / W, (3 * cell - (r + 0.5) * cell) / H, f"MRI axis {ax} at {pos:.2f} mm", rotation=90, ha="center",
                 va="center", fontsize=8)
    fig.savefig(a.o, dpi=150, facecolor="white")
    from PIL import Image
    Image.open(a.o).convert("L").save(a.o, optimize=True)                                     # grey only: smaller file
    print("wrote", a.o, "MRI planes", idx.tolist())


if __name__ == "__main__":
    main()
