#!/usr/bin/env python3
"""README registration figure in the MRI frame, read from the original files: three MRI array planes through the registered
OCT specimen centre, three panels each.

    MRI              the reference, as it is, on its own grey scale.
    registered OCT   the OCT through the run's transform and its smooth field, on the same plane and its own grey scale.
    checkerboard     the two panels beside it, cut into TILE_MM squares and interleaved. It is nothing else: the same two
                     arrays on the same two grey scales, not inverted, not matched to each other, not masked. So a structure
                     runs on across a square edge where the two agree and steps where they do not, and the reader can check
                     any square against the panel it came from.

The OCT is box-averaged to the MRI voxel size before it is sampled. What a checkerboard settles, and what it does not, is
measured in docs/METHOD.md under "Evaluation".

    python bench/fig_registration.py --run RUN --mask MASK --oct OCT --mri MRI -o FIG.png

RUN: octreg register output (result.json). MASK: the run's OCT specimen mask on the base grid, which places the three planes
(ablate prep/texture/oct_mask.nii.gz).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import ndimage

from octreg import geometry as G, io

TILE_MM = 8.0              # checkerboard square: wide enough that one square holds a structure to follow across its edge


def sample(vol, affine, world, order=1):
    """Values of vol (voxel -> world affine) at world points [4, N]."""
    return ndimage.map_coordinates(vol, (np.linalg.inv(affine) @ world)[:3], order=order, cval=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--run", "--mask", "--oct", "--mri", "-o"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    res = json.loads(open(f"{a.run}/result.json").read())
    T = np.array(res["T_oct2mri"])
    mri_img = nib.load(a.mri)
    mri, A_m = np.asarray(mri_img.dataobj, np.float32), mri_img.affine
    sp_m = float(np.linalg.norm(A_m[:3, 0]))
    m_img = nib.load(a.mask)
    spec, A_h = np.asarray(m_img.dataobj, np.float32), m_img.affine
    oct_, A_o = G.resample_iso(io.load_volume(a.oct), sp_m)                                   # OCT at the MRI voxel size
    warp = Path(a.run) / "oct2mri_warp.nii.gz"                                                # §6, when the run applied it
    field, A_u = io.load_field(warp) if warp.exists() else (None, None)

    c_oct = A_h @ np.append(ndimage.center_of_mass(spec > 0.5), 1.0)                          # OCT specimen centre, OCT world
    idx = np.rint((np.linalg.inv(A_m) @ T @ c_oct)[:3]).astype(int)                           # MRI planes
    sub = oct_[::3, ::3, ::3]
    lo_o, hi_o = np.percentile(sub[sub > 0], [0.5, 99.5])
    hi_m = float(np.percentile(mri[mri > 0], 99.5))
    tile = int(round(TILE_MM / sp_m))

    rows = []
    for ax in range(3):
        sl = np.take(mri, idx[ax], axis=ax)                                                    # native MRI slice
        others = [d for d in range(3) if d != ax]
        u, v = np.meshgrid(np.arange(sl.shape[0]), np.arange(sl.shape[1]), indexing="ij")
        ijk = np.zeros((4,) + u.shape)
        ijk[others[0]], ijk[others[1]], ijk[ax], ijk[3] = u, v, idx[ax], 1.0
        world = A_m @ ijk.reshape(4, -1)
        moved = world.copy()                                                                   # x + u(x), octreg.deform
        if field is not None:
            moved[:3] += np.stack([sample(c, A_u, world) for c in field])
        back = np.linalg.inv(T) @ moved                                                        # MRI world -> OCT world
        o = sample(oct_, A_o, back).reshape(u.shape)
        m = np.clip(sl / hi_m, 0, 1).T
        g = (np.clip((o - lo_o) / (hi_o - lo_o), 0, 1) * (o > 0)).T
        board = ((np.arange(m.shape[0])[:, None] // tile + np.arange(m.shape[1])[None, :] // tile) % 2) == 0
        rows.append({"ax": ax, "pos": idx[ax] * sp_m, "mri": m, "oct": g, "check": np.where(board, g, m)})

    cell, head, n = 3.3, 0.35, 3
    x_of = lambda j: 0.35 + j * cell
    W, H = x_of(n) + 0.05, head + 3 * cell
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor="white")
    heads = ("MRI", "registered OCT", f"the two interleaved, {TILE_MM:g} mm squares")
    for r, d in enumerate(rows):
        for j, img in enumerate((d["mri"], d["oct"], d["check"])):
            axh = fig.add_axes(((x_of(j) + 0.03) / W, (3 * cell - (r + 1) * cell + 0.03) / H, (cell - 0.06) / W, (cell - 0.06) / H))
            axh.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="antialiased")
            axh.set_axis_off()
            if r == 0:
                axh.set_title(heads[j], fontsize=9)
            if j == 0:
                bar = 10 if img.shape[1] * sp_m > 35 else 5
                axh.add_artist(AnchoredSizeBar(axh.transData, bar / sp_m, f"{bar} mm", "lower left", color="w", frameon=False,
                                               borderpad=0.5, size_vertical=img.shape[0] / 120, fontproperties={"size": 7}))
        fig.text(0.2 / W, (3 * cell - (r + 0.5) * cell) / H, f"MRI axis {d['ax']} at {d['pos']:.2f} mm", rotation=90,
                 ha="center", va="center", fontsize=8)
    fig.savefig(a.o, dpi=150, facecolor="white")
    from PIL import Image
    Image.open(a.o).convert("L").save(a.o, optimize=True)                # grey only: nothing is drawn on the data
    print("wrote", a.o, "MRI planes", idx.tolist())


if __name__ == "__main__":
    main()
