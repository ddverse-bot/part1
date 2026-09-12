#!/usr/bin/env python3
"""Publication-style figures, one per subject: 5 panels A-E, minimal in-figure text (letters + scale bars only).
A MRI slice with the located block (red);  B OCT mid-depth slice;  C registered MRI, same plane;
D checkerboard of B|C;  E edges of C (Canny) over B.  Grayscale, robust windowing, physical scale bars."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "octreg")
from octreg.common import to_t, apply_affine, sample_at_world, grid_points
from skimage.feature import canny
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms

def letter(ax, s):
    ax.text(0.02, 0.98, s, transform=ax.transAxes, ha="left", va="top", fontsize=20, fontweight="bold",
            color="white", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

def scalebar(ax, vox_mm, img_w, length_mm, label):
    px = length_mm / vox_mm
    x0 = 0.05 * img_w
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([x0, x0 + px], [0.045, 0.045], transform=tr, color="white", linewidth=4, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=6, foreground="black")])
    ax.text(x0 + px / 2, 0.075, label, transform=tr, ha="center", va="bottom", fontsize=12, color="white",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

CASES = [
    ("I46", "work/I46", "work/runs/I46_otsu"),
    ("I55", "work/I55", "work/runs/I55_otsu"),
    ("I38", "work/I38", "work/runs/I38_otsu"),
    ("I56", "work/I56", "work/runs/I56_otsu"),
    ("I62", "work/I62", "work/runs/I62_otsu"),
    ("I58_brainstem", "work/xiangrui_I58bs", "work/runs/xiangrui_I58bs_novasc"),
]
out_dir = Path("octreg/figures"); out_dir.mkdir(parents=True, exist_ok=True)
for name, W, R in CASES:
    W, R = Path(W), Path(R)
    mri = np.load(W / "mri.npy", mmap_mode="r"); A_m = np.load(W / "mri_affine.npy")
    o = np.load(W / "oct150.npy"); om = np.load(W / "oct150_mask.npy"); A_o = np.load(W / "oct150_affine.npy")
    T = np.load(R / "T_oct2mri.npy"); Tinv = np.linalg.inv(T)
    vox_m = float(np.linalg.norm(A_m[:3, :3], axis=0).mean()); vox_o = float(np.linalg.norm(A_o[:3, :3], axis=0).mean())
    D = o.shape[0]; z = D // 2
    c_o = (A_o @ np.r_[(np.array(o.shape) - 1) / 2.0, 1.0])[:3]; bc = (T @ np.r_[c_o, 1.0])[:3]
    i0 = int(round(np.linalg.solve(A_m[:3, :3], bc - A_m[:3, 3])[0])); i0 = max(0, min(mri.shape[0] - 1, i0))
    sl = np.asarray(mri[i0]).astype(np.float32)
    jj, kk = np.meshgrid(np.arange(mri.shape[1]), np.arange(mri.shape[2]), indexing="ij")
    pts = (A_m @ np.c_[np.full(jj.size, i0), jj.ravel(), kk.ravel(), np.ones(jj.size)].T).T[:, :3]
    with torch.no_grad():
        mo = sample_at_world(to_t(om.astype(np.float32))[None], to_t(A_o), apply_affine(to_t(Tinv), to_t(pts)))[0].reshape(jj.shape).cpu().numpy()
    pts_o = grid_points(to_t(A_o), o.shape).reshape(-1, 3)
    with torch.no_grad():
        m_in_o = sample_at_world(to_t(np.asarray(mri).astype(np.float32))[None], to_t(A_m), apply_affine(to_t(T), pts_o))[0].reshape(o.shape).cpu().numpy()
    vo = np.percentile(o[om], 99) if om.any() else float(o.max()); ml, mh = np.percentile(m_in_o[om], [1, 99])
    on = np.clip(o[z] / vo, 0, 1); mn = np.clip((m_in_o[z] - ml) / (mh - ml + 1e-6), 0, 1)
    t = max(8, min(on.shape) // 6); yy, xx = np.indices(on.shape); chk = np.where(((yy // t + xx // t) % 2).astype(bool), on, mn)
    from scipy.ndimage import binary_dilation, zoom
    up = 4 if min(on.shape) < 400 else 1                              # thin edges: detect on an upsampled slice for small images
    mn_up = zoom(mn, up, order=1) if up > 1 else mn
    ed = canny(mn_up, sigma=2.0 * up / 2, low_threshold=0.08, high_threshold=0.18)
    keep = binary_dilation(zoom(om[z].astype(float), up, order=0) > 0.5, iterations=6 * up)   # only near the block
    ed &= keep
    msl_hi = np.percentile(sl[sl > 0], 99.5)
    fig, ax = plt.subplots(1, 5, figsize=(27, 6.0))
    ax[0].imshow(np.clip(sl / msl_hi, 0, 1), cmap="gray"); ax[0].contour(mo > 0.5, levels=[0.5], colors="red", linewidths=1.6)
    letter(ax[0], "A"); scalebar(ax[0], vox_m, sl.shape[1], 10, "10 mm")
    ax[1].imshow(on, cmap="gray"); letter(ax[1], "B"); scalebar(ax[1], vox_o, on.shape[1], 5, "5 mm")
    ax[2].imshow(mn, cmap="gray"); letter(ax[2], "C"); scalebar(ax[2], vox_o, on.shape[1], 5, "5 mm")
    ax[3].imshow(chk, cmap="gray"); letter(ax[3], "D"); scalebar(ax[3], vox_o, on.shape[1], 5, "5 mm")
    ax[4].imshow(on, cmap="gray")
    rgba = np.zeros((*ed.shape, 4), dtype=np.float32); rgba[ed] = (1.0, 0.45, 0.05, 1.0)
    ax[4].imshow(rgba, interpolation="nearest", extent=(-0.5, on.shape[1] - 0.5, on.shape[0] - 0.5, -0.5))
    letter(ax[4], "E"); scalebar(ax[4], vox_o, on.shape[1], 5, "5 mm")
    for a_ in ax: a_.axis("off")
    plt.tight_layout(pad=0.4)
    plt.savefig(out_dir / f"fig_{name}.png", dpi=105, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved", name, flush=True)
print("done")
