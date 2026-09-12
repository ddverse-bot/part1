"""Evaluation of an OCT->MRI transform on the Costantini I46 pair using structures that the
registration never saw: MRI vessel labels vs OCT vessel segmentation, MRI GM/WM labels vs OCT
tissue classes; plus robustness/self-consistency; plus QC figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .common import DEVICE, apply_affine, sample_at_world, to_t


def vessel_distance_stats(T_oct2mri: np.ndarray, mri_vessels_ijk: np.ndarray, A_mri: np.ndarray,
                          oct_ves_dist_um: np.ndarray, A_ves: np.ndarray, rng_shift_mm: float = 3.0, n_ctrl: int = 30, seed=0):
    """MRI vessel voxel centres -> OCT space -> distance (um) to nearest OCT vessel (precomputed EDT on a
    pooled OCT vessel grid with affine A_ves).  Also a random-shift control (same points, random +-shift)."""
    pts_m = (A_mri @ np.c_[mri_vessels_ijk, np.ones(len(mri_vessels_ijk))].T).T[:, :3]
    Tinv = np.linalg.inv(T_oct2mri)
    pts_o = (Tinv @ np.c_[pts_m, np.ones(len(pts_m))].T).T[:, :3]
    dist_t = to_t(oct_ves_dist_um)[None]
    A_v = to_t(A_ves)

    def stats(p):
        v = apply_affine(torch.linalg.inv(A_v), to_t(p))
        inside = ((v >= 0) & (v <= to_t(np.array(oct_ves_dist_um.shape) - 1))).all(1)
        n = int(inside.sum())
        if n < 20:
            return {"n_inside": n}
        d = sample_at_world(dist_t, A_v, to_t(p)[inside])[0].cpu().numpy()
        return {"n_inside": n, "median_um": float(np.median(d)), "p75_um": float(np.percentile(d, 75)),
                "p95_um": float(np.percentile(d, 95)), "frac_within_150um": float((d <= 150).mean()),
                "frac_within_300um": float((d <= 300).mean())}

    res = {"registered": stats(pts_o)}
    rng = np.random.default_rng(seed)
    ctrl = [stats(pts_o + rng.uniform(-rng_shift_mm, rng_shift_mm, 3)) for _ in range(n_ctrl)]
    meds = [c["median_um"] for c in ctrl if "median_um" in c]
    res["random_shift_control"] = {"n": len(meds), "median_um_mean": float(np.mean(meds)) if meds else None,
                                   "median_um_min": float(np.min(meds)) if meds else None,
                                   "frac_within_300um_mean": float(np.mean([c["frac_within_300um"] for c in ctrl if "median_um" in c])) if meds else None}
    return res


def gm_wm_overlap(T_oct2mri: np.ndarray, oct_class: np.ndarray, oct_mask: np.ndarray, A_oct: np.ndarray,
                  labels4: np.ndarray, A_mri: np.ndarray):
    """oct_class: [D,H,W] in {0 bg,1 WM,2 GM} on the OCT grid; labels4 on the MRI grid (0,1 WM,2/3 GM).
    Returns Dice(WM), Dice(GM), fraction of OCT tissue landing on any MRI label, confusion counts."""
    idx = np.argwhere(oct_mask)
    p_o = (A_oct @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
    p_m = (T_oct2mri @ np.c_[p_o, np.ones(len(p_o))].T).T[:, :3]
    ijk = np.rint((np.linalg.inv(A_mri) @ np.c_[p_m, np.ones(len(p_m))].T).T[:, :3]).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(labels4.shape)), axis=1)
    lab = np.zeros(len(idx), dtype=np.int64)
    lab[ok] = labels4[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    lab_c = np.where(lab == 1, 1, np.where(np.isin(lab, (2, 3)), 2, 0))
    oc = oct_class[idx[:, 0], idx[:, 1], idx[:, 2]]
    out = {"frac_on_mri_labels": float((lab_c > 0).mean())}
    for c, name in ((1, "WM"), (2, "GM")):
        a = oc == c; b = lab_c == c
        out[f"dice_{name}"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    conf = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            conf[i, j] = int(((oc == i) & (lab_c == j)).sum())
    out["confusion_oct_rows_mri_cols"] = conf.tolist()
    return out


def transform_diff(T1: np.ndarray, T2: np.ndarray, c_o: np.ndarray, extent_mm: float = 7.0):
    """Centre displacement (mm), rotation difference (deg) and mean corner displacement for a block."""
    from .common import polar_rotation, rotation_geodesic_deg
    c = np.r_[c_o, 1.0]
    dc = np.linalg.norm((T1 @ c)[:3] - (T2 @ c)[:3])
    rot = rotation_geodesic_deg(polar_rotation(T1[:3, :3]), polar_rotation(T2[:3, :3]))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * extent_mm + c_o
    d = np.linalg.norm((T1 @ np.c_[corners, np.ones(8)].T).T[:, :3] - (T2 @ np.c_[corners, np.ones(8)].T).T[:, :3], axis=1)
    return {"centre_mm": float(dc), "rotation_deg": float(rot), "corner_mean_mm": float(d.mean()), "corner_max_mm": float(d.max())}


def qc_figure(path: Path, T: np.ndarray, oct_vol: np.ndarray, A_oct: np.ndarray, mri_vol: np.ndarray, A_mri: np.ndarray,
              oct_mask: np.ndarray, oct_class: np.ndarray | None = None, mri_labels: np.ndarray | None = None, title: str = ""):
    """OCT mid-slices with the MRI resampled into OCT space (checkerboard) and edge overlays."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, W = oct_vol.shape
    mri_t = to_t(mri_vol.astype(np.float32))[None]
    A_m = to_t(A_mri); A_o = to_t(A_oct); Tt = to_t(T)
    ii, jj, kk = torch.meshgrid(torch.arange(D, device=DEVICE, dtype=torch.float32), torch.arange(H, device=DEVICE, dtype=torch.float32),
                                torch.arange(W, device=DEVICE, dtype=torch.float32), indexing="ij")
    p = torch.stack([ii, jj, kk], -1).reshape(-1, 3)
    pm = apply_affine(Tt, apply_affine(A_o, p))
    mri_in_oct = sample_at_world(mri_t, A_m, pm)[0].reshape(D, H, W).cpu().numpy()
    lab_in_oct = None
    if mri_labels is not None:
        lab_in_oct = sample_at_world(to_t(mri_labels.astype(np.float32))[None], A_m, pm, mode="nearest")[0].reshape(D, H, W).cpu().numpy()
    fig, ax = plt.subplots(3, 4, figsize=(16, 11))
    vo = np.percentile(oct_vol[oct_mask], 99) if oct_mask.any() else oct_vol.max()
    vm_lo, vm_hi = np.percentile(mri_in_oct[oct_mask], [1, 99]) if oct_mask.any() else (mri_in_oct.min(), mri_in_oct.max())
    for r, sl in enumerate([(D // 2, slice(None), slice(None)), (slice(None), H // 2, slice(None)), (slice(None), slice(None), W // 2)]):
        o = oct_vol[sl] / vo; m = (mri_in_oct[sl] - vm_lo) / (vm_hi - vm_lo + 1e-6)
        ax[r, 0].imshow(np.clip(o, 0, 1), cmap="gray"); ax[r, 0].set_title("OCT")
        ax[r, 1].imshow(np.clip(m, 0, 1), cmap="gray"); ax[r, 1].set_title("MRI resampled into OCT space")
        cb = np.indices(o.shape).sum(0) // 12 % 2
        ax[r, 2].imshow(np.where(cb == 0, np.clip(o, 0, 1), np.clip(m, 0, 1)), cmap="gray"); ax[r, 2].set_title("checkerboard")
        ax[r, 3].imshow(np.clip(o, 0, 1), cmap="gray")
        if lab_in_oct is not None:
            L = lab_in_oct[sl]
            ax[r, 3].contour(L == 1, levels=[0.5], colors="r", linewidths=0.8)
            ax[r, 3].contour(np.isin(L, (2, 3)), levels=[0.5], colors="c", linewidths=0.8)
            ax[r, 3].set_title("MRI labels on OCT (WM red / GM cyan)")
        if oct_class is not None:
            ax[r, 3].contour(oct_class[sl] == 1, levels=[0.5], colors="y", linewidths=0.6, linestyles="dashed")
    for x in ax.ravel(): x.axis("off")
    fig.suptitle(title); plt.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); plt.savefig(path, dpi=100); plt.close()
