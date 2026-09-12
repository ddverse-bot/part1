#!/usr/bin/env python3
"""Document figures: one wide 4-panel PNG per subject.
[MRI + block location] [raw OCT slice] [registered MRI, same plane] [overlay: MRI + OCT boundaries/vessels]"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "octreg")
from octreg.common import to_t, apply_affine, sample_at_world, grid_points
from octreg.vascular import density_on_grid
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CASES = [  # (name, work, run, show_vessels)
    ("I46", "work/I46", "work/runs/I46_otsu", True),
    ("I55", "work/I55", "work/runs/I55_otsu", True),
    ("I38", "work/I38", "work/runs/I38_otsu", True),
    ("I56", "work/I56", "work/runs/I56_otsu", True),
    ("I62", "work/I62", "work/runs/I62_otsu", True),
    ("I58_brainstem", "work/xiangrui_I58bs", "work/runs/xiangrui_I58bs_novasc", False),
]
out_dir = Path("octreg/figures"); out_dir.mkdir(parents=True, exist_ok=True)
for name, W, R, show_ves in CASES:
    W, R = Path(W), Path(R)
    mri = np.load(W / "mri.npy", mmap_mode="r"); A_m = np.load(W / "mri_affine.npy")
    o = np.load(W / "oct150.npy"); om = np.load(W / "oct150_mask.npy"); A_o = np.load(W / "oct150_affine.npy")
    T = np.load(R / "T_oct2mri.npy"); Tinv = np.linalg.inv(T)
    oc = np.load(R / "oct_class150.npy") if (R / "oct_class150.npy").exists() else None
    D, H, Wd = o.shape; z = D // 2
    # panel 1: MRI slice through the block centre, with the OCT mask outline mapped in
    c_o = (A_o @ np.r_[(np.array(o.shape) - 1) / 2.0, 1.0])[:3]; bc = (T @ np.r_[c_o, 1.0])[:3]
    i0 = int(round(np.linalg.solve(A_m[:3, :3], bc - A_m[:3, 3])[0])); i0 = max(0, min(mri.shape[0] - 1, i0))
    sl = np.asarray(mri[i0]).astype(np.float32)
    jj, kk = np.meshgrid(np.arange(mri.shape[1]), np.arange(mri.shape[2]), indexing="ij")
    pts = (A_m @ np.c_[np.full(jj.size, i0), jj.ravel(), kk.ravel(), np.ones(jj.size)].T).T[:, :3]
    with torch.no_grad():
        mo = sample_at_world(to_t(om.astype(np.float32))[None], to_t(A_o), apply_affine(to_t(Tinv), to_t(pts)))[0].reshape(jj.shape).cpu().numpy()
    # panels 2-4 in the OCT frame at slice z
    pts_o = grid_points(to_t(A_o), o.shape).reshape(-1, 3)
    with torch.no_grad():
        m_in_o = sample_at_world(to_t(np.asarray(mri).astype(np.float32))[None], to_t(A_m), apply_affine(to_t(T), pts_o))[0].reshape(o.shape).cpu().numpy()
    dens = None
    if show_ves and (W / "octv_vessels.npy").exists():
        vm = to_t(np.load(W / "octv_vessels.npy"), dtype=torch.bool); A_v = np.load(W / "octv_affine.npy")
        sp_v = np.linalg.norm(A_v[:3, :3], axis=0) * 1000
        dens = density_on_grid(vm, A_v, A_o, o.shape, om, pool=int(max(1, round(150.0 / sp_v.mean())))).cpu().numpy(); del vm
    vo = np.percentile(o[om], 99) if om.any() else o.max(); ml, mh = np.percentile(m_in_o[om], [1, 99])
    msl_hi = np.percentile(sl[sl > 0], 99.5)
    fig, ax = plt.subplots(1, 4, figsize=(23, 6.2))
    ax[0].imshow(np.clip(sl / msl_hi, 0, 1), cmap="gray"); ax[0].contour(mo > 0.5, levels=[0.5], colors="red", linewidths=1.4)
    ax[0].set_title("MRI with the located OCT block (red)", fontsize=13)
    ax[1].imshow(np.clip(o[z] / vo, 0, 1), cmap="gray"); ax[1].set_title("OCT, mid-depth slice (raw)", fontsize=13)
    ax[2].imshow(np.clip((m_in_o[z] - ml) / (mh - ml + 1e-6), 0, 1), cmap="gray"); ax[2].set_title("MRI resampled to the same plane (registered)", fontsize=13)
    ax[3].imshow(np.clip((m_in_o[z] - ml) / (mh - ml + 1e-6), 0, 1), cmap="gray")
    if oc is not None: ax[3].contour(oc[z] == 1, levels=[0.5], colors="cyan", linewidths=0.9)
    if dens is not None: ax[3].contour(dens[z] > 0.35, levels=[0.5], colors="lime", linewidths=0.8)
    ax[3].contour(om[z], levels=[0.5], colors="white", linewidths=0.6)
    ax[3].set_title("Overlay: OCT WM/GM (cyan)" + (" + OCT vessels (green)" if dens is not None else " + block outline"), fontsize=13)
    for a_ in ax: a_.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_{name}.png", dpi=105, bbox_inches="tight"); plt.close()
    print("saved", name, flush=True)
print("all figures done")
