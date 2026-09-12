"""Label-free non-rigid refinement on top of the (vascular) affine: a smooth free-form displacement of the OCT block
(control grid over the block, trilinear), optimised with the same multi-channel masked NCC [WM, GM, vessel density]
plus smoothness / magnitude penalties.  Real data only, no annotations."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import to_t, apply_affine, sample_at_world, avg_pool_iso, DEVICE
from .refine import Refiner, masked_ncc


class GridDisp:
    """Smooth free-form displacement (mm, in MRI world axes) over the OCT block: control grid (gz,gy,gx), trilinear."""

    def __init__(self, shape_oct, A_oct, grid_shape, device=DEVICE):
        self.g = torch.zeros((1, 3, *grid_shape), device=device, requires_grad=True)
        self.A_inv = torch.linalg.inv(to_t(A_oct, device=device)); self.size = to_t(np.array(shape_oct) - 1, device=device)

    def __call__(self, pts_world_oct):
        v = apply_affine(self.A_inv, pts_world_oct) / self.size * 2 - 1
        grid = v[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.g, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(3, -1).T

    def smooth(self):
        g = self.g
        return ((g[..., 1:, :, :] - g[..., :-1, :, :]) ** 2).mean() + ((g[..., :, 1:, :] - g[..., :, :-1, :]) ** 2).mean() + ((g[..., :, :, 1:] - g[..., :, :, :-1]) ** 2).mean()

    def magnitude(self):
        return (self.g ** 2).mean()

    def state(self):
        return self.g.detach().cpu().numpy().copy()


def loss_with_disp(ref: Refiner, T: torch.Tensor, disp: GridDisp, subsample: int = 1):
    pts = ref.pts_o[::subsample]; w = ref.w[::subsample]; fo = ref.fo[:, ::subsample]
    pm = apply_affine(T, pts) + disp(pts)
    fm = sample_at_world(ref.FM, ref.A_M, pm)
    return 1.0 - masked_ncc(fm, fo, w, chan_w=ref.chan_w)


def nonrigid_refine(T: np.ndarray, FM: torch.Tensor, A_reg: np.ndarray, FO: torch.Tensor, A_oct: np.ndarray, MO: torch.Tensor,
                    grid=(5, 8, 8), iters: int = 300, lr: float = 0.05, lam_smooth: float = 5.0, lam_mag: float = 0.5,
                    chan_w=(1.0, 1.0, 1.0), factors=(2, 1), device=DEVICE):
    """Optimise a displacement grid on top of the affine T (fixed).  Returns (disp, info)."""
    Tt = to_t(T, device=device)
    disp = GridDisp(MO.shape[1:], A_oct, grid, device=device)
    opt = torch.optim.Adam([disp.g], lr=lr)
    hist = []
    for f in factors:
        if f == 1:
            ref = Refiner(FM, A_reg, FO, A_oct, MO, device=device, chan_w=list(chan_w))
        else:
            a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(MO, A_oct, f)
            ref = Refiner(a1, A1, a2, A2, a3, device=device, chan_w=list(chan_w))
        for it in range(iters):
            opt.zero_grad()
            L = loss_with_disp(ref, Tt, disp)
            tot = L + lam_smooth * disp.smooth() + lam_mag * disp.magnitude()
            tot.backward(); opt.step()
            if it % 50 == 0 or it == iters - 1:
                hist.append((f, it, float(L)))
    with torch.no_grad():
        mag = disp(ref.pts_o).norm(dim=1)
        info = {"final_data_loss": float(L), "disp_mean_mm": float(mag.mean()), "disp_p95_mm": float(torch.quantile(mag, 0.95)), "disp_max_mm": float(mag.max()), "history": hist,
                "ncc_channels": None}
        pm = apply_affine(Tt, ref.pts_o) + disp(ref.pts_o); fm = sample_at_world(ref.FM, ref.A_M, pm)
        info["ncc_channels"] = [float(masked_ncc(fm[c:c + 1], ref.fo[c:c + 1], ref.w)) for c in range(fm.shape[0])]
    return disp, info


@torch.no_grad()
def inverse_points(T: np.ndarray, disp: GridDisp, pts_mri: torch.Tensor, n_iter: int = 8):
    """OCT-world points p_o with T p_o + disp(p_o) = p_m, by fixed-point iteration (small smooth displacements)."""
    Tinv = torch.linalg.inv(to_t(T, device=pts_mri.device))
    p_o = apply_affine(Tinv, pts_mri)
    for _ in range(n_iter):
        p_o = apply_affine(Tinv, pts_mri - disp(p_o))
    return p_o
