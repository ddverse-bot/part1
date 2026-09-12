"""Differentiable rigid / similarity / affine refinement of a candidate pose.

Parameters: rotation vector (3), translation (3), log-scales (3), shears (3); the transform is
    x_m = M (x_o - c_o) + t,   M = R(r) · Shear(sh) · diag(exp(ls))
Loss: 1 - masked NCC of MRI features sampled at the transformed OCT voxel positions (OCT tissue
weighted, averaged over feature channels); multi-resolution; Adam.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .common import DEVICE, apply_affine, grid_points, matrix_to_rotvec, polar_rotation, rotvec_to_matrix, sample_at_world, to_t


def _shear(sh: torch.Tensor) -> torch.Tensor:
    S = torch.eye(3, dtype=sh.dtype, device=sh.device)
    S = S.clone()
    S[0, 1] = sh[0]; S[0, 2] = sh[1]; S[1, 2] = sh[2]
    return S


MIRROR = np.diag([1.0, 1.0, -1.0])


def params_from_matrix(T: np.ndarray, c_o: np.ndarray):
    """Decompose x_m = A x_o + b into (rotvec, translation about c_o, log-scales, shears, mirror).
    A = R · Shear · diag(exp(ls)) · diag(1,1,±1); exact for outputs of compose()."""
    A = np.asarray(T[:3, :3], dtype=float)
    mirror = bool(np.linalg.det(A) < 0)
    if mirror:
        A = A @ MIRROR                             # undo the reflection: A_proper = A · diag(1,1,-1)
    Q, U = np.linalg.qr(A)                         # A = Q U, U upper triangular
    d = np.sign(np.diag(U)); d[d == 0] = 1.0
    Q = Q * d[None, :]; U = d[:, None] * U         # make diag(U) > 0
    if np.linalg.det(Q) < 0:                       # improper: fall back to polar rotation
        Q = polar_rotation(A); U = Q.T @ A
    scales = np.diag(U).copy()
    scales[scales <= 1e-3] = 1e-3
    Sh = U / scales[None, :]                       # unit-diagonal upper triangular = shear
    sh = np.array([Sh[0, 1], Sh[0, 2], Sh[1, 2]])
    r = matrix_to_rotvec(Q)
    t = np.asarray(T[:3, :3], dtype=float) @ c_o + T[:3, 3]
    return r, t, np.log(scales), sh, mirror


def compose(r: torch.Tensor, t: torch.Tensor, ls: torch.Tensor, sh: torch.Tensor, c_o: torch.Tensor, mirror: bool = False) -> torch.Tensor:
    R = rotvec_to_matrix(r)
    M = R @ _shear(sh) @ torch.diag(torch.exp(ls))
    if mirror:
        M = M @ torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=r.dtype, device=r.device))
    T = torch.eye(4, dtype=r.dtype, device=r.device)
    T = T.clone()
    T[:3, :3] = M
    T[:3, 3] = t - M @ c_o
    return T


def masked_ncc(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, chan_w: torch.Tensor | None = None) -> torch.Tensor:
    """a, b: [C,N]; w: [N] weights -> (channel-weighted) mean over channels of weighted NCC."""
    N = w.sum() + eps
    ma = (a * w).sum(1, keepdim=True) / N
    mb = (b * w).sum(1, keepdim=True) / N
    da, db = a - ma, b - mb
    cov = (w * da * db).sum(1)
    va = (w * da * da).sum(1); vb = (w * db * db).sum(1)
    ncc = cov / torch.sqrt((va * vb).clamp(min=eps))
    if chan_w is None:
        return ncc.mean()
    return (ncc * chan_w).sum() / chan_w.sum()


class Refiner:
    def __init__(self, mri_feat: torch.Tensor, mri_affine: np.ndarray, oct_feat: torch.Tensor, oct_affine: np.ndarray,
                 oct_mask: torch.Tensor, device=DEVICE, chan_w=None):
        self.dev = device
        self.chan_w = None if chan_w is None else torch.as_tensor(np.asarray(chan_w, dtype=np.float32), device=device)
        self.FM = mri_feat.to(device).float()
        self.A_M = to_t(mri_affine, device=device)
        self.FO = oct_feat.to(device).float()
        self.A_O = to_t(oct_affine, device=device)
        m = oct_mask.to(device).float()[0]
        idx = torch.nonzero(m > 0.05)
        self.pts_o = apply_affine(self.A_O, idx.float())              # [N,3] world coords of OCT tissue voxels
        self.w = m[idx[:, 0], idx[:, 1], idx[:, 2]]
        self.fo = self.FO[:, idx[:, 0], idx[:, 1], idx[:, 2]]         # [C,N]
        d, h, w_ = m.shape
        corners = torch.tensor([[i, j, k] for i in (0, d - 1) for j in (0, h - 1) for k in (0, w_ - 1)], dtype=torch.float32, device=device)
        self.c_o = apply_affine(self.A_O, corners).mean(0)

    def loss_of(self, T: torch.Tensor, subsample: int = 1) -> torch.Tensor:
        pts = self.pts_o[::subsample]; w = self.w[::subsample]; fo = self.fo[:, ::subsample]
        pm = apply_affine(T, pts)
        fm = sample_at_world(self.FM, self.A_M, pm)                     # [C,N]
        return 1.0 - masked_ncc(fm, fo, w, chan_w=self.chan_w)

    @torch.no_grad()
    def ncc_channels(self, T: np.ndarray) -> list:
        """Per-channel masked NCC at the current level (diagnostic)."""
        Tt = to_t(T, device=self.dev)
        pm = apply_affine(Tt, self.pts_o)
        fm = sample_at_world(self.FM, self.A_M, pm)
        out = []
        for c in range(fm.shape[0]):
            out.append(float(masked_ncc(fm[c:c + 1], self.fo[c:c + 1], self.w).item()))
        return out

    def refine(self, T0: np.ndarray, dof: str = "affine", iters: int = 200, lr_rot: float = 0.02, lr_t: float = 0.3,
               lr_ls: float = 0.01, lr_sh: float = 0.01, subsample: int = 1, verbose: bool = False,
               ls_clamp: float = 0.15, sh_clamp: float = 0.15, reg: float = 2.0):
        c_o = self.c_o.detach().cpu().numpy()
        r0, t0, ls0, sh0, mirror = params_from_matrix(np.asarray(T0), c_o)
        r = to_t(r0, device=self.dev).requires_grad_(True)
        t = to_t(t0, device=self.dev).requires_grad_(True)
        ls = to_t(ls0, device=self.dev).requires_grad_(dof in ("similarity", "affine"))
        sh = to_t(sh0, device=self.dev).requires_grad_(dof == "affine")
        groups = [{"params": [r], "lr": lr_rot}, {"params": [t], "lr": lr_t}]
        if dof == "similarity":
            groups.append({"params": [ls], "lr": lr_ls})
        if dof == "affine":
            groups += [{"params": [ls], "lr": lr_ls}, {"params": [sh], "lr": lr_sh}]
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.0)
        best = (float("inf"), None)
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            if dof == "similarity":
                lsu = ls.mean().expand(3)                              # isotropic scale
                T = compose(r, t, lsu, sh, self.c_o, mirror)
            else:
                T = compose(r, t, ls, sh, self.c_o, mirror)
            data_loss = self.loss_of(T, subsample)
            loss = data_loss + reg * ((ls ** 2).sum() + (sh ** 2).sum())
            loss.backward()
            opt.step(); sched.step()
            with torch.no_grad():
                ls.clamp_(-ls_clamp, ls_clamp); sh.clamp_(-sh_clamp, sh_clamp)
            if data_loss.item() < best[0]:
                best = (data_loss.item(), T.detach().cpu().numpy().copy())
            if verbose and (it % 50 == 0 or it == iters - 1):
                print(f"    it {it:4d} loss {loss.item():.4f}", flush=True)
        return best[1], best[0]

    @torch.no_grad()
    def evaluate(self, T: np.ndarray) -> float:
        return float(self.loss_of(to_t(T, device=self.dev)).item())
