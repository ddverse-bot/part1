"""Affine refinement (method step 5): every search pose is fitted on the base grid with its polarity fixed, poses that leave
the MRI foreground are dropped, and the lowest loss wins.

Model x_mri = R(r) Sh(sh) diag(exp(ls)) (x_oct - c) + t, c = OCT grid box centre (mm); no mirror (handedness from the file frames).
S = mean over the two channel pairs of the OCT-weighted NCC between the OCT channels at the OCT voxels with weight > 0 (swapped
for polarity -1) and the MRI channels sampled through the pose. L = 1 - S + lam (sum ls^2 + sum sh^2); after every Adam step
(cosine schedule) |ls|, |sh| <= clamp (absolute).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import geometry as G
from .params import Params
from .search import EPS, box, to_torch


def compose(r, t, ls, sh, c):
    """Torch parameters (rotation vector r rad, t = image of c in mm, log-scales, shears Sh[0,1], Sh[0,2], Sh[1,2]; each [3]) and
    c (mm, [3]) -> 4x4 OCT world -> MRI world (torch, differentiable; the rotation is exp([r]x), also at r = 0)."""
    z, one = torch.zeros_like(r[0]), torch.ones_like(r[0])
    K = torch.stack([torch.stack([z, -r[2], r[1]]), torch.stack([r[2], z, -r[0]]), torch.stack([-r[1], r[0], z])])
    Sh = torch.stack([torch.stack([one, sh[0], sh[1]]), torch.stack([z, one, sh[2]]), torch.stack([z, z, one])])
    A = torch.linalg.matrix_exp(K) @ Sh @ torch.diag(torch.exp(ls))
    return torch.cat([torch.cat([A, (t - A @ c)[:, None]], 1), torch.eye(4, dtype=r.dtype, device=r.device)[3:]])


def decompose(T, c):
    """Inverse of compose for a 4x4 numpy affine: A = Q U (QR, diag U > 0) -> (r, t, ls, sh numpy [3]). A mirrored pose is refused."""
    A = np.asarray(T, float)[:3, :3]
    if np.linalg.det(A) <= 0:
        raise ValueError(f"mirrored or singular pose (det <= 0): the handedness comes from the file frames\n{T}")
    Q, U = np.linalg.qr(A)
    d = np.where(np.diag(U) < 0, -1.0, 1.0)
    Q, U = Q * d, d[:, None] * U
    s = np.diag(U)
    if not s.min() > 1e-9 * s.max():
        raise ValueError(f"singular pose\n{T}")
    Sh = U / s
    return (Rotation.from_matrix(Q).as_rotvec(), A @ np.asarray(c, float) + np.asarray(T, float)[:3, 3], np.log(s),
            np.array([Sh[0, 1], Sh[0, 2], Sh[1, 2]]))


def masked_ncc(a, b, w):
    """Mean over channels of the w-weighted NCC of a and b (torch [C, N], [C, N], [N])."""
    N = w.sum() + EPS
    da, db = a - (a * w).sum(1, keepdim=True) / N, b - (b * w).sum(1, keepdim=True) / N
    return ((w * da * db).sum(1) / torch.sqrt(((w * da * da).sum(1) * (w * db * db).sum(1)).clamp(min=EPS))).mean()


class BaseGrid:
    """The base grid on the device: OCT voxels with weight > 0 (world points, weights, channels) and the MRI channels and mask."""

    def __init__(self, mri, oct, device="cuda"):
        """mri = (v [2, D, H, W], mask [D, H, W], affine); oct = (u [2, d, h, w], w [d, h, w], affine); numpy or torch."""
        (v, m, self.A_M), (u, w, A_O) = mri, oct
        w = to_torch(w, device)
        idx = torch.nonzero(w > 0)
        if not len(idx):
            raise ValueError("OCT weight is empty on the base grid")
        self.c = box(A_O, w.shape)[0]
        self.pts = G.apply_affine(np.asarray(A_O, float), idx.to(torch.float32))
        self.w = w[tuple(idx.T)]
        self.u = to_torch(u, device)[(slice(None),) + tuple(idx.T)]
        self.v, self.m = to_torch(v, device), to_torch(m, device)[None]

    def score(self, T, polarity):
        """S of pose T (torch 4x4) with the OCT channels swapped for polarity -1."""
        a = self.u if polarity == 1 else self.u.flip(0)
        return masked_ncc(G.sample_world(self.v, self.A_M, G.apply_affine(T, self.pts)), a, self.w)

    @torch.no_grad()
    def overlap(self, T):
        """sum(w M(T x)) / sum(w) of pose T (numpy 4x4): fraction of the OCT weight on MRI foreground."""
        T = torch.as_tensor(np.asarray(T, float), dtype=torch.float32, device=self.w.device)
        return float((G.sample_world(self.m, self.A_M, G.apply_affine(T, self.pts))[0] * self.w).sum() / self.w.sum())


def fit(grid, T0, polarity, params: Params = Params()):
    """Adam (lr_rot rad, lr_t mm, lr_ls, lr_sh) with a cosine schedule over params.iters steps from pose T0 (numpy 4x4), all
    twelve affine parameters (r, t, 3 log-scales, 3 shears). -> (T numpy 4x4, S, L) of the iterate with the lowest L."""
    P, dev = params, grid.w.device
    f = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float32, device=dev, requires_grad=True)
    r, t, ls, sh = (f(x) for x in decompose(T0, grid.c))
    c = torch.tensor(np.asarray(grid.c, float), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([{"params": [r], "lr": P.lr_rot}, {"params": [t], "lr": P.lr_t}, {"params": [ls], "lr": P.lr_ls},
                            {"params": [sh], "lr": P.lr_sh}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.iters, eta_min=0.0)
    best = (math.inf, None, None)
    for _ in range(P.iters):
        opt.zero_grad(set_to_none=True)
        T = compose(r, t, ls, sh, c)
        S = grid.score(T, polarity)
        L = 1.0 - S + P.lam * ((ls ** 2).sum() + (sh ** 2).sum())
        L.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            ls.clamp_(-P.clamp, P.clamp)
            sh.clamp_(-P.clamp, P.clamp)
        l, s = torch.stack([L.detach(), S.detach()]).tolist()
        if l < best[0]:
            best = (l, T.detach().cpu().double().numpy(), s)
    return best[1], best[2], best[0]


def refine(candidates, mri, oct, params: Params = Params(), device="cuda", tau=0.0):
    """Fit every search pose (fit, its polarity fixed) on the base grid, drop the poses whose overlap (BaseGrid.overlap) is below
    tau (the search's admissibility rule also holds after refinement) and sort by L. candidates: search output (T, polarity);
    mri = (v [2, D, H, W], mask, affine), oct = (u [2, d, h, w], w, affine) on the base grid; tau: the search's overlap threshold.
    -> [{T, S, L, polarity, log_scales, shears, overlap, search_rank}] lowest L first.
    Raises ValueError when no pose keeps overlap >= tau."""
    if not candidates or any(c["polarity"] not in (1, -1) for c in candidates):
        raise ValueError("refine needs candidates with polarity +1 or -1")
    grid, poses = BaseGrid(mri, oct, device), []
    for i, c in enumerate(candidates):
        T, S, L = fit(grid, np.asarray(c["T"], float), int(c["polarity"]), params)
        _, _, ls, sh = decompose(T, grid.c)
        poses.append(dict(T=T, S=S, L=L, polarity=int(c["polarity"]), log_scales=ls.tolist(), shears=sh.tolist(),
                          overlap=grid.overlap(T), search_rank=i))
    poses = sorted((p for p in poses if p["overlap"] >= tau), key=lambda p: p["L"])
    if not poses:
        raise ValueError(f"no refined pose keeps overlap >= tau = {tau:.3f}")
    return poses
