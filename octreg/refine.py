"""Affine refinement (method step 5): every search pose is fitted on the base grid with its polarity fixed, and the lowest loss
wins.

Model x_mri = R(r) Sh(sh) diag(exp(ls)) (x_oct - c) + t, c = OCT grid box centre (mm); no mirror (handedness from the file frames).
S = (2 S_class + S_outline) / 3 as in the search: S_class the mean over the two channel pairs of the NCC between the OCT channels
at the specimen voxels (swapped for polarity -1) and the MRI channels sampled through the pose; S_outline the NCC between the
specimen mask at the measured OCT voxels and the MRI foreground sampled through the pose. L = 1 - S + lam (sum ls^2 + sum sh^2);
after every Adam step (cosine schedule) |ls|, |sh| <= clamp (absolute).
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
    """The base grid on the device: the measured OCT voxels (world points, measured fraction, specimen mask, channels) and the
    MRI channels and foreground."""

    def __init__(self, mri, oct, device="cuda"):
        """mri = (v [2, D, H, W], mask [D, H, W], affine); oct = (u [2, d, h, w], w [d, h, w] specimen mask, measured fraction
        [d, h, w], affine); numpy or torch."""
        (v, m, self.A_M), (u, w, q, A_O) = mri, oct
        w, q = to_torch(w, device), to_torch(q, device)
        idx = torch.nonzero(q > 0)
        if not float(w.sum()) > 0:
            raise ValueError("OCT specimen mask is empty on the base grid")
        self.c = box(A_O, w.shape)[0]
        self.pts = G.apply_affine(np.asarray(A_O, float), idx.to(torch.float32))
        self.q, self.w = q[tuple(idx.T)], w[tuple(idx.T)]
        self.u = to_torch(u, device)[(slice(None),) + tuple(idx.T)]
        self.spec = self.w > 0
        self.v = torch.cat([to_torch(v, device), to_torch(m, device)[None]])

    def score(self, T, polarity):
        """(S, S_class signed, S_outline) of pose T (torch 4x4); S = (2 polarity S_class + S_outline) / 3."""
        s = G.sample_world(self.v, self.A_M, G.apply_affine(T, self.pts))
        S_class = masked_ncc(s[:2, self.spec], self.u[:, self.spec], self.w[self.spec])
        S_outline = masked_ncc(s[2:], self.w[None], self.q)
        return (2 * polarity * S_class + S_outline) / 3, S_class, S_outline


def fit(grid, T0, polarity, params: Params = Params()):
    """Adam (lr_rot rad, lr_t mm, lr_ls, lr_sh) with a cosine schedule over params.iters steps from pose T0 (numpy 4x4), all
    twelve affine parameters (r, t, 3 log-scales, 3 shears). -> (T numpy 4x4, S, S_class, S_outline, L) of the iterate with the
    lowest L."""
    P, dev = params, grid.w.device
    f = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float32, device=dev, requires_grad=True)
    r, t, ls, sh = (f(x) for x in decompose(T0, grid.c))
    c = torch.tensor(np.asarray(grid.c, float), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([{"params": [r], "lr": P.lr_rot}, {"params": [t], "lr": P.lr_t}, {"params": [ls], "lr": P.lr_ls},
                            {"params": [sh], "lr": P.lr_sh}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.iters, eta_min=0.0)
    best = (math.inf,)
    for _ in range(P.iters):
        opt.zero_grad(set_to_none=True)
        T = compose(r, t, ls, sh, c)
        S, S_class, S_outline = grid.score(T, polarity)
        L = 1.0 - S + P.lam * ((ls ** 2).sum() + (sh ** 2).sum())
        L.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            ls.clamp_(-P.clamp, P.clamp)
            sh.clamp_(-P.clamp, P.clamp)
        l, *s = torch.stack([L.detach(), S.detach(), S_class.detach(), S_outline.detach()]).tolist()
        if l < best[0]:
            best = (l, T.detach().cpu().double().numpy(), *s)
    return (*best[1:], best[0])


def refine(candidates, mri, oct, params: Params = Params(), device="cuda"):
    """Fit every search pose (fit, its polarity fixed) on the base grid and sort by L. candidates: search output (T, polarity);
    mri = (v [2, D, H, W], mask, affine), oct = (u [2, d, h, w], w, measured fraction, affine) on the base grid.
    -> [{T, S, S_class, S_outline, L, polarity, log_scales, shears, search_rank}] lowest L first."""
    if not candidates or any(c["polarity"] not in (1, -1) for c in candidates):
        raise ValueError("refine needs candidates with polarity +1 or -1")
    grid, poses = BaseGrid(mri, oct, device), []
    for i, c in enumerate(candidates):
        T, S, S_class, S_outline, L = fit(grid, np.asarray(c["T"], float), int(c["polarity"]), params)
        _, _, ls, sh = decompose(T, grid.c)
        poses.append(dict(T=T, S=S, S_class=S_class, S_outline=S_outline, L=L, polarity=int(c["polarity"]),
                          log_scales=ls.tolist(), shears=sh.tolist(), search_rank=i))
    return sorted(poses, key=lambda p: p["L"])
