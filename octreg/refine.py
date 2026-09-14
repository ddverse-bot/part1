"""Affine ladder (method step 5): refine the search poses coarse to fine with each pose's polarity fixed.

Model x_mri = R(r) Sh(sh) diag(exp(ls)) M (x_oct - c) + t, M = diag(1, 1, -1) for a mirrored pose, c = OCT grid box centre (mm).
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
from .search import EPS, MIRROR, box, to_torch

LADDER = (("rigid", "similarity"), ("rigid", "affine"), ("affine",))     # degrees of freedom run at each ladder level, in order


def rodrigues(r):
    """Rotation vector r (rad, torch [3]) -> rotation matrix exp([r]x) (torch [3, 3], differentiable, also at r = 0)."""
    z = torch.zeros((), dtype=r.dtype, device=r.device)
    K = [torch.stack([z, -r[2], r[1]]), torch.stack([r[2], z, -r[0]]), torch.stack([-r[1], r[0], z])]
    return torch.linalg.matrix_exp(torch.stack(K))


def compose(r, t, ls, sh, c, mirror=False):
    """Torch parameters (r rad, t = image of c in mm, log-scales, shears Sh[0,1], Sh[0,2], Sh[1,2]; each [3]) and c (mm, [3])
    -> 4x4 OCT world -> MRI world (torch, differentiable)."""
    one, z = torch.ones_like(sh[0]), torch.zeros_like(sh[0])
    Sh = torch.stack([torch.stack([one, sh[0], sh[1]]), torch.stack([z, one, sh[2]]), torch.stack([z, z, one])])
    A = rodrigues(r) @ Sh @ torch.diag(torch.exp(ls))
    if mirror:
        A = A * torch.tensor([1.0, 1.0, -1.0], dtype=r.dtype, device=r.device)
    return torch.cat([torch.cat([A, (t - A @ c)[:, None]], 1), torch.eye(4, dtype=r.dtype, device=r.device)[3:]])


def decompose(T, c):
    """Inverse of compose for a 4x4 numpy affine: A M = Q U (QR, diag U > 0) -> (r, t, ls, sh numpy [3], mirror = det A < 0)."""
    A = np.asarray(T, float)[:3, :3]
    mirror = bool(np.linalg.det(A) < 0)
    Q, U = np.linalg.qr(A @ MIRROR if mirror else A)
    d = np.where(np.diag(U) < 0, -1.0, 1.0)
    Q, U = Q * d, d[:, None] * U
    s = np.diag(U)
    if not s.min() > 1e-9 * s.max():
        raise ValueError(f"singular pose\n{T}")
    Sh = U / s
    return (Rotation.from_matrix(Q).as_rotvec(), A @ np.asarray(c, float) + np.asarray(T, float)[:3, 3], np.log(s),
            np.array([Sh[0, 1], Sh[0, 2], Sh[1, 2]]), mirror)


def masked_ncc(a, b, w):
    """Mean over channels of the w-weighted NCC of a and b (torch [C, N], [C, N], [N])."""
    N = w.sum() + EPS
    da, db = a - (a * w).sum(1, keepdim=True) / N, b - (b * w).sum(1, keepdim=True) / N
    return ((w * da * db).sum(1) / torch.sqrt(((w * da * da).sum(1) * (w * db * db).sum(1)).clamp(min=EPS))).mean()


class Level:
    """One pyramid level on the device: OCT voxels with weight > 0 (world points, weights, channels) and the MRI channels and mask."""

    def __init__(self, mri, oct, device="cuda"):
        """mri = (v [2, D, H, W], mask [D, H, W], affine); oct = (u [2, d, h, w], w [d, h, w], affine); numpy or torch."""
        (v, m, self.A_M), (u, w, A_O) = mri, oct
        w = to_torch(w, device)
        idx = torch.nonzero(w > 0)
        if not len(idx):
            raise ValueError("OCT weight is empty on a refinement level")
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


def fit(level, T0, polarity, dof, iters, params: Params = Params()):
    """Adam (lr_rot rad, lr_t mm, lr_ls, lr_sh) with a cosine schedule over iters steps from pose T0 (numpy 4x4) at one level.
    dof 'rigid' (r, t) | 'similarity' (+ one isotropic log-scale) | 'affine' (+ 3 log-scales, 3 shears); the parameters not
    optimised keep their T0 values. -> (T numpy 4x4, S, L) of the iterate with the lowest L."""
    P, dev = params, level.w.device
    r0, t0, ls0, sh0, mirror = decompose(T0, level.c)
    if dof == "similarity":
        ls0 = np.full(3, ls0.mean())
    f = lambda x, grad: torch.tensor(np.asarray(x, float), dtype=torch.float32, device=dev, requires_grad=grad)
    r, t, ls, sh, c = f(r0, True), f(t0, True), f(ls0, dof != "rigid"), f(sh0, dof == "affine"), f(level.c, False)
    groups = [{"params": [r], "lr": P.lr_rot}, {"params": [t], "lr": P.lr_t}]
    groups += [{"params": [ls], "lr": P.lr_ls}] if dof != "rigid" else []
    groups += [{"params": [sh], "lr": P.lr_sh}] if dof == "affine" else []
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.0)
    best = (math.inf, None, None)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        T = compose(r, t, ls.mean().expand(3) if dof == "similarity" else ls, sh, c, mirror)
        S = level.score(T, polarity)
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


def refine(candidates, mri_pyr, oct_pyr, params: Params = Params(), device="cuda", tau=0.0):
    """The ladder over params.levels (coarse to fine): at each level every surviving pose runs LADDER's degrees of freedom in order
    (iters of that level each), poses whose overlap (Level.overlap) fell below tau are dropped (the search's admissibility rule
    holds along the ladder too) and the keep lowest-L poses survive. params.ladder False: one affine stage at levels[-1] from every
    search pose, keep[-1] survive. candidates: search output (T, polarity used); mri_pyr {level_mm: (v, mask, affine)}, oct_pyr
    {level_mm: (u, w, affine)}; tau: the search's overlap threshold. -> (best, finalists): finalists = the keep[-1] survivors of
    the last level, lowest L first, as dicts {T, S, L, polarity, mirror, log_scales, shears, overlap, search_rank}; best =
    finalists[0]. Raises ValueError when no pose keeps overlap >= tau."""
    P = params
    if not candidates:
        raise ValueError("no candidates to refine")
    stages = list(zip(P.levels, LADDER, P.iters, P.keep)) if P.ladder else [(P.levels[-1], ("affine",), P.iters[-1], P.keep[-1])]
    poses = [dict(T=np.asarray(c["T"], float), polarity=int(c["polarity"]), search_rank=i) for i, c in enumerate(candidates)]
    if any(p["polarity"] not in (1, -1) for p in poses):
        raise ValueError("candidate polarity must be +1 or -1")
    for level_mm, dofs, iters, keep in stages:
        level = Level(mri_pyr[level_mm], oct_pyr[level_mm], device)
        for p in poses:
            for dof in dofs:
                p["T"], p["S"], p["L"] = fit(level, p["T"], p["polarity"], dof, iters, P)
            p["overlap"] = level.overlap(p["T"])
        poses = sorted((p for p in poses if p["overlap"] >= tau), key=lambda p: p["L"])[:keep]
        if not poses:
            raise ValueError(f"no pose keeps overlap >= tau = {tau:.3f} along the ladder ({level_mm} mm)")
    for p in poses:
        _, _, ls, sh, mirror = decompose(p["T"], level.c)
        p.update(mirror=mirror, log_scales=ls.tolist(), shears=sh.tolist())
    return poses[0], poses
