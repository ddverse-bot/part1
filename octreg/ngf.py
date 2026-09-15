"""Fine-structure refinement (docs/METHOD.md §5): the best pose of §4 is refined on the internal structure of both volumes,
compared by normalised gradient fields (NGF; Haber and Modersitzki 2006), which need neither an intensity mapping nor the
polarity.

On its base grid each volume I with mask m is smoothed and differentiated inside the mask by normalised convolution,
g = grad(G_s(I m) / G_s(m)), so the mask edge adds no gradient; g is taken to world coordinates and n = g / sqrt(|g|^2 + eps^2), eps the median |g| over the MRI points used and over the OCT specimen mask. Both masks are eroded by ngf_erode_mm, so the outlines are left to §2-4. Over the MRI interior
points x (every second base voxel), with the OCT interior weight w,
    F(T) = sum_x w(T^-1 x) (n_O(x) . n_M(x))^2 / sum_x w(T^-1 x),   g_O(x) = A^-T g_OCT(T^-1 x) (A the linear part of T),
and L = 1 - F + lam (sum ls^2 + sum sh^2) (refine.compose, the prior of §4) is minimised with Adam from the §4 pose, one pass
per sigma in ngf_sigmas_mm (coarse to fine), learning rates a fifth of those of §4, |ls|, |sh| <= clamp; each pass keeps its
lowest-L iterate.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F_
from scipy import ndimage

from . import geometry as G
from .params import Params
from .refine import compose, decompose
from .search import box, to_torch

LR_FRACTION = 0.2          # learning rates of §5 as a fraction of those of §4 (the pose starts within a few mm)


def _gauss_1d(sigma_vox, order, device):
    r = int(math.ceil(4 * sigma_vox))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    g = g / g.sum()
    if order == 0:
        return g
    d = x * g                                          # -G'(x) up to a factor (conv3d correlates), scaled to be exact on ramps
    return d / (x * d).sum()


def _filter(vol, sigma_vox, orders):
    """Separable Gaussian (derivative orders per axis, 0 or 1) of vol [D, H, W] (torch), zero padding."""
    out = vol[None, None]
    for ax in range(3):
        k = _gauss_1d(sigma_vox, orders[ax], vol.device)
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = len(k) // 2
        out = F_.conv3d(F_.pad(out, pad), k.reshape(shape))
    return out[0, 0]


def gradient_field(arr, mask, affine, sigma_mm, device="cuda"):
    """World-frame gradient [3, D, H, W] (torch, per mm) of arr smoothed with sigma_mm inside mask by normalised convolution;
    zero outside the mask. arr, mask on an isotropic grid (affine voxel -> world)."""
    h = float(np.linalg.norm(np.asarray(affine)[:3, 0]))
    s = sigma_mm / h
    x, m = to_torch(arr, device), to_torch(mask, device)
    num, den = _filter(x * m, s, (0, 0, 0)), _filter(m, s, (0, 0, 0)).clamp(min=1e-3)
    g = torch.stack([(_filter(x * m, s, o) * den - num * _filter(m, s, o)) / (den * den)
                     for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]) * m
    Ainv = torch.as_tensor(np.linalg.inv(np.asarray(affine, float)[:3, :3]).T, dtype=torch.float32, device=device)
    return torch.einsum("ij,jklm->iklm", Ainv, g)


class NGFGrid:
    """F(T) at one sigma: the MRI interior points and their normalised gradients, and the OCT gradient field and interior weight."""

    def __init__(self, mri, oct, sigma_mm, params: Params = Params(), device="cuda"):
        """mri = (arr, foreground, affine), oct = (arr, specimen mask, affine) on isotropic base grids (numpy)."""
        (m_arr, m_mask, A_M), (o_arr, o_mask, self.A_O) = mri, oct
        h_m, h_o = (float(np.linalg.norm(np.asarray(A)[:3, 0])) for A in (A_M, self.A_O))
        inner_m = ndimage.binary_erosion(m_mask, iterations=max(1, round(params.ngf_erode_mm / h_m)))
        inner_m[1::2] = inner_m[:, 1::2] = inner_m[:, :, 1::2] = False                       # every second voxel
        inner_o = ndimage.binary_erosion(o_mask, iterations=max(1, round(params.ngf_erode_mm / h_o)))
        if not inner_m.any() or not inner_o.any():
            raise ValueError("§5: a mask has no interior after erosion by ngf_erode_mm")
        gM = gradient_field(m_arr, m_mask, A_M, sigma_mm, device)
        idx = torch.as_tensor(np.argwhere(inner_m), device=device)
        g = gM[(slice(None),) + tuple(idx.T)]
        del gM
        self.nM = g / torch.sqrt((g * g).sum(0) + g.norm(dim=0).median() ** 2)
        self.pts = G.apply_affine(np.asarray(A_M, float), idx.to(torch.float32))
        self.gO = gradient_field(o_arr, o_mask, self.A_O, sigma_mm, device)
        mag = self.gO.norm(dim=0)
        self.eps_O = mag[to_torch(o_mask, device) > 0].median()
        self.w = to_torch(inner_o, device)[None]
        self.c = box(self.A_O, np.shape(o_arr))[0]

    def F(self, T):
        """F of pose T (torch 4x4)."""
        Ti = torch.linalg.inv(T)
        x = G.apply_affine(Ti, self.pts)
        g = Ti[:3, :3].T @ G.sample_world(self.gO, self.A_O, x)
        w = G.sample_world(self.w, self.A_O, x)[0]
        n = g / torch.sqrt((g * g).sum(0) + self.eps_O ** 2)
        return (w * ((n * self.nM).sum(0)) ** 2).sum() / w.sum().clamp(min=1.0)


def fit_ngf(grid: NGFGrid, T0, params: Params = Params()):
    """Adam on L = 1 - F + lam (sum ls^2 + sum sh^2) from T0 (numpy 4x4). -> (T numpy, F, L) of the lowest-L iterate."""
    P, dev = params, grid.w.device
    f = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float32, device=dev, requires_grad=True)
    r, t, ls, sh = (f(x) for x in decompose(T0, grid.c))
    c = torch.tensor(np.asarray(grid.c, float), dtype=torch.float32, device=dev)
    k = LR_FRACTION
    opt = torch.optim.Adam([{"params": [r], "lr": k * P.lr_rot}, {"params": [t], "lr": k * P.lr_t},
                            {"params": [ls], "lr": k * P.lr_ls}, {"params": [sh], "lr": k * P.lr_sh}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.ngf_iters, eta_min=0.0)
    best = (math.inf,)
    for _ in range(P.ngf_iters + 1):
        opt.zero_grad(set_to_none=True)
        T = compose(r, t, ls, sh, c)
        Fv = grid.F(T)
        L = 1.0 - Fv + P.lam * ((ls ** 2).sum() + (sh ** 2).sum())
        l, fv = float(L.detach()), float(Fv.detach())
        if l < best[0]:
            best = (l, T.detach().cpu().double().numpy(), fv)
        L.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            ls.clamp_(-P.clamp, P.clamp)
            sh.clamp_(-P.clamp, P.clamp)
    return best[1], best[2], best[0]


def refine_ngf(mri, oct, T0, params: Params = Params(), device="cuda"):
    """§5 from the §4 pose T0: one fit_ngf pass per sigma in params.ngf_sigmas_mm. mri = (arr, foreground, affine), oct = (arr,
    specimen mask, affine) on the base grids. -> (T, {F_start, F at the finest sigma, L, sigmas, seconds})."""
    t0, T, info = time.time(), np.asarray(T0, float), {"sigmas_mm": list(params.ngf_sigmas_mm)}
    for i, s in enumerate(params.ngf_sigmas_mm):
        grid = NGFGrid(mri, oct, s, params, device)
        if i == len(params.ngf_sigmas_mm) - 1:
            with torch.no_grad():
                info["F_start"] = float(grid.F(torch.tensor(np.asarray(T0, float), dtype=torch.float32, device=grid.w.device)))
        T, Fv, L = fit_ngf(grid, T, params)
        del grid
    info.update(F=Fv, L=L, seconds=time.time() - t0)
    return T, info
