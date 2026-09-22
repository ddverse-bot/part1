"""Fine-structure refinement (docs/METHOD.md §5): the best pose of §4 is refined on the internal structure of both volumes,
compared by normalised gradient fields (NGF; Haber and Modersitzki 2006), which need neither an intensity mapping nor the
polarity.

On its base grid each volume I with mask m is smoothed and differentiated inside the mask by normalised convolution,
g = grad(G_s(I m) / G_s(m)), so the mask edge adds no gradient; g is taken to world coordinates and n = g / sqrt(|g|^2 + eps^2), eps the median |g| over the MRI points used and over the OCT specimen mask. Both masks are eroded by ngf_erode_mm, so the outlines are left to §2-4. Over the MRI interior
points x (every second interior MRI voxel along each axis), with the OCT interior weight w,
    F(T) = sum_x w(T^-1 x) (n_O(x) . n_M(x))^2 / sum_x w(T^-1 x),   g_O(x) = A^-T g_OCT(T^-1 x) (A the linear part of T),
and L = 1 - F + lam (sum ls^2 + sum sh^2) (refine.compose, the prior of §4) is minimised with Adam from the §4 pose, one pass
per sigma in ngf_sigmas_mm (coarse to fine), learning rates a fifth of those of §4, |ls|, |sh| <= clamp; each pass keeps its
lowest-L iterate.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy import ndimage

from . import geometry as G
from .params import Params
from .refine import fit_adam
from .search import box, to_torch

LR_FRACTION = 0.2          # learning rates of §5 as a fraction of those of §4 (the pose starts within a few mm)


def gradient_field(arr, mask, affine, sigma_mm, device="cuda"):
    """World-frame gradient [3, D, H, W] (torch, per mm) of arr smoothed with sigma_mm inside mask by normalised convolution;
    zero outside the mask. arr, mask on an isotropic grid (affine voxel -> world)."""
    h = float(np.linalg.norm(np.asarray(affine)[:3, 0]))
    s = sigma_mm / h
    x, m = to_torch(arr, device), to_torch(mask, device)
    g = G.nc_gradient(x * m, m, s)
    Ainv = torch.as_tensor(np.linalg.inv(np.asarray(affine, float)[:3, :3]).T, dtype=torch.float32, device=device)
    return torch.einsum("ij,jklm->iklm", Ainv, g)


class NGFGrid:
    """F(T) at one sigma: the MRI interior points and their normalised gradients, and the OCT gradient field and interior weight."""

    def __init__(self, mri, oct, sigma_mm, params: Params = Params(), device="cuda"):
        """mri = (arr, foreground, affine), oct = (arr, specimen mask, affine) on isotropic base grids (numpy)."""
        (m_arr, m_mask, A_M), (o_arr, o_mask, self.A_O) = mri, oct
        h_m, h_o = (float(np.linalg.norm(np.asarray(A)[:3, 0])) for A in (A_M, self.A_O))
        inner_m = ndimage.binary_erosion(m_mask, iterations=max(1, round(params.ngf_erode_mm / h_m)))
        inner_m[1::2] = inner_m[:, 1::2] = inner_m[:, :, 1::2] = False                       # every second voxel along each axis
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


def refine_ngf(mri, oct, T0, params: Params = Params(), device="cuda"):
    """§5 from the §4 pose T0: one refine.fit_adam pass on F per sigma in params.ngf_sigmas_mm. mri = (arr, foreground, affine), oct = (arr,
    specimen mask, affine) on the base grids. -> (T, {F_start, F at the finest sigma, L, sigmas, seconds})."""
    t0, T, info = time.time(), np.asarray(T0, float), {"sigmas_mm": list(params.ngf_sigmas_mm)}
    for i, s in enumerate(params.ngf_sigmas_mm):
        grid = NGFGrid(mri, oct, s, params, device)
        if i == len(params.ngf_sigmas_mm) - 1:
            with torch.no_grad():
                info["F_start"] = float(grid.F(torch.tensor(np.asarray(T0, float), dtype=torch.float32, device=grid.w.device)))
        T, Fv, L = fit_adam(grid, T, lambda x: (grid.F(x),), params.ngf_iters + 1, params.ngf_iters,
                            LR_FRACTION, params)
        del grid
    info.update(F=Fv, L=L, seconds=time.time() - t0)
    return T, info
