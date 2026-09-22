"""Grids, rotations, world-space sampling and the separable Gaussian filtering §5 and §6 share.

Arrays are indexed (i, j, k); an affine maps voxel index -> world mm; a transform T is a 4x4 world -> world matrix (mm).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.transform import Rotation

from . import io

CHUNK = 1 << 22                                  # destination voxels per resample_to chunk (memory only)


def _np(A) -> np.ndarray:
    return A.detach().cpu().double().numpy() if isinstance(A, torch.Tensor) else np.asarray(A, float)


def rotations(n, seed) -> np.ndarray:
    """n rotations uniform on SO(3) from normalised Gaussian quaternions (w, x, y, z) of numpy default_rng(seed); R[0] = identity.
    -> [n, 3, 3] float64."""
    q = np.random.default_rng(seed).normal(size=(n, 4))
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    R[0] = np.eye(3)
    return R


def apply_affine(T, pts):
    """T [4, 4], pts [..., 3] (numpy or torch; the result is of the same kind) -> T applied to the points [..., 3]."""
    if isinstance(pts, torch.Tensor):
        T = torch.as_tensor(T, dtype=pts.dtype, device=pts.device)
        return pts @ T[:3, :3].T + T[:3, 3]
    T = _np(T)
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def resample_iso(vol: io.Volume, level_mm, binary=False):
    """Isotropic grid of spacing level_mm (mm) aligned with the volume axes, streamed plane by plane (memory: the output plus two
    planes). Per axis box average over k = max(1, round(level_mm / spacing)) voxels (trailing remainder dropped), then trilinear
    onto floor(pooled extent / level_mm) voxels from the first pooled voxel centre, blending with 0 beyond the last. k and the
    grid size use the spacing to 6 significant digits (a float32 20 um header gives an exact 2^3 mean at 0.04 mm). binary:
    average the indicator voxel > 0 (a mask file). -> (float32 [d, h, w], affine)."""
    planes = ((k, p > 0) for k, p in io.iter_planes(vol)) if binary else io.iter_planes(vol)
    return _iso(planes, vol.shape, vol.affine, level_mm)


def pool_iso(arr, affine, level_mm):
    """resample_iso for an in-memory array [D, H, W] or [C, D, H, W] on the grid `affine`, whose column norms are its voxel
    edge. -> (float32 array, affine)."""
    arr = np.asarray(arr)
    if arr.ndim == 4:
        outs = [pool_iso(a, affine, level_mm) for a in arr]
        return np.stack([o for o, _ in outs]), outs[0][1]
    return _iso(((k, arr[:, :, k]) for k in range(arr.shape[2])), arr.shape, affine, level_mm)


def _iso(planes, shape, affine, level_mm):
    A = _np(affine)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    spg = np.array([float(f"{s:.6g}") for s in sp])                            # grid arithmetic ignores float32 header noise
    k = np.maximum(1, np.rint(np.round(level_mm / spg, 9))).astype(int)       # round half to even
    n = np.array(shape[:3]) // k
    step = level_mm / (spg * k)                                                # output step in pooled voxels
    N = np.floor(np.round(n / step, 9)).astype(int)
    if N.min() < 2:
        raise ValueError(f"volume {tuple(shape)} with spacing {sp} mm has fewer than 2 voxels per axis at {level_mm} mm")
    A_out = A.copy()
    A_out[:3, 3] += A[:3, :3] @ ((k - 1) / 2.0)
    A_out[:3, :3] = A[:3, :3] / sp * level_mm
    taps = []
    for a in range(3):
        u = np.arange(N[a]) * step[a]
        i0 = np.minimum(np.floor(u).astype(int), n[a] - 1)
        taps.append((i0, u - i0))
    out, stream = np.empty(N, np.float32), _pooled(planes, k, n)
    lo = hi = None
    have = -1                                                                  # pooled planes have - 1 (lo) and have (hi) held
    for m, (q, f) in enumerate(zip(*taps[2])):
        while have < min(q + 1, n[2] - 1):
            lo, hi, have = hi, next(stream), have + 1
        plane = lo * (1 - f) + hi * f if have > q else hi * (1 - f)            # past the last pooled plane: blend with 0
        out[:, :, m] = _lin(_lin(plane, 0, *taps[0]), 1, *taps[1])
    return out, A_out


def _pooled(planes, k, n):
    """float64 [n0, n1] box means of consecutive k[2] planes of the (index, plane [D, H]) stream."""
    acc = 0.0
    for q, plane in planes:
        acc = acc + np.asarray(plane[:n[0] * k[0], :n[1] * k[1]], np.float64).reshape(n[0], k[0], n[1], k[1]).sum((1, 3))
        if q % k[2] == k[2] - 1:
            yield acc / k.prod()
            acc = 0.0


def _lin(x, axis, i0, f):
    """Linear interpolation of the 2-D x along axis at i0 + f, zero beyond the last voxel."""
    if len(i0) == x.shape[axis] and not f.any():
        return x
    x = np.concatenate([x, np.zeros_like(x.take([0], axis))], axis)
    f = f.reshape((-1, 1) if axis == 0 else (1, -1))
    return x.take(i0, axis) * (1 - f) + x.take(i0 + 1, axis) * f


def sample_world(field, affine, pts):
    """Trilinear sample of field (torch [C, D, H, W], voxel -> world affine) at world points pts (torch [..., 3], mm);
    torch grid_sample with align_corners=True, blending with 0 outside. -> torch [C, ...]."""
    C, D, H, W = field.shape
    Ainv = torch.as_tensor(np.linalg.inv(_np(affine)), dtype=pts.dtype, device=pts.device)
    ijk = pts @ Ainv[:3, :3].T + Ainv[:3, 3]
    size = torch.tensor([D - 1, H - 1, W - 1], dtype=pts.dtype, device=pts.device)
    grid = (ijk / size * 2 - 1)[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3).to(field.dtype)
    out = F.grid_sample(field[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(C, *pts.shape[:-1])


def resample_to(arr, affine_src, shape_dst, affine_dst, T_dst_world_to_src_world, order=1, field=None, affine_field=None):
    """Resample arr (numpy [D, H, W], voxel -> world affine_src) onto the grid (shape_dst, affine_dst): every destination
    voxel goes to world, through T to the source world and is sampled there, order 1 (trilinear, float32 out) or 0 (nearest,
    arr dtype out); 0 outside (scipy map_coordinates 'grid-constant', the same values as sample_world). -> numpy shape_dst.
    field (numpy [3, d, h, w], mm along the destination world axes, on the grid affine_field in the destination world): a
    pull-back displacement (octreg.deform): the destination point x is sampled at T (x + u(x)), u trilinear in the field and
    constant beyond its grid. CHUNK destination voxels at a time, so the memory does not grow with the destination grid."""
    arr = np.asarray(arr)
    src = arr.view(np.uint8) if arr.dtype == bool else arr
    A_d, B = _np(affine_dst), np.linalg.inv(_np(affine_src)) @ _np(T_dst_world_to_src_world)   # B: destination world -> source voxel
    M = B @ A_d                                                                                # destination -> source voxel
    shape = tuple(int(s) for s in shape_dst)
    out = np.zeros(shape, np.float32 if order else src.dtype)
    jk = np.stack(np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij"), 0).reshape(2, -1).astype(float)
    step = max(1, (CHUNK // (1 if field is None else 4)) // jk.shape[1])
    if field is not None:
        field, Fm = np.asarray(field, np.float32), np.linalg.inv(_np(affine_field)) @ A_d      # destination -> field voxel
    for i0 in range(0, shape[0], step):
        i = np.repeat(np.arange(i0, min(shape[0], i0 + step), dtype=float), jk.shape[1])
        c = M[:3, :1] * i + np.tile(M[:3, 1:3] @ jk + M[:3, 3:], len(i) // jk.shape[1])
        if field is not None:
            f = Fm[:3, :1] * i + np.tile(Fm[:3, 1:3] @ jk + Fm[:3, 3:], len(i) // jk.shape[1])
            c += B[:3, :3] @ np.stack([ndimage.map_coordinates(u, f, output=np.float64, order=1, mode="nearest") for u in field])
        out[i0:i0 + len(i) // jk.shape[1]] = ndimage.map_coordinates(src, c, output=out.dtype, order=order, mode="grid-constant",
                                                                     cval=0).reshape(-1, shape[1], shape[2])
    return out.view(bool) if arr.dtype == bool else out

def gauss_1d(sigma_vox, order, device):
    """1-D Gaussian of sigma_vox voxels, or its first derivative scaled to be exact on a ramp (torch, on device)."""
    r = int(math.ceil(4 * sigma_vox))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    g = g / g.sum()
    if order == 0:
        return g
    d = x * g                                          # -G'(x) up to a factor (conv3d correlates), scaled to be exact on ramps
    return d / (x * d).sum()


def filter_sep(vol, sigma_vox, orders):
    """Separable Gaussian (derivative orders per axis, 0 or 1) of vol [D, H, W] (torch), zero padding."""
    out = vol[None, None]
    for ax in range(3):
        k = gauss_1d(sigma_vox, orders[ax], vol.device)
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = len(k) // 2
        out = F.conv3d(F.pad(out, pad), k.reshape(shape))
    return out[0, 0]


def nc_gradient(xm, m, sigma_vox):
    """Gradient of the normalised convolution G(x m) / G(m) at sigma_vox voxels, in array axes, zero outside the mask: the
    quotient rule on filter_sep, so that the edge of the mask adds no gradient. xm is x already multiplied by m (torch)."""
    num, den = filter_sep(xm, sigma_vox, (0, 0, 0)), filter_sep(m, sigma_vox, (0, 0, 0)).clamp(min=1e-3)
    return torch.stack([(filter_sep(xm, sigma_vox, o) * den - num * filter_sep(m, sigma_vox, o)) / (den * den)
                        for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]) * m

