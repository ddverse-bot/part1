"""Geometry and resampling: rotations, the affine model, world-space sampling, pooling, pose distances.

Arrays are indexed (i, j, k) in file order; an affine maps voxel index -> world mm; transforms are 4x4 world -> world.
torch tensors for everything differentiable or on the GPU, numpy elsewhere (each function says which).
"""
from __future__ import annotations

import numpy as np


def rotations(n, seed) -> np.ndarray:
    """n rotations uniform on SO(3) from unit quaternions of numpy default_rng(seed); R[0] = identity. -> [n, 3, 3] float64."""
    raise NotImplementedError


def rodrigues(r):
    """Rotation vector r (rad, torch [3]) -> rotation matrix (torch [3, 3]), differentiable, finite at r = 0."""
    raise NotImplementedError


def compose(r, t, ls, sh, c, mirror=False):
    """Refinement model x_mri = R(r) Sh(sh) diag(exp(ls)) M (x_oct - c) + t, with M = diag(1, 1, -1) if mirror else I.

    r: rotation vector (rad) [3]; t: image of c (mm) [3]; ls: log-scales [3]; sh: shears (Sh[0,1], Sh[0,2], Sh[1,2]) [3];
    c: OCT foreground centroid (mm) [3]; torch tensors. -> torch [4, 4] OCT world -> MRI world (mm), differentiable.
    """
    raise NotImplementedError


def decompose(T, c):
    """Inverse of compose: T numpy 4x4 (det != 0), c (mm) [3] -> (r, t, ls, sh, mirror) numpy [3] x 4 and bool (mirror = det < 0).
    Exact round trip for any T = compose(...); any other affine is represented exactly as well (QR with positive diagonal)."""
    raise NotImplementedError


def apply_affine(A, pts):
    """A [4, 4], pts [..., 3] (numpy or torch, result of the same kind) -> A applied to the points [..., 3]."""
    raise NotImplementedError


def sample_at_world(vol, affine, pts, mode="bilinear"):
    """Sample vol (torch [C, D, H, W]) with voxel -> world affine [4, 4] at world points pts (torch [..., 3], mm) by trilinear
    ('bilinear') or 'nearest' interpolation (torch grid_sample, align_corners=True). Outside the volume = 0. -> torch [C, ...]."""
    raise NotImplementedError


def box_pool(vol, affine, k):
    """Box average over k = (k0, k1, k2) voxels per axis (int or triple; trailing remainder dropped), affine moved to the pooled
    voxel centres. vol: numpy [D, H, W] (chunked, float32 out) or torch [C, D, H, W]. -> (pooled, affine 4x4 numpy)."""
    raise NotImplementedError


def resample(vol, affine, level_mm, device="cpu"):
    """Exactly isotropic grid of spacing level_mm aligned with vol's own array axes: per axis box average over
    k = max(1, round(level_mm / spacing)) voxels, then trilinear onto the isotropic grid whose origin is the first pooled voxel
    centre, floor(pooled extent / level_mm) voxels per axis (v1 prep_subject.py L181-184).
    vol: array-like [D, H, W] read plane by plane (memmap / plane reader), affine: 4x4 voxel -> world mm.
    -> (float32 numpy [d, h, w], affine 4x4 voxel -> world mm)."""
    raise NotImplementedError


def gaussian(vol, sigma_mm, spacing_mm, truncate):
    """Separable Gaussian blur with sigma_mm (mm) on a grid of spacing_mm (mm, scalar or per axis), kernel truncated at
    truncate sigmas (Params.gauss_truncate), replicate padding. vol: numpy [D, H, W] or torch [C, D, H, W]; same kind out."""
    raise NotImplementedError


def pose_distance(T1, T2, points) -> dict:
    """Difference of two 4x4 poses measured on points (numpy [N, 3], OCT world mm, e.g. foreground points).
    -> {'mean_mm', 'max_mm': displacement |T1 x - T2 x| over the points; 'rot_deg': angle (deg) of R1 R2^T with R_i the rotation
    factors of decompose (mirror diag(1, 1, -1) in OCT axes removed first, so a mirrored pair gets the angle of its proper parts),
    None if a linear part is singular; 'handedness_differs': det(A1) det(A2) < 0}."""
    raise NotImplementedError


def radius_of_gyration(points, weights=None) -> float:
    """sqrt(weighted mean squared distance to the weighted centroid) of points (numpy [N, 3], mm) -> mm."""
    raise NotImplementedError


def apply_transform(moving, reference, T, nearest=False, device="cpu") -> np.ndarray:
    """Resample an image onto a reference grid through x_ref = T x_moving (4x4 world -> world mm).
    moving: (array [D, H, W], affine 4x4); reference: (shape (3 ints), affine 4x4). Trilinear, or nearest for labels; chunked
    over the reference's first axis. Outside the moving image = 0. -> numpy array of reference shape (float32, or moving dtype
    when nearest)."""
    raise NotImplementedError
