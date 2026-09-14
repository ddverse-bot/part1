"""Orientation search (method step 4) on the search grid (Params.search_mm).

For every rotation of a fixed uniform set the OCT template is correlated with the MRI crop over all translations by FFT
(weighted NCC with a Padfield-style mask). Rotations only: two physical specimens are never mirror images, so the handedness is
that of the input file frames and is not searched (a mirrored stack has to be fixed in its header).
The score is the mean over three channel pairs. Two are the two-class maps: the OCT channels (p, 1 - p) with the specimen mask as
weight against the MRI channels (p M, (1 - p) M); swapping the OCT classes gives exactly -S_class. The third is the specimen
outline: the OCT specimen mask against the MRI foreground M, weighted by the measured OCT voxels, so the embedding has to fall on
MRI background; it does not depend on the polarity. S = (2 |S_class| + S_outline) / 3, and the sign of S_class is the polarity.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G
from .params import Params

EPS = 1e-6          # numerical guard of weight sums and NCC denominators
VAR_FLOOR = 0.02    # local MRI variance >= this x its variance (foreground; whole grid for M): round-off guard of the FFT sums
DIMS = (-3, -2, -1)


def to_torch(x, device):
    """numpy or torch -> float32 tensor on device."""
    return torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x, np.float32)).to(device=device, dtype=torch.float32)


def box(affine, shape):
    """World centre (mm) and bounding-sphere radius (mm) of the voxel centres of a grid (affine 4x4, shape 3 ints)."""
    ijk = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    xyz = G.apply_affine(np.asarray(affine, float), ijk)
    return xyz.mean(0), float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max())


def angle_deg(R1, R2):
    """Rotation angle (deg) between two orthogonal 3x3 matrices; 180 when their handedness differs."""
    R = np.asarray(R1, float) @ np.asarray(R2, float).T
    return 180.0 if np.linalg.det(R) < 0 else math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def combined(S_class, S_outline, polarity):
    """S = (2 polarity S_class + S_outline) / 3; polarity 0 takes |S_class| (the method: the polarity is the sign of S_class)."""
    return (2 * (S_class.abs() if polarity == 0 else polarity * S_class) + S_outline) / 3


class Searcher:
    """Score maps of one orientation at a time against the zero-padded MRI grid."""

    def __init__(self, mri_v, mri_mask, mri_affine, oct_u, oct_w, oct_valid, oct_affine, device="cuda"):
        """mri_v [2, D, H, W] MRI channels, mri_mask [D, H, W] foreground fraction M, mri_affine voxel -> world (mm, isotropic
        grid); oct_u [2, d, h, w] OCT channels (p, 1 - p), oct_w [d, h, w] specimen mask (the two-class weight and the outline
        channel), oct_valid [d, h, w] measured OCT fraction (the outline weight), oct_affine. numpy or torch.
        The template is centred on the OCT grid box centre c_o and covers its bounding sphere (+3 voxels); the MRI is zero-padded
        by half a template, so a block over the edge of the crop is still scored (outside the crop counts as background)."""
        self.dev = device
        I, M, u, w, q = (to_torch(x, device) for x in (mri_v, mri_mask, oct_u, oct_w, oct_valid))
        A_M, A_O = np.asarray(mri_affine, float), np.asarray(oct_affine, float)
        if not float(w.sum()) > 0:
            raise ValueError("OCT specimen mask is empty on the search grid")
        self.c_o, radius = box(A_O, w.shape)
        self.n = n = math.ceil(2 * radius / float(np.linalg.norm(A_M[:3, :3], axis=0).mean())) + 3
        pad = n // 2 + 1
        self.A_M = A_M.copy()
        self.A_M[:3, 3] -= A_M[:3, :3] @ np.full(3, float(pad))
        I = F.pad(torch.cat([I, M[None]]), [pad] * 6)                 # channels p M, (1 - p) M, M
        self.shape = sh = tuple(I.shape[1:])
        fg = I[2] > 0.5
        if not bool(fg.any()):
            raise ValueError("no MRI foreground on the search grid")
        self.var_floor = torch.stack([I[0][fg].var(), I[1][fg].var(), I[2].var()]).reshape(3, 1, 1, 1) * VAR_FLOOR
        self.FI, self.FI2 = torch.fft.rfftn(I, dim=DIMS), torch.fft.rfftn(I * I, dim=DIMS)
        del I, fg
        # template source: u edge-replicated, w and the measured fraction zero-padded by one voxel, so wherever the sampled weight
        # is > 0 the two sampled channels sum to 1 and the swapped-class map is exactly -S_class (also at the OCT array faces)
        self.src = torch.cat([F.pad(u[None], [1] * 6, mode="replicate")[0], F.pad(torch.stack([w, q])[None], [1] * 6)[0]])
        self.A_src = A_O.copy()
        self.A_src[:3, 3] -= A_O[:3, :3] @ np.ones(3)
        k = torch.arange(n, dtype=torch.float32, device=device) - (n - 1) / 2.0
        L_M = torch.as_tensor(self.A_M[:3, :3], dtype=torch.float32, device=device)
        self.tgrid = torch.stack(torch.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3) @ L_M.T     # offsets on MRI axes, mm
        self.c_t = torch.as_tensor(self.c_o, dtype=torch.float32, device=device)
        self.valid = torch.zeros(sh, dtype=torch.bool, device=device)                                 # template inside the grid
        self.valid[:sh[0] - n + 1, :sh[1] - n + 1, :sh[2] - n + 1] = True

    def _ft(self, x):
        """Conjugate FFT of a template x [..., n, n, n] placed at the grid origin (correlation over all translations)."""
        xp = x.new_zeros(x.shape[:-3] + self.shape)
        xp[..., :self.n, :self.n, :self.n] = x
        return torch.fft.rfftn(xp, dim=DIMS).conj()

    def _ncc(self, T, w, ch):
        """Weighted NCC maps [C, D', H', W'] of templates T [C, n, n, n] with weight w [n, n, n] against MRI channels ch (slice)."""
        N, Fw, wT = w.sum() + EPS, self._ft(w), w * T
        SI, SII, SIT = (torch.fft.irfftn(a * b, s=self.shape, dim=DIMS)
                        for a, b in ((Fw, self.FI[ch]), (Fw, self.FI2[ch]), (self._ft(wT), self.FI[ch])))
        ST, STT = wT.sum(DIMS).reshape(-1, 1, 1, 1), (wT * T).sum(DIMS).reshape(-1, 1, 1, 1)
        varI = torch.maximum(SII - SI * SI / N, self.var_floor[ch] * N)
        return (SIT - ST * SI / N) / torch.sqrt(((STT - ST * ST / N) * varI).clamp(min=EPS))

    @torch.no_grad()
    def score(self, R):
        """Orientation R (numpy 3x3) -> (S_class [D', H', W'], the signed two-class NCC, mean over its two channel pairs;
        S_outline, the NCC of the specimen mask with M over the measured voxels). Index u means x_mri = R (x_oct - c_o) + t_u;
        the template lies inside the grid at the indices of self.valid."""
        n = self.n
        pts = self.c_t + self.tgrid @ torch.as_tensor(np.asarray(R), dtype=torch.float32, device=self.dev)   # x_o = c_o + R^T y
        vals = G.sample_world(self.src, self.A_src, pts).reshape(4, n, n, n)
        return self._ncc(vals[:2], vals[2], slice(0, 2)).mean(0), self._ncc(vals[2:3], vals[3], slice(2, 3))[0]

    def pose(self, R, index):
        """4x4 OCT world -> MRI world (mm) of orientation R at flat translation index: x_mri = R (x_oct - c_o) + t."""
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, float)
        centre = np.array(np.unravel_index(int(index), self.shape), float) + (self.n - 1) / 2.0     # template centre, MRI voxels
        T[:3, 3] = G.apply_affine(self.A_M, centre) - T[:3, :3] @ self.c_o
        return T


def search(mri_v, mri_mask, mri_affine, oct_u, oct_w, oct_valid, oct_affine, params: Params = Params(), device="cuda", polarity=0):
    """Orientation search at one grid (arguments as Searcher). Rotations: geometry.rotations(n_rot, seed), proper rotations only.
    Per orientation the argmax over the translations of S = combined(S_class, S_outline, polarity); with polarity 0 (the method)
    the polarity is the sign of S_class there, +1 / -1 force it. Greedy NMS on S: a pose duplicates a kept one iff centres are
    closer than nms_mm and rotations closer than nms_deg; the best topk are kept.
    -> (candidates [{T 4x4, S, S_class signed, S_outline, polarity +1|-1}] best first,
        info {top1, top2 (S of the first two), n_orientations, seconds})."""
    P, t0 = params, time.time()
    if polarity not in (0, 1, -1):
        raise ValueError(f"polarity must be 0 (sign of the score), +1 or -1, got {polarity!r}")
    s = Searcher(mri_v, mri_mask, mri_affine, oct_u, oct_w, oct_valid, oct_affine, device)
    Rs = G.rotations(P.n_rot, P.seed)
    found = []
    for i, R in enumerate(Rs):
        S_class, S_outline = s.score(R)
        X = combined(S_class, S_outline, polarity).masked_fill(~s.valid, float("-inf")).reshape(-1)
        j = torch.argmax(X)
        x, Sc, So, j = torch.stack([X[j].double(), S_class.reshape(-1)[j].double(), S_outline.reshape(-1)[j].double(),
                                    j.double()]).tolist()
        found.append((x, i, int(j), Sc, So, polarity or (1 if Sc >= 0 else -1)))
    kept = []
    for x, i, j, Sc, So, pol in sorted(found, key=lambda f: -f[0]):
        T = s.pose(Rs[i], j)
        centre = G.apply_affine(T, s.c_o)
        if any(np.linalg.norm(centre - c) < P.nms_mm and angle_deg(Rs[i], R) < P.nms_deg for _, c, R in kept):
            continue
        kept.append((dict(T=T, S=x, S_class=Sc, S_outline=So, polarity=pol), centre, Rs[i]))
        if len(kept) >= P.topk:
            break
    info = dict(top1=kept[0][0]["S"], top2=kept[1][0]["S"] if len(kept) > 1 else None, n_orientations=len(Rs),
                seconds=time.time() - t0)
    return [k[0] for k in kept], info
