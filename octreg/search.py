"""Global pose search without a location prior: masked normalized cross-correlation of the OCT
feature template against the MRI feature volume over *all* translations (FFT) for a dense set of
rotations (uniform on SO(3)), optionally a few isotropic scales.  Returns the top-K distinct poses.

The score is overlap-normalized (Padfield-style masked NCC): only OCT-tissue voxels contribute, so
a pose that hangs off the edge of the MRI or over non-tissue cannot win by accident; a minimum
tissue-overlap fraction is enforced.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import DEVICE, apply_affine, random_rotations, rotation_geodesic_deg, sample_at_world, to_t


def _corr_fft(Fw_conj: torch.Tensor, Fimg: torch.Tensor, shape) -> torch.Tensor:
    return torch.fft.irfftn(Fw_conj * Fimg, s=shape)


class FFTSearcher:
    def __init__(self, mri_feat: torch.Tensor, mri_affine: np.ndarray, mri_tissue: torch.Tensor,
                 oct_feat: torch.Tensor, oct_affine: np.ndarray, oct_mask: torch.Tensor, spacing: float,
                 device=DEVICE, min_overlap: float = 0.85):
        """mri_feat [C,D,H,W] on a world-axis-aligned isotropic grid (spacing); oct_feat [C,d,h,w] with its
        own affine (any orientation); oct_mask [1,d,h,w] in [0,1]."""
        self.dev = device
        self.C = mri_feat.shape[0]
        self.s = float(spacing)
        self.min_overlap = min_overlap
        self.oct_feat = oct_feat.to(device).float()
        self.oct_mask = oct_mask.to(device).float()
        self.A_O = to_t(oct_affine, device=device)
        # OCT block centre and template size (voxels) covering the block's bounding sphere
        d, h, w = oct_mask.shape[1:]
        corners = torch.tensor([[i, j, k] for i in (0, d - 1) for j in (0, h - 1) for k in (0, w - 1)], dtype=torch.float32, device=device)
        cw = apply_affine(self.A_O, corners)
        self.c_o = cw.mean(0)                                                    # world centre of the block
        radius = float((cw - self.c_o).norm(dim=1).max())
        self.n = int(math.ceil(2 * radius / self.s)) + 3
        # zero-pad the MRI grid by half a template on every side (features 0, tissue 0): a block that sits near or
        # partly over the edge of a cropped MRI is still scored; the overlap test decides what is acceptable
        self.pad = self.n // 2 + 1
        A_M = np.asarray(mri_affine, dtype=float).copy(); A_M[:3, 3] = A_M[:3, 3] - A_M[:3, :3] @ (np.ones(3) * self.pad)
        self.A_M = A_M
        P = [self.pad] * 6
        I = F.pad(mri_feat.to(device).float(), P); tis = F.pad(mri_tissue.to(device).float(), P)
        self.shape = tuple(I.shape[1:])
        self.FI = torch.stack([torch.fft.rfftn(I[c]) for c in range(self.C)])            # [C,...]
        self.FI2 = torch.stack([torch.fft.rfftn(I[c] * I[c]) for c in range(self.C)])
        self.Ftis = torch.fft.rfftn(tis[0])
        # per-channel variance inside the MRI tissue: floor for the local variance (nearly constant windows must not
        # produce huge normalised scores through round-off in the FFT sums)
        tm = tis[0] > 0.5
        self.var_floor = torch.stack([I[c][tm].var() for c in range(self.C)]) * 0.02 if tm.any() else torch.zeros(self.C, device=device)
        del I, tm, tis
        assert all(self.n <= sdim for sdim in self.shape), f"template {self.n} larger than MRI grid {self.shape}"
        q = torch.arange(self.n, device=device, dtype=torch.float32) - (self.n - 1) / 2.0
        # template voxel offsets expressed in WORLD mm through the MRI grid's linear part, so the template
        # lives on the same (possibly permuted/flipped) axes as the MRI array we correlate against
        L_M = to_t(self.A_M[:3, :3], device=device)
        self.tgrid = torch.stack(torch.meshgrid(q, q, q, indexing="ij"), -1) @ L_M.T   # [n,n,n,3]

    def _template(self, R: torch.Tensor):
        """Rotate the template grid into OCT space and sample features/mask -> T [C,n,n,n], w [n,n,n]."""
        pts = self.c_o + self.tgrid.reshape(-1, 3) @ R          # x_o = c_o + R^T y   (R rows act as R^T on row vectors)
        T = sample_at_world(self.oct_feat, self.A_O, pts).reshape(self.C, self.n, self.n, self.n)
        w = sample_at_world(self.oct_mask, self.A_O, pts).reshape(self.n, self.n, self.n)
        return T, w

    @torch.no_grad()
    def score_rotation(self, R: np.ndarray, scale: float = 1.0):
        Rt = to_t(R, device=self.dev)
        if scale != 1.0:
            # scaling the OCT block by `scale` in MRI space == sampling the template on a grid shrunk by 1/scale
            pts = self.c_o + (self.tgrid.reshape(-1, 3) / scale) @ Rt
            T = sample_at_world(self.oct_feat, self.A_O, pts).reshape(self.C, self.n, self.n, self.n)
            w = sample_at_world(self.oct_mask, self.A_O, pts).reshape(self.n, self.n, self.n)
        else:
            T, w = self._template(Rt)
        D, H, W = self.shape
        wp = torch.zeros(self.shape, device=self.dev); wp[:self.n, :self.n, :self.n] = w
        Fw = torch.fft.rfftn(wp).conj()
        N = w.sum() + 1e-6
        SI = _corr_fft(Fw, self.FI, self.shape)                 # [C,D,H,W]
        SII = _corr_fft(Fw, self.FI2, self.shape)
        overlap = _corr_fft(Fw, self.Ftis, self.shape) / N      # fraction of template mass on MRI tissue
        score = torch.zeros(self.shape, device=self.dev)
        for c in range(self.C):
            wT = torch.zeros(self.shape, device=self.dev); wT[:self.n, :self.n, :self.n] = w * T[c]
            STI = _corr_fft(torch.fft.rfftn(wT).conj(), self.FI[c], self.shape)
            ST = (w * T[c]).sum(); STT = (w * T[c] * T[c]).sum()
            cov = STI - ST * SI[c] / N
            varT = STT - ST * ST / N
            varI = (SII[c] - SI[c] * SI[c] / N).clamp(min=float(self.var_floor[c]) * float(N))
            score += cov / torch.sqrt((varT * varI).clamp(min=1e-6))
        score /= self.C
        # valid translations: template fully inside the grid
        valid = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        valid[:D - self.n + 1, :H - self.n + 1, :W - self.n + 1] = True
        score = torch.where(valid & (overlap >= self.min_overlap), score, torch.full_like(score, -2.0))
        best = torch.argmax(score)
        u = np.array(np.unravel_index(int(best), self.shape))
        return float(score.reshape(-1)[best]), u, float(overlap.reshape(-1)[best])

    def pose_from(self, R: np.ndarray, u: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """4x4 world transform x_m = s*R (x_o - c_o) + t_c for correlation index u."""
        centre_idx = np.asarray(u, dtype=float) + (self.n - 1) / 2.0
        t_c = (self.A_M @ np.r_[centre_idx, 1.0])[:3]
        T = np.eye(4)
        T[:3, :3] = scale * np.asarray(R)
        T[:3, 3] = t_c - T[:3, :3] @ self.c_o.detach().cpu().numpy()
        return T

    def run(self, n_rot: int = 3000, scales=(1.0,), topk: int = 32, seed: int = 0, log_every: int = 500,
            min_sep_mm: float = 3.0, min_sep_deg: float = 10.0, mirror: bool = True):
        """mirror=True also scores the reflected orientations R·diag(1,1,-1): the handedness of a microscopy
        stack relative to physical space is not guaranteed."""
        Rs = random_rotations(n_rot, seed=seed)
        if mirror:
            Rs = np.concatenate([Rs, Rs @ np.diag([1.0, 1.0, -1.0])], 0)
        cands = []
        t0 = time.time()
        for i, R in enumerate(Rs):
            for sc in scales:
                s, u, ov = self.score_rotation(R, sc)
                cands.append((s, i, sc, u, ov))
            if log_every and (i + 1) % log_every == 0:
                best = max(cands, key=lambda c: c[0])
                print(f"  [search] {i + 1}/{len(Rs)} orientations, best score {best[0]:.4f}, {time.time() - t0:.0f}s", flush=True)
        cands.sort(key=lambda c: -c[0])
        # non-maximum suppression in pose space
        kept = []
        c_o = np.r_[self.c_o.detach().cpu().numpy(), 1.0]
        for s, i, sc, u, ov in cands:
            T = self.pose_from(Rs[i], u, sc)
            centre = (T @ c_o)[:3]
            dup = False
            for k in kept:
                if np.linalg.norm(centre - k["centre"]) < min_sep_mm and rotation_geodesic_deg(Rs[i], k["R"]) < min_sep_deg:
                    dup = True; break
            if not dup:
                kept.append({"score": s, "rot_index": int(i), "scale": sc, "u": u.tolist(), "overlap": ov, "R": Rs[i], "T": T, "centre": centre,
                             "mirror": bool(np.linalg.det(Rs[i]) < 0)})
            if len(kept) >= topk:
                break
        return kept, {"n_rot": n_rot, "mirror": mirror, "n_orientations": len(Rs), "scales": list(scales), "seconds": time.time() - t0, "template_n": self.n,
                      "top1": kept[0]["score"] if kept else None, "top2": kept[1]["score"] if len(kept) > 1 else None}
