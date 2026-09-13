"""Fine stage (octreg v1.1, spec 2.4): sub-mm polish of the final pose with ONE sharp loss and label-free verification.

Loss: sign-corrected masked NCC of 3 mm-bias-flattened raw intensities (P2's winner, contrast 0.54) inside a specimen mask
(eroded MRI tissue mapped through the pose & OCT mask & OCT validity).  The optimiser works on a DELTA transform D about
identity with T = T_start @ D, so refine.py's absolute ls/sh clamps and its reg*(|ls|^2+|sh|^2) term become RELATIVE to the
start pose without editing refine.py (P0.4).  Independent measures (LCC / LCC^2, MI-32) are verifiers, never summed into the
loss (P0.3); [WM,GM] class NCC (ref015) is only a protection gate.  A rejected stage returns the start pose (P0.5: normal).

Gate geometry (deviation from the spec's literal transform_diff(...)['corner_mean_mm'], reported): moves are measured at the
block's OWN 8 corners (pose_move) and the rotation of the delta is capped (cfg.max_rot); transform_diff's fixed 7 mm cube
under-reads the rotation of a 29 mm block ~3x and its rotation_deg is meaningless for mirrored poses.  Restarts are REQUIRED
(gate (d) fails with n_restarts = 0), U_mm must not exceed max_move (g), and a dense LCC loss whose box is wider than the mask
(degenerate) skips the level / drops the witness instead of producing a NaN loss.

Options after the P7/P8 probes (all default to the v1.1 behaviour above; the call sequence is byte-identical when unset):
  fixed_mask      P7's configuration: m_spec is chosen ONCE per level at T_start and every chain (main run, restarts, LCC witness,
                  backward ICE mask) optimises on that fixed OCT point set, AND (fixed_weight 'auto') the ncc loss's detached MRI-tissue
                  point weight is frozen at T_start as well, so the effective point set (points with non-zero weight) cannot follow the
                  pose either.  v1.1 re-derived m_spec from the CURRENT pose at every level start ('mask follows pose'), which rewards
                  drifting into basins 6-8 mm away (P7).
  fixed_weight    'auto' (default) = frozen iff fixed_mask; 'on' = (ncc loss only) the weight is frozen at the pose the mask was derived
                  from even without fixed_mask (the current pose at the level start); 'off' = re-sampled through the current pose at every
                  evaluation (v1.1; with fixed_mask this is the fixed-mask-only variant of the first I58 runs (ii)/(iii), in which points
                  that drift off the MRI tissue silently drop out of the correlation).  Frozen: points that leave the MRI tissue see the
                  flattened MRI's 0 background and decorrelate instead (P7's 'fixed detached tissue weight').  The dense LCC / LCC2
                  witnesses and the backward ICE problem keep their validity(T) mask in every case.  weight_fixed(cfg) resolves the switch.
  restart_mm / restart_deg / restart_logscale   size of the V1 restart perturbations: angle U(0.4, 1.2) x restart_deg about a random
                  axis, translation U(-2/3, 2/3) x restart_mm per axis (RMS norm 2/3 restart_mm, max 1.15 restart_mm), log-scale
                  U(-1, 1) x restart_logscale per axis; the defaults 5 deg / 3 mm / 0.03 are v1.1's U(2,6) deg / U(-2,2) mm / U(-0.03,0.03).
                  The rigid basin of the I58 loss captures ~1.5 mm / 3 deg (P7): 1 mm / 2 deg restarts test convergence, 3 mm / 5 deg
                  ones test the (absent) global basin.
  restart_tol_mm  a restart counts as converged when its block corners lie within this distance of T_fine (default 0.3 mm = the v1.1
                  gate; recorded as info.restart_tol_mm / restarts.tol_mm).  An explicit option, not a slope on restart_mm: the asked-for
                  max(0.3, 0.3 x restart_mm) would move the default gate to 0.9 mm and change validated accept/reject outcomes; pass it
                  explicitly (e.g. 0.3 x restart_mm) when that scaling is wanted for restarts larger than 1 mm.
  exclude_fov_mm  MRI tissue within this distance of a field-of-view face is removed from t_m before everything that derives from it
                  (flattening mask, tissue weight, eroded m_spec seed, guard / MI tissue).  FOV faces = the faces of the MRI region
                  [lo,hi) the stage sees: the mri.npy array faces, or the --crop-centre box faces when that is set (the stage's MRI box
                  is clipped to the region, so tissue is truncated there exactly as at an array face; info.fov_faces records which
                  flagged faces are array faces).  Box faces inside the region do not count.  The I58 MRI is a whole-brain crop whose
                  specimen touches 3 array faces (P8), so the truncated partial-volume rind must not attract the pose.  erode_ball already
                  treats the box faces as background, so the eroded seed loses exclude_fov_mm + erode_mm at a FOV face (I58 at 0.32 mm,
                  1.0 mm: m_spec 230.7k -> 227.7k voxels); the tissue WEIGHT loses exactly the exclude_fov_mm rind.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .common import DEVICE, apply_affine, avg_pool_iso, resample_to_grid, sample_at_world, to_t, world_bbox_to_voxel
from .evaluate import transform_diff
from .refine import Refiner, compose, masked_ncc, params_from_matrix

AX = ("z", "y", "x")                                        # OCT array axes (depth, y, x)


@dataclass
class FineContext:
    """What the fine stage needs from register.py (arrays exactly as loaded there; the MRI stays an mmap)."""
    mri: np.ndarray; mri_tissue: np.ndarray; A_mri: np.ndarray; lo: np.ndarray; hi: np.ndarray
    oct150: np.ndarray; oct_mask: np.ndarray; A_oct: np.ndarray; c_o: np.ndarray
    BLOCK_R: float; VOX_M: float; VOX_O: float
    ref015: object = None                                   # Refiner on [WM,GM] at 0.15 mm (structural gate); None = gate skipped
    device: object = DEVICE


@dataclass
class FineConfig:
    sim: str = "auto"; levels: tuple = (0.3, 0.15); flatten_mm: float = 3.0; erode_mm: float = 1.3; lcc_mm: float = 4.5
    clamp: float = 0.15; reg: float = 0.5; iters: int = 200; n_restarts: int = 8; max_move: float = 3.0; subsample: int = 1
    verify: str = "basic"; rigid_iters: int = 150; n_points: int = 50000
    max_rot: float = 5.0                                    # deg, rotation of the delta vs the start pose (the loss is flat in rotation: P2 caveat 4)
    fixed_mask: bool = False                                # P7: m_spec per level chosen once at T_start for every chain, ncc weight frozen there too (fixed_weight auto); False = v1.1 'mask follows pose'
    fixed_weight: object = "auto"                           # ncc weight: "auto" = frozen iff fixed_mask; "on"/True = frozen at the mask pose; "off"/False = re-sampled through T (v1.1); weight_fixed(cfg) resolves it
    restart_mm: float = 3.0; restart_deg: float = 5.0; restart_logscale: float = 0.03    # V1 restart size (defaults = v1.1's hard-coded U(-2,2) mm / U(2,6) deg / 0.03)
    restart_tol_mm: float = 0.3                             # mm at the block corners for a converged restart (v1.1 gate; NOT scaled with restart_mm, see the header)
    exclude_fov_mm: float = 0.0                             # mm; > 0 drops MRI tissue this close to a field-of-view face from t_m / m_spec (0 = v1.1)
    dof: str = "affine"                                     # degrees of freedom of every polish step after the first level's rigid step: rigid | similarity | affine (v1.1 = affine; R10: the affine polish over-stretches the I46 depth axis)


def restart_tol_mm(cfg: FineConfig) -> float:
    """Block-corner tolerance (mm) for a converged restart = cfg.restart_tol_mm (an explicit option; default = the v1.1 gate 0.3 mm)."""
    return float(cfg.restart_tol_mm)


def weight_fixed(cfg: FineConfig) -> bool:
    """Resolved ncc tissue-weight switch: fixed_weight 'auto' / None follows fixed_mask (so fixed_mask alone is P7's fixed point set
    WITH a fixed detached weight), 'on' / True freezes the weight at the mask pose regardless, 'off' / False re-samples it through the
    current pose at every evaluation (v1.1; with fixed_mask = the fixed-mask-only variant)."""
    w = cfg.fixed_weight
    if w is None or (isinstance(w, str) and w.strip().lower() == "auto"): return bool(cfg.fixed_mask)
    return bool(w) if isinstance(w, (bool, int)) else str(w).strip().lower() in ("on", "true", "yes", "1")


# ----------------------------------------------------------------------------- small helpers
def mri_box(mri, mri_tissue, A_mri, lo, hi, bc, half_mm):
    """MRI box bc +- half_mm clipped to the feature region [lo,hi) -> (vlo, vhi, A_box, mri_box f32, tis_box bool)."""
    vlo, vhi = world_bbox_to_voxel(A_mri, mri.shape, bc - half_mm, bc + half_mm); A_v = np.asarray(A_mri).copy()
    vlo = np.maximum(vlo, lo); vhi = np.minimum(vhi, hi); A_v[:3, 3] = (A_mri @ np.r_[vlo, 1.0])[:3]
    mri_v = np.asarray(mri[vlo[0]:vhi[0], vlo[1]:vhi[1], vlo[2]:vhi[2]]).astype(np.float32)
    tis_v = np.asarray(mri_tissue[vlo[0]:vhi[0], vlo[1]:vhi[1], vlo[2]:vhi[2]]).astype(bool)
    return vlo, vhi, A_v, mri_v, tis_v


def _gauss1d(sigma, device):
    r = int(math.ceil(3 * sigma)); x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2); return k / k.sum()


def gauss3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """[D,H,W] -> separable Gaussian (kernel radius 3 sigma, zero padding)."""
    k = _gauss1d(sigma, x.device); r = (k.numel() - 1) // 2; y = x[None, None]
    for ax in range(3):
        s = [1, 1, 1, 1, 1]; s[2 + ax] = k.numel(); p = [0, 0, 0]; p[ax] = r; y = F.conv3d(y, k.view(s), padding=tuple(p))
    return y[0, 0]


def box3d(x: torch.Tensor, w: int) -> torch.Tensor:
    """Separable box mean (== avg_pool3d w^3, count_include_pad, zero padding); x [D,H,W]; 10-15x cheaper than the cube."""
    y = x[None, None]
    for ax in range(3):
        k = [1, 1, 1]; k[ax] = w; p = [0, 0, 0]; p[ax] = w // 2
        y = F.avg_pool3d(y, tuple(k), stride=1, padding=tuple(p), count_include_pad=True)
    return y[0, 0]


def erode_vox(mask: torch.Tensor, r: int) -> torch.Tensor:
    """bool [D,H,W] eroded by r voxels (max-pool of the complement)."""
    if r <= 0: return mask
    return ~(F.max_pool3d((~mask).float()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0.5)


def erode_ball(mask_np: np.ndarray, r_mm: float, vox_mm: float) -> np.ndarray:
    """Erosion by a physical radius: EDT(mask) * vox >= r_mm (array faces count as background, as binary_erosion does)."""
    m = np.pad(np.asarray(mask_np, bool), 1)
    return (ndimage.distance_transform_edt(m) * vox_mm >= r_mm)[1:-1, 1:-1, 1:-1]


def fov_face_distance(shape_p, f: int, n_orig, vox_mm, face_lo, face_hi) -> np.ndarray:
    """Distance (mm) of every voxel centre of a grid pooled by f from the nearest field-of-view face of the unpooled box it came from
    (n_orig voxels of size vox_mm per axis; face_lo / face_hi flag the box faces that are FOV faces).  The faces are axis-aligned
    planes, so the EDT from them is the minimum over axes of the per-axis distances (exact, no padding / pooling artefacts):
    pooled voxel i covers unpooled voxels [i f, (i+1) f), its centre is (i + 0.5) f voxels from the lo face plane."""
    d = np.full(tuple(int(s) for s in shape_p), np.inf, np.float32)
    for k in range(3):
        c = (np.arange(shape_p[k], dtype=np.float64) + 0.5) * f; dk = np.full(shape_p[k], np.inf)
        if face_lo[k]: dk = np.minimum(dk, c * vox_mm[k])
        if face_hi[k]: dk = np.minimum(dk, (n_orig[k] - c) * vox_mm[k])
        shp = [1, 1, 1]; shp[k] = -1; d = np.minimum(d, dk.reshape(shp).astype(np.float32))
    return d


def _zs(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    v = x[m]; return (x - v.mean()) / (v.std() + 1e-6)


def _pct(x: torch.Tensor, q: float) -> float:
    x = x.flatten(); k = int(max(1, min(x.numel(), round(q / 100.0 * x.numel())))); return float(x.kthvalue(k).values)


def flatten_masked(I: torch.Tensor, m: torch.Tensor, sigma_vox: float, pooled: bool = False) -> torch.Tensor:
    """Bias-flattened intensity F = I / (masked local mean + 1e-3 mean) - 1, z-scored inside m, clamped to [-4,4], 0 outside m.
    pooled: the low-pass is computed on the grid pooled by 2 and trilinearly upsampled (finest level)."""
    mf = m.float(); eps = 1e-3 * float(I[m].mean())
    if pooled:
        num = gauss3d(F.avg_pool3d((I * mf)[None, None], 2)[0, 0], sigma_vox / 2); den = gauss3d(F.avg_pool3d(mf[None, None], 2)[0, 0], sigma_vox / 2)
        lm = F.interpolate((num / (den + 1e-6))[None, None], size=tuple(I.shape), mode="trilinear", align_corners=False)[0, 0]
    else:
        lm = gauss3d(I * mf, sigma_vox) / (gauss3d(mf, sigma_vox) + 1e-6)
    Fx = I / (lm + eps) - 1.0
    return _zs(Fx, m).clamp(-4, 4) * mf


def point_disp_stats(T1: np.ndarray, T2: np.ndarray, pts: np.ndarray) -> dict:
    """Displacement |T1 p - T2 p| over world points pts [N,3] -> mean / p95 / max (mm)."""
    d = _point_disp(T1, T2, pts)
    return {"mean_mm": float(d.mean()), "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max())}


def _point_disp(T1, T2, pts):
    P = np.c_[pts, np.ones(len(pts))]; return np.linalg.norm((P @ T1.T)[:, :3] - (P @ T2.T)[:, :3], axis=1)


def _rodrigues(u, deg):
    th = math.radians(deg); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def perturb_matrix(kind: str, u: np.ndarray, amt: float, c_o: np.ndarray) -> np.ndarray:
    """OCT-world perturbation P (apply as T @ P): translation (mm) along u, rotation (deg) about u through c_o, or log-scale along u about c_o."""
    P = np.eye(4)
    if kind == "trans": P[:3, 3] = amt * u
    else:
        M = _rodrigues(u, amt) if kind == "rot" else np.eye(3) + (math.exp(amt) - 1.0) * np.outer(u, u)
        P[:3, :3] = M; P[:3, 3] = c_o - M @ c_o
    return P


def stretch_ijk(T: np.ndarray, A_oct: np.ndarray) -> list:
    R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
    return [round(float(np.linalg.norm(T[:3, :3] @ R[:, q])), 3) for q in range(3)]


def block_corners(A_oct: np.ndarray, shape) -> np.ndarray:
    """The 8 world corners of the OCT block [8,3] (the points BLOCK_R is measured on)."""
    c = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    return (np.asarray(A_oct, float) @ np.c_[c, np.ones(8)].T).T[:, :3]


def rotation_deg(M: np.ndarray) -> float:
    """Rotation angle (deg) of the polar factor of a 3x3 linear map (a delta between poses of the same handedness has det > 0;
    an improper map is flipped on its last singular axis).  transform_diff's rotation_deg is NOT usable for mirrored poses:
    polar_rotation flips a singular axis that changes with the anisotropic stretch (I46 vascular vs structural reads 129 deg)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float)); R = U @ Vt
    if np.linalg.det(R) < 0: R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return float(math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))))


def pose_move(T_new: np.ndarray, T_ref: np.ndarray, c_o: np.ndarray, corners: np.ndarray) -> dict:
    """transform_diff (spec's fixed 7 mm cube, kept for comparability with v1 numbers) + the move at the block's OWN corners and
    the rotation of the delta D = inv(T_ref) T_new.  The 7 mm cube under-reads rotations of a large block (I58 BLOCK_R 29 mm: a
    12 deg rotation is 2.1 mm at 7 mm but 6.2 mm at the block corners), so the gates use block_corner_mean_mm / delta_rotation_deg."""
    T_new = np.asarray(T_new, float); T_ref = np.asarray(T_ref, float); d = transform_diff(T_new, T_ref, c_o)
    Q = np.c_[corners, np.ones(len(corners))]; db = np.linalg.norm((Q @ T_new.T)[:, :3] - (Q @ T_ref.T)[:, :3], axis=1)
    D = np.linalg.inv(T_ref) @ T_new
    d.update(block_corner_mean_mm=float(db.mean()), block_corner_max_mm=float(db.max()), delta_rotation_deg=rotation_deg(D[:3, :3]))
    return d


# ----------------------------------------------------------------------------- one pyramid level (pose independent)
class Level:
    """Pooled OCT / MRI-box arrays of one level, flattened once; the specimen mask is built per pose by spec_mask().
    fov (optional, cfg.exclude_fov_mm > 0): (n_orig, vox_mm, face_lo, face_hi) of the unpooled MRI box -> tissue within
    exclude_fov_mm of a field-of-view face is dropped from t_m before anything is derived from it (n_fov_excluded voxels)."""

    def __init__(self, O: torch.Tensor, MK: torch.Tensor, A_oct, box, L: float, cfg: FineConfig, VOX_O: float, VOX_M: float, dev, fov=None):
        mri_b, tis_b, A_box = box; self.mm = float(L); self.dev = dev; self.n_fov_excluded = 0
        self.f_o = max(1, int(round(L / VOX_O))); self.f_m = max(1, int(round(L / VOX_M)))
        pos = (O > 0).float()
        if self.f_o > 1:
            I_o, A_o = avg_pool_iso(O, A_oct, self.f_o); m_o = avg_pool_iso(MK, A_oct, self.f_o)[0][0] > 0.5
            valid = avg_pool_iso(pos, A_oct, self.f_o)[0][0] > 0.99
        else:
            I_o, A_o = O, np.asarray(A_oct).copy(); m_o = MK[0] > 0.5; valid = pos[0] > 0.5
        self.valid_o = erode_vox(valid, 1 if L >= 0.3 else 2)                    # zeros are missing data (P0.6)
        self.I_o = I_o[0]; self.A_o = A_o; self.A_o_t = to_t(A_o, device=dev); self.shape_o = tuple(self.I_o.shape); self.m_o = m_o
        self.mo = m_o & self.valid_o; self.mo_f = self.mo.float()[None]
        if self.f_m > 1:
            I_m, A_m = avg_pool_iso(mri_b, A_box, self.f_m); t_m = avg_pool_iso(tis_b, A_box, self.f_m)[0][0] > 0.5
        else:
            I_m, A_m = mri_b, np.asarray(A_box).copy(); t_m = tis_b[0] > 0.5
        if fov is not None and cfg.exclude_fov_mm > 0:                              # FOV rind: EDT from the field-of-view faces < exclude_fov_mm
            far = to_t(fov_face_distance(t_m.shape, self.f_m, *fov), device=dev) >= cfg.exclude_fov_mm
            self.n_fov_excluded = int((t_m & ~far).sum()); t_m = t_m & far
        self.I_m = I_m[0]; self.A_m = A_m; self.A_m_t = to_t(A_m, device=dev); self.shape_m = tuple(self.I_m.shape); self.t_m = t_m
        self.t_m_f = t_m.float()[None]
        self.vox_o = float(np.linalg.norm(A_o[:3, :3], axis=0).mean()); self.vox_m = float(np.linalg.norm(A_m[:3, :3], axis=0).mean())
        pooled = L < 0.3
        self.F_o = flatten_masked(self.I_o, self.mo, cfg.flatten_mm / self.vox_o, pooled)[None]
        self.F_m = flatten_masked(self.I_m, self.t_m, cfg.flatten_mm / self.vox_m, pooled)[None]
        er = erode_ball(t_m.cpu().numpy(), cfg.erode_mm, self.vox_m)
        self.t_m_er = to_t(er.astype(np.float32), device=dev)[None]; self.t_m_er_b = to_t(er, dtype=torch.bool, device=dev)
        self.raw_m = torch.cat([self.I_m[None], self.t_m_f], 0)                  # [raw MRI, tissue] for the guard / MI

    @torch.no_grad()
    def spec_mask(self, T) -> torch.Tensor:
        """m_spec at pose T: eroded MRI tissue mapped through T (> 0.5) & OCT mask & OCT validity  (bool [d,h,w])."""
        er = resample_to_grid(self.t_m_er, self.A_m_t, self.A_o_t, self.shape_o, T_grid_to_vol=to_t(T, device=self.dev))[0] > 0.5
        return er & self.mo

    @torch.no_grad()
    def mri_at(self, T) -> torch.Tensor:
        """[raw MRI, tissue] sampled on the OCT grid through T -> [2,d,h,w]."""
        return resample_to_grid(self.raw_m, self.A_m_t, self.A_o_t, self.shape_o, T_grid_to_vol=to_t(T, device=self.dev))


# ----------------------------------------------------------------------------- polarity guard
@torch.no_grad()
def polarity_guard(I_o, A_o, I_m, A_m, m_spec, T, tis_m=None, min_vox: int = 2000, n_tiles: int = 4, level=None):
    """Regional sign of the raw OCT/MRI intensity relation at T: masked NCC (z-scored raw intensities) in a 4x4x4 tiling of the
    m_spec bounding box, blocks with > min_vox voxels.  sign -1 if frac_neg >= 0.60 and volume-weighted mean <= -0.05;
    +1 if frac_neg <= 0.40 and mean >= +0.05; else 0 (mixed).  I58 at 0.3 mm (P2): 26/37 negative, mean -0.15 -> -1."""
    if isinstance(level, Level):
        s = level.mri_at(T); I_m_s, tis = s[0], s[1] > 0.5
    else:
        dev = I_o.device; vol = I_m[None] if tis_m is None else torch.cat([I_m[None], tis_m.float()[None]], 0)
        s = resample_to_grid(vol, to_t(A_m, device=dev), to_t(A_o, device=dev), tuple(I_o.shape), T_grid_to_vol=to_t(T, device=dev))
        I_m_s = s[0]; tis = s[1] > 0.5 if tis_m is not None else torch.ones_like(m_spec)
    m = m_spec & tis
    out = {"n_blocks": 0, "n_neg": 0, "frac_neg": None, "mean": None, "n_vox": int(m.sum()), "block_ncc": []}
    if int(m.sum()) < min_vox: return 0, out
    a = _zs(I_o, m) * m; b = _zs(I_m_s, m) * m
    idx = torch.nonzero(m); lo = idx.min(0).values.cpu().numpy(); hi = idx.max(0).values.cpu().numpy() + 1
    edges = [np.linspace(lo[k], hi[k], n_tiles + 1).round().astype(int) for k in range(3)]
    vals, ns = [], []
    for i in range(n_tiles):
        for j in range(n_tiles):
            for k in range(n_tiles):
                sl = (slice(edges[0][i], edges[0][i + 1]), slice(edges[1][j], edges[1][j + 1]), slice(edges[2][k], edges[2][k + 1]))
                mm = m[sl]; n = int(mm.sum())
                if n <= min_vox: continue
                aa = a[sl][mm]; bb = b[sl][mm]; aa = aa - aa.mean(); bb = bb - bb.mean()
                vals.append(float((aa * bb).sum() / torch.sqrt((aa * aa).sum() * (bb * bb).sum() + 1e-12))); ns.append(n)
    if not vals: return 0, out
    v = np.array(vals); n = np.array(ns, float); frac_neg = float((v < 0).mean()); mean = float((v * n).sum() / n.sum())
    sign = -1 if (frac_neg >= 0.60 and mean <= -0.05) else (1 if (frac_neg <= 0.40 and mean >= 0.05) else 0)
    out.update(n_blocks=len(v), n_neg=int((v < 0).sum()), frac_neg=frac_neg, mean=mean, block_ncc=[round(x, 4) for x in vals], sign=sign)
    return sign, out


# ----------------------------------------------------------------------------- refiners on a DELTA about identity
class _DeltaRefiner(Refiner):
    """Refiner whose optimised transform is a delta D about identity with T = T0 @ D (Refiner.refine is reused verbatim:
    params_from_matrix(I) gives ls0 = sh0 = 0, so its clamps and regulariser act on the CHANGE relative to T0)."""

    failed_steps: int = 0                                   # refine() steps that saw no finite loss (returned D = None): no move, flagged
    degenerate: bool = False                                # FineDense: no voxel has more than half its LCC box inside the mask (loss undefined)

    def set_start(self, T):
        self.T0_np = np.asarray(T, float).copy(); self.T0 = to_t(self.T0_np, device=self.dev)

    def refine_delta(self, dof: str, iters: int, lr_rot: float, lr_t: float, lr_ls: float = 0.0, lr_sh: float = 0.0,
                     clamp: float = 0.15, reg: float = 0.5, subsample: int = 1):
        """One optimisation step from the current T0; returns (T0 @ D_best, loss) and moves T0 there (chained levels keep the clamp relative).
        Refiner.refine keeps the best FINITE data loss; when every iteration was NaN/inf it returns (None, inf) -> treated as
        'no improvement' (T0 kept, failed_steps += 1) instead of crashing the stage (P0.5: a rejected stage is normal behaviour)."""
        D, loss = self.refine(np.eye(4), dof=dof, iters=iters, lr_rot=lr_rot, lr_t=lr_t, lr_ls=lr_ls, lr_sh=lr_sh, subsample=subsample,
                              ls_clamp=clamp, sh_clamp=clamp, reg=reg)
        if D is None or not np.isfinite(loss):
            self.failed_steps += 1; return self.T0_np.copy(), float(self.evaluate_T(self.T0_np))
        T = self.T0_np @ D; self.set_start(T); return T, float(loss)

    @torch.no_grad()
    def evaluate_T(self, T) -> float:
        """Data loss at an absolute pose T (D = inv(T0) @ T)."""
        return float(self.loss_of(to_t(np.linalg.inv(self.T0_np) @ np.asarray(T, float), device=self.dev)).item())


class FineRefiner(_DeltaRefiner):
    """sim = ncc: point-based masked NCC of the flattened intensities, sign-corrected (FM = sign * F_m); the MRI tissue sampled
    through T is a detached point weight (the moving-tissue intersection carries no gradient).  T_weight (weight_fixed(cfg)): the
    weight is sampled ONCE through this pose and frozen (None = v1.1: re-sampled through the current pose at every evaluation)."""

    def __init__(self, F_m, A_m, F_o, A_o, mask, tis_m, T_start, sign, device=DEVICE, T_weight=None):
        super().__init__(sign * F_m, A_m, F_o, A_o, mask, device)
        self.FMT = torch.cat([self.FM, tis_m.to(device).float()], 0); self.set_start(T_start); self.w_fix = None
        if T_weight is not None:
            with torch.no_grad(): self.w_fix = (sample_at_world(self.FMT[1:], self.A_M, apply_affine(to_t(T_weight, device=self.dev), self.pts_o))[0] > 0.5).float()

    def loss_of(self, D, subsample: int = 1):
        pts = self.pts_o[::subsample]; w = self.w[::subsample]; fo = self.fo[:, ::subsample]
        if self.w_fix is not None:
            s = sample_at_world(self.FMT[:1], self.A_M, apply_affine(self.T0 @ D, pts)); return 1.0 - masked_ncc(s, fo, w * self.w_fix[::subsample])
        s = sample_at_world(self.FMT, self.A_M, apply_affine(self.T0 @ D, pts))
        tis = (s[1] > 0.5).float().detach()
        return 1.0 - masked_ncc(s[:1], fo, w * tis)


class FineDense(_DeltaRefiner):
    """Dense similarity on the fixed grid (all voxels are the points): sim = lcc (signed local correlation), lcc2 (sign-free
    cov^2/(var var)) with masked local statistics in a separable box of w voxels, or ncc (global masked NCC, used for the
    backward MRI->OCT problem).  Moving image + validity sampled through T; m = m_fix & validity(T) (detached).
    Inputs are globally z-scored inside the mask at construction (moving side: sampled at T_start).
    Local statistics are only defined where more than half the box lies inside the mask (n > 0.5); a mask thinner than half the
    box (4.5 mm box: specimen < ~2.3 mm thick or < ~0.05 cm3 at 0.15 mm, split halves, restart masks) has NO such voxel -> the
    loss falls back to a finite constant with zero gradient (no move) and `degenerate` is set at construction so callers can skip
    the level / flag the verifier instead of aborting on a NaN loss."""

    def __init__(self, F_fix, A_fix, m_fix, F_mov, A_mov, valid_mov, T_start, sign, sim, w_vox, device=DEVICE):
        super().__init__(torch.cat([F_mov, valid_mov.to(F_mov.device).float()], 0), A_mov, F_fix[None], A_fix, torch.ones_like(m_fix, dtype=torch.float32)[None], device)
        self.shape = tuple(m_fix.shape); self.m_fix = m_fix.to(device); self.sign = float(sign); self.sim = sim; self.w_box = int(w_vox) | 1
        self.a = _zs(F_fix.to(device), self.m_fix) * self.m_fix.float(); self.set_start(T_start)
        with torch.no_grad():
            s = self._sample(self.T0); m = self.m_fix & (s[1] > 0.5); v = s[0][m]
            self.mu_b = float(v.mean()) if v.numel() else 0.0; self.sd_b = float(v.std()) + 1e-6 if v.numel() > 1 else 1.0
            self.n_valid0 = int(m.sum()) if sim == "ncc" else int((m & (box3d(m.float(), self.w_box) > 0.5)).sum())
        self.degenerate = self.n_valid0 == 0

    def _sample(self, T):
        return sample_at_world(self.FM, self.A_M, apply_affine(T, self.pts_o)).reshape(2, *self.shape)

    def loss_of(self, D, subsample: int = 1):
        s = self._sample(self.T0 @ D); m = self.m_fix & (s[1] > 0.5).detach(); mf = m.float()
        a = self.a * mf; b = (s[0] - self.mu_b) / self.sd_b * mf
        if self.sim == "ncc":
            N = mf.sum() + 1e-6; ma = a.sum() / N; mb = b.sum() / N; da = (a - ma) * mf; db = (b - mb) * mf
            return 1.0 - self.sign * (da * db).sum() / torch.sqrt(((da * da).sum() * (db * db).sum()).clamp(min=1e-6))
        w = self.w_box; n = box3d(mf, w).clamp(min=1e-6); mu_a = box3d(a, w) / n; mu_b = box3d(b, w) / n
        va = (box3d(a * a, w) / n - mu_a ** 2).clamp(min=0); vb = (box3d(b * b, w) / n - mu_b ** 2).clamp(min=0)
        cov = box3d(a * b, w) / n - mu_a * mu_b; valid = m & (n > 0.5)
        if not bool(valid.any()): return 1.0 + 0.0 * s[0].sum()                  # undefined local statistics: finite, zero gradient, stays in the graph
        if self.sim == "lcc": return 1.0 - self.sign * (cov / torch.sqrt(va * vb + 1e-2))[valid].mean()
        return 1.0 - (cov ** 2 / (va * vb + 1e-2))[valid].mean()


def make_refiner(L: Level, m_spec, T, sign, sim, cfg: FineConfig, dev, T_mask=None):
    """T_mask: pose m_spec was derived from (None = T); with weight_fixed(cfg) (fixed_mask, or fixed_weight 'on') the ncc tissue
    weight is frozen at that pose (fixed_mask: T_start for every chain)."""
    if sim == "ncc": return FineRefiner(L.F_m, L.A_m, L.F_o, L.A_o, m_spec.float()[None], L.t_m_f, T, sign, dev, (T if T_mask is None else T_mask) if weight_fixed(cfg) else None)
    return FineDense(L.F_o[0], L.A_o, m_spec, L.F_m, L.A_m, L.t_m_f, T, sign, sim, int(round(cfg.lcc_mm / L.vox_o)) | 1, dev)


def _schedule(kind: str, i: int, L: float, cfg: FineConfig):
    """Per-level steps [(dof, iters, lrs)]: first level rigid -> cfg.dof, later levels cfg.dof (v1.1: affine); lrs halved below 0.25 mm.
    With cfg.dof == "rigid" the first level runs two consecutive rigid steps (rigid_iters, then iters), which is the same as one longer one."""
    s = 1.0 if L >= 0.25 else 0.5; dof = cfg.dof if cfg.dof in ("rigid", "similarity", "affine") else "affine"
    if kind == "main": steps = ([("rigid", cfg.rigid_iters)] if i == 0 else []) + [(dof, cfg.iters)]
    else: steps = ([("rigid", 100)] if i == 0 else []) + [(dof, 100)]
    return [(dof, n, dict(lr_rot=0.01 * s, lr_t=0.15 * s, lr_ls=0.005 * s, lr_sh=0.005 * s)) for dof, n in steps if n > 0]


def run_steps(ref: _DeltaRefiner, steps, cfg: FineConfig):
    T, loss = ref.T0_np.copy(), None
    for dof, n, lrs in steps: T, loss = ref.refine_delta(dof, n, clamp=cfg.clamp, reg=cfg.reg, subsample=cfg.subsample, **lrs)
    return T, (ref.evaluate_T(T) if loss is None else loss)


def run_chain(levels, T_init, signs, sims, cfg: FineConfig, kind: str, dev, notes: list | None = None, masks=None, T_mask=None):
    """Full level schedule from T_init with the main run's per-level sign / sim (None = level skipped); kind 'main' | 'lcc'.
    notes (optional list) collects 'degenerate@L' / 'failed@L' for levels whose refiner could not move (FineDense window too large
    for the mask, or no finite loss); the pose is then carried through unchanged at that level.
    masks: per-level m_spec to optimise on (cfg.fixed_mask: the T_start masks of the main run); None = m_spec at the chain's current pose;
    T_mask: the pose those masks were derived from (weight_fixed(cfg) freezes the ncc weight there)."""
    T = np.asarray(T_init, float).copy()
    for i, L in enumerate(levels):
        if sims[i] is None: continue
        sim = sims[i] if kind == "main" else ("lcc" if signs[i] != 0 else "lcc2")
        ref = make_refiner(L, masks[i] if masks is not None else L.spec_mask(T), T, signs[i], sim, cfg, dev, T_mask if masks is not None else None)
        if ref.degenerate:
            if notes is not None: notes.append(f"degenerate@{L.mm}")
            del ref; continue
        T, _ = run_steps(ref, _schedule(kind, i, L.mm, cfg), cfg)
        if ref.failed_steps and notes is not None: notes.append(f"failed@{L.mm}")
        del ref
    return T


# ----------------------------------------------------------------------------- MI-32 (sign-free witness)
class MI32:
    """32-bin joint-histogram MI of raw OCT vs raw MRI sampled through T at the m_spec voxels (& MRI tissue); bin edges fixed."""

    def __init__(self, L: Level, m_spec, dev, bins: int = 32):
        idx = torch.nonzero(m_spec); self.pts = apply_affine(L.A_o_t, idx.float()); self.a = L.I_o[idx[:, 0], idx[:, 1], idx[:, 2]]
        self.L = L; self.bins = bins; self.edges = None

    @torch.no_grad()
    def sample(self, T):
        s = sample_at_world(self.L.raw_m, self.L.A_m_t, apply_affine(to_t(T, device=self.a.device), self.pts)); return s[0], s[1] > 0.5

    @torch.no_grad()
    def fix_edges(self, T):
        b, m = self.sample(T); self.edges = ((_pct(self.a[m], 0.5), _pct(self.a[m], 99.5)), (_pct(b[m], 0.5), _pct(b[m], 99.5)))
        return self.edges

    @torch.no_grad()
    def __call__(self, T) -> float:
        b, m = self.sample(T); a = self.a[m]; b = b[m]; (a0, a1), (b0, b1) = self.edges; n = self.bins
        ia = ((a - a0) / (a1 - a0 + 1e-12) * n).long().clamp(0, n - 1); ib = ((b - b0) / (b1 - b0 + 1e-12) * n).long().clamp(0, n - 1)
        h = torch.bincount(ia * n + ib, minlength=n * n).float().view(n, n); p = h / h.sum().clamp(min=1)
        pa = p.sum(1); pb = p.sum(0); nz = p > 0
        return float((p[nz] * torch.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())


def mi_polish(mi: MI32, T_fine, c_o, maxfev: int = 300):
    """Nelder-Mead over 6 rigid params (rotvec deg about c_o, t mm; OCT world) from T_fine maximising MI32."""
    from scipy.optimize import minimize

    def T_of(p):
        P = np.eye(4); ang = np.linalg.norm(p[:3]); R = _rodrigues(p[:3] / ang, ang) if ang > 1e-9 else np.eye(3)
        P[:3, :3] = R; P[:3, 3] = p[3:] + c_o - R @ c_o; return T_fine @ P
    x0 = np.zeros(6); S = np.vstack([x0] + [x0 + np.eye(6)[i] * (1.0 if i < 3 else 0.5) for i in range(6)])
    res = minimize(lambda p: -mi(T_of(p)), x0, method="Nelder-Mead", options={"initial_simplex": S, "maxfev": maxfev, "xatol": 0.05, "fatol": 1e-6})
    return T_of(res.x), {"mi": float(-res.fun), "nfev": int(res.nfev), "params": res.x.tolist()}


# ----------------------------------------------------------------------------- verification pieces
def landscape(eval_fn, T, c_o, axes_u, trans=(0.25, 0.5, 1.0, 2.0), rots=(1.0, 2.0, 5.0), lscales=(0.03, 0.06)) -> dict:
    """1-D profiles of eval_fn (higher = better) along the 3 OCT axes: translations (mm), rotations (deg) through c_o and
    log-scales, each with argmax offset (parabolic through the 3 points around the max) and a unimodality flag."""
    out = {}; v0 = float(eval_fn(T))
    for kind, amts in (("trans", trans), ("rot", rots), ("lscale", lscales)):
        for k in range(3):
            xs = sorted({0.0} | {float(a) for a in amts} | {-float(a) for a in amts})
            ys = [v0 if x == 0 else float(eval_fn(T @ perturb_matrix(kind, axes_u[k], x, c_o))) for x in xs]; i = int(np.argmax(ys)); pk = xs[i]
            if 0 < i < len(xs) - 1:
                c = np.polyfit(np.array(xs[i - 1:i + 2]), np.array(ys[i - 1:i + 2]), 2)
                if c[0] < 0: pk = float(np.clip(-c[1] / (2 * c[0]), xs[i - 1], xs[i + 1]))
            loc = [j for j in range(1, len(xs) - 1) if ys[j] > ys[j - 1] and ys[j] > ys[j + 1]]
            out[f"{kind}_{AX[k]}"] = {"offsets": xs, "values": [float(y) for y in ys], "argmax_grid": xs[i], "argmax_offset": float(pk),
                                      "unimodal": bool(all(j == i for j in loc)), "at_end": bool(i in (0, len(xs) - 1))}
    return {"per_axis": out}


def restarts(levels, T_start, T_fine, signs, sims, cfg: FineConfig, rng, c_o, corners, pts, dev, masks=None, T_mask=None) -> dict:
    """V1: perturbed restarts (rotation U(0.4,1.2) x restart_deg, translation U(-2/3,2/3) x restart_mm per axis, log-scale
    U(-1,1) x restart_logscale per axis; defaults = v1.1's U(2,6) deg, U(-2,2) mm, U(-0.03,0.03)) through the full schedule;
    converged iff the mean displacement of the BLOCK corners (Tr vs T_fine) < restart_tol_mm(cfg) = cfg.restart_tol_mm (0.3 mm default;
    the spec's 7 mm cube tolerates ~2.5 deg on I58, recorded as converged_7mm); specimen-point displacement stats of each Tr vs
    T_fine; n_within: restarts within 0.3 / 0.5 / 1 mm at the block corners (cluster size).  masks: see run_chain."""
    r0, t0, ls0, sh0, mir = params_from_matrix(T_start, c_o); det = []; pooled = []
    tol = restart_tol_mm(cfg); a_lo, a_hi = cfg.restart_deg * 2.0 / 5.0, cfg.restart_deg * 6.0 / 5.0; t_h = cfg.restart_mm * 2.0 / 3.0
    for i in range(cfg.n_restarts):
        ang = np.deg2rad(rng.uniform(a_lo, a_hi)); ax = rng.normal(size=3); ax /= np.linalg.norm(ax); tp = rng.uniform(-t_h, t_h, 3); lp = rng.uniform(-cfg.restart_logscale, cfg.restart_logscale, 3)
        if cfg.dof == "rigid": lp[:] = 0.0                  # a rigid polish cannot undo a scale perturbation (3 % = 0.9 mm at the I58 block corners), so the restarts must not apply one
        elif cfg.dof == "similarity": lp[:] = lp[0]         # similarity: isotropic only
        Tp = compose(to_t(r0 + ax * ang, device=dev), to_t(t0 + tp, device=dev), to_t(ls0 + lp, device=dev), to_t(sh0, device=dev), to_t(c_o, device=dev), mir).cpu().numpy()
        t1 = time.time(); notes = []; Tr = run_chain(levels, Tp, signs, sims, cfg, "main", dev, notes, masks, T_mask); d = _point_disp(Tr, T_fine, pts)
        e = {"perturb_deg": float(np.rad2deg(ang)), "perturb_mm": float(np.linalg.norm(tp)), "perturb_logscale": lp.tolist(), **pose_move(Tr, T_fine, c_o, corners),
             "disp_mean_mm": float(d.mean()), "disp_p95_mm": float(np.percentile(d, 95)), "notes": notes, "seconds": time.time() - t1, "T": Tr.tolist()}
        e["converged"] = bool(e["block_corner_mean_mm"] < tol); e["converged_7mm"] = bool(e["corner_mean_mm"] < tol); det.append(e)
        if e["converged"]: pooled.append(d)
    p95 = float(np.percentile(np.concatenate(pooled), 95)) if pooled else None
    within = {f"{r:g}": sum(e["block_corner_mean_mm"] < r for e in det) for r in (0.3, 0.5, 1.0)}
    return {"n": len(det), "n_converged": sum(e["converged"] for e in det), "n_converged_7mm": sum(e["converged_7mm"] for e in det), "detail": det, "p95_disp_converged_mm": p95,
            "tol_mm": tol, "n_within_mm": within, "size": {"deg_range": [a_lo, a_hi], "mm_per_axis": t_h, "logscale": cfg.restart_logscale}}


def split_half(L: Level, m_spec, T_fine, sign, sim, cfg: FineConfig, pts, dev, T_mask=None) -> dict:
    """V3: per OCT array axis split m_spec at the median coordinate; affine 100 iters from T_fine on each half; disagreement of the
    two half solutions over the specimen points (mean mm); loho_max_mm = max over axes.  T_mask: see make_refiner."""
    idx = torch.nonzero(m_spec); s = 1.0 if L.mm >= 0.25 else 0.5; per, detail = [], []
    for k in range(3):
        med = float(idx[:, k].float().median()); shp = [1, 1, 1]; shp[k] = -1
        coord = torch.arange(m_spec.shape[k], device=m_spec.device).view(shp); Ts = []; deg = []
        for h in (m_spec & (coord <= med), m_spec & (coord > med)):
            ref = make_refiner(L, h, T_fine, sign, sim, cfg, dev, T_mask); deg.append(bool(ref.degenerate))
            Th, _ = ref.refine_delta("affine", 100, lr_rot=0.01 * s, lr_t=0.15 * s, lr_ls=0.005 * s, lr_sh=0.005 * s, clamp=cfg.clamp, reg=cfg.reg, subsample=cfg.subsample)
            Ts.append(Th); del ref
        d = point_disp_stats(Ts[0], Ts[1], pts); ok = not any(deg)                  # a degenerate half (LCC box wider than the half) cannot move: no test
        per.append(d["mean_mm"] if ok else None); detail.append({"axis": AX[k], **d, "median_index": med, "degenerate_halves": deg})
    have = [p for p in per if p is not None]
    return {"per_axis_mm": per, "loho_max_mm": float(max(have)) if have else None, "detail": detail}


def inverse_consistency(levels, T_start, T_fine, signs, sims, sim_f, cfg: FineConfig, pts, dev) -> dict:
    """V5: backward problem MRI -> OCT with FineDense roles swapped (fixed = MRI level grid, mask = eroded MRI tissue & OCT mask
    through inv(T_start); moving = flattened OCT with its validity), started at inv(T_start), same sign; ICE = |T_fine p - inv(T_bwd) p|.
    cfg.fixed_mask: the backward mask is taken at inv(T_start) on every level (v1.1: at the current backward pose)."""
    T_b = np.linalg.inv(T_start); notes = []; ran = 0
    for i, L in enumerate(levels):
        if sims[i] is None: continue
        om = resample_to_grid(L.mo_f, L.A_o_t, L.A_m_t, L.shape_m, T_grid_to_vol=to_t(np.linalg.inv(T_start) if cfg.fixed_mask else T_b, device=dev))[0] > 0.5
        sim_b = sim_f if signs[i] != 0 else "lcc2"
        ref = FineDense(L.F_m[0], L.A_m, L.t_m_er_b & om, L.F_o, L.A_o, L.mo_f, T_b, signs[i], sim_b, int(round(cfg.lcc_mm / L.vox_m)) | 1, dev)
        if ref.degenerate: notes.append(f"degenerate@{L.mm}"); del ref; continue
        T_b, _ = run_steps(ref, _schedule("lcc", i, L.mm, cfg), cfg); ran += 1
        if ref.failed_steps: notes.append(f"failed@{L.mm}")
        del ref
    if ran == 0:                                                                     # the backward problem never moved: no consistency test
        return {"mean_mm": None, "p95_mm": None, "max_mm": None, "T_bwd": T_b.tolist(), "notes": notes, "degenerate": True}
    d = _point_disp(T_fine, np.linalg.inv(T_b), pts)
    return {"mean_mm": float(d.mean()), "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()), "T_bwd": T_b.tolist(), "notes": notes, "degenerate": False}


# ----------------------------------------------------------------------------- the stage
def fine_stage(T_start, ctx: FineContext, cfg: FineConfig, rng, probe_only: bool = False):
    """-> (T_accepted, T_candidate, info).  Runs the level schedule from T_start, then the label-free verification (V1-V5) and
    the acceptance gate (a)-(f); rejected -> T_accepted = T_start with every number logged.  probe_only: guard + loss at T_start
    per level, no optimisation (test_fine.py --init-only)."""
    t0 = time.time(); dev = ctx.device; T_start = np.asarray(T_start, float).copy(); c_o = np.asarray(ctx.c_o, float)
    axes_u = [ctx.A_oct[:3, k] / np.linalg.norm(ctx.A_oct[:3, k]) for k in range(3)]; corners = block_corners(ctx.A_oct, ctx.oct150.shape)
    info = {"mode": "on", "sim": cfg.sim, "levels_mm": [float(x) for x in cfg.levels], "verify": cfg.verify, "max_move_mm": cfg.max_move, "max_rot_deg": cfg.max_rot,
            "fixed_mask": bool(cfg.fixed_mask), "fixed_weight": weight_fixed(cfg), "fixed_weight_option": cfg.fixed_weight if isinstance(cfg.fixed_weight, str) else str(cfg.fixed_weight),
            "restart_mm": cfg.restart_mm, "restart_deg": cfg.restart_deg, "restart_logscale": cfg.restart_logscale,
            "restart_tol_mm": restart_tol_mm(cfg), "exclude_fov_mm": cfg.exclude_fov_mm, "dof": cfg.dof}
    bc = (T_start @ np.r_[c_o, 1.0])[:3]
    vlo, vhi, A_box, mri_b, tis_b = mri_box(ctx.mri, ctx.mri_tissue, ctx.A_mri, ctx.lo, ctx.hi, bc, ctx.BLOCK_R + 8.0)
    box = (to_t(mri_b, device=dev)[None], to_t(tis_b.astype(np.float32), device=dev)[None], A_box); del mri_b, tis_b
    info["mri_box_ijk"] = [vlo.tolist(), vhi.tolist()]
    fov = None
    if cfg.exclude_fov_mm > 0:                               # box faces that ARE field-of-view faces = faces of the region [lo,hi) (mri.npy array faces, or the --crop-centre box faces)
        f_lo = [bool(vlo[k] <= ctx.lo[k]) for k in range(3)]; f_hi = [bool(vhi[k] >= ctx.hi[k]) for k in range(3)]
        a_lo = [bool(f and vlo[k] <= 0) for k, f in enumerate(f_lo)]; a_hi = [bool(f and vhi[k] >= ctx.mri.shape[k]) for k, f in enumerate(f_hi)]   # which flagged faces are true array faces
        fov = (np.asarray(vhi - vlo, float), np.linalg.norm(np.asarray(A_box)[:3, :3], axis=0), f_lo, f_hi)
        info["fov_faces"] = {"lo": f_lo, "hi": f_hi, "array_lo": a_lo, "array_hi": a_hi, "region_ijk": [np.asarray(ctx.lo).tolist(), np.asarray(ctx.hi).tolist()],
                             "region": "array" if (np.all(np.asarray(ctx.lo) == 0) and np.all(np.asarray(ctx.hi) == np.asarray(ctx.mri.shape))) else "crop"}
    O = to_t(ctx.oct150, device=dev)[None]; MK = to_t(ctx.oct_mask.astype(np.float32), device=dev)[None]
    levels = [Level(O, MK, ctx.A_oct, box, L, cfg, ctx.VOX_O, ctx.VOX_M, dev, fov) for L in cfg.levels]; del O, MK, box
    info["level_grids"] = [{"level": L.mm, "oct_shape": list(L.shape_o), "oct_vox_mm": L.vox_o, "mri_shape": list(L.shape_m), "mri_vox_mm": L.vox_m,
                            **({"fov_excluded_vox": L.n_fov_excluded, "fov_excluded_cm3": L.n_fov_excluded * L.vox_m ** 3 / 1000} if fov is not None else {})} for L in levels]
    info["seconds_prep"] = time.time() - t0

    # ---- main run: guard -> similarity -> delta refinement, level by level (m_spec fixed within a level; cfg.fixed_mask: at T_start on every level,
    #      and the ncc weight frozen there too unless fixed_weight 'off')
    T_cur = T_start.copy(); signs, sims, guard, lv, spec_masks = [], [], [], [], []; last = None
    for i, L in enumerate(levels):
        tl = time.time(); m_spec = L.spec_mask(T_start if cfg.fixed_mask else T_cur); spec_masks.append(m_spec)
        sign, g = polarity_guard(L.I_o, L.A_o, L.I_m, L.A_m, m_spec, T_cur, min_vox=int(2000 * (0.3 / L.mm) ** 3), level=L); g["level"] = L.mm; guard.append(g)
        sim = cfg.sim if cfg.sim != "auto" else ("ncc" if sign != 0 else "lcc2")
        e = {"level": L.mm, "sim": sim, "sign": sign, "n_spec": int(m_spec.sum()), "spec_cm3": float(m_spec.sum()) * L.vox_o ** 3 / 1000, "skipped": None,
             "mask_pose": "start" if cfg.fixed_mask else "current",
             "ncc_weight": (("frozen@start" if cfg.fixed_mask else "frozen@current") if weight_fixed(cfg) else "follows_pose") if sim == "ncc" else None}
        if e["n_spec"] < 1000: e["skipped"] = "specimen_mask_too_small"
        elif sim in ("ncc", "lcc") and sign == 0: e["skipped"] = "mixed_polarity"
        if e["skipped"]:
            signs.append(0); sims.append(None); lv.append(e); continue
        ref = make_refiner(L, m_spec, T_cur, sign, sim, cfg, dev, T_start if cfg.fixed_mask else None)
        if ref.degenerate:                                   # LCC box wider than the specimen: the dense loss is undefined everywhere -> skip, do not crash
            e.update(skipped="lcc_window_too_large", lcc_box_vox=ref.w_box, lcc_box_mm=ref.w_box * L.vox_o); signs.append(0); sims.append(None); lv.append(e); del ref; continue
        e["loss_before"] = ref.evaluate_T(T_cur); e["ncc_before"] = 1.0 - e["loss_before"]
        if probe_only:
            signs.append(sign); sims.append(sim); lv.append(e); last = (L, m_spec, ref, sign, sim); continue
        T_new, loss1 = run_steps(ref, _schedule("main", i, L.mm, cfg), cfg); mv = pose_move(T_new, T_cur, c_o, corners)
        e.update(loss_after=float(loss1), ncc_after=1.0 - float(loss1), moved_mm=mv["corner_mean_mm"], moved_block_mm=mv["block_corner_mean_mm"],
                 rotated_deg=mv["delta_rotation_deg"], failed_steps=ref.failed_steps, seconds=time.time() - tl)
        signs.append(sign); sims.append(sim); lv.append(e); last = (L, m_spec, ref, sign, sim); T_cur = T_new
    T_fine = T_cur; masks = spec_masks if cfg.fixed_mask else None; T_mask = T_start if cfg.fixed_mask else None   # chains re-derive m_spec at their own pose unless fixed_mask
    info.update(sim_per_level=[l["sim"] if not l["skipped"] else None for l in lv], sign_per_level=signs, guard=guard, levels=lv)
    if probe_only or last is None:
        info.update(accepted=False, reasons=["probe_only"] if probe_only else ["no_level"], U_mm=None, seconds=time.time() - t0)
        return T_start, T_start, info

    # ---- verification at the finest executed level
    Lf, m_spec_f, ref_f, sign_f, sim_f = last; tv = time.time()
    idx = torch.nonzero(m_spec_f); sel = rng.choice(len(idx), min(cfg.n_points, len(idx)), replace=False)
    pts = apply_affine(Lf.A_o_t, idx[torch.as_tensor(np.sort(sel), device=dev)].float()).cpu().numpy(); del idx
    loss_start_f, loss_fine_f = ref_f.evaluate_T(T_start), ref_f.evaluate_T(T_fine)
    struct_before = struct_after = None
    if ctx.ref015 is not None: struct_before = 1.0 - ctx.ref015.evaluate(T_start); struct_after = 1.0 - ctx.ref015.evaluate(T_fine)
    mi = MI32(Lf, m_spec_f, dev); mi.fix_edges(T_fine); mi_after = mi(T_fine); mi_before = mi(T_start)
    vs = pose_move(T_fine, T_start, c_o, corners); D_tot = np.linalg.inv(T_start) @ T_fine
    vs["centre_shift_oct_axes_mm"] = [float(np.dot((D_tot @ np.r_[c_o, 1.0])[:3] - c_o, u)) for u in axes_u]
    ls_s = params_from_matrix(T_start, c_o)[2]; ls_f = params_from_matrix(T_fine, c_o)[2]
    info.update(finest_level=Lf.mm, finest_sim=sim_f, finest_sign=sign_f, loss_start_finest=loss_start_f, loss_fine_finest=loss_fine_f,
                struct_ncc_before=struct_before, struct_ncc_after=struct_after, mi_before=mi_before, mi_after=mi_after, mi_edges=mi.edges, vs_start=vs,
                stretch_ijk_before=stretch_ijk(T_start, ctx.A_oct), stretch_ijk_after=stretch_ijk(T_fine, ctx.A_oct), cum_log_scale=(ls_f - ls_s).tolist(),
                fine_large_move=bool(vs["block_corner_mean_mm"] > 1.5))
    # V1 restarts
    t1 = time.time(); rs = restarts(levels, T_start, T_fine, signs, sims, cfg, rng, c_o, corners, pts, dev, masks, T_mask) if cfg.n_restarts > 0 else {"n": 0, "n_converged": 0, "n_converged_7mm": 0, "detail": [], "p95_disp_converged_mm": None, "tol_mm": restart_tol_mm(cfg)}
    rs["seconds"] = time.time() - t1; info["restarts"] = rs
    # V2 multi-similarity: dense LCC chain from T_start, MI polish from T_fine (a degenerate LCC chain = no independent witness, not a disagreement)
    t1 = time.time(); lcc_notes = []; T_lcc = run_chain(levels, T_start, signs, sims, cfg, "lcc", dev, lcc_notes, masks, T_mask); T_mi, mi_info = mi_polish(mi, T_fine, c_o)
    lcc_ok = not any(n.startswith("degenerate") for n in lcc_notes)
    info["multi_sim"] = {"fine_lcc": point_disp_stats(T_fine, T_lcc, pts) if lcc_ok else None, "fine_mi": point_disp_stats(T_fine, T_mi, pts), "lcc_mi": point_disp_stats(T_lcc, T_mi, pts) if lcc_ok else None,
                         "lcc_vs_start": pose_move(T_lcc, T_start, c_o, corners), "lcc_notes": lcc_notes, "mi_polish": mi_info, "T_lcc": T_lcc.tolist(), "T_mi": T_mi.tolist(), "seconds": time.time() - t1}
    # V3 split-half
    t1 = time.time(); info["split_half"] = split_half(Lf, m_spec_f, T_fine, sign_f, sim_f, cfg, pts, dev, T_mask); info["split_half"]["seconds"] = time.time() - t1
    # V4 landscapes at T_fine (fine loss sign-corrected, higher = better; MI32 on the same 9 axes)
    t1 = time.time()
    info["landscape"] = {"fine": landscape(lambda T: 1.0 - ref_f.evaluate_T(T), T_fine, c_o, axes_u), "mi": landscape(mi, T_fine, c_o, axes_u), "seconds": time.time() - t1}
    # V5 inverse consistency (full only)
    info["ice"] = None
    if cfg.verify == "full":
        t1 = time.time(); info["ice"] = inverse_consistency(levels, T_start, T_fine, signs, sims, sim_f, cfg, pts, dev); info["ice"]["seconds"] = time.time() - t1
    info["seconds_verify"] = time.time() - tv

    # ---- acceptance gate (all must hold).  (c) is measured at the block's own corners plus a rotation cap on the delta (the spec's
    # 7 mm transform_diff cube under-reads rotations of a large block and the loss is flat in rotation); (d) is REQUIRED (no restarts =
    # unverified); (g) the self-consistency spread must not exceed the allowed move; (h) no optimisation step may have failed.
    reasons = []
    if not (loss_fine_f <= loss_start_f - 0.002): reasons.append(f"(a) loss gain {loss_start_f - loss_fine_f:.4f} < 0.002")
    if struct_after is not None and struct_after < struct_before - 0.02: reasons.append(f"(b) structural NCC {struct_before:.4f} -> {struct_after:.4f} (drop > 0.02)")
    if vs["block_corner_mean_mm"] > cfg.max_move: reasons.append(f"(c) moved {vs['block_corner_mean_mm']:.2f} mm at the block corners > {cfg.max_move} (7 mm cube: {vs['corner_mean_mm']:.2f})")
    if vs["delta_rotation_deg"] > cfg.max_rot: reasons.append(f"(c) rotated {vs['delta_rotation_deg']:.2f} deg > {cfg.max_rot}")
    need = int(math.ceil(0.625 * rs["n"]))                                            # 5 of 8
    if rs["n"] == 0: reasons.append("(d) no restarts run (n_restarts = 0): convergence unverified")
    elif rs["n_converged"] < need: reasons.append(f"(d) restarts converged {rs['n_converged']}/{rs['n']} < {need} (block corners < {rs['tol_mm']:g} mm; 7 mm cube: {rs['n_converged_7mm']})")
    if mi_after < mi_before - 0.001: reasons.append(f"(e) MI32 {mi_before:.4f} -> {mi_after:.4f} (drop > 0.001)")
    if lv[-1]["skipped"]: reasons.append(f"(f) finest level skipped ({lv[-1]['skipped']})")
    ms = info["multi_sim"]; ms_have = [ms[k]["mean_mm"] for k in ("fine_lcc", "fine_mi", "lcc_mi") if ms[k] is not None]
    cands = [rs["p95_disp_converged_mm"], max(ms_have) if ms_have else None, info["split_half"]["loho_max_mm"]]
    U = [c for c in cands if c is not None]; U_mm = float(max(U)) if U else None
    if U_mm is not None and U_mm > cfg.max_move: reasons.append(f"(g) U_mm {U_mm:.2f} > {cfg.max_move} (self-consistency spread exceeds the allowed move)")
    failed = [l["level"] for l in lv if l.get("failed_steps")]
    if failed: reasons.append(f"(h) optimisation returned no finite loss at level(s) {failed}")
    accepted = not reasons
    info.update(accepted=bool(accepted), reasons=reasons, U_mm=U_mm, U_components={"restart_p95": cands[0], "multi_sim_mean_max": cands[1], "loho_max": cands[2]},
                seconds=time.time() - t0)
    return (T_fine if accepted else T_start), T_fine, info
