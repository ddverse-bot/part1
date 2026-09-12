"""Shared utilities: physical-space affines, GPU resampling, OCT preprocessing, I/O.

Conventions
-----------
* A volume is a torch tensor [C, D, H, W] (or numpy [D, H, W]) with a 4x4 affine ``A`` mapping voxel
  index (i, j, k) = (D, H, W) axes to world millimetres.  For NIfTI loaded with nibabel the array is
  (i, j, k) with ``img.affine`` — we keep that order (no transposition), so D=i, H=j, W=k.
* All transforms between volumes are 4x4 world->world matrices in millimetres.
* ``sample_at_world`` samples a source volume at arbitrary world points with trilinear
  interpolation (torch.grid_sample), which is the single primitive that search, refinement and
  evaluation are built on.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- affines
def to_t(a, dtype=torch.float32, device=DEVICE):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def homog(p: torch.Tensor) -> torch.Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)


def apply_affine(A: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """A: [4,4]; p: [..., 3] -> [..., 3]."""
    return (homog(p) @ A.T)[..., :3]


def voxel_size(A) -> np.ndarray:
    A = np.asarray(A)
    return np.linalg.norm(A[:3, :3], axis=0)


def iso_grid_affine(origin_mm, spacing_mm) -> np.ndarray:
    """World-axis-aligned isotropic grid: voxel (i,j,k) -> origin + spacing*(i,j,k)."""
    A = np.eye(4)
    A[:3, :3] = np.eye(3) * spacing_mm
    A[:3, 3] = np.asarray(origin_mm, dtype=float)
    return A


def rotvec_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """Rodrigues; r: [3] -> [3,3] (differentiable)."""
    theta = torch.linalg.norm(r) + 1e-12
    k = r / theta
    K = torch.zeros(3, 3, dtype=r.dtype, device=r.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    I = torch.eye(3, dtype=r.dtype, device=r.device)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = math.acos(cos)
    if theta < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(theta))
    return w * theta


def random_rotations(n: int, seed: int = 0) -> np.ndarray:
    """Uniform random rotations on SO(3) via random unit quaternions -> [n,3,3]."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1).reshape(n, 3, 3)
    R[0] = np.eye(3)
    return R


def rotation_geodesic_deg(R1, R2) -> float:
    R = np.asarray(R1) @ np.asarray(R2).T
    if np.linalg.det(R) < 0:          # different handedness: not comparable
        return 180.0
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def polar_rotation(M: np.ndarray) -> np.ndarray:
    """Closest rotation to a 3x3 linear map (polar decomposition, det>0)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ----------------------------------------------------------------------------- resampling
def sample_at_world(vol: torch.Tensor, A_vol: torch.Tensor, pts_world: torch.Tensor,
                    mode: str = "bilinear", padding: float = 0.0) -> torch.Tensor:
    """Sample vol [C,D,H,W] (voxel->world affine A_vol) at world points [...,3] -> [C, ...].

    Points outside the volume get ``padding`` (implemented through zeros padding + a validity mask
    when padding != 0).  Trilinear ('bilinear') or 'nearest'.
    """
    C, D, H, W = vol.shape
    Ainv = torch.linalg.inv(A_vol)
    ijk = apply_affine(Ainv, pts_world)                                # voxel coords (float)
    # grid_sample wants normalized coords in (x=W, y=H, z=D) order, align_corners=True: -1..1 maps 0..N-1
    size = torch.tensor([D - 1, H - 1, W - 1], dtype=ijk.dtype, device=ijk.device)
    norm = ijk / size * 2 - 1
    grid = norm[..., [2, 1, 0]]                                          # (x, y, z)
    shp = grid.shape[:-1]
    grid = grid.reshape(1, 1, 1, -1, 3)
    out = F.grid_sample(vol[None], grid, mode=mode, padding_mode="zeros", align_corners=True)
    out = out.reshape(C, *shp)
    if padding != 0.0:
        inside = ((norm >= -1) & (norm <= 1)).all(-1)
        out = torch.where(inside[None], out, torch.full_like(out, padding))
    return out


def grid_points(A_grid: torch.Tensor, shape) -> torch.Tensor:
    """World coordinates of every voxel of a grid -> [D,H,W,3]."""
    D, H, W = shape
    ii, jj, kk = torch.meshgrid(torch.arange(D, device=A_grid.device, dtype=A_grid.dtype),
                                torch.arange(H, device=A_grid.device, dtype=A_grid.dtype),
                                torch.arange(W, device=A_grid.device, dtype=A_grid.dtype), indexing="ij")
    p = torch.stack([ii, jj, kk], -1)
    return apply_affine(A_grid, p)


def resample_to_grid(vol: torch.Tensor, A_vol: torch.Tensor, A_grid: torch.Tensor, shape,
                     T_grid_to_vol: torch.Tensor | None = None, mode="bilinear", chunk=8_000_000) -> torch.Tensor:
    """Resample vol onto a target grid (A_grid, shape). Optionally apply a world transform
    (grid world -> vol world) first.  Chunked over voxels to bound memory."""
    pts = grid_points(A_grid, shape).reshape(-1, 3)
    if T_grid_to_vol is not None:
        pts = apply_affine(T_grid_to_vol, pts)
    C = vol.shape[0]
    out = torch.empty((C, pts.shape[0]), dtype=vol.dtype, device=vol.device)
    for s in range(0, pts.shape[0], chunk):
        out[:, s:s + chunk] = sample_at_world(vol, A_vol, pts[s:s + chunk], mode=mode)
    return out.reshape(C, *shape)


def avg_pool_iso(vol: torch.Tensor, A_vol: np.ndarray, factor: int) -> tuple[torch.Tensor, np.ndarray]:
    """Integer-factor average pooling of [C,D,H,W] with the affine updated (voxel centres)."""
    v = F.avg_pool3d(vol[None], factor, factor, ceil_mode=False)[0]
    A = np.asarray(A_vol).copy()
    A[:3, 3] = A[:3, 3] + A[:3, :3] @ (np.ones(3) * (factor - 1) / 2.0)
    A[:3, :3] = A[:3, :3] * factor
    return v, A


def crop_volume(vol: torch.Tensor, A_vol: np.ndarray, lo, hi) -> tuple[torch.Tensor, np.ndarray]:
    lo = np.asarray(lo, int); hi = np.asarray(hi, int)
    sub = vol[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    A = np.asarray(A_vol).copy()
    A[:3, 3] = (np.asarray(A_vol) @ np.r_[lo, 1.0])[:3]
    return sub, A


def world_bbox_to_voxel(A_vol: np.ndarray, shape, wlo, whi):
    """Voxel bbox (lo inclusive, hi exclusive) covering a world-aligned box, clipped to the volume."""
    corners = np.array([[x, y, z] for x in (wlo[0], whi[0]) for y in (wlo[1], whi[1]) for z in (wlo[2], whi[2])])
    inv = np.linalg.inv(np.asarray(A_vol))
    v = (inv @ np.c_[corners, np.ones(8)].T).T[:, :3]
    lo = np.clip(np.floor(v.min(0)).astype(int), 0, np.asarray(shape) - 1)
    hi = np.clip(np.ceil(v.max(0)).astype(int) + 1, 1, np.asarray(shape))
    return lo, hi


# ----------------------------------------------------------------------------- OCT preprocessing
def oct_slab_normalize(oct_zyx: np.ndarray, tissue_thresh: float = 30.0, window: int = 100, out_dtype=np.float32) -> tuple[np.ndarray, dict]:
    """Remove the periodic per-section depth attenuation of serial-sectioning OCT.

    For every z-slice compute the tissue median; divide the slice by it and multiply by a slowly
    varying reference (running mean of the medians over `window` slices, i.e. one section period),
    so slow trends are kept and the ~100-slice sawtooth is removed.  Returns float32 volume.
    """
    z = oct_zyx.shape[0]
    med = np.full(z, np.nan)
    for i in range(z):
        s = oct_zyx[i]
        t = s[s > tissue_thresh]
        if t.size > 2000:
            med[i] = np.median(t)
    good = np.isfinite(med)
    idx = np.arange(z)
    med_f = np.interp(idx, idx[good], med[good]) if good.any() else np.ones(z)
    from scipy.ndimage import uniform_filter1d
    ref = uniform_filter1d(med_f, size=window, mode="nearest")
    scale = np.where(med_f > 1e-3, ref / med_f, 1.0).astype(np.float32)
    # float16 output must not overflow (max 65504): put the tissue median at ~16k and cap bright outliers.
    gain = 1.0
    if np.dtype(out_dtype) == np.float16:
        gain = 16384.0 / max(float(np.nanmedian(ref)), 1e-3); cap = 60000.0
    out = np.empty(oct_zyx.shape, dtype=out_dtype)                   # one output volume; scaled slice by slice
    for i in range(z):
        v = np.asarray(oct_zyx[i], dtype=np.float32) * (scale[i] * gain)
        if gain != 1.0: np.clip(v, 0, cap, out=v)
        out[i] = v.astype(out_dtype, copy=False)
    return out, {"per_slice_median": med.tolist(), "reference": ref.tolist(), "scale": scale.tolist(),
                 "window": window, "tissue_thresh": tissue_thresh, "gain": float(gain)}


def histogram_tissue_threshold(sub: np.ndarray, bins: int = 256, prom: float = 0.05, deep: float = 0.5):
    """Foreground/tissue threshold from a 1-D intensity sample (positive, outlier-clipped): find the tissue peak
    (rightmost prominent mode) and walk left over prominent peaks; cut at the first valley deeper than `deep` x the
    smaller neighbouring peak.  No deep valley (all-tissue field of view) -> a floor that keeps everything.
    Returns (threshold, info)."""
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    from skimage.filters import threshold_multiotsu
    sub = np.asarray(sub); sub = sub[np.isfinite(sub) & (sub > 0)]
    hi = np.percentile(sub, 99.5); sub = sub[sub < hi]
    h, e = np.histogram(sub, bins=bins); c = 0.5 * (e[1:] + e[:-1]); hs = uniform_filter1d(h.astype(float), 5)
    pk, _ = find_peaks(hs, prominence=prom * hs.max(), distance=5)
    try: mo = float(threshold_multiotsu(sub, classes=3)[0])
    except Exception: mo = float(np.percentile(sub, 10))
    thr = float(min(np.percentile(sub, 1), 0.5 * mo)); vr = None
    if len(pk) >= 2:
        i_tis = pk[-1]
        for i_left in pk[:-1][::-1]:
            seg = hs[i_left:i_tis + 1]; i_val = i_left + int(np.argmin(seg))
            r = float(hs[i_val] / max(min(hs[i_left], hs[i_tis]), 1.0))
            if r < deep:
                thr = float(c[i_val]); vr = r; break
            i_tis = i_left
    return thr, {"n_peaks": int(len(pk)), "peak_pos": [float(c[i]) for i in pk], "valley_ratio": vr}


def oct_tissue_mask(vol_zyx: np.ndarray, thresh: float | None = None, closing_iter: int = 2) -> tuple[np.ndarray, float]:
    """Tissue mask of a (pooled) OCT block: threshold (Otsu on positive voxels by default) +
    closing + hole filling + largest connected component."""
    from scipy import ndimage
    if thresh is None:
        thresh, _ = histogram_tissue_threshold(vol_zyx[::2, ::2, ::2])
    m = vol_zyx > thresh
    if closing_iter:
        m = ndimage.binary_closing(m, iterations=closing_iter)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m, thresh


def pool_mean_np(a: np.ndarray, k) -> np.ndarray:
    """Mean-pool a (Z,Y,X) array by integer factor k (scalar or per-axis (kz,ky,kx)); trailing remainder dropped."""
    kz, ky, kx = (int(v) for v in np.broadcast_to(np.asarray(k), (3,)))
    z, y, x = a.shape[0] // kz * kz, a.shape[1] // ky * ky, a.shape[2] // kx * kx
    out = np.empty((z // kz, y // ky, x // kx), dtype=np.float32)
    step = max(1, 64 // kz) * kz                                     # chunk over z to bound memory on big stacks
    for z0 in range(0, z, step):
        blk = a[z0:min(z0 + step, z), :y, :x].astype(np.float32)
        out[z0 // kz:(z0 + blk.shape[0]) // kz] = blk.reshape(blk.shape[0] // kz, kz, y // ky, ky, x // kx, kx).mean(axis=(1, 3, 5))
    return out


def pooled_affine(A: np.ndarray, k) -> np.ndarray:
    """Affine of the array mean-pooled by k (scalar or per-axis) from an array with affine A (voxel-centre convention)."""
    kz = np.broadcast_to(np.asarray(k, dtype=float), (3,))
    Ap = A.copy(); Ap[:3, 3] = A[:3, 3] + A[:3, :3] @ ((kz - 1) / 2.0); Ap[:3, :3] = A[:3, :3] * kz[None, :]
    return Ap


def layout_affine(shape_zyx, vox_mm, layout: str = "SPR", center_origin: bool = False) -> np.ndarray:
    """World affine for an array stored (Z,Y,X); layout letters give the RAS direction of Z, Y, X.
    vox_mm: scalar or per-axis (z,y,x) voxel size in mm."""
    sign = {"R": (0, +1), "L": (0, -1), "A": (1, +1), "P": (1, -1), "S": (2, +1), "I": (2, -1)}
    vz = np.broadcast_to(np.asarray(vox_mm, dtype=float), (3,))
    A = np.zeros((4, 4)); A[3, 3] = 1
    for ax, letter in enumerate(layout):
        w, s = sign[letter]; A[w, ax] = s * vz[ax]
    if center_origin:
        c = (np.array(shape_zyx, dtype=float) - 1) / 2.0
        A[:3, 3] = -(A[:3, :3] @ c)
    return A


# ----------------------------------------------------------------------------- I/O
def save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path, dtype=None) -> None:
    import nibabel as nib
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    img = nib.Nifti1Image(a, np.asarray(affine))
    img.header.set_xyzt_units("mm")
    img.header.set_qform(np.asarray(affine), code=1)
    img.header.set_sform(np.asarray(affine), code=1)
    nib.save(img, str(path))


def load_nifti(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), np.asarray(img.affine)


def write_json(obj, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return np.asarray(o.detach().cpu() if isinstance(o, torch.Tensor) else o).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def lta_text(M: np.ndarray, src_desc: str = "oct", dst_desc: str = "mri") -> str:
    """FreeSurfer-style LINEAR_RAS_TO_RAS text (src world -> dst world)."""
    rows = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in np.asarray(M))
    return f"type      = 1  # LINEAR_RAS_TO_RAS\nnxforms   = 1\nmean      = 0.0000 0.0000 0.0000\nsigma     = 1.0000\n1 4 4\n{rows}\n# src: {src_desc}\n# dst: {dst_desc}\n"
