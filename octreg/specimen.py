"""OCT specimen mask for scatterer-doped agarose blocks (probe P1 recipe g): texture markers + rim watershed.

Intensity cannot separate the specimen from doped agarose (equal medians), but the agarose artefacts are anisotropic
(tile stripes along x, per-section sawtooth along z) while tissue texture is isotropic: the min-over-axes directional
std of the band-passed octv, normalised by the local mean and pooled to a ~0.16 mm working grid W, gives inner/outer
markers; a watershed on the 0.04 mm rim strength (max-pooled to W) snaps the boundary to the bright specimen rim where
one exists; exterior pockets enclosed by the specimen are reassigned; closing / largest component / hole filling; the
W mask is resampled to the 0.15 mm grid and to the octv grid.  All sizes are given in mm and converted with the octv
voxel size.  CPU only: multiprocessing.Pool(n_workers) over z-chunks of octv (forked workers read the parent's array).
Reference: work/probe_xr/mask/scripts/{p1_aniso04,p1_rim04,p1_final,p1_common}.py (I58: 18.05 cm3, 1 CC, ~4 min).
"""
from __future__ import annotations
import os, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter, map_coordinates, binary_fill_holes, label

# recipe constants (mm); P1 validated at 0.04 mm octv / 0.16 mm working grid (4 / 9 / 4 / 3 / 5 / 4 / 3 voxels or cells)
BP_SIGMA_MM = 0.08     # Gaussian band-pass before the directional std (removes speckle)
STD_WIN_MM = 0.36      # masked 1-D running-std window per axis, and the 3-D box of the local mean
GRID_MM = 0.16         # working grid W = octv mean-pooled by kW = round(GRID_MM / vox)
SMOOTH_MM = 0.48       # masked Gaussian smoothing of the pooled fields on W
RIM_WIN_MM = 0.20      # 3-D local-std window of the rim strength (then Gaussian sigma = 1 octv voxel, max-pool by kW)
OPEN_MM = 0.64         # ball radius of the marker opening
CLOSE_MM = 0.48        # ball radius of the cleanup closing (edge-padded)

_SRC = None            # source volume shared with the forked workers (set right before a Pool is created)
_AUX = None            # auxiliary array for the resample workers (mask on W as float32)


# ----------------------------------------------------------------------------- small helpers
def _odd(x: float) -> int:
    n = max(1, int(round(x))); return n if n % 2 else n + 1

def ball(r: int) -> np.ndarray:
    z, y, x = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]; return (z * z + y * y + x * x) <= r * r

def pooled_affine_np(A, k) -> np.ndarray:
    """Affine of an array mean-pooled by k (voxel-centre convention; same as common.pooled_affine, torch-free)."""
    kz = np.broadcast_to(np.asarray(k, dtype=float), (3,)); A = np.asarray(A, float)
    Ap = A.copy(); Ap[:3, 3] = A[:3, 3] + A[:3, :3] @ ((kz - 1) / 2.0); Ap[:3, :3] = A[:3, :3] * kz[None, :]
    return Ap

def _pool(a: np.ndarray, k: int, how: str) -> np.ndarray:
    """Block-pool a (Z,Y,X) array by k along all axes (trailing remainder dropped); how = 'mean' | 'max'."""
    z, y, x = (s // k * k for s in a.shape); b = a[:z, :y, :x].reshape(z // k, k, y // k, k, x // k, k)
    return b.mean(axis=(1, 3, 5)) if how == "mean" else b.max(axis=(1, 3, 5))

def largest_cc(m: np.ndarray) -> tuple[np.ndarray, int]:
    """(largest connected component of m, number of components before the filter)."""
    lab, n = label(m)
    if n <= 1: return m.copy(), int(n)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(np.argmax(sizes)), int(n)

def cleanup(m: np.ndarray, close_r: int = 0) -> tuple[np.ndarray, int]:
    """Optional closing on an edge-padded array (plain binary_closing erodes a mask cut by the box faces), largest
    component, 3-D hole filling -> (mask, n_components_before_largest)."""
    if close_r > 0:
        mp = np.pad(m, close_r, mode="edge"); mp = ndimage.binary_closing(mp, structure=ball(close_r))
        m = mp[close_r:-close_r, close_r:-close_r, close_r:-close_r]
    m, n = largest_cc(m); return binary_fill_holes(m), n

def erode_mm(mask: np.ndarray, r_mm: float, vox_mm: float) -> np.ndarray:
    """Erosion by a ball of radius r_mm through the Euclidean distance transform (cheap for large radii); the box
    faces count as boundary (1-voxel False padding), so the result is defined even for an all-True mask."""
    d = ndimage.distance_transform_edt(np.pad(np.asarray(mask, bool), 1))[1:-1, 1:-1, 1:-1]
    return d * vox_mm >= r_mm

def gmm2_1d(v, iters: int = 100, n_sub: int = 500_000):
    """2-component 1-D Gaussian mixture by EM (numpy; init means p25/p75, equal sd = std/2, weights 0.5) on a strided
    subsample -> (equal-weighted-density crossing between the two means, {log_means, sds, weights})."""
    v = np.asarray(v, np.float64); v = v[::max(1, v.size // n_sub)]
    mu = np.percentile(v, [25, 75]); sd = np.array([v.std() / 2] * 2); w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = np.stack([w[k] / sd[k] * np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) for k in range(2)], 1) + 1e-300
        r = ll / ll.sum(1, keepdims=True); n = r.sum(0); w = n / n.sum(); mu = (r * v[:, None]).sum(0) / n
        sd = np.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / n) + 1e-6
    lo, hi = np.argsort(mu); grid = np.linspace(mu[lo], mu[hi], 400)
    post = [w[k] / sd[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) for k in range(2)]
    return float(grid[np.argmin(np.abs(post[lo] - post[hi]))]), {"log_means": mu.tolist(), "sds": sd.tolist(), "weights": w.tolist()}

def grid_coords(M, shape, z0: int = 0) -> np.ndarray:
    """Source-voxel coordinates (3, N) float64 of every voxel (i+z0, j, k) of a target block through the 4x4 voxel->voxel map M."""
    M = np.asarray(M, float); n, Y, X = shape
    zz = np.arange(z0, z0 + n, dtype=np.float64)[:, None, None]; yy = np.arange(Y, dtype=np.float64)[None, :, None]; xx = np.arange(X, dtype=np.float64)[None, None, :]
    q = np.empty((3, n, Y, X), np.float64)
    for r in range(3): q[r] = M[r, 0] * zz + M[r, 1] * yy + M[r, 2] * xx + M[r, 3]
    return q.reshape(3, -1)


# ----------------------------------------------------------------------------- chunk workers (forked; numpy/scipy only)
def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"

def _run(fn, jobs, src, n_workers: int, aux=None):
    """Run fn over jobs; workers are forked after _SRC/_AUX are set so they read the parent's arrays without copying."""
    global _SRC, _AUX
    _SRC, _AUX = src, aux
    try:
        if n_workers <= 1:
            for j in jobs: yield fn(j)
        else:
            with get_context("fork").Pool(n_workers, initializer=_init_worker) as p:
                for r in p.imap_unordered(fn, jobs): yield r
    finally:
        _SRC, _AUX = None, None

def _field_chunk(args):
    """Step 1: band-pass, masked directional running std per axis, masked local mean; weighted mean-pool by kW."""
    z0, z1, pad, sig_bp, win, kW = args
    v = _SRC; Z = v.shape[0]; a, b = max(0, z0 - pad), min(Z, z1 + pad)
    blk = np.asarray(v[a:b]).astype(np.float32); m = (blk > 0).astype(np.float32)
    bs = gaussian_filter(blk * m, sig_bp) / np.maximum(gaussian_filter(m, sig_bp), 1e-3); bs[m == 0] = 0; del blk
    sl = slice(z0 - a, z1 - a); outs = []
    for axn in range(3):
        sz = [1, 1, 1]; sz[axn] = win
        den = uniform_filter(m, sz); mu = uniform_filter(bs * m, sz) / np.maximum(den, 1e-3)
        var = uniform_filter(bs * bs * m, sz) / np.maximum(den, 1e-3) - mu * mu
        s = np.sqrt(np.maximum(var, 0)); s[den < 0.5] = 0
        outs.append(_pool(s[sl] * m[sl], kW, "mean")); del den, mu, var, s
    mu3 = uniform_filter(bs * m, win) / np.maximum(uniform_filter(m, win), 1e-3)
    outs += [_pool(mu3[sl] * m[sl], kW, "mean"), _pool(m[sl], kW, "mean")]
    return z0, np.stack(outs, 0).astype(np.float32)

def _rim_chunk(args):
    """Step 3: 3-D local std (window win, zeros included) -> Gaussian sigma 1 voxel -> max-pool by kW."""
    z0, z1, pad, win, kW = args
    v = _SRC; Z = v.shape[0]; a, b = max(0, z0 - pad), min(Z, z1 + pad)
    blk = np.asarray(v[a:b]).astype(np.float32)
    mloc = uniform_filter(blk, win); var = uniform_filter(blk * blk, win) - mloc * mloc; del mloc
    t = gaussian_filter(np.sqrt(np.maximum(var, 0)), 1.0)[z0 - a:z1 - a]
    return z0, _pool(t, kW, "max").astype(np.float32)

def _resample_chunk(args):
    """Step 7 (octv grid): trilinear W-mask > 0.5 & (octv > 0) for slices [z0, z1), written into the bool memmap."""
    z0, z1, M, path = args
    v = _SRC; Y, X = v.shape[1], v.shape[2]
    f = map_coordinates(_AUX, grid_coords(M, (z1 - z0, Y, X), z0), order=1, mode="nearest").reshape(z1 - z0, Y, X)
    m = (f > 0.5) & (np.asarray(v[z0:z1]) > 0)
    out = np.load(path, mmap_mode="r+"); out[z0:z1] = m; out.flush(); del out
    return z0, int(m.sum())


# ----------------------------------------------------------------------------- the mask
def oct_specimen_mask(octv: np.ndarray, A_v, o150: np.ndarray, A150, work, n_workers: int = 4):
    """Texture-marker + rim-watershed specimen mask of a slab-normalised OCT block (zeros = missing data).

    octv: (Z,Y,X) float16/float32 array or memmap at the vessel level (~0.04 mm) with voxel->world affine A_v;
    o150/A150: the 0.15 mm level (only o150 > 0 and its grid are used); work: directory for octv_mask_texture.npy.
    Returns (mask150_tex bool, path of the octv-grid bool memmap, info dict)."""
    t0 = time.time(); work = Path(work); work.mkdir(parents=True, exist_ok=True); A_v = np.asarray(A_v, float); A150 = np.asarray(A150, float)
    vox = float(np.linalg.norm(A_v[:3, :3], axis=0).mean()); vox150 = float(np.linalg.norm(A150[:3, :3], axis=0).mean())
    kW = max(1, int(round(GRID_MM / vox))); vox_W = vox * kW; A_W = pooled_affine_np(A_v, kW)
    Z, Y, X = octv.shape; ZW, YW, XW = Z // kW, Y // kW, X // kW
    sig_bp = BP_SIGMA_MM / vox; win = _odd(STD_WIN_MM / vox); win_rim = _odd(RIM_WIN_MM / vox); sig_s = SMOOTH_MM / vox_W
    r_open = max(1, int(round(OPEN_MM / vox_W))); r_close = max(1, int(round(CLOSE_MM / vox_W)))
    CH = max(kW, (128 // kW) * kW); pad_f = max(16, int(np.ceil(4 * sig_bp)) + win + 2); pad_r = max(8, win_rim // 2 + 6)
    params = {"vox_v_mm": vox, "kW": kW, "vox_W_mm": vox_W, "shape_W": [ZW, YW, XW], "bp_sigma_vox": sig_bp, "std_window_vox": win, "rim_window_vox": win_rim,
              "smooth_sigma_cells": sig_s, "open_r_cells": r_open, "close_r_cells": r_close, "chunk": CH, "n_workers": n_workers}
    jobs = [(z0, min(Z, z0 + CH)) for z0 in range(0, ZW * kW, CH)]
    # ---- step 1: texture field F on W
    fields = np.zeros((5, ZW, YW, XW), np.float32)
    for z0, arr in _run(_field_chunk, [(a, b, pad_f, sig_bp, win, kW) for a, b in jobs], octv, n_workers):
        fields[:, z0 // kW:z0 // kW + arr.shape[1]] = arr
    den = np.maximum(fields[4], 1e-3); valid = fields[4] > 0.5; mv = valid.astype(np.float32)
    def msmooth(f): return gaussian_filter(f * mv, sig_s) / np.maximum(gaussian_filter(mv, sig_s), 1e-3)
    Ss = [msmooth(fields[i] / den) for i in range(3)]; MU = msmooth(fields[3] / den)
    F = np.minimum(np.minimum(Ss[0], Ss[1]), Ss[2]) / np.maximum(MU, 1.0); F[~valid] = 0; del fields, Ss, MU, den
    t1 = time.time()
    # ---- step 2: threshold from a log-GMM (never Otsu: it splits at the rim class)
    sel = valid & (F > 0)
    if sel.sum() >= 1000:
        lthr, ginfo = gmm2_1d(np.log(F[sel])); thr = float(np.exp(lthr))
    else:
        thr, ginfo = float("nan"), {"log_means": None, "sds": None, "weights": None}
    # ---- step 3: rim strength E on W
    E = np.zeros((ZW, YW, XW), np.float32)
    for z0, arr in _run(_rim_chunk, [(a, b, pad_r, win_rim, kW) for a, b in jobs], octv, n_workers):
        E[z0 // kW:z0 // kW + arr.shape[0]] = arr
    E[~valid] = 0; t2 = time.time()
    # ---- steps 4-6: markers, watershed, enclosed exterior pockets, cleanup
    from skimage.segmentation import watershed
    if np.isfinite(thr):
        inner = ndimage.binary_opening((F > thr) & valid, structure=ball(r_open)); outer = ndimage.binary_opening((F < thr) & valid, structure=ball(r_open)) | ~valid
    else:
        inner = np.zeros_like(valid); outer = np.ones_like(valid)
    mk = np.zeros(F.shape, np.int32); mk[outer] = 1; mk[inner] = 2
    lab = watershed(E, mk)
    ext = (lab == 1) & valid; le, ne = label(ext); touch = np.zeros(ne + 1, bool)
    for f in (le[0], le[-1], le[:, 0], le[:, -1], le[:, :, 0], le[:, :, -1]): touch[np.unique(f)] = True
    touch[np.unique(le[ndimage.binary_dilation(~valid, structure=ball(1))])] = True; touch[0] = False
    mask_W = valid & ~np.isin(le, np.where(touch)[0])                 # specimen label + exterior pockets that reach neither a face nor a zero cell
    mask_W, n_cc_W = cleanup(mask_W, close_r=r_close); del lab, ext, le, mk, inner, outer, E
    t3 = time.time()
    # ---- step 7: resample W -> 0.15 mm grid and -> octv grid (bool memmap)
    mf = mask_W.astype(np.float32)
    f150 = map_coordinates(mf, grid_coords(np.linalg.inv(A_W) @ A150, o150.shape), order=1, mode="nearest").reshape(o150.shape)
    mask150 = (f150 > 0.5) & (np.asarray(o150) > 0); mask150, n_cc_150 = cleanup(mask150); del f150
    maskv_path = work / "octv_mask_texture.npy"
    np.lib.format.open_memmap(str(maskv_path), mode="w+", dtype=bool, shape=(Z, Y, X)).flush()
    Mv = np.linalg.inv(A_W) @ A_v; n_v = 0
    for _, n in _run(_resample_chunk, [(z0, min(Z, z0 + 32), Mv, str(maskv_path)) for z0 in range(0, Z, 32)], octv, n_workers, aux=mf):
        n_v += n
    t4 = time.time()
    info = {"V_cm3_150": float(mask150.sum()) * vox150 ** 3 / 1000.0, "V_cm3_W": float(mask_W.sum()) * vox_W ** 3 / 1000.0, "V_cm3_v": n_v * vox ** 3 / 1000.0,
            "voxels150": int(mask150.sum()), "voxels_v": int(n_v), "frac150": float(mask150.mean()), "thr": thr, "gmm": ginfo,
            "n_cc_before_largest": n_cc_W, "n_cc_150_before_largest": n_cc_150, "seconds_field": t1 - t0, "seconds_rim": t2 - t1, "seconds_ws": t3 - t2,
            "seconds_resample": t4 - t3, "seconds_total": t4 - t0, "params": params}
    return mask150, maskv_path, info


# ----------------------------------------------------------------------------- mask choice (prep_subject.py step 8)
def want_texture(mode: str, V_int_cm3: float, V_mri_cm3):
    """Does the mask rule call for the texture mask?  -> (bool, reason).  auto: only when the intensity mask is
    over-inclusive (V_int > 1.5 V_mri); without an MRI volume auto stays with the intensity mask."""
    if mode == "intensity": return False, "intensity_default"
    if mode == "texture": return True, "forced"
    if V_mri_cm3 is None: return False, "no_mri_volume"
    if V_int_cm3 > 1.5 * V_mri_cm3: return True, "auto: V_int > 1.5 V_mri"
    return False, "intensity_default"

def choose_oct_mask(mask_int, V_int_cm3, mask_tex, V_tex_cm3, V_mri_cm3, mode: str = "auto", band=(0.5, 1.75)):
    """Pick the 0.15 mm mask -> (mask, mask_mode, mask_reason).  The texture mask (None = not computed) is accepted only
    inside band x V_mri (or >= 1e5 voxels when V_mri is unknown and texture is forced) and non-empty; otherwise intensity."""
    want, reason = want_texture(mode, V_int_cm3, V_mri_cm3)
    if not want: return mask_int, "intensity", reason
    if mask_tex is None: return mask_int, "intensity", "texture_not_computed"
    n_tex = int(mask_tex.sum())
    ok = (band[0] * V_mri_cm3 <= V_tex_cm3 <= band[1] * V_mri_cm3) if V_mri_cm3 is not None else n_tex >= 1e5
    if ok and n_tex > 0: return mask_tex, "texture", reason
    return mask_int, "intensity", "texture_out_of_band"


# ----------------------------------------------------------------------------- QC figure
def mask_qc_png(o150, mask_int, mask_chosen, path, title: str = ""):
    """Three mid-planes of oct150 with the intensity mask contour (red) and the chosen mask contour (green)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    o150 = np.asarray(o150); pos = o150[o150 > 0]; vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    for r in range(3):
        i = o150.shape[r] // 2; sl = lambda m: np.take(np.asarray(m), i, axis=r)
        ax[r].imshow(sl(o150), cmap="gray", vmin=0, vmax=vmax)
        for m, col in ((mask_int, "r"), (mask_chosen, "lime")):
            if m is not None and 0 < sl(m).sum() < sl(m).size: ax[r].contour(sl(m).astype(float), levels=[0.5], colors=col, linewidths=0.8)
        ax[r].set_title(f"axis {r} idx {i}  (red = intensity mask, green = chosen)", fontsize=9); ax[r].axis("off")
    if title: fig.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.savefig(str(path), dpi=80); plt.close(fig)
