"""Shared helpers for probe P1 (OCT specimen mask). Writes only under work/probe_xr/mask/."""
import numpy as np, os, json, time, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import uniform_filter, gaussian_filter, label, binary_fill_holes, binary_closing, binary_opening, generate_binary_structure, map_coordinates
W = "work/xiangrui_I58bs"; OUT = "work/probe_xr/mask"; os.makedirs(OUT, exist_ok=True)
VOX150 = 0.15; VOX04 = 0.04
MRI_CM3 = 13.315028
_t0 = time.time()
def say(*a): print(*a, f"[{time.time()-_t0:.0f}s]", flush=True)

def local_std(x, w):
    x = x.astype(np.float32); m = uniform_filter(x, w); v = uniform_filter(x * x, w) - m * m
    return np.sqrt(np.maximum(v, 0)), m

def otsu(v, n=2_000_000):
    from skimage.filters import threshold_otsu
    v = np.asarray(v).ravel(); v = v[::max(1, v.size // n)]
    return float(threshold_otsu(v))

def ball(r):
    z, y, x = np.mgrid[-r:r+1, -r:r+1, -r:r+1]; return (z*z + y*y + x*x) <= r*r

def largest_cc(m):
    lab, n = label(m)
    if n <= 1: return m.copy()
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(np.argmax(sizes))

def cleanup(m, close_r=0):
    """3D: optional closing, largest connected component, fill 3D holes."""
    if close_r > 0:   # pad with edge values so a mask cut by the box faces is not eroded at the faces
        mp = np.pad(m, close_r, mode="edge"); mp = binary_closing(mp, structure=ball(close_r)); m = mp[close_r:-close_r, close_r:-close_r, close_r:-close_r]
    m = largest_cc(m); m = binary_fill_holes(m); return m

def exterior_fill(passable, zero, face_only_seed=True):
    """Flood the exterior from the box faces through `passable | zero` voxels; specimen = complement.
    Returns (specimen_raw, exterior)."""
    ok = passable | zero
    lab, n = label(ok)
    faces = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
    seeds = np.unique(faces[faces > 0])
    ext = np.isin(lab, seeds)
    return ~ext, ext

def mri_tissue_in_oct150():
    p = f"{OUT}/mri_tissue_in_oct150.npy"
    if os.path.exists(p): return np.load(p)
    A150 = np.load(f"{W}/oct150_affine.npy"); Am = np.load(f"{W}/mri_affine.npy"); T = np.load("work/runs/xiangrui_I58bs_novasc/T_oct2mri.npy")
    mt = np.load(f"{W}/mri_tissue.npy", mmap_mode="r")
    shp = np.load(f"{W}/oct150.npy", mmap_mode="r").shape
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shp], indexing="ij")
    P = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], 0).astype(np.float64)
    M = np.linalg.inv(Am) @ T @ A150
    q = (M @ P)[:3]
    out = map_coordinates(np.asarray(mt).astype(np.uint8), q, order=0, mode="constant", cval=0).reshape(shp).astype(bool)
    np.save(p, out); return out

def report(mask, name, o=None, old=None, mri_oct=None, extra=None, vox=VOX150, fine=None):
    """Numbers + PNG (oct150 slices at 25/50/75% per axis with the mask contour; MRI-in-OCT contour in blue)."""
    if o is None: o = np.load(f"{W}/oct150.npy")
    if old is None: old = np.load(f"{W}/oct150_mask.npy")
    if mri_oct is None: mri_oct = mri_tissue_in_oct150()
    vol = float(mask.sum()) * vox**3 / 1000.0
    d = {"name": name, "voxels": int(mask.sum()), "cm3": round(vol, 3), "ratio_to_mri": round(vol / MRI_CM3, 3), "fraction": round(float(mask.mean()), 4),
         "keep_of_old": round(float((mask & old).sum() / old.sum()), 4), "outside_old_frac": round(float((mask & ~old).sum() / max(1, mask.sum())), 4),
         "n_cc": int(label(mask)[1]), "dice_vs_mri_in_oct": round(float(2 * (mask & mri_oct).sum() / (mask.sum() + mri_oct.sum())), 4),
         "mri_in_oct_cm3": round(float(mri_oct.sum()) * vox**3 / 1000, 3)}
    if extra: d.update(extra)
    say(json.dumps(d))
    with open(f"{OUT}/{name}.json", "w") as f: json.dump(d, f, indent=1)
    vmax = float(np.percentile(o[o > 0], 99.5))
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    for r, ax_ in enumerate(range(3)):
        for c, frac in enumerate((0.25, 0.5, 0.75)):
            i = int(o.shape[ax_] * frac)
            sl = lambda a: np.take(a, i, axis=ax_)
            a = ax[r, c]; a.imshow(sl(o), cmap="gray", vmax=vmax)
            a.contour(sl(mask).astype(float), levels=[0.5], colors="r", linewidths=0.8)
            a.contour(sl(mri_oct).astype(float), levels=[0.5], colors="c", linewidths=0.6, linestyles="dashed")
            if fine is not None: a.contour(sl(fine).astype(float), levels=[0.5], colors="y", linewidths=0.5)
            a.set_title(f"{name} axis{ax_} idx{i} (red=mask, cyan=MRI tissue via T)", fontsize=8); a.axis("off")
        i = o.shape[ax_] // 2
        a = ax[r, 3]; a.imshow(np.take(old, i, axis=ax_).astype(float) + np.take(mask, i, axis=ax_), cmap="viridis"); a.set_title(f"old mask + new (mid, axis{ax_})", fontsize=8); a.axis("off")
    fig.suptitle(f"{name}: {vol:.2f} cm3 (MRI 13.32), frac {mask.mean():.3f}, keep-of-old {d['keep_of_old']:.3f}, Dice vs MRI-in-OCT {d['dice_vs_mri_in_oct']:.3f}", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=70); plt.close(fig)
    return d
