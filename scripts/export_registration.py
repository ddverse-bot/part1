#!/usr/bin/env python3
"""Export a registration in the coordinate frames of the ORIGINAL input NIfTI files, for FreeSurfer / freeview users.

    python scripts/export_registration.py --work WORK --run RUN --oct-nifti OCT.nii.gz --mri-nifti MRI.nii.gz --out OUT [--T RUN/T_oct2mri.npy]

Why: register.py's T_oct2mri maps the PIPELINE OCT world (prep's layout_affine of the raw array, oct_native_affine.npy) to the MRI
NIfTI world, and its T_oct2mri.lta says so.  That frame is not the OCT NIfTI's own RAS frame, so the .lta cannot be applied to the
original OCT file.  This script composes
    T_nii = T_oct2mri @ A_pipe_native @ inv(A_nii_native)        (OCT NIfTI RAS mm -> MRI NIfTI RAS mm)
where A_pipe_native = WORK/oct_native_affine.npy and A_nii_native = the OCT NIfTI header affine (both voxel -> world for the same
raw array index order), and writes, into OUT:
  T_octnii_to_mrinii.txt / .npy      4x4 RAS-to-RAS (mm), OCT NIfTI world -> MRI NIfTI world (and the inverse)
  octnii_to_mrinii.lta               FreeSurfer LINEAR_RAS_TO_RAS with src (OCT) / dst (MRI) volume geometry
  oct150_octnii_frame.nii.gz         the preprocessed 0.15 mm OCT (oct150.npy) with an affine in the OCT NIfTI frame: overlays the
                                     original OCT file directly in freeview
  oct_in_mri.nii.gz                  oct150 resampled onto the MRI NIfTI grid (linear): overlays the original MRI file directly
  mri_in_octnii_frame.nii.gz         the MRI resampled onto a --mri-in-oct-mm grid aligned with the OCT NIfTI voxel axes and covering the
                                     OCT field of view (linear): overlays the original OCT file directly
  export.json, README.md             provenance, checks, pose summary, freeview commands
Checks (export.json 'checks'): the OCT header frame and the pipeline layout frame differ only by a signed axis permutation and an
origin (same spacing); the composed transform maps the OCT FOV corners to the same MRI points as T_oct2mri does (numerical); oct_in_mri.nii.gz equals register.py's own oct_in_mri_region.nii.gz where both exist (NCC and max
abs difference over the overlap, independent code path: scipy here, torch there).  The corner check is numerical only (the
composition is exact by construction); the frame check reports whether the header spacing equals the spacing prep used.  Reads only; CPU only; memory ~ a few GB."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--oct-nifti", type=Path, required=True, help="the original OCT NIfTI given to prep_subject.py --oct")
ap.add_argument("--mri-nifti", type=Path, required=True, help="the original MRI NIfTI given to prep_subject.py --mri")
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--T", type=Path, default=None, help="default RUN/T_oct2mri.npy")
ap.add_argument("--mri-in-oct-mm", type=float, default=0.08, help="grid spacing (mm) of mri_in_octnii_frame.nii.gz")
ap.add_argument("--chunk", type=int, default=24, help="slices per resampling chunk")
a = ap.parse_args(); t0 = time.time(); a.out.mkdir(parents=True, exist_ok=True)
def say(*x): print(*x, f"[{time.time() - t0:.0f}s]", flush=True)
import nibabel as nib

T_pipe = np.load(a.T or a.run / "T_oct2mri.npy").astype(np.float64)
A_pipe_nat = np.load(a.work / "oct_native_affine.npy").astype(np.float64)
oct_img = nib.load(str(a.oct_nifti)); A_nii_nat = np.asarray(oct_img.affine, np.float64); oct_shape = tuple(int(s) for s in oct_img.shape[:3])
mri_img = nib.load(str(a.mri_nifti)); A_mri_hdr = np.asarray(mri_img.affine, np.float64); mri_shape = tuple(int(s) for s in mri_img.shape[:3])
A_mri = np.load(a.work / "mri_affine.npy").astype(np.float64); mri = np.load(a.work / "mri.npy", mmap_mode="r")
A150 = np.load(a.work / "oct150_affine.npy").astype(np.float64); oct150 = np.load(a.work / "oct150.npy", mmap_mode="r")
prep = json.load(open(a.work / "prep.json")) if (a.work / "prep.json").exists() else {}
checks = {}

# ---- 1. frames
if not np.allclose(A_mri, A_mri_hdr, atol=1e-6): raise SystemExit(f"WORK/mri_affine.npy differs from the MRI NIfTI header affine:\n{A_mri}\n{A_mri_hdr}")
if tuple(mri.shape) != mri_shape: raise SystemExit(f"WORK/mri.npy shape {mri.shape} != MRI NIfTI shape {mri_shape}")
raw_shape = prep.get("oct", {}).get("raw_shape") or prep.get("oct", {}).get("shape_native")
if raw_shape and tuple(int(s) for s in raw_shape) != oct_shape: raise SystemExit(f"prep raw OCT shape {raw_shape} != OCT NIfTI shape {oct_shape}")
R = A_nii_nat @ np.linalg.inv(A_pipe_nat)                         # pipeline OCT world -> OCT NIfTI world (both index the same raw array)
R3 = R[:3, :3]; perm_like = bool(np.allclose(np.sort(np.abs(R3), axis=1), [[0, 0, 1]] * 3, atol=1e-6) and np.allclose(np.abs(R3).sum(0), 1, atol=1e-6))
checks["oct_frames"] = {"A_pipeline_native": A_pipe_nat.tolist(), "A_nifti_native": A_nii_nat.tolist(), "pipeline_world_to_nifti_world": np.round(R, 9).tolist(),
                        "signed_permutation_same_spacing": perm_like, "det": float(np.linalg.det(R3)),
                        "note": "prep's layout_affine ignores the OCT header orientation; a signed permutation with unit scale means the two frames differ only in axis direction / order / origin, i.e. the header spacing equals the spacing prep used"}
if not perm_like: say("WARNING: the pipeline OCT frame and the OCT NIfTI frame differ by more than a signed permutation (spacing mismatch between the header and prep's --oct-spacing-um?)", np.round(R3, 6).tolist())
T_nii = T_pipe @ A_pipe_nat @ np.linalg.inv(A_nii_nat)             # OCT NIfTI world -> MRI NIfTI world
def corners(shape):
    return np.array([[i, j, k, 1.0] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])
C = corners(oct_shape)
err = np.abs((T_nii @ (A_nii_nat @ C.T)).T - (T_pipe @ (A_pipe_nat @ C.T)).T)[:, :3].max()
checks["corner_consistency_max_mm (numerical)"] = float(err)
if err > 1e-6: raise SystemExit(f"composed transform inconsistent at the OCT corners: {err} mm")
np.save(a.out / "T_octnii_to_mrinii.npy", T_nii); np.save(a.out / "T_mrinii_to_octnii.npy", np.linalg.inv(T_nii))
np.savetxt(a.out / "T_octnii_to_mrinii.txt", T_nii, fmt="%.9f", header="4x4 RAS-to-RAS (mm): OCT NIfTI world -> MRI NIfTI world")
np.savetxt(a.out / "T_mrinii_to_octnii.txt", np.linalg.inv(T_nii), fmt="%.9f", header="4x4 RAS-to-RAS (mm): MRI NIfTI world -> OCT NIfTI world")
sv = np.linalg.svd(T_nii[:3, :3])[1]; det = float(np.linalg.det(T_nii[:3, :3]))
say(f"T_nii singular values {np.round(sv, 4).tolist()}, det {det:.4f}; corner consistency {err:.2e} mm")

# ---- 2. LTA with volume geometry (FreeSurfer: c_ras = vox2ras @ dims/2)
def vol_info(path, A, shape):
    vs = np.linalg.norm(A[:3, :3], axis=0); D = A[:3, :3] / vs; c = A[:3, :3] @ (np.array(shape, float) / 2.0) + A[:3, 3]
    f = lambda v: " ".join(f"{x:.15e}" for x in v)
    return (f"valid = 1  # volume info valid\nfilename = {Path(path).resolve()}\nvolume = {shape[0]} {shape[1]} {shape[2]}\nvoxelsize = {f(vs)}\n"
            f"xras   = {f(D[:, 0])}\nyras   = {f(D[:, 1])}\nzras   = {f(D[:, 2])}\ncras   = {f(c)}\n")
rows = "\n".join(" ".join(f"{v:.15e}" for v in r) for r in T_nii)
lta = (f"# transform file {a.out / 'octnii_to_mrinii.lta'}\n# created by octreg scripts/export_registration.py\ntype      = 1 # LINEAR_RAS_TO_RAS\nnxforms   = 1\n"
       f"mean      = 0.0000 0.0000 0.0000\nsigma     = 1.0000\n1 4 4\n{rows}\nsrc volume info\n{vol_info(a.oct_nifti, A_nii_nat, oct_shape)}"
       f"dst volume info\n{vol_info(a.mri_nifti, A_mri_hdr, mri_shape)}subject unknown\nfscale 0.100000\n")
(a.out / "octnii_to_mrinii.lta").write_text(lta)

# ---- 3. images
def save(arr, A, name):
    img = nib.Nifti1Image(np.asarray(arr, np.float32), A); img.header.set_xyzt_units("mm"); img.set_qform(A, code=1); img.set_sform(A, code=1)
    nib.save(img, str(a.out / name)); say("wrote", name, arr.shape)
A150_nii = A_nii_nat @ np.linalg.inv(A_pipe_nat) @ A150              # oct150 voxel -> OCT NIfTI world
save(oct150, A150_nii, "oct150_octnii_frame.nii.gz")

def resample(src, A_src, shape_out, A_out, T_out2src_world):
    """Linear resampling: output voxel -> world_out -> T -> world_src -> src voxel."""
    Minv = np.linalg.inv(A_src) @ T_out2src_world @ A_out; out = np.zeros(shape_out, np.float32); src = np.asarray(src, np.float32)
    jj, kk = np.meshgrid(np.arange(shape_out[1], dtype=np.float64), np.arange(shape_out[2], dtype=np.float64), indexing="ij")
    for i0 in range(0, shape_out[0], a.chunk):
        i1 = min(shape_out[0], i0 + a.chunk); ii = np.arange(i0, i1, dtype=np.float64)[:, None, None]
        P = np.stack([np.broadcast_to(ii, (i1 - i0, *jj.shape)), np.broadcast_to(jj, (i1 - i0, *jj.shape)), np.broadcast_to(kk, (i1 - i0, *jj.shape))], 0).reshape(3, -1)
        V = Minv[:3, :3] @ P + Minv[:3, 3:4]
        out[i0:i1] = ndimage.map_coordinates(src, V, order=1, mode="constant", cval=0.0).reshape(i1 - i0, *jj.shape)
    return out
oct_in_mri = resample(oct150, A150_nii, mri_shape, A_mri_hdr, np.linalg.inv(T_nii)); save(oct_in_mri, A_mri_hdr, "oct_in_mri.nii.gz")
f = a.mri_in_oct_mm / np.linalg.norm(A_nii_nat[:3, :3], axis=0)          # native -> export grid factor per axis
shape_g = tuple(int(np.ceil(s / fk)) for s, fk in zip(oct_shape, f))
A_g = A_nii_nat @ np.diag([*f, 1.0]); A_g[:3, 3] = A_nii_nat[:3, :3] @ ((f - 1) / 2.0) + A_nii_nat[:3, 3]   # export voxel centres cover the native FOV
mri_in_oct = resample(mri, A_mri_hdr, shape_g, A_g, T_nii); save(mri_in_oct, A_g, "mri_in_octnii_frame.nii.gz")

# ---- 4. independent check against register.py's own export
ref = a.run / "oct_in_mri_region.nii.gz"
if ref.exists():
    r_img = nib.load(str(ref)); R = np.asarray(r_img.dataobj, np.float32); A_r = np.asarray(r_img.affine)
    off = np.rint(np.linalg.inv(A_mri_hdr) @ A_r[:, 3])[:3].astype(int)
    if np.allclose(A_r[:3, :3], A_mri_hdr[:3, :3], atol=1e-6) and np.all(off >= 0):
        sub = oct_in_mri[off[0]:off[0] + R.shape[0], off[1]:off[1] + R.shape[1], off[2]:off[2] + R.shape[2]]
        m = (sub > 0) & (R > 0); x, y = sub[m].astype(np.float64), R[m].astype(np.float64)
        ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-12)) if m.sum() > 100 else None
        checks["vs_register_oct_in_mri_region"] = {"n_overlap": int(m.sum()), "ncc": ncc, "median_abs_rel_diff": float(np.median(np.abs(x - y) / (np.abs(y) + 1e-6))) if m.sum() else None,
                                                   "support_dice": float(2 * m.sum() / max(1, (sub > 0).sum() + (R > 0).sum()))}
        say("check vs register.py oct_in_mri_region:", checks["vs_register_oct_in_mri_region"])
    else: checks["vs_register_oct_in_mri_region"] = "skipped: grids not aligned"

res = json.load(open(a.run / "result.json")) if (a.run / "result.json").exists() else {}
exp = {"run": str(a.run), "work": str(a.work), "T_source": str(a.T or a.run / "T_oct2mri.npy"), "oct_nifti": str(a.oct_nifti), "mri_nifti": str(a.mri_nifti),
       "oct_shape": oct_shape, "mri_shape": mri_shape, "T_octnii_to_mrinii": T_nii.tolist(), "singular_values": sv.tolist(), "det": det,
       "mri_in_oct_grid": {"shape": shape_g, "affine": A_g.tolist(), "mm": a.mri_in_oct_mm}, "checks": checks,
       "result_summary": {k: res.get(k) for k in ("search_top1_top2", "oct_wm_bright", "mirror", "stretch_ijk", "final_ncc") if k in res}, "seconds": time.time() - t0}
json.dump(exp, open(a.out / "export.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
(a.out / "README.md").write_text(f"""# Registration export ({time.strftime('%Y-%m-%d')})

Run: `{a.run}`  (prep: `{a.work}`)

| file | what |
|---|---|
| `T_octnii_to_mrinii.txt` / `.npy` | 4x4 RAS-to-RAS (mm) from the world of `{a.oct_nifti.name}` to the world of `{a.mri_nifti.name}`; inverse in `T_mrinii_to_octnii.*` |
| `octnii_to_mrinii.lta` | the same transform as a FreeSurfer LTA with source/destination volume geometry |
| `oct_in_mri.nii.gz` | preprocessed OCT (0.15 mm) resampled onto the MRI grid; opens on top of the original MRI |
| `mri_in_octnii_frame.nii.gz` | MRI resampled onto a {a.mri_in_oct_mm} mm grid in the OCT file's frame; opens on top of the original OCT |
| `oct150_octnii_frame.nii.gz` | preprocessed OCT (0.15 mm) in the OCT file's frame |

freeview, MRI space: `freeview -v {a.mri_nifti} oct_in_mri.nii.gz:colormap=heat:opacity=0.4`
freeview, OCT space: `freeview -v {a.oct_nifti} mri_in_octnii_frame.nii.gz:colormap=heat:opacity=0.4`
FreeSurfer resampling of the original OCT into MRI space: `mri_vol2vol --mov {a.oct_nifti} --targ {a.mri_nifti} --lta octnii_to_mrinii.lta --o oct_native_in_mri.nii.gz`

Checks: corner consistency {err:.1e} mm; comparison with register.py's own resampling: {checks.get('vs_register_oct_in_mri_region')}.
""")
say("done", a.out)
