"""`octreg register` (method steps 1-5 on two files, then the outputs) and `octreg apply`.

A transform T is a 4x4 matrix (mm) from the OCT world to the MRI world; worlds are the NIfTI header frames (array frame for
TIFF / NPY). The OCT is only ever streamed: its largest array in memory is the fine grid (Params.fine_mm).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from . import __version__, geometry as G, io, preprocess as pp
from .params import Params
from .refine import refine
from .search import search

OUTPUTS = ("T_oct2mri.txt", "T_mri2oct.txt", "oct2mri.lta", "oct2mri_itk.txt", "oct_in_mri.nii.gz", "mri_in_oct.nii.gz", "qc.png",
           "result.json")


def register(oct_path, mri_path, out_dir, oct_spacing_um=None, oct_mask=None, mri_mask=None, params: Params = Params(),
             device="cuda") -> dict:
    """Register an OCT block to an MRI cropped around it and write OUTPUTS into out_dir.

    oct_path: NIfTI, or TIFF / OME-TIFF / NPY with spacing; mri_path: NIfTI. oct_spacing_um: (z, y, x) um (io.load_volume).
    oct_mask, mri_mask: optional mask files (voxels > 0, in the world of their image; oct_spacing_um applies to the OCT mask
    too) replacing the texture specimen mask and the histogram foreground. device: 'cuda' | 'cpu'.
    oct_in_mri.nii.gz: the OCT box-averaged to the MRI voxel size (not finer than fine_mm), trilinear through T onto the MRI grid
    (= apply(out_dir, oct_path, mri_path)); mri_in_oct.nii.gz: the MRI through T onto the OCT base grid; qc.png: three planes
    through the OCT mask centroid. -> the result.json dict. Raises ValueError for a refused input."""
    P, out, t0, sec = params, Path(out_dir), time.time(), {}
    h, cuda = P.base_mm, str(device).startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is not available: use device 'cpu'")
    vo, vm = io.load_volume(oct_path, oct_spacing_um), io.load_volume(mri_path)
    mo = None if oct_mask is None else io.load_volume(oct_mask, oct_spacing_um)
    mm = None if mri_mask is None else io.load_volume(mri_mask)
    out.mkdir(parents=True, exist_ok=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def lap(name):
        sec[name] = time.time() - t0 - sum(sec.values())
        print(f"octreg [{time.time() - t0:6.0f} s] {name} done in {sec[name]:.0f} s", flush=True)

    # step 1-2, MRI: base grid and histogram foreground
    mri_h, A_m = G.resample_iso(vm, h)
    mask_m, fg_m = pp.foreground(mri_h, h, P) if mm is None else (_mask_on(mm, mri_h.shape, A_m, h), {"source": mm.path})
    lap("mri")
    # step 1-2, OCT: fine grid, specimen mask (innovation 1), base grid
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay = G.pool_iso(fine, A_f, P.fine_mm, max(_mm(vm.spacing_mm.min()), P.fine_mm))       # for oct_in_mri
    lap("oct_fine_grid")
    mask_f, fg_o = fine_mask(fine, A_f, P, mo)
    oct_h, A_o = G.pool_iso(fine, A_f, P.fine_mm, h)
    mask_o, valid_o = (G.pool_iso(x, A_f, P.fine_mm, h)[0] > 0.5 for x in (mask_f, fine > 0))
    del fine, mask_f
    lap("oct_mask")
    # step 3: two-class maps (innovation 2)
    u, w = pp.oct_channels(pp.two_class(oct_h, mask_o, h, P), mask_o)
    v = pp.mri_channels(pp.two_class(mri_h, mask_m, h, P), mask_m)
    lap("two_class")
    # steps 4-5: orientation search in the crop and affine refinement (innovation 3)
    poses, info = align((u, w, A_o), (v, mask_m, A_m), P, device)
    lap("search_refine")

    best = poses[0]
    T = best["T"]
    pose = {k: best[k] for k in ("S", "L", "polarity", "mirror", "log_scales", "shears", "overlap", "search_rank")}
    pose["scale_per_oct_axis"] = np.linalg.norm(T[:3, :3] @ (vo.affine[:3, :3] / vo.spacing_mm), axis=0).tolist()
    flags = ["mri_foreground_no_valley"] * (fg_m.get("status") == "no_valley") + ["overlap_floor"] * info["search"]["overlap_floor"]
    flags += ["nondefault_params"] * (P != Params())
    flags += ["clamp_saturated"] * bool(np.abs([*best["log_scales"], *best["shears"]]).max() >= P.clamp * (1 - 1e-3))
    inputs = {"oct": {**vars(vo), "path": _abs(vo.path)}, "mri": {**vars(vm), "path": _abs(vm.path)},
              "oct_spacing_um": oct_spacing_um, "oct_mask": oct_mask, "mri_mask": mri_mask, "device": str(device)}
    result = {"octreg_version": __version__, "inputs": inputs, "params": P.to_dict(), "params_hash": P.hash(), "T_oct2mri": T,
              "pose": pose, "flags": flags, "boundary_mm": _boundary_mm(mask_o, valid_o, A_o, mask_m, A_m, T),
              "foreground": {"oct": fg_o, "mri": {"source": "histogram", **fg_m}}, **info}
    io.write_transform_txt(T, out / "T_oct2mri.txt")
    io.write_transform_txt(np.linalg.inv(T), out / "T_mri2oct.txt")
    io.write_lta(T, vo, vm, out / "oct2mri.lta")
    io.write_itk(T, vo, vm, out / "oct2mri_itk.txt")
    io.write_json(result, out / "result.json")                  # before the overlays, so a failing overlay loses no result
    io.save_nifti(G.resample_to(*overlay, vm.shape, vm.affine, np.linalg.inv(T)), vm.affine, out / "oct_in_mri.nii.gz")
    mri_in_oct = G.resample_to(mri_h, A_m, oct_h.shape, A_o, T)
    io.save_nifti(mri_in_oct, A_o, out / "mri_in_oct.nii.gz")
    _qc_png(out / "qc.png", oct_h, mri_in_oct, mask_o, pose, h)
    lap("outputs")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
    result.update(seconds={**sec, "total": time.time() - t0}, peak_rss_gb=rss,
                  gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None)
    io.write_json(result, out / "result.json")
    return result


def fine_mask(fine, affine, params: Params = Params(), mask=None):
    """OCT specimen mask on the fine grid (fine [D, H, W] of spacing params.fine_mm, voxel -> world affine): the mask file `mask`
    (io.Volume) if given, else the isotropic-texture mask; AND fine > 0 (zeros are missing data). -> (bool [D, H, W], info)."""
    if mask is not None:
        m, info = _mask_on(mask, fine.shape, affine, params.fine_mm), {"source": mask.path}
    else:
        m, info = pp.specimen_mask(fine, params.fine_mm, params)
        info["source"] = "texture"
    return np.logical_and(m, fine > 0, out=m), info


def align(oct_base, mri_base, params: Params = Params(), device="cuda", polarity=0):
    """Steps 4-5 on base-grid channels, oct_base = (u [2, D, H, W], weight, affine) and mri_base = (v [2, D', H', W'], mask,
    affine): box average to search_mm, search.search there (polarity 0 reads the polarity off the sign of the score, +1 / -1
    force it), then refine.refine of every search pose on the base grid, gated by the search tau.
    -> (refined poses lowest L first, {'search': search info, 'refine': {n_poses, n_below_tau, seconds}})."""
    P = params
    coarse = lambda c, m, A: (G.pool_iso(c, A, P.base_mm, P.search_mm)[0], *G.pool_iso(m, A, P.base_mm, P.search_mm))
    candidates, info = search(*coarse(*mri_base), *coarse(*oct_base), P, device, polarity)
    t0 = time.time()
    poses = refine(candidates, mri_base, oct_base, P, device, info["tau"])
    return poses, {"search": info, "refine": {"n_poses": len(candidates), "n_below_tau": len(candidates) - len(poses),
                                              "seconds": time.time() - t0}}


def apply(run_dir, moving, reference, out, inverse=False) -> Path:
    """Resample the volume `moving` (OCT frame; MRI frame with inverse) onto the grid of `reference` through run_dir/T_oct2mri.txt
    (T_mri2oct.txt with inverse): streamed box average to the reference voxel size (never finer than the moving voxel size),
    then trilinear, 0 outside. moving, reference: image files (io.load_volume; the run's OCT file with the run's
    oct_spacing_um from result.json); out: NIfTI path. -> out."""
    run = Path(run_dir)
    T = np.loadtxt(run / ("T_mri2oct.txt" if inverse else "T_oct2mri.txt"))
    inp = json.loads((run / "result.json").read_text())["inputs"]
    mv, rf = (io.load_volume(f, inp["oct_spacing_um"] if _abs(f) == inp["oct"]["path"] else None) for f in (moving, reference))
    arr, A = G.resample_iso(mv, max(_mm(rf.spacing_mm.min()), _mm(mv.spacing_mm.min())))
    io.save_nifti(G.resample_to(arr, A, rf.shape, rf.affine, np.linalg.inv(T)), rf.affine, out)
    return Path(out)


def _mm(x):
    """A length to 6 significant digits (float32 header noise)."""
    return float(f"{x:.6g}")


def _abs(path):
    return str(Path(path).resolve())


def _mask_on(vol, shape, affine, voxel_mm):
    """A mask file (io.Volume, voxels > 0) on a grid (shape, affine, spacing voxel_mm) in the same world: its indicator
    streamed and box-averaged to max(voxel_mm, mask voxel), trilinear onto the grid, > 0.5. -> bool array."""
    frac, A = G.resample_iso(vol, max(voxel_mm, _mm(vol.spacing_mm.min())), binary=True)
    return G.resample_to(frac, A, shape, affine, np.eye(4)) > 0.5


def _boundary_mm(mask_o, valid_o, A_o, mask_m, A_m, T):
    """Boundary agreement: median distance (mm) from the OCT mask outline through T to the MRI foreground outline. An outline
    voxel has a 6-neighbour outside its mask; grid faces (a crop may cut the tissue) and neighbours without OCT data are not
    outside. Over the OCT outline voxels landing inside the MRI grid; None if there are none (the block has no real outline)."""
    o = mask_o & ~ndimage.binary_erosion(mask_o | ~valid_o, border_value=1)
    m = mask_m & ~ndimage.binary_erosion(mask_m, border_value=1)
    c = G.apply_affine(np.linalg.inv(A_m) @ T @ A_o, np.argwhere(o).astype(float))
    c = c[np.all((c >= 0) & (c <= np.array(m.shape) - 1), axis=1)]
    if not (len(c) and m.any()):
        return None
    d = ndimage.distance_transform_edt(~m, sampling=np.linalg.norm(A_m[:3, :3], axis=0))
    return float(np.median(ndimage.map_coordinates(d, c.T, order=1)))


def _qc_png(path, oct_h, mri_in_oct, mask, pose, h):
    """3 x 3 figure: planes through the mask centroid normal to each OCT array axis; OCT, MRI through T, 2 mm checkerboard (MRI
    inverted inside the mask when the polarity is -1). Grey levels: p1-p99 inside the mask."""
    from matplotlib.figure import Figure
    o, m = (np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1) for x in (oct_h, mri_in_oct) for lo, hi in [np.percentile(x[mask], [1, 99])])
    centre, tile, inv = np.argwhere(mask).mean(0).round().astype(int), max(2, round(2.0 / h)), pose["polarity"] < 0
    fig = Figure(figsize=(10, 10.5))
    for a in range(3):
        po, pm, pw = (np.take(x, centre[a], a) for x in (o, m, mask))
        board = (np.arange(po.shape[0])[:, None] // tile + np.arange(po.shape[1]) // tile) % 2 == 0
        board = np.where(board, po, np.where(pw & inv, 1 - pm, pm))
        for j, (img, name) in enumerate(((po, "OCT"), (pm, "MRI through T"), (board, "checkerboard" + " (MRI inverted)" * inv))):
            ax = fig.add_subplot(3, 3, 3 * a + j + 1)
            ax.imshow(img.T, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_title(f"{name}, OCT axis {a} = {centre[a]}", fontsize=9)
            ax.set_axis_off()
    fig.suptitle(f"S {pose['S']:.4f}   polarity {pose['polarity']:+d}   overlap {pose['overlap']:.2f}   "
                 f"scale per OCT axis {np.round(pose['scale_per_oct_axis'], 3).tolist()}")
    fig.savefig(path, dpi=90)
