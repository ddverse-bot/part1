"""`octreg register` (method steps 1-5 on two files, then the outputs) and `octreg apply`.

A transform T is a 4x4 matrix (mm) from the OCT world to the MRI world; worlds are the NIfTI header frames (array frame for
TIFF / NPY). The OCT is only ever streamed: its largest array in memory is the fine grid (Params.fine_mm).
"""
from __future__ import annotations

import json
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
    oct_in_mri.nii.gz: the OCT box-averaged to the MRI voxel size (not finer than fine_mm), trilinear through T onto the MRI
    grid (apply(out_dir, oct_path, mri_path) gives the same). mri_in_oct.nii.gz: the MRI at levels[-1] through T onto the OCT
    grid of levels[-1]. qc.png: three planes through the OCT mask centroid (OCT, MRI through T, checkerboard).
    -> the result.json dict. Raises ValueError for a refused input."""
    P, out, t0 = params, Path(out_dir), time.time()
    h, sec, clock = P.levels[-1], {}, [t0]
    cuda = str(device).startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is not available: use device 'cpu'")
    vo, vm = io.load_volume(oct_path, oct_spacing_um), io.load_volume(mri_path)
    mo = None if oct_mask is None else io.load_volume(oct_mask, oct_spacing_um)
    mm = None if mri_mask is None else io.load_volume(mri_mask)
    out.mkdir(parents=True, exist_ok=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def lap(name):
        now = time.time()
        sec[name], clock[0] = now - clock[0], now
        print(f"octreg [{now - t0:6.0f} s] {name} done in {sec[name]:.0f} s", flush=True)

    # MRI: base grid and foreground
    mri_h, A_m = G.resample_iso(vm, h)
    if mm is None:
        mask_m, fg_m = pp.foreground(mri_h, h, P)
        fg_m["source"] = "histogram"
    else:
        mask_m, fg_m = _mask_on(mm, mri_h.shape, A_m, h), {"source": mm.path}
    lap("mri")
    # OCT: fine grid, specimen mask (innovation 1), stripe flat field, base grid
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay_mm = max(_mm(vm.spacing_mm.min()), P.fine_mm)
    overlay = G.pool_iso(fine, A_f, P.fine_mm, overlay_mm)                                  # raw OCT for oct_in_mri
    lap("oct_fine_grid")
    mask_f, fg_o = fine_mask(fine, A_f, P, mo)
    valid_o = G.pool_iso(fine > 0, A_f, P.fine_mm, h)[0] > 0.5
    lap("oct_mask")
    fine, section = pp.destripe(fine, mask_f, P.fine_mm, P)
    oct_h, A_o = G.pool_iso(fine, A_f, P.fine_mm, h)
    mask_o = G.pool_iso(mask_f, A_f, P.fine_mm, h)[0] > 0.5
    del fine, mask_f
    lap("destripe")
    oct_pyr, mri_pyr = pyramids(oct_h, mask_o, A_o, mri_h, mask_m, A_m, P)
    lap("two_class")
    # orientation search in the crop and affine ladder (innovation 3)
    candidates, info = search(*mri_pyr[P.levels[0]], *oct_pyr[P.levels[0]], P, device)
    lap("search")
    best, _ = refine(candidates, mri_pyr, oct_pyr, P, device, info["tau"])
    del oct_pyr, mri_pyr
    lap("refine")

    T = best["T"]
    pose = {k: best[k] for k in ("S", "L", "polarity", "mirror", "log_scales", "shears", "overlap", "search_rank")}
    pose["scale_per_oct_axis"] = np.linalg.norm(T[:3, :3] @ (vo.affine[:3, :3] / vo.spacing_mm), axis=0).tolist()
    flags = [f"foreground_no_valley:{k}" for k, f in (("oct", fg_o), ("mri", fg_m)) if f.get("status") == "no_valley"]
    flags += ["overlap_floor"] * bool(info["overlap_floor"]) + ["nondefault_params"] * (P != Params())
    flags += ["clamp_saturated"] * bool(np.abs([*best["log_scales"], *best["shears"]]).max() >= P.clamp * (1 - 1e-3))
    result = {"octreg_version": __version__, "inputs": {"oct": {**vars(vo), "path": _abs(vo.path)},
              "mri": {**vars(vm), "path": _abs(vm.path)}, "oct_spacing_um": oct_spacing_um, "oct_mask": oct_mask, "mri_mask": mri_mask,
              "device": str(device)},
              "params": P.to_dict(), "params_hash": P.hash(), "T_oct2mri": T, "pose": pose, "flags": flags,
              "boundary_mm": _boundary_mm(mask_o, valid_o, A_o, mask_m, A_m, T), "foreground": {"oct": fg_o, "mri": fg_m},
              "section": section, "search": info}
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
    result.update(seconds={**sec, "total": time.time() - t0}, peak_rss_gb=_peak_rss_gb(),
                  gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None)
    io.write_json(result, out / "result.json")
    return result


def fine_mask(fine, affine, params: Params = Params(), mask=None):
    """OCT specimen mask on the fine grid (fine [D, H, W] of spacing params.fine_mm, voxel -> world affine): the mask file
    `mask` (io.Volume) if given, else the isotropic-texture mask, or for oct_foreground 'intensity' the histogram foreground
    of the OCT box-averaged to levels[-1] (nearest back); always AND fine > 0 (zeros are missing data).
    -> (bool [D, H, W], info with 'source')."""
    P = params
    if mask is not None:
        m, info = _mask_on(mask, fine.shape, affine, P.fine_mm), {"source": mask.path}
    elif P.oct_foreground == "texture":
        m, info = pp.specimen_mask(fine, P.fine_mm, P)
        info["source"] = "texture"
    else:
        raw_h, A_h = G.pool_iso(fine, affine, P.fine_mm, P.levels[-1])
        m_h, info = pp.foreground(raw_h, P.levels[-1], P)
        m, info["source"] = G.resample_to(m_h, A_h, fine.shape, affine, np.eye(4), order=0), "intensity"
    return np.logical_and(m, fine > 0, out=m), info


def pyramids(oct_h, oct_mask, oct_affine, mri_h, mri_mask, mri_affine, params: Params = Params()):
    """Two-class maps at levels[-1] (innovation 2) and their pyramids. oct_h, mri_h: images [D, H, W] on isotropic levels[-1] grids
    with bool foreground masks and voxel -> world affines. OCT: p and the mask are box-averaged to each level, channels
    u = (p, 1 - p) with weight w = pooled mask; MRI: channels v = (p M, (1 - p) M) box-averaged, and the pooled mask.
    -> (oct_pyr {level_mm: (u [2, ...], w, affine)}, mri_pyr {level_mm: (v [2, ...], mask fraction, affine)})."""
    P, h = params, params.levels[-1]
    p_o = pp.two_class(oct_h, oct_mask, h, P, flatten=P.oct_flatten)
    v_m = pp.mri_channels(pp.two_class(mri_h, mri_mask, h, P, flatten=P.mri_flatten), mri_mask, P.features)
    oct_pyr, mri_pyr = {}, {}
    for lv in P.levels:
        (p, A_o), (w, _) = _pool(p_o, oct_affine, h, lv), _pool(oct_mask, oct_affine, h, lv)
        (v, A_m), (m, _) = _pool(v_m, mri_affine, h, lv), _pool(mri_mask, mri_affine, h, lv)
        oct_pyr[lv], mri_pyr[lv] = (*pp.oct_channels(p, w, P.features), A_o), (v, m, A_m)
    return oct_pyr, mri_pyr


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


def _pool(x, A, h, level):
    """Box average of [D, H, W] or [C, D, H, W] on the base grid (h mm) to a pyramid level. -> (float32 array, affine)."""
    return (np.asarray(x, np.float32), A) if level == h else G.pool_iso(x, A, h, level)


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


def _peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)


def _qc_png(path, oct_h, mri_in_oct, mask, pose, h):
    """3 x 3 figure: planes through the mask centroid normal to each OCT array axis; OCT, MRI through T, 2 mm checkerboard (MRI
    inverted inside the mask when the polarity is -1). Grey levels: p1-p99 inside the mask."""
    from matplotlib.figure import Figure
    o, m = ((x - lo) / max(hi - lo, 1e-12) for x in (oct_h, mri_in_oct) for lo, hi in [np.percentile(x[mask], [1, 99])])
    o, m = np.clip(o, 0, 1), np.clip(m, 0, 1)
    centre, tile = np.argwhere(mask).mean(0).round().astype(int), max(2, round(2.0 / h))
    fig = Figure(figsize=(10, 10.5))
    for ax_i in range(3):
        po, pm, pw = (np.take(x, centre[ax_i], ax_i) for x in (o, m, mask))
        board = (np.arange(po.shape[0])[:, None] // tile + np.arange(po.shape[1]) // tile) % 2 == 0
        inv = pose["polarity"] < 0
        board_img = np.where(board, po, np.where(pw & inv, 1 - pm, pm))
        for j, (img, name) in enumerate(((po, "OCT"), (pm, "MRI through T"), (board_img, "checkerboard" + " (MRI inverted)" * inv))):
            ax = fig.add_subplot(3, 3, 3 * ax_i + j + 1)
            ax.imshow(img.T, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_title(f"{name}, OCT axis {ax_i} = {centre[ax_i]}", fontsize=9)
            ax.set_axis_off()
    fig.suptitle(f"S {pose['S']:.4f}   polarity {pose['polarity']:+d}   overlap {pose['overlap']:.2f}   "
                 f"scale per OCT axis {np.round(pose['scale_per_oct_axis'], 3).tolist()}")
    fig.savefig(path, dpi=90)
