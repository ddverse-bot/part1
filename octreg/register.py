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
           "qc_montage.png", "result.json")


def register(oct_path, mri_path, out_dir, oct_spacing_um=None, oct_mask=None, mri_mask=None, params: Params = Params(),
             device="cuda") -> dict:
    """Register an OCT block to an MRI cropped around it and write OUTPUTS into out_dir.

    oct_path: NIfTI, or TIFF / OME-TIFF / NPY with spacing; mri_path: NIfTI. oct_spacing_um: (z, y, x) um (io.load_volume).
    oct_mask, mri_mask: optional mask files (voxels > 0, in the world of their image; oct_spacing_um applies to the OCT mask
    too) replacing the texture specimen mask and the histogram foreground. device: 'cuda' | 'cpu'.
    oct_in_mri.nii.gz: the OCT box-averaged to the MRI voxel size (not finer than fine_mm), trilinear through T onto the MRI grid
    (= apply(out_dir, oct_path, mri_path)); mri_in_oct.nii.gz: the MRI through T onto the OCT base grid; qc.png, qc_montage.png:
    the visual QC (qc_figures). -> the result.json dict. Raises ValueError for a refused input."""
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
    (mri_h, mask_m, A_m), fg_m = prepare_mri(vm, P, mm)
    lap("mri")
    # step 1-2, OCT: fine grid, specimen mask (innovation 1), base grid
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay = G.pool_iso(fine, A_f, P.fine_mm, max(_mm(vm.spacing_mm.min()), P.fine_mm))       # for oct_in_mri
    lap("oct_fine_grid")
    (oct_h, mask_o, A_o), valid_o, fg_o = prepare_oct(fine, A_f, P, mo)
    del fine
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
    io.save_nifti(G.resample_to(mri_h, A_m, oct_h.shape, A_o, T), A_o, out / "mri_in_oct.nii.gz")
    qc_figures(out / "qc", (oct_h, mask_o, A_o), (mri_h, mask_m, A_m), T, pose["polarity"], vo.affine,
               f"S {pose['S']:.4f}   overlap {pose['overlap']:.2f}")
    lap("outputs")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
    result.update(seconds={**sec, "total": time.time() - t0}, peak_rss_gb=rss,
                  gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None)
    io.write_json(result, out / "result.json")
    return result


def prepare_mri(vol, params: Params = Params(), mask=None):
    """Steps 1-2 for the MRI (io.Volume): base grid and foreground, the mask file `mask` (io.Volume) if given, else the histogram
    rule. -> ((mri_h float32 [D, H, W], foreground bool [D, H, W], affine), info)."""
    h = params.base_mm
    arr, A = G.resample_iso(vol, h)
    fg, info = pp.foreground(arr, h, params) if mask is None else (_mask_on(mask, arr.shape, A, h), {"source": mask.path})
    return (arr, fg, A), info


def prepare_oct(fine, affine, params: Params = Params(), mask=None):
    """Steps 1-2 for the OCT fine grid (fine [D, H, W] of spacing params.fine_mm, affine): specimen mask (fine_mask), then the
    OCT, the mask and the measured fraction box-averaged to base_mm, the last two > 0.5.
    -> ((oct_h float32, mask bool, base affine), measured bool, info)."""
    P = params
    mask_f, info = fine_mask(fine, affine, P, mask)
    arr, A = G.pool_iso(fine, affine, P.fine_mm, P.base_mm)
    m, valid = (G.pool_iso(x, affine, P.fine_mm, P.base_mm)[0] > 0.5 for x in (mask_f, fine > 0))
    return (arr, m, A), valid, info


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


def qc(run_dir, oct_path, mri_path, T=None, prefix=None, oct_spacing_um=None, oct_mask=None, mri_mask=None):
    """`octreg qc`: the visual QC of a run (qc_figures) for its saved T or for another transform file T (4x4 OCT world -> MRI
    world, text or .npy; a candidate pose), without registering: base grids and masks as in register with the run's params, the
    run's polarity. oct_spacing_um and the mask files default to the run's inputs; prefix to run_dir/qc, or run_dir/qc_<T file
    stem> for a given T. -> (prefix.png, prefix_montage.png) Paths. Raises ValueError for a refused input."""
    run = Path(run_dir)
    for f in (run / "result.json", T):
        if f is not None and not Path(f).is_file():
            raise ValueError(f"{f}: no such file")
    res = json.loads((run / "result.json").read_text())
    inp, P = res["inputs"], Params.from_dict(res["params"])
    spacing = inp["oct_spacing_um"] if oct_spacing_um is None else oct_spacing_um
    vo, vm = io.load_volume(oct_path, spacing), io.load_volume(mri_path)
    mo, mm = (io.load_volume(f, sp) if f else None for f, sp in ((oct_mask or inp["oct_mask"], spacing),
                                                                  (mri_mask or inp["mri_mask"], None)))
    T_run, pose = np.asarray(res["T_oct2mri"]), res["pose"]
    Tm = T_run if T is None else np.load(T) if str(T).endswith(".npy") else np.loadtxt(T)
    if Tm.shape != (4, 4):
        raise ValueError(f"{T}: expected a 4x4 matrix")
    label = f"S {pose['S']:.4f}   overlap {pose['overlap']:.2f}" if np.array_equal(Tm, T_run) else f"T {Path(T).name}"
    mri_grid = prepare_mri(vm, P, mm)[0]
    oct_grid = prepare_oct(*G.resample_iso(vo, P.fine_mm), P, mo)[0]
    prefix = prefix or run / ("qc" if T is None else f"qc_{Path(T).stem}")
    return qc_figures(prefix, oct_grid, mri_grid, Tm, pose["polarity"], vo.affine, label)


QC_COLOURS, QC_CELL_IN, QC_HEAD_IN, QC_DPI = ("#ff3030", "#30d0ff"), 2.8, 0.4, 100


def qc_figures(prefix, oct_grid, mri_grid, T, polarity, oct_file_affine, label=""):
    """Visual QC, the primary evaluation: prefix.png, one plane per OCT array axis through the specimen mask centroid, and
    prefix_montage.png, 4 evenly spaced planes per axis inside the specimen. Columns: OCT | MRI through T | checkerboard of the
    two (~2 mm squares) | OCT with the outlines of the MRI foreground through T (red) and of the OCT specimen mask (cyan). Grey
    levels p1-p99 inside the OCT mask (MRI: inside the mask and its foreground); with polarity -1 the MRI is inverted inside
    its foreground, so tissue shows the same contrast in both and the MRI background stays dark. Rows are titled by the plane
    position (mm from the first voxel of the OCT file along that axis) and are QC_CELL_IN inches high at QC_DPI; the lower
    remaining OCT axis runs to the right, the higher one up; scale bar in the first column. oct_grid = (oct_h, mask, affine)
    on the base grid; mri_grid = (mri_h, foreground, affine); T: OCT world -> MRI world; oct_file_affine: of the OCT file.
    -> (prefix.png, prefix_montage.png) Paths."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from PIL import Image
    arr, mask, A_o = oct_grid
    h, sp = _mm(np.linalg.norm(A_o[:3, 0])), np.linalg.norm(oct_file_affine[:3, :3], axis=0)
    mri, fg = (G.resample_to(np.asarray(x, np.float32), mri_grid[2], arr.shape, A_o, T) for x in mri_grid[:2])
    tissue = fg > 0.5
    both = mask & tissue if (mask & tissue).any() else mask
    o, m = (np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
            for x, w in ((arr, mask), (mri, both)) for lo, hi in [np.percentile(x[w], [1, 99])])
    m = np.where(tissue, 1 - m, m) if polarity < 0 else m
    origin = (np.linalg.inv(oct_file_affine) @ A_o)[:3, 3] * sp
    centre = np.rint(ndimage.center_of_mass(mask)).astype(int)
    span = [np.flatnonzero(mask.any(axis=tuple({0, 1, 2} - {a})))[[0, -1]] for a in range(3)]
    views = {"": [(a, centre[a]) for a in range(3)],
             "_montage": [(a, int(round(lo + (hi - lo) * t / 5))) for a, (lo, hi) in enumerate(span) for t in range(1, 5)]}
    heads = ("OCT", "MRI through T" + ", inverted" * (polarity < 0), "checkerboard", "MRI foreground (red), OCT mask (cyan)")
    scales = np.round(np.linalg.norm(T[:3, :3] @ (oct_file_affine[:3, :3] / sp), axis=0), 3).tolist()
    tile, paths, palette = max(2, round(2.0 / h)), [], Image.new("P", (1, 1))        # PNG: 254 greys and the outline colours
    palette.putpalette(np.repeat(np.linspace(0, 255, 254).round().astype(np.uint8), 3).tobytes()
                       + bytes.fromhex("".join(c[1:] for c in QC_COLOURS)))
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for suffix, planes in views.items():
        W, H = 0.3 + 4 * QC_CELL_IN, QC_HEAD_IN + len(planes) * QC_CELL_IN
        fig = Figure(figsize=(W, H), dpi=QC_DPI)
        fig.suptitle(f"{label}   polarity {polarity:+d}   scale per OCT axis {scales}", y=1 - 0.06 / H, va="top", fontsize=9)
        for r, (a, c) in enumerate(planes):
            po, pm, pf, pw = (np.take(x, c, a).T.astype(np.float32) for x in (o, m, fg, mask))
            board = (np.arange(po.shape[0])[:, None] // tile + np.arange(po.shape[1]) // tile) % 2 == 0
            y = H - QC_HEAD_IN - (r + 1) * QC_CELL_IN
            fig.text(0.2 / W, (y + QC_CELL_IN / 2) / H, f"axis {a} at {origin[a] + c * h:.2f} mm", rotation=90, ha="center",
                     va="center", fontsize=8)
            for j, img in enumerate((po, pm, np.where(board, po, pm), po)):
                ax = fig.add_axes(((0.34 + j * QC_CELL_IN) / W, (y + 0.04) / H, (QC_CELL_IN - 0.08) / W, (QC_CELL_IN - 0.08) / H))
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower")
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(heads[j], fontsize=8)
                if j == 0:
                    bar = max([b for b in (0.5, 1, 2, 5, 10, 20) if b <= po.shape[1] * h / 4], default=0.5)
                    ax.add_artist(AnchoredSizeBar(ax.transData, bar / h, f"{bar:g} mm", "lower left", color="w", frameon=False,
                                                  size_vertical=max(1, po.shape[0] / 100), fontproperties={"size": 7}))
            for x, colour in zip((pf, pw), QC_COLOURS):
                if x.min() < 0.5 < x.max():
                    ax.contour(x, [0.5], colors=colour, linewidths=1.2)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        paths.append(Path(f"{prefix}{suffix}.png"))
        Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3]).quantize(palette=palette, dither=0).save(paths[-1], optimize=True)
    return tuple(paths)
