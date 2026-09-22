"""`octreg register` (docs/METHOD.md §1-6 on two files, then the outputs) and `octreg apply`.

A transform T is a 4x4 matrix (mm) from the OCT world to the MRI world; worlds are the NIfTI header frames (array frame for
TIFF / NPY). The OCT is only ever streamed: its largest array in memory is the fine grid (Params.fine_mm). The smooth
deformation of §6 is a pull-back field u on the MRI base grid (octreg.deform): the registered OCT at the MRI point x is
OCT(T^-1 (x + u(x))). T stays the primary result, and all transform files hold T alone.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from . import __version__, geometry as G, io, preprocess as pp
from .deform import smooth_deformation
from .ngf import refine_ngf
from .params import Params
from .refine import decompose, refine
from .search import box, search

MIRROR = np.diag([1.0, 1.0, -1.0, 1.0])       # the other handedness of an OCT frame: world z negated

OUTPUTS = ("T_oct2mri.txt", "T_mri2oct.txt", "oct2mri.lta", "oct2mri_itk.txt", "oct_in_mri.nii.gz", "oct_in_mri_affine.nii.gz",
           "mri_in_oct.nii.gz", "qc.png", "qc_montage.png", "result.json")
WARP = "oct2mri_warp.nii.gz"
OUTPUTS_DEFORM = (WARP, "qc_deform.png")       # written only when the smooth deformation (§6) is applied


def peak_rss_gb():
    """Peak resident memory of this process (GiB): ru_maxrss on Unix, the peak working set on Windows; None if unavailable."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
    except ImportError:
        pass
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):                           # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c, k32 = Counters(), ctypes.WinDLL("kernel32")
    c.cb = ctypes.sizeof(c)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    return c.PeakWorkingSetSize / 2 ** 30 if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else None


def register(oct_path, mri_path, out_dir, oct_spacing_um=None, oct_mask=None, mri_mask=None, params: Params = Params(),
             device="cuda") -> dict:
    """Register an OCT block to an MRI cropped around it and write OUTPUTS into out_dir.

    oct_path: NIfTI, or TIFF / OME-TIFF / NPY with spacing; mri_path: NIfTI. oct_spacing_um: (z, y, x) um (io.load_volume).
    oct_mask, mri_mask: optional mask files (voxels > 0, in the world of their image; oct_spacing_um applies to the OCT mask
    too) replacing the texture specimen mask and the histogram foreground. device: 'cuda' | 'cpu'.
    oct_in_mri.nii.gz: the OCT box-averaged to the MRI voxel size (not finer than fine_mm), trilinear through T and the field
    of §6 onto the MRI grid (= apply(out_dir, oct_path, mri_path)); oct_in_mri_affine.nii.gz: the same through T alone (the
    same file when no field is applied); mri_in_oct.nii.gz: the MRI through T onto the OCT base grid; qc.png, qc_montage.png:
    the visual QC of the affine (qc_figures). With the field applied also OUTPUTS_DEFORM: oct2mri_warp.nii.gz, the field on
    the MRI base grid (io.save_field; convention in octreg.deform), and qc_deform.png (qc_deform_figure).
    -> the result.json dict. Raises ValueError for a refused input."""
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

    # §1, MRI: base grid and histogram foreground
    (mri_h, mask_m, A_m), fg_m = prepare_mri(vm, P, mm)
    lap("mri")
    # §1, OCT: fine grid, specimen mask, base grid
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay = G.pool_iso(fine, A_f, max(_mm(vm.spacing_mm.min()), P.fine_mm))                   # for oct_in_mri
    lap("oct_fine_grid")
    (oct_h, mask_o, A_o), valid_o, fg_o = prepare_oct(fine, A_f, P, mo)
    del fine
    lap("oct_mask")
    # §2: two-class maps
    u, w = pp.oct_channels(pp.two_class(oct_h, mask_o, h, P), mask_o)
    v = pp.mri_channels(pp.two_class(mri_h, mask_m, h, P), mask_m)
    lap("two_class")
    # §3-5: search and affine refinement, then fine-structure refinement of the best pose. Orientation headers on both files fix
    # the handedness; otherwise (TIFF, NPY, NIfTI without sform/qform) both are tried and the pose with the higher F wins
    hands = {}
    for hand, F in ((1, np.eye(4)), (-1, MIRROR))[:1 if vo.frame == vm.frame == "header" else 2]:
        poses, info = align((u, w, valid_o, F @ A_o), (v, mask_m, A_m), P, device)
        T5, info["ngf"] = refine_ngf((mri_h, mask_m, A_m), (oct_h, mask_o, F @ A_o), poses[0]["T"], P, device)
        hands[hand] = (poses[0], T5, F, info)
    lap("search_refine_ngf")
    hand = max(hands, key=lambda k: hands[k][3]["ngf"]["F"])
    best, T5, F, info = hands[hand]
    T = T5 @ F
    io.write_transform_txt(T, out / "T_oct2mri.txt")            # the affine, the primary result: written before §6 runs
    io.write_transform_txt(np.linalg.inv(T), out / "T_mri2oct.txt")
    io.write_lta(T, vo, vm, out / "oct2mri.lta")
    io.write_itk(T, vo, vm, out / "oct2mri_itk.txt")
    # §6: smooth deformation on top of the affine, in the MRI world (where the handedness does not appear)
    field, info["deform"] = smooth_deformation((mri_h, mask_m, A_m), (oct_h, mask_o, A_o), T, P, device)
    applied = info["deform"]["status"] == "applied"
    lap("deform")

    _, _, ls, sh = decompose(T5, box(F @ A_o, oct_h.shape)[0])
    pose = {k: best[k] for k in ("S", "S_class", "S_outline", "L", "polarity", "search_rank")}
    corners = G.apply_affine(vo.affine, np.array([[i, j, k] for i in (0, vo.shape[0] - 1) for j in (0, vo.shape[1] - 1)
                                                  for k in (0, vo.shape[2] - 1)], float))
    shift = float(np.linalg.norm(G.apply_affine(T, corners) - G.apply_affine(best["T"] @ F, corners), axis=1).mean())
    pose.update(handedness=hand, T_before_ngf=best["T"] @ F, NGF_start=info["ngf"]["F_start"], NGF=info["ngf"]["F"], ngf_shift_mm=shift,
                NGF_other_handedness=hands[-hand][3]["ngf"]["F"] if -hand in hands else None, log_scales=ls.tolist(),
                shears=sh.tolist(), scale_per_oct_axis=np.linalg.norm(T[:3, :3] @ (vo.affine[:3, :3] / vo.spacing_mm), axis=0).tolist())
    flags = ["mri_foreground_no_valley"] * (fg_m.get("status") == "no_valley")
    flags += ["nondefault_params"] * (P != Params())
    flags += ["clamp_saturated"] * bool(np.abs([*best["log_scales"], *best["shears"], *ls, *sh]).max() >= P.clamp * (1 - 1e-3))
    flags += ["mirrored_oct_frame"] * (hand == -1)
    flags += ["deformation_not_supported"] * (not applied)
    inputs = {"oct": {**vars(vo), "path": _abs(vo.path)}, "mri": {**vars(vm), "path": _abs(vm.path)},
              "oct_spacing_um": oct_spacing_um, "oct_mask": oct_mask, "mri_mask": mri_mask, "device": str(device)}
    result = {"octreg_version": __version__, "inputs": inputs, "params": P.to_dict(), "params_hash": P.hash(), "T_oct2mri": T,
              "pose": pose, "flags": flags,
              "foreground": {"oct": fg_o, "mri": {"source": "histogram", **fg_m}}, **info}
    io.write_json(result, out / "result.json")                  # before the overlays, so a failing overlay loses no result
    for f in OUTPUTS_DEFORM:                                    # of an earlier run into the same directory
        (out / f).unlink(missing_ok=True)
    io.save_nifti(G.resample_to(*overlay, vm.shape, vm.affine, np.linalg.inv(T)), vm.affine, out / "oct_in_mri_affine.nii.gz")
    if applied:
        io.save_field(field, A_m, out / WARP)
        io.save_nifti(G.resample_to(*overlay, vm.shape, vm.affine, np.linalg.inv(T), field=field, affine_field=A_m), vm.affine,
                      out / "oct_in_mri.nii.gz")
    else:
        shutil.copyfile(out / "oct_in_mri_affine.nii.gz", out / "oct_in_mri.nii.gz")
    io.save_nifti(G.resample_to(mri_h, A_m, oct_h.shape, A_o, T), A_o, out / "mri_in_oct.nii.gz")
    qc_figures(out / "qc", (oct_h, mask_o, A_o), (mri_h, mask_m, A_m), T, pose["polarity"], vo.affine,
               _label(pose))
    if applied:
        qc_deform_figure(out / "qc_deform.png", (oct_h, mask_o, A_o), (mri_h, mask_m, A_m), T, field, pose["polarity"],
                         deform_label(info["deform"]))
    lap("outputs")
    result.update(seconds={**sec, "total": time.time() - t0}, peak_rss_gb=peak_rss_gb(),
                  gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None)
    io.write_json(result, out / "result.json")
    return result


def prepare_mri(vol, params: Params = Params(), mask=None):
    """§1 for the MRI (io.Volume): base grid and foreground, the mask file `mask` (io.Volume) if given, else the histogram
    rule. -> ((mri_h float32 [D, H, W], foreground bool [D, H, W], affine), info)."""
    h = params.base_mm
    arr, A = G.resample_iso(vol, h)
    fg, info = pp.foreground(arr, h, params) if mask is None else (_mask_on(mask, arr.shape, A, h), {"source": mask.path})
    return (arr, fg, A), info


def prepare_oct(fine, affine, params: Params = Params(), mask=None):
    """§1 for the OCT fine grid (fine [D, H, W] of spacing params.fine_mm, affine): specimen mask (fine_mask), then the
    OCT, the mask and the measured fraction box-averaged to base_mm, the last two > 0.5.
    -> ((oct_h float32, mask bool, base affine), measured bool, info)."""
    P = params
    mask_f, info = fine_mask(fine, affine, P, mask)
    arr, A = G.pool_iso(fine, affine, P.base_mm)
    m, valid = (G.pool_iso(x, affine, P.base_mm)[0] > 0.5 for x in (mask_f, fine > 0))
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
    """§3-4 on base-grid arrays, oct_base = (u [2, D, H, W], specimen mask, measured fraction, affine) and mri_base =
    (v [2, D', H', W'], mask, affine): box average to search_mm, search.search there (polarity 0 reads the polarity off the sign
    of the two-class score, +1 / -1 force it), then refine.refine of every search pose on the base grid.
    -> (refined poses lowest L first, {'search': search info, 'refine': {n_poses, seconds}})."""
    P = params
    pool = lambda x, A: G.pool_iso(x, A, P.search_mm)[0]
    (u, w, q, A_o), (v, m, A_m) = oct_base, mri_base
    candidates, info = search(pool(v, A_m), *G.pool_iso(m, A_m, P.search_mm), pool(u, A_o), pool(w, A_o),
                              *G.pool_iso(q, A_o, P.search_mm), P, device, polarity)
    t0 = time.time()
    poses = refine(candidates, mri_base, oct_base, P, device)
    return poses, {"search": info, "refine": {"n_poses": len(candidates), "seconds": time.time() - t0}}


def apply(run_dir, moving, reference, out, inverse=False, affine_only=False, oct_spacing_um=None) -> Path:
    """Resample the volume `moving` (OCT frame; MRI frame with inverse) onto the grid of `reference` through run_dir/T_oct2mri.txt
    (T_mri2oct.txt with inverse): streamed box average to the reference voxel size (never finer than the moving voxel size),
    then trilinear, 0 outside. When the run holds oct2mri_warp.nii.gz (§6), an OCT-frame volume goes through T and that field
    onto the MRI-frame reference, as oct_in_mri.nii.gz did; affine_only leaves the field out. The field is not inverted: with
    inverse the affine alone is used and a note is printed. moving, reference: image files (io.load_volume; the run's OCT file
    with the run's oct_spacing_um from result.json). oct_spacing_um: (z, y, x) um for the file in the OCT frame (`moving`, or
    `reference` with inverse), overriding the recorded spacing: a TIFF or NPY run needs it once its file has moved. out: NIfTI
    path. -> out. Raises ValueError for a refused input."""
    run = Path(run_dir)
    T_file = run / ("T_mri2oct.txt" if inverse else "T_oct2mri.txt")
    for f in (T_file, run / "result.json"):
        if not f.is_file():
            raise ValueError(f"{f}: no such file")
    T = np.loadtxt(T_file)
    inp = json.loads((run / "result.json").read_text())["inputs"]
    oct_i = int(bool(inverse))                                  # the file in the OCT frame: moving, reference with inverse
    mv, rf = (io.load_volume(f, oct_spacing_um if oct_spacing_um is not None and i == oct_i else
                             inp["oct_spacing_um"] if _abs(f) == inp["oct"]["path"] else None)
              for i, f in enumerate((moving, reference)))
    field, A_field = io.load_field(run / WARP) if (run / WARP).is_file() and not (inverse or affine_only) else (None, None)
    if inverse and (run / WARP).is_file():
        print(f"octreg: note: {WARP} is not inverted; --inverse resamples through the affine alone", flush=True)
    arr, A = G.resample_iso(mv, max(_mm(rf.spacing_mm.min()), _mm(mv.spacing_mm.min())))
    io.save_nifti(G.resample_to(arr, A, rf.shape, rf.affine, np.linalg.inv(T), field=field, affine_field=A_field), rf.affine, out)
    return Path(out)


def deform_label(d):
    """One line of the §6 read-outs (info of deform.smooth_deformation): status; when a field is applied the lattice spacing and
    the chosen membrane weight, the residuals of both kinds on the affine -> on the warped OCT, the field magnitude and the
    strain, else the evidence found."""
    if d["status"] != "applied":
        return f"§6: {d['status']} ({d['n_interior']} interior matches, {d['n_boundary']} boundary points)"
    r, arrow = d["residual"], lambda v, s=1.0, nd=2: " -> ".join("n/a" if x is None else f"{s * x:.{nd}f}" for x in v)
    return (f"§6: lattice {d['grid_mm']:g} mm, lam {d['lam']:.2f}; interior {arrow(r['interior_mm'])} mm, "
            f"boundary {arrow(r['boundary_mm'])} mm (within 0.3 mm: {arrow(r['boundary_within_0.3mm'], 100, 0)} %); field median {d['field']['median_mm']:.2f}, max "
            f"{d['field']['max_mm']:.2f} mm; max strain {d['max_strain']:.2f}")


def _label(pose):
    ngf = f"; §5: NGF {pose['NGF_start']:.4f} -> {pose['NGF']:.4f}" if "NGF" in pose else ""
    return f"§4: S {pose['S']:.4f} (class {pose['S_class']:+.4f}, outline {pose['S_outline']:.4f}){ngf}"


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
    label = _label(pose) if np.array_equal(Tm, T_run) else f"T {Path(T).name}"
    mri_grid = prepare_mri(vm, P, mm)[0]
    oct_grid = prepare_oct(*G.resample_iso(vo, P.fine_mm), P, mo)[0]
    prefix = prefix or run / ("qc" if T is None else f"qc_{Path(T).stem}")
    return qc_figures(prefix, oct_grid, mri_grid, Tm, pose["polarity"], vo.affine, label)


QC_COLOURS, QC_CELL_IN, QC_HEAD_IN, QC_DPI = ("#ff3030", "#30d0ff"), 2.8, 0.4, 100


def _match(x, y, where, n=256):
    """x mapped by rank onto the grey scale of y over the voxels where both show tissue, so that the two columns can be read
    against each other. Display only."""
    if where.sum() < 100:
        return x
    q = np.linspace(0, 1, n)
    return np.interp(x, np.quantile(x[where], q), np.quantile(y[where], q))


def qc_figures(prefix, oct_grid, mri_grid, T, polarity, oct_file_affine, label=""):
    """Visual QC, the primary evaluation: prefix.png, one plane per OCT array axis through the specimen mask centroid, and
    prefix_montage.png, 4 evenly spaced planes per axis inside the specimen. Columns: OCT | MRI through T | checkerboard of the
    two (~2 mm squares) | OCT with the outlines of the MRI foreground through T (red) and of the OCT specimen mask (cyan). Grey
    levels p1-p99 inside the OCT mask (MRI: inside the mask and its foreground); with polarity -1 the MRI is inverted inside
    its foreground, and inside its foreground it is then mapped by rank onto the grey scale of the OCT in the tissue both show
    (_match), so that a structure keeps its grey level from one tile of the checkerboard to the next. The checkerboard is drawn
    wherever the specimen mask or the MRI foreground reaches. The only voxels it leaves out are those both maps call
    background, where by definition the two cannot disagree, so the support cannot hide a misplacement: it grows, not shrinks,
    as the pose gets worse, and tissue in the tiles of one volume against nothing in the tiles of the other stays visible.
    Rows are titled by the plane
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
    m = np.where(tissue, _match(m, o, mask & tissue), m)
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
            for j, img in enumerate((po, pm, np.where(board, po, pm) * ((pw > 0.5) | (pf > 0.5)), po)):
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


def qc_deform_figure(path, oct_grid, mri_grid, T, field, polarity, label=""):
    """Visual QC of the smooth deformation (§6), on the MRI base grid: one plane per MRI array axis through the centroid of the
    OCT specimen mask through T inside the MRI foreground (of the MRI foreground alone if they do not meet). Columns: MRI | OCT through T | OCT through T and
    the field | field magnitude inside the MRI foreground (mm, colour bar), each with the outline of the MRI foreground (red).
    Grey levels as in qc_figures (p1-p99 inside the masks, MRI inverted inside its foreground with polarity -1 and mapped by
    rank onto the grey scale of the OCT through T in the tissue both show); rows are titled
    by the plane position (mm from the first voxel of the MRI base grid), QC_CELL_IN inches high at QC_DPI. oct_grid, mri_grid:
    (array, mask, affine) on the base grids; field: [3, D, H, W] on the MRI base grid (octreg.deform). -> Path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image
    (o_arr, o_mask, A_o), (mri, fg, A_m) = oct_grid, mri_grid
    h, Ti = _mm(np.linalg.norm(A_m[:3, 0])), np.linalg.inv(T)
    inside = G.resample_to(np.asarray(o_mask, np.float32), A_o, mri.shape, A_m, Ti) > 0.5
    octs = [G.resample_to(o_arr, A_o, mri.shape, A_m, Ti, **kw) for kw in ({}, {"field": field, "affine_field": A_m})]
    both = inside & fg if (inside & fg).any() else fg
    lo, hi = np.percentile(octs[0][inside if inside.any() else fg], [1, 99])
    octs = [np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1) for x in octs]
    lo, hi = np.percentile(mri[both], [1, 99])
    m = np.clip((mri - lo) / max(hi - lo, 1e-12), 0, 1)
    m = np.where(fg, 1 - m, m) if polarity < 0 else m
    m = np.where(fg, _match(m, octs[0], inside & fg), m)
    mag = np.where(fg, np.linalg.norm(field, axis=0), np.nan)
    top = max(float(np.nanmax(mag)) if fg.any() else 0.0, 1e-3)
    centre = np.rint(ndimage.center_of_mass(both)).astype(int)
    heads = ("MRI" + ", inverted" * (polarity < 0), "OCT through T", "OCT through T and the field", "field magnitude (mm)")
    W, H = 0.3 + 4 * QC_CELL_IN + 0.55, QC_HEAD_IN + 3 * QC_CELL_IN
    fig = Figure(figsize=(W, H), dpi=QC_DPI)
    fig.suptitle(label, y=1 - 0.06 / H, va="top", fontsize=8)
    for a in range(3):
        y = H - QC_HEAD_IN - (a + 1) * QC_CELL_IN
        fig.text(0.2 / W, (y + QC_CELL_IN / 2) / H, f"axis {a} at {centre[a] * h:.2f} mm", rotation=90, ha="center", va="center",
                 fontsize=8)
        contour = np.take(fg, centre[a], a).T.astype(np.float32)
        for j, x in enumerate((m, *octs, mag)):
            ax = fig.add_axes(((0.34 + j * QC_CELL_IN) / W, (y + 0.04) / H, (QC_CELL_IN - 0.08) / W, (QC_CELL_IN - 0.08) / H))
            ax.set_facecolor("k")
            im = ax.imshow(np.take(x, centre[a], a).T, origin="lower", **({"cmap": "gray", "vmin": 0, "vmax": 1} if j < 3 else
                                                                         {"cmap": "viridis", "vmin": 0, "vmax": top}))
            ax.set_xticks([])
            ax.set_yticks([])
            if contour.min() < 0.5 < contour.max():
                ax.contour(contour, [0.5], colors=QC_COLOURS[0], linewidths=1.0)
            if a == 0:
                ax.set_title(heads[j], fontsize=8)
        bar = fig.colorbar(im, cax=fig.add_axes(((0.34 + 4 * QC_CELL_IN) / W, (y + 0.04) / H, 0.1 / W, (QC_CELL_IN - 0.08) / H)))
        bar.ax.tick_params(labelsize=7)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3]).save(path, optimize=True)
    return Path(path)
