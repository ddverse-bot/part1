"""Volumes in their file frames, read plane by plane, and the output writers.

An affine maps voxel index (i, j, k) -> world mm; a transform T is a 4x4 world -> world matrix (mm).
Volume axes follow the file's storage order so that the last axis k is always the plane axis the file is streamed along:
NIfTI (i, j, k) as in the header; TIFF and NPY (x, y, z), the numpy axes (z, y, x) reversed (the ITK / SimpleITK index order).
NIfTI world = sform, else qform (frame 'header'); neither coded -> diag(pixdim) (frame 'array').
TIFF / OME-TIFF / NPY world = diag(spacing_x, spacing_y, spacing_z) (frame 'array'), the physical frame ITK gives such files.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

UNIT_MM = {"nm": 1e-6, "um": 1e-3, "µm": 1e-3, "μm": 1e-3, "micron": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1e3, "meter": 1e3}


@dataclasses.dataclass(eq=False)
class Volume:
    """A 3-D image on disk (header only; voxels are read with iter_planes). shape: voxels along (i, j, k); affine: 4x4
    voxel -> world mm; spacing_mm: voxel edges along (i, j, k), the affine column norms; frame: 'header' | 'array'."""
    path: str
    shape: tuple
    affine: np.ndarray
    spacing_mm: np.ndarray
    frame: str


def _kind(path) -> str:
    name = str(path).lower()
    for kind, ends in (("nii", (".nii", ".nii.gz")), ("tif", (".tif", ".tiff")), ("npy", (".npy",))):
        if name.endswith(ends):
            return kind
    raise ValueError(f"{path}: unsupported format (use .nii, .nii.gz, .tif, .tiff, .ome.tif or .npy)")


def _shape3(path, shape, trailing_ok) -> tuple:
    """The 3-D shape; trailing singleton dimensions are accepted only when trailing_ok (NIfTI)."""
    shape = tuple(int(s) for s in shape)
    if not (len(shape) >= 3 and min(shape[:3]) >= 2 and (all(s == 1 for s in shape[3:]) if trailing_ok else len(shape) == 3)):
        raise ValueError(f"{path}: image of shape {shape}; octreg needs one 3-D volume with >= 2 voxels per axis")
    return shape[:3]


def load_volume(path, spacing_um=None) -> Volume:
    """Open a 3-D image without reading voxels.

    spacing_um: (z, y, x) um, i.e. along (k, j, i) (numpy order for TIFF / NPY, the CLI's Z,Y,X); overrides every other
    source (a NIfTI header keeps its directions and origin). Otherwise NIfTI: sform / qform / pixdim in the header's length
    unit ('unknown' = mm); OME-TIFF: PhysicalSize X / Y / Z (unit absent = um); plain TIFF / NPY need spacing_um.
    Numpy axes are (z, y, x) unless the TIFF series names them.
    Raises ValueError for a missing file, an unsupported format, a non-3-D image, missing spacing or a singular affine."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{p}: no such file")
    kind = _kind(p)
    cli = None if spacing_um is None else np.asarray(spacing_um, float).ravel()[::-1] / 1000.0      # along (i, j, k)
    if cli is not None and (cli.shape != (3,) or not np.all(cli > 0)):
        raise ValueError(f"{p}: spacing_um must be three positive numbers (um, Z,Y,X), got {spacing_um}")
    if kind == "nii":
        import nibabel as nib
        hdr = nib.load(str(p)).header
        shape = _shape3(p, hdr.get_data_shape(), True)
        unit = UNIT_MM.get(hdr.get_xyzt_units()[0], 1.0)
        (S, scode), (Q, qcode) = hdr.get_sform(coded=True), hdr.get_qform(coded=True)
        if scode or qcode:
            A, frame = np.array(S if scode else Q, float), "header"
            A[:3] *= unit
        else:
            A, frame = np.diag([*(np.asarray(hdr.get_zooms()[:3], float) * unit), 1.0]), "array"
        if cli is not None:
            A[:3, :3] *= cli / np.linalg.norm(A[:3, :3], axis=0)
    else:
        letters, ome = "zyx", None
        if kind == "npy":
            shape = _shape3(p, np.load(p, mmap_mode="r").shape, False)[::-1]
        else:
            import tifffile
            with tifffile.TiffFile(str(p)) as tif:
                s = tif.series[0]
                shape, axes = _shape3(p, s.shape, False)[::-1], s.axes.lower()
                if set(axes) & set("cs"):
                    raise ValueError(f"{p}: TIFF axes {s.axes} have channels or samples; octreg needs one 3-D volume")
                letters = axes if sorted(axes) == ["x", "y", "z"] else letters
                ome = _ome_spacing(tif.ome_metadata) if tif.is_ome else None
        if cli is None and ome is None:
            raise ValueError(f"{p}: no voxel spacing (no OME PhysicalSize X / Y / Z); pass spacing_um (--oct-spacing-um Z,Y,X)")
        A, frame = np.diag([*(cli if cli is not None else [ome[a] for a in letters[::-1]]), 1.0]), "array"
    A[3] = (0.0, 0.0, 0.0, 1.0)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    if not (np.all(np.isfinite(A)) and np.all(sp > 0) and abs(np.linalg.det(A[:3, :3])) > 1e-9 * np.prod(sp)):
        raise ValueError(f"{p}: singular or missing voxel spacing in affine\n{A}\npass spacing_um (--oct-spacing-um Z,Y,X)")
    return Volume(str(p), tuple(shape), A, sp, frame)


def _ome_spacing(xml):
    """{'x', 'y', 'z': mm} from the first OME Pixels element, or None unless X, Y and Z all have a length unit."""
    px = next((e for e in ET.fromstring(xml).iter() if e.tag.split("}")[-1] == "Pixels"), None)
    out = {}
    for a in "xyz":
        v, u = (None, None) if px is None else (px.get(f"PhysicalSize{a.upper()}"), px.get(f"PhysicalSize{a.upper()}Unit", "µm"))
        if v is None or u not in UNIT_MM or not float(v) > 0:
            return None
        out[a] = float(v) * UNIT_MM[u]
    return out


def iter_planes(vol: Volume):
    """Yield (k, plane float32 [shape_i, shape_j]) for every plane along the last axis, one plane in memory at a time, in
    file order (NIfTI: the (gzip) byte stream; TIFF: pages or memmap; NPY: memmap). NIfTI scl_slope / scl_inter are applied;
    non-finite voxels are yielded as 0."""
    kind, (ni, nj, nk) = _kind(vol.path), vol.shape
    if kind == "nii":
        import nibabel as nib
        img = nib.load(vol.path)
        dt, off, slope, inter = img.header.get_data_dtype(), int(img.dataobj.offset), float(img.dataobj.slope), float(img.dataobj.inter)
        nbytes = ni * nj * dt.itemsize
        with (gzip.open if vol.path.lower().endswith(".gz") else open)(vol.path, "rb") as f:
            f.read(off)                                        # nibabel resets vox_offset to 0 in .gz headers: use dataobj.offset
            for k in range(nk):
                buf = f.read(nbytes)
                if len(buf) != nbytes:
                    raise ValueError(f"{vol.path}: data truncated at plane {k}")
                yield k, _finish(np.frombuffer(buf, dt).reshape((ni, nj), order="F"), slope, inter)
        return
    if kind == "npy":
        arr = np.load(vol.path, mmap_mode="r")
    else:
        import tifffile
        try:
            arr = tifffile.memmap(vol.path, series=0, mode="r").reshape(nk, nj, ni)
        except ValueError:                                     # compressed or non-contiguous: one page per plane
            with tifffile.TiffFile(vol.path) as tif:
                pages = tif.series[0].pages
                if len(pages) != nk:
                    raise ValueError(f"{vol.path}: TIFF is neither memory-mappable nor one page per plane")
                for k, page in enumerate(pages):
                    yield k, _finish(page.asarray().reshape(nj, ni).T, 1.0, 0.0)
            return
    for k in range(nk):
        yield k, _finish(arr[k].T, 1.0, 0.0)


def _finish(plane, slope, inter):
    x = np.asarray(plane, np.float64) * slope + inter if (slope, inter) != (1.0, 0.0) else np.asarray(plane)
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def save_nifti(arr, affine, path) -> None:
    """NIfTI-1 with sform = qform = affine (code 1, unit mm); arr in (i, j, k) order."""
    import nibabel as nib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(affine, float)
    img = nib.Nifti1Image(np.asarray(arr), A)
    img.header.set_xyzt_units("mm")
    img.header.set_qform(A, code=1)
    img.header.set_sform(A, code=1)
    nib.save(img, str(path))


def save_field(field, affine, path) -> None:
    """A displacement field (numpy [3, D, H, W], mm along the world axes of `affine`, the grid's voxel -> world) as a NIfTI-1
    vector image: float32 of shape [D, H, W, 1, 3], intent 'vector', sform = qform = affine. The components follow the NIfTI
    world axes (RAS+ for a standard header); ITK and ANTs read displacement components as LPS, the first two negated."""
    import nibabel as nib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(affine, float)
    img = nib.Nifti1Image(np.ascontiguousarray(np.moveaxis(np.asarray(field, np.float32), 0, -1)[:, :, :, None, :]), A)
    img.header.set_intent("vector")
    img.header.set_xyzt_units("mm")
    img.header.set_qform(A, code=1)
    img.header.set_sform(A, code=1)
    nib.save(img, str(path))


def load_field(path):
    """A displacement field written by save_field -> (float32 [3, D, H, W], affine 4x4). Raises ValueError for another shape."""
    import nibabel as nib
    img = nib.load(str(path))
    if len(img.shape) != 5 or img.shape[3:] != (1, 3):
        raise ValueError(f"{path}: shape {img.shape}; a displacement field has shape [D, H, W, 1, 3]")
    return np.moveaxis(np.asarray(img.dataobj, np.float32)[:, :, :, 0, :], -1, 0), np.array(img.affine, float)


def write_transform_txt(T, path) -> None:
    """4x4 matrix as plain text, one row per line, 17 significant digits (exact float64 round trip with np.loadtxt)."""
    with open(path, "w", newline="\n") as f:                    # LF on every OS (np.savetxt(path) writes CRLF on Windows)
        np.savetxt(f, np.asarray(T, float), fmt="%.17g")


def _vol_info(vol: Volume) -> str:
    """FreeSurfer volume geometry: direction cosines, voxel size, c_ras = affine (shape / 2)."""
    A = np.asarray(vol.affine, float)
    vs = np.linalg.norm(A[:3, :3], axis=0)
    D, c = A[:3, :3] / vs, A[:3, :3] @ (np.array(vol.shape, float) / 2.0) + A[:3, 3]
    f = lambda v: " ".join(f"{x:.15e}" for x in v)
    return (f"valid = 1  # volume info valid\nfilename = {Path(vol.path).resolve()}\nvolume = {vol.shape[0]} {vol.shape[1]} "
            f"{vol.shape[2]}\nvoxelsize = {f(vs)}\nxras   = {f(D[:, 0])}\nyras   = {f(D[:, 1])}\nzras   = {f(D[:, 2])}\ncras   = {f(c)}\n")


def write_lta(T, src: Volume, dst: Volume, path) -> None:
    """FreeSurfer LTA, LINEAR_RAS_TO_RAS, x_dst = T x_src (world mm), with source and destination volume geometry."""
    rows = "\n".join(" ".join(f"{v:.15e}" for v in r) for r in np.asarray(T, float))
    with open(path, "w", newline="\n") as f:                    # LF on every OS
        f.write(f"# transform file {path}\n# created by octreg\ntype      = 1 # LINEAR_RAS_TO_RAS\nnxforms   = 1\n"
                f"mean      = 0.0000 0.0000 0.0000\nsigma     = 1.0000\n1 4 4\n{rows}\nsrc volume info\n{_vol_info(src)}"
                f"dst volume info\n{_vol_info(dst)}subject unknown\nfscale 0.100000\n")


def write_itk(T, src: Volume, dst: Volume, path) -> None:
    """ITK AffineTransform_double_3_3 text for resampling src onto dst with ITK / ANTs (antsApplyTransforms -i src -r dst -t path):
    the physical point map dst -> src, F_src inv(T) F_dst, where F = diag(-1, -1, 1) for NIfTI (ITK reads RAS headers as LPS)
    and the identity for TIFF / NPY (the array frame is ITK's physical frame)."""
    F_src, F_dst = (np.diag([-1.0, -1.0, 1.0, 1.0] if _kind(v.path) == "nii" else [1.0] * 4) for v in (src, dst))
    L = F_src @ np.linalg.inv(np.asarray(T, float)) @ F_dst
    q = " ".join(f"{v:.17g}" for v in [*L[:3, :3].ravel(), *L[:3, 3]])
    with open(path, "w", newline="\n") as f:                    # LF on every OS
        f.write(f"#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: {q}\n"
                "FixedParameters: 0 0 0\n")


def write_json(obj, path) -> None:
    """JSON (indent 1) of dicts, lists, dataclasses, numpy arrays and scalars, Paths; NaN and inf become null."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:                 # LF on every OS, like the transform files
        f.write(json.dumps(_plain(obj), indent=1, allow_nan=False))


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _plain(dataclasses.asdict(o))
    if isinstance(o, (np.ndarray, np.generic)):
        return _plain(o.tolist())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if o is None or isinstance(o, (str, int, float)):
        return o
    raise TypeError(f"write_json: cannot serialise {type(o).__name__}")
