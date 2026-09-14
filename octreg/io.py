"""Input and output in the world frames of the input files.

NIfTI: voxel -> world from the sform, else the qform (both codes 0: frame 'array', handedness 'unknown'). TIFF/OME-TIFF/NPY:
world = diag(spacing) in array order, handedness 'unknown'. Transforms are 4x4 world -> world in mm (RAS for NIfTI worlds).
"""
from __future__ import annotations

import numpy as np

from .types import Volume


def load_volume(path, spacing_um=None, axes=None, layout=None) -> Volume:
    """Open a 3-D image lazily (memmap or plane reader; never loaded whole) and hash the file.

    path: .nii / .nii.gz, .tif / .tiff / .ome.tif(f) or .npy. spacing_um: (s0, s1, s2) in um per array axis, overriding every
    other source (spacing_source 'cli'). axes: array axis letters of a TIFF/NPY, e.g. 'zyx' (default: OME DimensionOrder, else
    'zyx'). layout: v1 RAS letters for the array axes, e.g. 'SPR' (compat M1 only; frame 'layout', corner origin).
    Spacing sources in order: spacing_um, NIfTI header, OME PhysicalSize (length units only), BIDS sidecar JSON.
    Returns Volume(reader, affine 4x4 voxel -> world mm, spacing_mm, frame, handedness, spacing_source, path, sha256).
    Raises InputRefused: no spacing source; not 3-D (a 4-D input lists its volumes and is never squeezed); unreadable file.
    """
    raise NotImplementedError


def read_transform(path) -> np.ndarray:
    """4x4 world -> world (mm) from a plain 4x4 text matrix, a FreeSurfer LTA (LINEAR_RAS_TO_RAS) or an ITK text transform
    written by write_itk; always returned in the sense of the T the matching writer was given. Raises ValueError otherwise."""
    raise NotImplementedError


def write_matrix(T, path) -> None:
    """4x4 matrix as plain text, one row per line, 17 significant digits (exact float64 round trip)."""
    raise NotImplementedError


def write_lta(T, src_geom, dst_geom, path) -> None:
    """FreeSurfer LTA, type LINEAR_RAS_TO_RAS, x_dst = T x_src in mm, with the src and dst volume geometry blocks.
    src_geom, dst_geom: (shape (3 ints), affine 4x4 voxel -> world mm) of the source (OCT) and destination (MRI) files."""
    raise NotImplementedError


def write_itk(T, path) -> None:
    """ITK text transform (AffineTransform_double_3_3) that resamples the source onto the destination with ITK/ANTs tools
    (antsApplyTransforms -i OCT -r MRI -t path): the LPS point map destination -> source, i.e. inv(T) conjugated by diag(-1, -1, 1).
    T: 4x4 RAS world x_dst = T x_src (mm)."""
    raise NotImplementedError


def save_nifti(arr, affine, path, dtype=None) -> None:
    """NIfTI-1 with sform = qform = affine (code 1, units mm); arr in array order (i, j, k); dtype casts first when given."""
    raise NotImplementedError


def write_json(obj, path) -> None:
    """JSON with indent 1, numpy-safe: arrays -> lists, numpy scalars -> Python numbers, Path -> str, dataclasses -> dicts
    (Volume.reader dropped), NaN/inf -> null. Written to path + '.tmp' and renamed, so a crash never leaves half a file."""
    raise NotImplementedError


def sha256_file(path, chunk_bytes=1 << 24) -> str:
    """Hex sha256 of a file read in chunks of chunk_bytes."""
    raise NotImplementedError
