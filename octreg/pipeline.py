"""The registration pipeline (spec steps 0-9) and `octreg apply`."""
from __future__ import annotations

from pathlib import Path

from .params import Params
from .types import Result


def register(oct, mri, out, oct_spacing_um=None, oct_mask=None, mri_mask=None, init=None, params: Params = Params(),
             device="cuda") -> Result:
    """Register an OCT block to an MRI and write every output into out.

    oct, mri: image paths (io.load_volume). out: output directory (created). oct_spacing_um: (s0, s1, s2) um per OCT array axis.
    oct_mask, mri_mask: user foreground masks on the image grids (source 'user'). init: prior pose file (io.read_transform,
    OCT world -> MRI world) restricting the same search. params: any difference from Params() sets flag nondefault_params.
    device: 'cuda' | 'cpu'.
    Steps 0-5 are cached under out/cache, keyed by the input sha256s and params.hash(stages=('prep',)); memory is freed between
    stages and seconds, peak RSS and peak GPU memory are recorded per stage.
    Writes T_oct2mri.txt, T_mri2oct.txt, oct2mri.lta, oct2mri_itk.txt, hypotheses/<hand>_<polarity>/T_oct2mri.txt (all four),
    oct_in_mri.nii.gz (MRI grid, block radius + out_margin_mm), mri_in_oct.nii.gz (OCT frame at h), qc.png, log.txt, result.json.
    -> Result. Raises types.InputRefused for a refused input; no transform is written then."""
    raise NotImplementedError


def apply(run, moving, reference, out, inverse=False, nearest=False, device="cpu") -> Path:
    """`octreg apply`: resample the NIfTI moving onto the grid of reference through run/T_oct2mri.txt (OCT -> MRI), or through
    run/T_mri2oct.txt with inverse; nearest for label images. run: a register() output directory; out: output NIfTI path. -> out."""
    raise NotImplementedError
