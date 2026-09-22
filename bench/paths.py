"""Machine roots of the bench scripts. The package (octreg/) never reads this.

    OCTREG_PROJECT_ROOT   where the runs live (bench_runs/...); unset, it is D:/Projects/oct-mri-registration
    OCTREG_DATA_ROOT      where the data live; unset, it is <project root>/oct-mri-registration/data when that directory
                          exists and D:/Datasets/oct-mri-registration otherwise

Those two are the only overrides: no other machine root is tried. Set both on any other machine. Nothing else in the
benchmark hard-codes a path.
"""
from __future__ import annotations

import os
from pathlib import Path


def _root(var, first, fallback):
    return Path(os.environ[var]) if os.environ.get(var) else first if first.is_dir() else Path(fallback)


PROJECT_ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or "D:/Projects/oct-mri-registration")
DATA_ROOT = _root("OCTREG_DATA_ROOT", PROJECT_ROOT / "oct-mri-registration/data", "D:/Datasets/oct-mri-registration")
BENCH_RUNS = PROJECT_ROOT / "bench_runs"
DANDI = DATA_ROOT / "costantini/dandi-000026"
OCT_I58 = DATA_ROOT / "xiangrui/OCT_to_MRI/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz"
MRI_I58 = DATA_ROOT / "xiangrui/OCT_to_MRI/I58_brainstem_MRI_cropped_to_OCT.nii.gz"
