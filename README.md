# octreg

Label-free affine registration of a serial-sectioning OCT block to an ex-vivo MRI that is already cropped around the block.

The OCT is a scattering or intensity volume of a tissue block embedded in scatterer-doped agarose, with section stripes along one
array axis and a tile mosaic in the section plane. Its contrast may be inverted relative to the MRI. octreg finds the specimen
from its isotropic texture (the agarose has the same intensity but only anisotropic artefacts), turns both volumes into one soft
two-class map so that the contrast polarity comes out as the sign of the best FFT score, searches the block orientation inside
the MRI crop, and refines the pose with a rigid, similarity and affine ladder under a scale prior. No labels or landmarks are used.

## Install

```
pip install .
```

Python 3.9 or newer with torch, numpy, scipy, scikit-image, nibabel, tifffile and matplotlib. A CUDA GPU is expected for real
volumes; `--device cpu` works for small ones.

## Use

```
octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--device cuda|cpu] [--params F]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse]
```

The OCT can be NIfTI, or TIFF / OME-TIFF / NPY with its voxel spacing (from OME metadata or `--oct-spacing-um`, given along
the numpy axes). The MRI is NIfTI. The OCT is streamed plane by plane and never held whole in memory.

`OUT` gets the transform from the OCT file frame to the MRI file frame as `T_oct2mri.txt` (4x4, mm; `T_mri2oct.txt` is its
inverse), the same transform for FreeSurfer (`oct2mri.lta`) and ITK / ANTs (`oct2mri_itk.txt`), the two overlays
`oct_in_mri.nii.gz` and `mri_in_oct.nii.gz`, a `qc.png`, and `result.json` with the score, polarity, search scores, scale per OCT
axis, overlap, boundary agreement, flags, runtime and memory. The report is descriptive; it is not a pass/fail gate.

From Python: `from octreg.register import register, apply`. All method constants are in `octreg.params.Params`.

## Status

This is the 1.0 release branch. The method is specified in [docs/design/spec_v1.0_lean.md](docs/design/spec_v1.0_lean.md);
the validation on Xiangrui's I58 brainstem pair and the ablations are run with `bench/` (see [bench/README.md](bench/README.md)).
The previous research code (v1.1) is kept at the git tag `v1.1-archive`, and its documents are in [docs/archive/](docs/archive/).

Tests: `python -m pytest -q tests` (CPU, synthetic data, about a minute).
