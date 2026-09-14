# octreg

Label-free affine registration of serial-section OCT blocks to ex-vivo MRI.

octreg aligns an OCT volume of a tissue block embedded in scatterer-doped agarose to an MRI already cropped around the block.
It writes the transform between the two input files, each volume resampled into the other's frame, and QC images.

![Registration of Xiangrui's I58 brainstem pair](docs/figures/fig_registration_xiangrui.png)

*Xiangrui's I58 brainstem pair, one plane per OCT axis: OCT, MRI through the transform (contrast inverted), checkerboard of
the two, and the MRI foreground (red) and OCT mask (cyan) outlines.*

## Installation

```
pip install -e .
```

Requires Python 3.9 or newer and PyTorch; a CUDA GPU is recommended for full-size volumes.

## Usage

```
octreg register OCT MRI -o OUT
octreg qc --run OUT --oct OCT --mri MRI [--T F]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse]
```

`register` runs the registration. The OCT can be NIfTI, TIFF, OME-TIFF or NPY (`--oct-spacing-um Z,Y,X` when the file has no
spacing), the MRI is NIfTI, and both files must have the correct handedness (for TIFF and NPY, x, y, z = numpy axes 2, 1, 0).
`--oct-mask` and `--mri-mask` replace the automatic masks, `--params` reads parameter overrides from JSON, and `--device cpu`
runs without a GPU. `qc` renders the QC images again, for the run's transform or another one given with `--T`. `apply`
resamples an OCT-frame volume (MRI-frame with `--inverse`) onto the grid of `--reference`.

## Outputs

| file | content |
|---|---|
| `T_oct2mri.txt`, `T_mri2oct.txt` | 4×4 affine in mm between the worlds of the input files |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same transform for FreeSurfer and ITK |
| `oct_in_mri.nii.gz`, `mri_in_oct.nii.gz` | OCT on the MRI grid, MRI on a 0.15 mm grid in the OCT frame |
| `qc.png`, `qc_montage.png` | visual QC; in every plane the OCT should lie on the same MRI anatomy, with structures continuing across the checkerboard |
| `result.json` | scores, contrast polarity, scales, flags, runtime |

## Method

1. **Specimen mask from isotropic texture.** Agarose artefacts vary along one array axis, tissue texture along all three.
2. **One score for structure and outline.** Both scans become bright/dark tissue maps, compared together with the specimen
   outline; the sign of the structure term gives the contrast polarity.
3. **Orientation search in the crop and a bounded affine fit.** FFT search over 8,000 rotations, then a 12-parameter affine fit
   of the best poses under a scale and shear prior.

Details, evaluation and parameters: [docs/METHOD.md](docs/METHOD.md).
