# octreg

Label-free affine registration of serial-section OCT blocks to ex-vivo MRI, with a small smooth deformation on top.

octreg aligns an OCT volume of a tissue block embedded in scatterer-doped agarose to an MRI roughly cropped around the block.
It writes the affine transform between the two input files, a smooth displacement field on top of it where the data support
one, each volume resampled into the other's frame, and QC images.

![Registration of the I58 brainstem pair](docs/figures/registration_I58.png)

*I58 brainstem pair, one plane per MRI array axis through the registered specimen, read from the original files: the MRI, the
OCT through the transform and the smooth field, and those two panels cut into 8 mm squares and interleaved. The checkerboard
is nothing but the two of them, on the same two grey scales, not inverted, not matched to each other and not masked, so every
square can be checked against the panel it came from and a structure runs on across a square edge where the two agree. What a
checkerboard settles, and what it does not, is measured in docs/METHOD.md; the panel that resolves a few millimetres is the
fourth column of the run's qc.png. Two places where the two volumes do not agree are disclosed there as well: the cerebellar
pieces that tore during sectioning, and the sections at one end of the block.*

## Installation

```
pip install -e .
```

Requires Python 3.9 or newer and PyTorch; a CUDA GPU is recommended for full-size volumes.

## Usage

```
octreg register OCT MRI -o OUT
octreg qc --run OUT --oct OCT --mri MRI [--T F]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--affine-only] [--oct-spacing-um Z,Y,X]
```

`register` runs the registration. The OCT can be NIfTI, TIFF, OME-TIFF or NPY (`--oct-spacing-um Z,Y,X` when the file has no
spacing), the MRI is NIfTI. When both files carry an orientation header (NIfTI sform or qform), it fixes the handedness and must
be correct; a TIFF or NPY stack (x, y, z = numpy axes 2, 1, 0) or a NIfTI file without one has none, so both handednesses are
tried and the one whose fine structure matches the MRI is kept (flag `mirrored_oct_frame` when it is the mirrored one).
`--oct-mask` and `--mri-mask` replace the automatic masks, `--params` reads parameter overrides from JSON, and `--device cpu`
runs without a GPU. `qc` renders the QC images of the affine again, for the run's transform or another one given with `--T`.
`apply` resamples an OCT-frame volume onto the grid of an MRI-frame `--reference`, through the affine and, when the run wrote
one, the deformation field (`--affine-only` leaves the field out). With `--inverse` it resamples an MRI-frame volume onto an
OCT-frame reference through the affine alone, since the field is not inverted. `--oct-spacing-um` gives the spacing of the
OCT-frame file when it is a TIFF or NPY that no longer sits where the run read it.

## Outputs

| file | content |
|---|---|
| `T_oct2mri.txt`, `T_mri2oct.txt` | 4×4 affine in mm between the worlds of the input files |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same transform for FreeSurfer and ITK |
| `oct2mri_warp.nii.gz` | the smooth deformation on top of the affine, only when one is applied: displacement u (mm, MRI world axes) on the 0.15 mm MRI grid as a NIfTI vector image. The registered OCT at the MRI point x is OCT(T⁻¹(x + u(x))) |
| `oct_in_mri.nii.gz` | OCT on the MRI grid through the affine and the deformation |
| `oct_in_mri_affine.nii.gz` | OCT on the MRI grid through the affine alone (the same image when no deformation is applied) |
| `mri_in_oct.nii.gz` | MRI on a 0.15 mm grid in the OCT frame, through the affine |
| `qc.png`, `qc_montage.png` | visual QC of the affine, four columns: OCT, MRI through the transform, a checkerboard of the two, and the OCT with the MRI foreground (red) and specimen mask (cyan) outlines. The outlines are what decides a pose; in every plane the MRI boundary should follow the edge of the OCT specimen and the OCT should lie on the same MRI anatomy |
| `qc_deform.png` | only with a deformation: MRI, OCT through the affine, OCT through affine and deformation, and the field magnitude, on three planes with the MRI outline |
| `result.json` | scores, fine-structure agreement, handedness, contrast polarity, scales, deformation read-outs, flags, runtime |

## Method

1. **Specimen mask from isotropic texture.** Agarose artefacts vary along one array axis, tissue texture along all three.
2. **One score for structure and outline.** Both scans become bright/dark tissue maps, compared together with the specimen
   outline (OCT tissue must lie on MRI tissue; the MRI may hold tissue beyond the block); the sign of the structure term gives
   the contrast polarity.
3. **Orientation search in the crop.** FFT search over 8,000 rotations and all translations inside the MRI crop.
4. **Prior-bounded affine refinement.** A 12-parameter affine fit of the best poses under a scale and shear prior.
5. **Fine-structure refinement.** The best pose is refined on the gradient orientations inside both scans (normalised gradient
   fields), which follow fibre bundles and vessels and need no intensity mapping.
6. **Smooth deformation.** A small displacement field on top of the affine, fitted to block matches of the interior structure
   and to the offset between the surface edges of the two scans along the MRI boundary normals. Its smoothness is chosen by
   held-out error under a strain limit, and without a held-out gain, or with no weight that keeps the limit, no field is
   applied. The affine stays the primary result.

The method assumes an OCT block embedded in scatterer-doped agarose and an MRI cropped around it. On the brainstem pair above
it places the block to about 0.3 mm over the specimen. On the cortex slabs of DANDI:000026, which are not embedded, the outline
carries almost nothing and the search finds the block on one of eight. Within that set the runs say so in their own scores, but
the scores are not a threshold that carries from one kind of specimen to another, and the overlay is what decides.

Details, evaluation and parameters: [docs/METHOD.md](docs/METHOD.md).
