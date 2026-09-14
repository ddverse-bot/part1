# octreg

octreg registers a serial-sectioning OCT block to an ex-vivo MRI with one affine transform, without labels or landmarks. It is
built for data like Xiangrui's I58 brainstem pair. The OCT is a scattering or intensity volume of a tissue block embedded in
scatterer-doped agarose, with section stripes, tile seams and a contrast that may be inverted relative to the MRI. The MRI has
already been cropped to a region containing the block. octreg writes the transform between the two input files, overlays in
both frames and QC images for visual inspection.

## Three innovations

**Specimen mask from isotropic texture.** Doped agarose scatters about as strongly as tissue, so an intensity threshold takes
in the embedding. Tissue texture varies along every direction, while the agarose is uniform apart from section stripes and tile
seams, each of which varies along a single array axis. octreg measures the local coefficient of variation along each array axis
and keeps the minimum, which is high only in tissue.

**One score for structure and outline, polarity from a sign.** Both volumes become the same soft map of bright against dark
tissue. The score is the mean correlation over three channel pairs: the two classes, with the OCT entering as (p, 1 − p) under
its specimen mask, and the specimen outline, the OCT mask against the MRI foreground. Swapping the two OCT classes negates the
class correlation exactly and leaves the outline unchanged, so one correlation covers both contrast polarities and the sign
tells whether the OCT contrast is inverted.

**Orientation search inside the crop, then a bounded affine fit.** The crop already fixes the position, so octreg searches
8,000 rotations and scores all translations inside the crop at once with FFTs. Mirror images are not searched: the handedness
comes from the file headers, because a nearly symmetric specimen can score better mirrored. The 24 best distinct poses are each
refined by a 12-parameter affine fit with a penalty and a hard bound on scale and shear, since both voxel sizes are known, and
the lowest loss wins.

The full description is in [docs/METHOD.md](docs/METHOD.md).

## Install

```
pip install -e .
```

Python 3.9 or newer with torch, numpy, scipy, scikit-image, nibabel, tifffile and matplotlib. A CUDA GPU is expected for real
volumes. `--device cpu` works for small ones.

## Usage

```
octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--device cuda|cpu] [--params F]
octreg qc --run OUT --oct OCT --mri MRI [--T F] [-o PREFIX] [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse]
```

`register` runs the method. The OCT can be NIfTI, or TIFF, OME-TIFF or NPY with its voxel spacing (from OME metadata or
`--oct-spacing-um`, along the numpy axes), and the MRI is NIfTI. The OCT is streamed plane by plane and never held whole in
memory. `--oct-mask` and `--mri-mask` replace the automatic masks, and `--params` reads `Params` overrides from a JSON file
(flagged `nondefault_params`). `qc` renders the QC images again without registering, for the run's transform or for another
transform between the same files given with `--T` (a 4×4 matrix in mm, as text in the format of `T_oct2mri.txt` or as .npy).
With `--T` the images are named after the transform file, so several candidate poses can be compared in one run directory.
`apply` resamples a volume in the OCT frame onto a grid in the MRI frame, or the reverse with `--inverse`.

From Python: `from octreg.register import register, apply, qc`. The method constants are in `octreg.params.Params`.

Both headers must have the right handedness. If the OCT stack is mirrored, for example with its section order reversed, fix the
header first.

## Outputs

| file | content |
|---|---|
| `T_oct2mri.txt`, `T_mri2oct.txt` | 4×4 affine in mm from the OCT file world to the MRI file world, and its inverse |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same transform for FreeSurfer and for ITK / ANTs |
| `oct_in_mri.nii.gz` | the OCT resampled onto the MRI grid |
| `mri_in_oct.nii.gz` | the MRI resampled onto a 0.15 mm grid in the OCT world |
| `qc.png` | one plane per OCT axis through the specimen centre: OCT, MRI through the transform, checkerboard, mask outlines |
| `qc_montage.png` | the same for four planes per axis spread across the specimen |
| `result.json` | transform, score S with S_class and S_outline, polarity, scale per OCT axis, flags, masks, runtime and memory |

The worlds are the frames of the input files (NIfTI sform or qform, diag(spacing) for TIFF and NPY), so the transform applies
to the original files as they are.

## How to evaluate a result

Visual inspection is the evaluation. These pairs have no labels, and the label-free numbers can prefer a wrong pose, so a
registration is accepted when its overlays show the same anatomy in both scans. Open `qc_montage.png`, then load the overlays in
freeview (`freeview MRI OUT/oct_in_mri.nii.gz`, or `freeview OCT OUT/mri_in_oct.nii.gz`) and check four things.

1. In every plane of all three axes, not only through the centre, the OCT specimen lies on the same anatomy in the MRI and the
   two outlines largely agree. The last column draws the MRI foreground in red and the OCT specimen mask in cyan.
2. Internal structures, such as fibre tracts, continue across the checkerboard squares.
3. Cut faces and detached or folded pieces of tissue correspond, and lie on the same side in both scans.
4. The contrast is consistently inverted, or consistently not, over the whole specimen. When the reported polarity is −1 the
   QC images show the MRI inverted inside its foreground, so tissue should then look alike in both scans.

When more than one pose is plausible, for example the result and an earlier transform, render each with `octreg qc --T` and
compare them side by side.

The numbers in `result.json` support this judgement. S_outline tells how well the specimen outlines agree, the scales per OCT
axis should stay near 1, and any flag deserves a look. They point to failures but do not certify a pose. On I58, switching off
the scale prior raises S from 0.2747 to 0.3610 with a distorted block, and the mirrored pose scores better than the result
while its anatomy is on the wrong side.

## Result on Xiangrui's I58 brainstem pair

![Result and the best pose of the other handedness on I58](docs/figures/fig_handedness_xiangrui.png)

One plane per OCT axis through the specimen centre: the OCT, then for the octreg result and for the best pose of the other
handedness, the MRI through the transform (inverted inside its foreground, polarity −1) and a 2 mm checkerboard.

In all three planes the MRI outline of the result follows the OCT specimen, and the cerebellar folia of the MRI lie on the
folded folia pieces of the OCT: at the lower right of the axis-1 plane and at the upper right of the axis-2 plane, where the
round nucleus at the top of the specimen also corresponds. The best pose of the other handedness fits the outline as well (S_outline
0.6438 against 0.5891), but its folia lie at the upper right of the axis-1 plane and are missing from the upper right of the
axis-2 plane. In the montage of the run the MRI outline follows the OCT specimen in every plane apart from the torn pieces,
although the OCT mask outline takes in a margin of agarose in some planes (most in axis 1 at 8.32 mm and axis 2 at 6.37 mm).
The folia also correspond in the axis-0 planes at 17.47 and 23.17 mm and in the axis-1 plane at 24.07 mm. The QC images of the run are
[fig_qc_xiangrui.png](docs/figures/fig_qc_xiangrui.png) and [fig_qc_montage_xiangrui.png](docs/figures/fig_qc_montage_xiangrui.png).

`octreg register` ran on the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) in 9 min 36 s
with 5.19 GiB of peak RAM and 1.8 GiB of GPU memory. S is 0.2747 (S_class −0.1175, S_outline 0.5891) with polarity −1
(inverted OCT contrast), the scales are 1.007 / 0.971 / 0.970, and there are no flags. Raw OCT values mapped through the file
header and the transform correlate with the exported overlay at Spearman 0.991, against at most 0.270 for any
flipped axis. Forcing the other polarity, dropping the scale prior or the outline term, thresholding the OCT by intensity or mirroring it moves the pose by 30 to 48 mm. Details and ablations are in [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Limitations

- Affine only. Torn or folded pieces that moved during embedding or sectioning cannot be matched.
- Demonstrated on one pair so far, and the texture smoothing (`texture_smooth_mm`) was set on that pair.
- The handedness is taken from the file headers and cannot be checked by the score.

## Layout

```
octreg/                      the package (method constants in params.py)
tests/                       synthetic CPU tests: python -m pytest -q tests
bench/                       I58 benchmark and ablations, run on the server (bench/README.md)
docs/METHOD.md               method, evaluation protocol and parameters
docs/BENCHMARK.md            I58 results and ablations
docs/figures/                figures
docs/results/xiangrui_I58/   result.json, eval.json and ablations.json of the I58 run
docs/design/                 specifications
docs/archive/                documents of the research code (git tag v1.1-archive)
```

## 中文摘要

octreg 把琼脂包埋的连续切片 OCT 组织块无标签地仿射配准到已裁剪到组织块附近的离体 MRI。三个创新点：用各向同性纹理分割标本（掺杂琼脂强度与组织相近，但它的伪影只沿单一轴变化）；一个打分同时比较两类结构图和标本轮廓，交换 OCT 两类恰好使结构相关变号而轮廓项不变，所以对比度极性就是结构相关的符号；在 MRI 裁剪范围内做 FFT 朝向搜索（只搜旋转，手性以文件头为准），再做带尺度先验的仿射精配准。评估以视觉检查为主：在 qc_montage.png 和 freeview 叠加图中检查标本轮廓、纤维束、切面和折叠碎片是否对应并位于同一侧，result.json 的数值只作辅助。I58 上从两份原始文件一条命令跑完，耗时 9 分 36 秒；三个方向的所有切面里，除撕裂、折叠后移位的小脑碎片外，MRI 标本轮廓都贴合 OCT 标本，小脑叶片碎片和圆形核团落在对应位置。另一手性的最佳位姿打分反而更高，但解剖结构位于错误的一侧，所以手性以文件头为准而不交给打分。
