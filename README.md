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

**Two-class maps, polarity from the sign of one score.** Both volumes become the same soft map of bright against dark tissue.
The OCT enters the score as the channel pair (p, 1 − p) with its mask as a weight. Swapping the two OCT classes negates the
weighted normalised cross-correlation exactly, so one correlation covers both contrast polarities. Its magnitude ranks the pose
and its sign tells whether the OCT contrast is inverted.

**Orientation search inside the crop, then a bounded affine fit.** The crop already fixes the position, so octreg searches
16,000 orientations (8,000 rotations and their mirror images) and scores all translations inside the crop at once with FFTs. The
24 best distinct poses are each refined by a 12-parameter affine fit with a penalty and a hard bound on scale and shear, since
both voxel sizes are known. A refined pose is kept only if the OCT specimen still lies on the MRI foreground as much as the
search required, and the lowest loss wins.

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

## Outputs

| file | content |
|---|---|
| `T_oct2mri.txt`, `T_mri2oct.txt` | 4×4 affine in mm from the OCT file world to the MRI file world, and its inverse |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same transform for FreeSurfer and for ITK / ANTs |
| `oct_in_mri.nii.gz` | the OCT resampled onto the MRI grid |
| `mri_in_oct.nii.gz` | the MRI resampled onto a 0.15 mm grid in the OCT world |
| `qc.png` | one plane per OCT axis through the specimen centre: OCT, MRI through the transform, checkerboard, mask outlines |
| `qc_montage.png` | the same for four planes per axis spread across the specimen |
| `result.json` | transform, score S, polarity, scale per OCT axis, overlap, flags, masks, runtime and memory |

The worlds are the frames of the input files (NIfTI sform or qform, diag(spacing) for TIFF and NPY), so the transform applies
to the original files as they are.

## How to evaluate a result

Visual inspection is the evaluation. These pairs have no labels, and the label-free numbers can prefer a wrong pose, so a
registration is accepted when its overlays show the same anatomy in both scans. Open `qc_montage.png`, then load the overlays in
freeview (`freeview MRI OUT/oct_in_mri.nii.gz`, or `freeview OCT OUT/mri_in_oct.nii.gz`) and check four things.

1. In every plane of all three axes, not only through the centre, the OCT specimen lies on the same anatomy in the MRI and the
   two outlines largely agree. The last column draws the MRI foreground in red and the OCT specimen mask in cyan.
2. Internal structures, such as fibre tracts, continue across the checkerboard squares.
3. Cut faces and detached or folded pieces of tissue correspond.
4. The contrast is consistently inverted, or consistently not, over the whole specimen. When the reported polarity is −1 the
   QC images show the MRI inverted inside its foreground, so tissue should then look alike in both scans.

When more than one pose is plausible, for example the result and an earlier transform, render each with `octreg qc --T` and
compare them side by side.

The numbers in `result.json` support this judgement. S is the weighted correlation of the two-class maps, the scales per OCT axis
should stay near 1, the overlap should be clearly above the gate `search.tau`, and any flag deserves a look. They point to
failures but do not certify a pose. On I58, switching off the scale prior raises S from 0.1455 to 0.2421 while squeezing the OCT
to 0.55 and 0.42 of its length along two axes.

## Result on Xiangrui's I58 brainstem pair

![Final pose and R5 on I58](docs/figures/fig_visual_final_vs_R5.png)

One plane per OCT axis through the specimen centre: the OCT, then for the octreg pose and for R5, the pose of our earlier
research pipeline, the MRI through the transform (inverted inside its foreground, polarity −1) and a 2 mm checkerboard.

In all three planes the final pose puts the MRI over the whole OCT specimen, and the fibre striations, the notch on the right
side and the folded piece correspond. R5 leaves a large part of the OCT uncovered in the axis-0 plane, puts cerebellar folia
inside the OCT body in the axis-1 plane and covers only part of the specimen in the axis-2 plane. The two poses lie 10.3 mm
apart at the block corners (mean) and 25.05° apart in rotation. The QC images of the run, with the mask outlines, are
[fig_qc_xiangrui.png](docs/figures/fig_qc_xiangrui.png) and
[fig_qc_montage_xiangrui.png](docs/figures/fig_qc_montage_xiangrui.png).

`octreg register` ran on the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) in 8 min 42 s
with 5.19 GiB of peak RAM and 1.6 GiB of GPU memory. S is 0.1455 with polarity −1 (inverted OCT contrast), the overlap 0.586,
the scales 1.000 / 0.967 / 0.985, and there are no flags. Raw OCT values mapped through the file header and the transform
correlate with the exported overlay at Spearman 0.990, against at most 0.205 for any flipped axis. Replacing the texture mask,
the two-class maps, the polarity rule or the scale prior, or switching off OCT flattening, moves the pose by 12 to 44 mm.
Details and ablations are in [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Limitations

- Affine only.
- Demonstrated on one pair so far.
- The overlap gate (`overlap_rho`) and the texture smoothing (`texture_smooth_mm`) were set on that pair.
- Handedness is not analysed. The search includes mirrored orientations and keeps the best pose.
- The texture mask can keep interior holes where the smoothed texture is weak. A few are visible in the I58 QC planes.

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

octreg 把琼脂包埋的连续切片 OCT 组织块无标签地仿射配准到已裁剪到组织块附近的离体 MRI。三个创新点：用各向同性纹理分割标本（掺杂琼脂强度与组织相近，但它的伪影只沿单一轴变化）；两类结构图，交换 OCT 两类恰好使加权 NCC 变号，所以对比度极性就是一次打分的符号；在 MRI 裁剪范围内做 FFT 朝向搜索，再做带尺度先验、受重叠门限约束的仿射精配准。评估以视觉检查为主：在 qc_montage.png 和 freeview 叠加图中检查标本轮廓、纤维束、切面和折叠碎片是否对应，result.json 的数值只作辅助。I58 上耗时 8 分 42 秒，最终位姿在三个平面上都让 MRI 覆盖整个 OCT 标本，纤维条纹、右侧缺口和折叠碎片都对得上；旧参考位姿 R5 在轴 0 平面留下大片 OCT 未覆盖，在轴 1 平面把小脑叶片放进 OCT 主体，在轴 2 平面只盖住部分标本。
