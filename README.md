# octreg

octreg registers a serial-sectioning OCT block to an ex-vivo MRI with one affine transform, without labels or landmarks. It is
built for data like Xiangrui's I58 brainstem pair: an OCT volume of a tissue block embedded in scatterer-doped agarose, with
section stripes, tile seams and a contrast that may be inverted, and an MRI already cropped to a region containing the block.

## Method

1. **Specimen mask from isotropic texture.** Doped agarose is as bright as tissue, but its artefacts vary along a single array
   axis, while tissue texture varies along all three. The minimum over the axes of the local coefficient of variation is high
   only in tissue.
2. **One score for structure and outline.** Both volumes become the same soft map of bright against dark tissue, scored
   together with the specimen outline (OCT mask against MRI foreground). Swapping the two OCT classes negates the structure
   term and leaves the outline unchanged, so one correlation covers both contrasts and its sign is the polarity.
3. **Orientation search inside the crop, then a bounded affine fit.** 8,000 rotations, all translations at once by FFT, and a
   12-parameter affine fit of the 24 best poses under a scale and shear prior. Mirror images are not searched: the handedness
   is that of the file headers, because a nearly symmetric specimen can score better mirrored.

Details and parameters: [docs/METHOD.md](docs/METHOD.md).

## Install and use

```
pip install -e .
octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--device cuda|cpu] [--params F]
octreg qc --run OUT --oct OCT --mri MRI [--T F] [-o PREFIX]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse]
```

The OCT can be NIfTI, TIFF, OME-TIFF or NPY and is streamed plane by plane; the MRI is NIfTI. Both files must have the right
handedness (for TIFF and NPY set by the array axis order, x, y, z = numpy axes 2, 1, 0). A CUDA GPU is expected for real
volumes. `register` writes `T_oct2mri.txt` and `T_mri2oct.txt` (4×4, mm, between the worlds of the input files), the same
transform as `oct2mri.lta` (FreeSurfer) and `oct2mri_itk.txt` (ITK), `oct_in_mri.nii.gz`, `mri_in_oct.nii.gz`, the QC images
`qc.png` and `qc_montage.png`, and `result.json`. `qc --T` renders the same QC for another transform, so candidate poses can be
compared side by side. `apply` resamples a volume through the registration.

## Checking a result

A result is accepted by looking at `qc_montage.png` and the overlays, since label-free scores can prefer a wrong pose. In every
plane the MRI outline (red) should follow the OCT specimen, fibre tracts should continue across the checkerboard, cut faces and
folded pieces should correspond and lie on the same side, and the contrast should be consistently inverted or not.

## Result on Xiangrui's I58 pair

![Result and the best pose of the other handedness on I58](docs/figures/fig_handedness_xiangrui.png)

The MRI outline of the result follows the OCT specimen apart from the torn and folded cerebellar pieces, which have moved, and
the cerebellar folia of the MRI lie on the folded folia pieces of the OCT, at the lower right of the axis-1 plane and the upper
right of the axis-2 plane. The best mirrored pose fits the outline as well and has the lower loss, but its folia lie at the upper
right of the axis-1 plane and are missing from the upper right of the axis-2 plane. The montage of all planes is
[fig_qc_montage_xiangrui.png](docs/figures/fig_qc_montage_xiangrui.png). The run took 9 min 36 s on the original files
(OCT 1457×2013×1595 at 20 µm, MRI 343×489×495 at 0.08 mm) with 5.2 GiB of RAM. Removing the texture mask, the outline term,
the polarity rule or the scale prior, or mirroring the OCT, moves the pose by 30 to 48 mm.

## Limitations

- Affine only; torn or folded pieces that moved cannot be matched.
- Validated on one pair so far.
- The handedness is taken from the file headers and cannot be checked by the score.

## 中文摘要

octreg 把琼脂包埋的连续切片 OCT 组织块无标签地仿射配准到已裁剪的离体 MRI。三个创新点：用各向同性纹理分割标本；一个打分同时比较两类结构图和标本轮廓，结构项的正负号就是对比度极性；在 MRI 裁剪范围内做 FFT 朝向搜索（只搜旋转，手性以文件头为准），再做带尺度先验的仿射精配准。结果以看图为准。I58 上一条命令 9 分 36 秒跑完，MRI 轮廓贴合 OCT 标本（撕裂移位的小脑碎片除外），小脑叶片和核团落在对应位置。
