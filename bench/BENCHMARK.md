# Benchmark: Xiangrui's I58 brainstem pair

octreg 1.0 registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash 8f731954d824ccac). The pair has no labels, so every number here is label-free. The reference R5 is the pose of the earlier research pipeline (v1.1) converted into the header frames. It is a body-level pose, not ground truth. The numbers are read from the run's result.json and eval.json and from ablations.json (copies in bench/results/xiangrui_I58/). Commands: `bash bench/run_xiangrui.sh` (see bench/README.md).

## Visual result

The result is judged by visual inspection of the overlays (docs/METHOD.md, "Evaluation"). The numbers below support that judgement.

![Result and the best pose of the other handedness](figures/fig_handedness_xiangrui.png)

Planes through the centroid of the specimen mask, normal to each OCT array axis: the OCT, then for the result and for the best pose of the other handedness (ablation A8) the MRI through the transform (inverted inside its foreground, polarity -1) and a 2 mm checkerboard. The MRI outline of the result follows the OCT specimen in all three planes, and the cerebellar folia of the MRI lie on the folded folia pieces of the OCT, at the lower right of the axis-1 plane and at the upper right of the axis-2 plane. The mirrored pose fits the outline better, with the higher S_outline (0.6867 against 0.6277) and the lower loss (L 0.7035 against 0.7192), but its folia lie at the upper right of the axis-1 plane and are missing from the upper right of the axis-2 plane.

![QC of the result](figures/fig_qc_xiangrui.png)

qc.png of the run, with the MRI foreground through the transform (red) and the OCT specimen mask (cyan) in the fourth column. qc_montage.png, with four planes per axis, is [fig_qc_montage_xiangrui.png](figures/fig_qc_montage_xiangrui.png). In every plane the MRI outline follows the OCT specimen apart from the torn and folded cerebellar pieces, although the OCT mask outline takes in a margin of agarose in several planes (most in axis 1 at 8.32 mm and axis 2 at 6.37 mm). The folia also correspond in the axis-0 planes at 17.47 and 23.17 mm and in the axis-1 plane at 24.07 mm.

## Main result

| | |
|---|---|
| pose vs R5, mean (max) over the v1.1 specimen mask | 13.78 (36.47) mm |
| pose vs R5 at the block corners, mean (max) | 30.20 (44.77) mm |
| rotation vs R5 | n/a deg |
| raw-data frame check (Spearman) | 0.990 at the pose; axis flips <= 0.266; 2 mm shifts <= 0.476; pass |
| score S = (2 polarity S_class + S_outline) / 3; S_class, S_outline; polarity | 0.2855; -0.1144, 0.6277; -1 |
| search top-1 / top-2 | 0.2855 / 0.2777 |
| scale per OCT array axis | 0.994 / 0.966 / 0.970 |
| flags | none |
| pose vs the previous run, mean / corners mean / corners max | 0.14 / 0.23 / 0.40 mm |
| rim boundary agreement forward / reverse, method masks | 1.22 / 1.66 mm; R5 3.04 / 3.53; previous 1.26 / 1.70 |
| the same with the v1.1 masks | 1.80 / 1.35 mm; R5 2.02 / 1.05; previous 1.80 / 1.38 |
| OCT specimen mask | 19.24 cm3, Dice 0.919 against the v1.1 mask (18.05 cm3) |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 10.5 min, 5.2 GiB, 1.27 GiB (wall clock 10.5 min, nvidia-smi peak 2.2 GiB) |
| time per step (s) | mri 4, oct fine grid 197, oct mask 167, two class 2, search 39, refinement 183, outputs 34 |

The raw-data frame check passes. R5 is a reference of the earlier pipeline, not a success criterion.

R5 in the header frames (T_R5 @ A_spr @ inv(A_hdr)) agrees with the stored header-frame export to 0.000000 mm.

## Ablations

Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). Pose change is against base (the method through the same driver), as the mean over the v1.1 specimen-mask points and the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every variant, so it reflects the pose only.

| variant | change | pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | the method | 0.00 / 0.00 / 0.00 | 44.77 | 0.2855 | 0.7192 | -1 | 0.994 / 0.966 / 0.970 | 1.22 / 1.66 | 19.24 (0.919) | 225 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 19.85 / 41.96 / 62.69 | 46.67 | 0.0752 | 0.9255 | -1 | 0.996 / 0.992 / 0.989 | 2.00 / 2.29 | 29.52 (0.759) | 249 |
| A0b | v1.1 rim-watershed specimen mask given as the OCT mask | 1.07 / 2.26 / 3.07 | 45.83 | 0.2949 | 0.7131 | -1 | 0.994 / 0.957 / 0.962 | 1.17 / 1.64 | 18.04 (1.000) | 223 |
| A0c | texture specimen mask with holes filled in 3-D only (method: in every array plane) | 1.01 / 2.18 / 2.87 | 45.64 | 0.2870 | 0.7202 | -1 | 0.995 / 0.958 / 0.965 | 1.19 / 1.65 | 16.88 (0.910) | 222 |
| A1 | MRI flattening off | 0.10 / 0.18 / 0.24 | 44.83 | 0.2878 | 0.7171 | -1 | 0.996 / 0.966 / 0.968 | 1.22 / 1.65 | 19.24 (0.919) | 224 |
| A2 | OCT flattening off | 1.08 / 2.41 / 3.02 | 45.47 | 0.2837 | 0.7232 | -1 | 0.990 / 0.959 / 0.965 | 1.19 / 1.63 | 19.24 (0.919) | 226 |
| A4 | standardised intensity channels (z, -z) instead of two-class maps | 0.57 / 0.79 / 1.29 | 45.19 | 0.3024 | 0.7007 | -1 | 1.007 / 0.974 / 0.977 | 1.22 / 1.71 | 19.24 (0.919) | 224 |
| A5+1 | polarity forced +1 | 19.57 / 41.12 / 43.92 | 51.66 | 0.2516 | 0.7560 | 1 | 0.953 / 0.970 / 0.982 | 1.96 / 2.68 | 19.24 (0.919) | 226 |
| A5-1 | polarity forced -1 | 0.00 / 0.00 / 0.00 | 44.77 | 0.2855 | 0.7192 | -1 | 0.994 / 0.966 / 0.970 | 1.22 / 1.66 | 19.24 (0.919) | 226 |
| A6 | no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15) | 16.02 / 38.73 / 48.32 | 52.16 | 0.5018 | 0.4982 | -1 | 0.768 / 0.606 / 0.456 | 2.73 / 2.26 | 19.24 (0.919) | 223 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 15.35 / 29.55 / 42.74 | 57.83 | 0.3128 | 0.7035 | -1 | 0.947 / 0.940 / 0.965 | 1.33 / 1.56 | 19.24 (0.919) | 222 |
| A9 | no outline term: S = 2 S_class / 3 in the search and the refinement | 19.55 / 41.69 / 62.52 | 47.42 | 0.0977 | 0.9032 | -1 | 0.990 / 0.991 / 0.987 | 1.91 / 2.19 | 19.24 (0.919) | 185 |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch | 0.14 / 0.23 / 0.40 | 45.02 | 0.2747 | 0.7296 | -1 | 1.007 / 0.970 / 0.970 | 1.26 / 1.70 | 19.24 (0.919) | 165 |
| A11 | simulated cut face: specimen mask removed beyond 70 % of its extent along OCT axis 1 (data kept as embedding) | 0.28 / 0.40 / 0.76 | 44.90 | 0.3488 | 0.6550 | -1 | 1.000 / 0.978 / 0.968 | 1.29 / 1.68 | 19.24 (0.919) | 216 |
| A11b | the same cut face with the two-sided outline | 2.47 / 4.69 / 6.55 | 46.87 | 0.3223 | 0.6804 | -1 | 1.016 / 1.009 / 0.980 | 1.36 / 1.70 | 19.24 (0.919) | 157 |

Driver check: base through bench/ablate.py lies 0.00 mm (corners max 0.00 mm) from the CLI run.

Deletion rule: a step goes when removing it moves the pose by at most 0.5 mm (mean over the specimen-mask points) and changes no other metric beyond noise. Removing MRI flattening (A1, 0.10 mm) stays within 0.5 mm. Removing per-plane hole filling (A0c, 1.01 mm), OCT flattening (A2, 1.08 mm), the two-class maps (A4, 0.57 mm), the scale prior (A6, 16.02 mm) or the outline term (A9, 19.55 mm) moves the pose further.

### Removed steps

Copied from the first ablation run, whose base still had the section-stripe flat field and the ladder (and the earlier score, two-class maps under an overlap gate), so pose changes in these rows are against that base. Removing either step moved the pose by less than the 0.5 mm deletion threshold and both were deleted.

| variant | change | pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A3 | section-stripe flat field off | 0.07 / 0.15 / 0.24 | 16.23 | 0.1454 | 0.8576 | -1 | 1.000 / 0.968 / 0.984 | 1.88 / 1.24 | 16.88 (0.910) | 147 |
| A7 | affine directly at the finest level from every search pose, no ladder | 0.00 / 0.00 / 0.00 | 16.14 | 0.1486 | 0.8544 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 (0.910) | 121 |

The present base lies 12.16 mm (corners mean 29.91 mm, corners max 45.62 mm, rotation n/a deg) from the base of that run. Source: /root/autodl-tmp/bench_runs/xiangrui_I58/ablate/ablations.json.

## Runtime and memory of the ablation driver

| step | seconds | peak RAM (GiB) |
|---|---|---|
| OCT fine grid 728x1006x797 (streamed once) | 197 | 3.3 |
| prep intensity | 68 | 3.8 |
| prep texture | 158 | 5.0 |
| prep texture3d | 158 | 5.0 |
| prep v11mask | 88 | 5.5 |
| prep mri | 5 | 0.7 |
| all variants and preprocessing | 3888 | 5.5 |

Peak GPU memory allocated by torch over the variants: 1.4 GiB.

<!-- reading: written by hand below this line; bench/report.py keeps it when it rewrites the file -->

## Reading

The outline term and the file-header handedness decide this pair. Without the outline term (A9, S = 2 S_class / 3 in the search and the refinement) the pose turns over by 164 degrees and lies 41.7 mm away. The mirrored OCT (A8) gives a pose 29.5 mm away with a lower loss, whose anatomy the overlays show on the wrong side, so handedness is taken from the headers and not from the score. Forcing the polarity to +1 (A5+1), dropping the scale prior (A6) or thresholding the OCT by intensity (A0, 29.5 cm3 against 19.2 cm3) also turns the pose over, 39 to 42 mm away. Without the scale prior S rises to 0.5018 while the OCT is scaled to 0.46 of its length along one axis. All distances are block-corner means.

The outline is one-sided for a roughly cropped MRI: agarose over MRI tissue is not a mismatch, since the MRI can hold tissue beyond the block. On the full pair the two-sided outline of the previous release (A10) gives the same pose to 0.23 mm; it is the previous run of the table. With a cut face simulated by removing the last 30 % of the specimen mask along OCT axis 1 and keeping the data there as embedding, the method moves 0.40 mm (A11) and the two-sided outline moves 4.69 mm (A11b). The same cut along OCT axis 0 or 2 moves the pose by 25 to 47 mm with either outline (separate runs, not in the table), so a truncation that large is not recovered on this pair by either crop model.

OCT flattening off (A2), the watershed mask of the earlier pipeline (A0b) and holes filled in 3-D only (A0c) end in nearly the same pose (0.35 to 0.69 mm apart), 5.2 to 5.7 degrees from the result (2.2 to 2.4 mm at the corners, 1.0 to 1.1 mm over the specimen). Scored with the method's maps and mask, the A2 pose has the loss 0.7198 against 0.7192; refined from the A2 and A0c poses the loss stops at 0.7197, while the A0b pose refines back to 0.7192. The A2 overlay (octreg qc --T) places the torn cerebellar pieces of the axis-1 plane at 32.02 mm worse than the result. The loss does not separate poses this close, so these three rows measure that precision as much as the value of the steps. With the two-sided outline they stayed within 0.10, 0.32 and 1.69 mm. Standardised intensities (A4) move the pose by 0.79 mm and MRI flattening off (A1) by 0.18 mm.

R5, the pose of the earlier research pipeline, lies 30.2 mm away and has the other handedness in the header frames; it is not a success criterion. Its outlines agree worse than the result's with the method masks (rim medians 3.04 / 3.53 mm against 1.22 / 1.66 mm).

The section-stripe flat field and the refinement ladder were removed after the first ablation run. Switching off the flat field moved that run's pose by 0.15 mm (A3), and one affine refinement from every search pose gave the ladder's pose to within 0.0015 mm (A7). The run needs 10 min 28 s and 5.2 GiB.
