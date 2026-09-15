# Benchmark: Xiangrui's I58 brainstem pair

octreg 1.0 registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash 7d01b8a167a83631). The pair has no labels, so every number here is label-free. The reference R5 is the pose of the earlier research pipeline (v1.1) converted into the header frames. It is a body-level pose, not ground truth. The numbers are read from the run's result.json and eval.json and from ablations.json (copies in bench/results/xiangrui_I58/). Commands: `bash bench/run_xiangrui.sh` (see bench/README.md).

## Visual result

The result is judged by visual inspection of the overlays (docs/METHOD.md, "Evaluation"). The numbers below support that judgement.

![Result and the best pose of the other handedness](figures/fig_handedness_xiangrui.png)

Planes through the centroid of the specimen mask, normal to each OCT array axis: the OCT, then for the result and for the best pose of the other handedness (ablation A8, also refined by §5) the MRI through the transform (inverted inside its foreground, polarity -1) and a 2 mm checkerboard. The MRI outline of the result follows the OCT specimen in all three planes, and the cerebellar folia of the MRI lie on the folded folia pieces of the OCT, at the lower right of the axis-1 plane and at the upper right of the axis-2 plane. The mirrored pose fits the outline better (S_outline 0.6867 against 0.6277) and has the lower §4 loss (L 0.7035 against 0.7192), but its fine structure matches less (F 0.078 against 0.103), and its folia lie at the upper right of the axis-1 plane and are missing from the upper right of the axis-2 plane.

![QC of the result](figures/fig_qc_xiangrui.png)

qc.png of the run, with the MRI foreground through the transform (red) and the OCT specimen mask (cyan) in the fourth column. qc_montage.png, with four planes per axis, is [fig_qc_montage_xiangrui.png](figures/fig_qc_montage_xiangrui.png). In every plane the MRI outline follows the OCT specimen apart from the torn and folded cerebellar pieces and debris; the OCT mask takes in a margin of agarose in several planes (most in axis 1 at 8.32 mm and axis 2 at 6.37 mm). The folia correspond in the axis-0 planes at 5.92, 17.47 and 23.17 mm, in the axis-1 planes at 24.07 and 32.02 mm and in the axis-2 planes at 12.67 and 19.12 mm, and the round nucleus at the top of the axis-2 planes lies on its MRI counterpart.

At the MRI voxel size (21 MRI planes with MRI, OCT and a colour fusion on one grid) vessels, fissures and fibre-bundle edges of the brainstem agree to a median of about 0.3 mm. Two parts misfit: the torn cerebellar pieces (1 to 2 mm) and the sections at one end of the block, below about 12 mm along MRI axis 1, which are 10 to 14 % larger in plane than the MRI and misfit by up to about 2 mm (1.8 mm at the scalloped lower edge of the axis-0 planes) while vessels a few millimetres further in match within 0.1 to 0.3 mm. In a blinded comparison of the §4 and §5 poses on 15 MRI planes (A and B in random order, internal structure and boundaries judged separately by language-model image readers) the §5 pose was preferred 20 times, the §4 pose 3 times, with 7 ties.

## Main result

| | |
|---|---|
| pose vs R5, mean (max) over the v1.1 specimen mask | 14.22 (37.75) mm |
| pose vs R5 at the block corners, mean (max) | 29.71 (45.13) mm |
| rotation vs R5 | n/a deg |
| raw-data frame check (Spearman) | 0.991 at the pose; axis flips <= 0.263; 2 mm shifts <= 0.473; pass |
| score S of the §4 pose = (2 polarity S_class + S_outline) / 3; S_class, S_outline; polarity | 0.2855; -0.1144, 0.6277; -1 |
| search top-1 / top-2 | 0.2855 / 0.2777 |
| fine-structure agreement F (§5), at the §4 pose -> refined; handedness | 0.0878 -> 0.1026; 1 |
| pose change of §5 (block-corner mean) | 2.36 mm |
| scale per OCT array axis | 0.977 / 0.974 / 0.985 |
| flags | none |
| pose vs the previous run, mean / corners mean / corners max | 1.49 / 2.36 / 3.63 mm |
| rim boundary agreement forward / reverse, method masks | 1.12 / 1.68 mm; R5 3.04 / 3.53; previous 1.22 / 1.66 |
| the same with the v1.1 masks | 1.70 / 1.36 mm; R5 2.02 / 1.05; previous 1.80 / 1.35 |
| OCT specimen mask | 19.24 cm3, Dice 0.919 against the v1.1 mask (18.05 cm3) |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 10.7 min, 5.3 GiB, 1.27 GiB (wall clock 10.7 min, nvidia-smi peak 2.6 GiB) |
| time per step (s) | mri 4, oct fine grid 194, oct mask 172, two class 2, search 40, refinement 183, fine-structure refinement 9, outputs 35 |

The raw-data frame check passes. R5 is a reference of the earlier pipeline, not a success criterion.

R5 in the header frames (T_R5 @ A_spr @ inv(A_hdr)) agrees with the stored header-frame export to 0.000000 mm.

## Ablations

Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). Pose change is against base (the method through the same driver), as the mean over the v1.1 specimen-mask points and the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every variant, so it reflects the pose only.

| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | the method | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 45.13 | 0.2855 | 0.7192 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 (0.919) | 235 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 19.85 / 41.96 | 23.36 / 42.92 / 66.12 | 46.89 | 0.0752 | 0.9255 | -1 | 0.994 / 0.994 / 1.000 | 2.61 / 2.11 | 29.52 (0.759) | 258 |
| A0b | v1.1 rim-watershed specimen mask given as the OCT mask | 1.07 / 2.26 | 0.13 / 0.24 / 0.41 | 44.97 | 0.2949 | 0.7131 | -1 | 0.973 / 0.975 / 0.988 | 1.10 / 1.65 | 18.04 (1.000) | 232 |
| A0c | texture specimen mask with holes filled in 3-D only (method: in every array plane) | 1.01 / 2.18 | 0.11 / 0.17 / 0.34 | 44.95 | 0.2870 | 0.7202 | -1 | 0.971 / 0.974 / 0.986 | 1.11 / 1.65 | 16.88 (0.910) | 231 |
| A1 | MRI flattening off | 0.10 / 0.18 | 0.00 / 0.00 / 0.00 | 45.13 | 0.2878 | 0.7171 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 (0.919) | 233 |
| A2 | OCT flattening off | 1.08 / 2.41 | 0.00 / 0.00 / 0.00 | 45.13 | 0.2837 | 0.7232 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 (0.919) | 234 |
| A4 | standardised intensity channels (z, -z) instead of two-class maps | 0.57 / 0.79 | 0.00 / 0.00 / 0.00 | 45.13 | 0.3024 | 0.7007 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 (0.919) | 233 |
| A5+1 | polarity forced +1 | 19.57 / 41.12 | 20.02 / 42.27 / 46.88 | 52.86 | 0.2516 | 0.7560 | 1 | 0.992 / 0.995 / 0.994 | 1.79 / 2.51 | 19.24 (0.919) | 236 |
| A5-1 | polarity forced -1 | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 45.13 | 0.2855 | 0.7192 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 (0.919) | 234 |
| A6 | no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15) | 16.02 / 38.73 | 15.09 / 35.82 / 46.29 | 51.54 | 0.5018 | 0.4982 | -1 | 0.558 / 0.515 / 0.552 | 4.46 / 2.21 | 19.24 (0.919) | 233 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 15.35 / 29.55 | 16.89 / 30.49 / 47.39 | 55.25 | 0.3128 | 0.7035 | -1 | 0.995 / 0.992 / 0.996 | 1.50 / 2.23 | 19.24 (0.919) | 231 |
| A9 | no outline term: S = 2 S_class / 3 in the search and the refinement | 19.55 / 41.69 | 19.76 / 40.91 / 63.67 | 47.32 | 0.0977 | 0.9032 | -1 | 0.990 / 0.997 / 0.994 | 2.28 / 2.64 | 19.24 (0.919) | 197 |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch | 0.14 / 0.23 | 0.00 / 0.00 / 0.00 | 45.13 | 0.2747 | 0.7296 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 (0.919) | 172 |
| A11 | simulated cut face: specimen mask removed beyond 70 % of its extent along OCT axis 1 (data kept as embedding) | 0.28 / 0.40 | 0.09 / 0.11 / 0.21 | 45.04 | 0.3488 | 0.6550 | -1 | 0.974 / 0.975 / 0.986 | 1.12 / 1.67 | 19.24 (0.919) | 226 |
| A11b | the same cut face with the two-sided outline | 2.47 / 4.69 | 2.38 / 3.87 / 5.87 | 46.54 | 0.3223 | 0.6804 | -1 | 0.986 / 0.988 / 0.990 | 1.18 / 1.68 | 19.24 (0.919) | 167 |
| A12 | no fine-structure refinement (§5): the pose of §4 | 0.00 / 0.00 | 1.49 / 2.36 / 3.63 | 44.77 | 0.2855 | 0.7192 | -1 | 0.994 / 0.966 / 0.970 | 1.22 / 1.66 | 19.24 (0.919) | 226 |

Driver check: base through bench/ablate.py lies 0.00 mm (corners max 0.00 mm) from the CLI run.

Deletion rule: a step goes when removing it moves the pose by at most 0.5 mm (mean over the specimen-mask points; the §4 pose for steps of §1-4, the final pose for §5) and changes no other metric beyond noise. Removing MRI flattening (A1, 0.10 mm) stays within 0.5 mm. Removing per-plane hole filling (A0c, 1.01 mm), OCT flattening (A2, 1.08 mm), the two-class maps (A4, 0.57 mm), the scale prior (A6, 16.02 mm), the outline term (A9, 19.55 mm) or the fine-structure refinement (A12, 1.49 mm) moves the pose further.

### Removed steps

Copied from the first ablation run, whose base still had the section-stripe flat field and the ladder (and the earlier score, two-class maps under an overlap gate), so pose changes in these rows are against that base. Removing either step moved the pose by less than the 0.5 mm deletion threshold and both were deleted.

| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | section-stripe flat field off | n/a / n/a | 0.07 / 0.15 / 0.24 | 16.23 | 0.1454 | 0.8576 | -1 | 1.000 / 0.968 / 0.984 | 1.88 / 1.24 | 16.88 (0.910) | 147 |
| A7 | affine directly at the finest level from every search pose, no ladder | n/a / n/a | 0.00 / 0.00 / 0.00 | 16.14 | 0.1486 | 0.8544 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 (0.910) | 121 |

The present base lies 12.15 mm (corners mean 29.38 mm, corners max 46.12 mm, rotation n/a deg) from the base of that run. Source: /root/autodl-tmp/bench_runs/xiangrui_I58/ablate/ablations.json.

## Runtime and memory of the ablation driver

| step | seconds | peak RAM (GiB) |
|---|---|---|
| OCT fine grid 728x1006x797 (streamed once) | 197 | 3.3 |
| prep intensity | 68 | 3.8 |
| prep texture | 158 | 5.0 |
| prep texture3d | 158 | 5.0 |
| prep v11mask | 88 | 5.5 |
| prep mri | 5 | 0.7 |
| all variants and preprocessing | 4252 | 5.5 |

Peak GPU memory allocated by torch over the variants: 1.4 GiB.

<!-- reading: written by hand below this line; bench/report.py keeps it when it rewrites the file -->

## Reading

Every variant except A12 ends with §5 on its own best pose; the table gives the change of the pose after §4 against the base's §4 pose and of the final pose against the base. All distances below are block-corner means.

A specimen mask rather than the intensity foreground, the outline term, the polarity rule, the scale prior and the handedness decide this pair. With the intensity-threshold OCT mask (A0, 29.5 cm3 against 19.2 cm3), without the outline term (A9), with the polarity forced to +1 (A5+1) or without the scale prior (A6) the block turns over, 36 to 43 mm from the result, and the mirrored OCT (A8) lies 30 mm away; §5, which only refines, does not bring any of them back. Without the scale prior S rises to 0.5018 (§4 pose) while the final OCT shrinks to 0.51-0.56 of its length along all three axes. The mirrored pose has the lower §4 loss but the lower F (0.078 against 0.103) and its anatomy on the wrong side.

§5 (A12) moves the result by 2.36 mm (1.49 mm over the specimen, 5.3 degrees). OCT flattening off (A2, 2.41 mm after §4), holes filled in 3-D only (A0c, 2.18), the rim-watershed mask of the earlier pipeline (A0b, 2.26), standardised intensities (A4, 0.79) and the two-sided outline (A10, 0.23) all change the §4 pose, and §5 settles them to 0.00-0.24 mm. MRI flattening off (A1) moves the §4 pose by 0.18 mm and the final pose by 0.00 mm, within the deletion rule; it is kept as an exception, so that both volumes pass through the same class map.

With a cut face simulated by removing the specimen mask beyond 70 % of its extent along OCT axis 1 and keeping the data there as embedding, the method moves 0.40 mm after §4 and 0.11 mm after §5 (A11), while the two-sided outline moves 4.69 mm and still 3.87 mm after §5 (A11b): §5 does not replace the one-sided outline for a roughly cropped MRI.

R5, the pose of the earlier research pipeline, lies 29.7 mm away and has the other handedness in the header frames; it is not a success criterion. The "previous run" in the table is the release run before §5; it lies 2.36 mm away. The section-stripe flat field and the refinement ladder were removed after the first ablation run: switching off the flat field moved that run's pose by 0.15 mm (A3), and one affine refinement from every search pose gave the ladder's pose to within 0.0015 mm (A7).
