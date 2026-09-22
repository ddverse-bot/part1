# Benchmark: Xiangrui's I58 brainstem pair

octreg registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash 892a1f3b4fd6f7ed). The pair has no labels, so every number here is label-free. The numbers are read from the run's result.json and eval.json and from ablations.json (copies in bench/results/xiangrui_I58/). Commands: `python bench/run_xiangrui.py` (see bench/README.md).

## Visual result

The result is judged by visual inspection of the overlays (docs/METHOD.md, "Evaluation"). The numbers below support that judgement.

![Result and the best pose of the other handedness](figures/fig_handedness_xiangrui.png)

Planes through the centroid of the specimen mask, normal to each OCT array axis: the OCT, then for the result and for the best pose of the other handedness (ablation A8, also refined by §5) the MRI through the transform (inverted inside its foreground, polarity -1) and a 2 mm checkerboard. The MRI outline of the result follows the OCT specimen in all three planes, and the cerebellar folia of the MRI lie on the folded folia pieces of the OCT, at the lower right of the axis-1 plane and at the upper right of the axis-2 plane. The mirrored pose fits the outline better (S_outline 0.6867 against 0.6277) and has the lower §4 loss (L 0.7035 against 0.7192), but its fine structure matches less (F 0.078 against 0.103), and its folia lie at the upper right of the axis-1 plane and are missing from the upper right of the axis-2 plane.

![QC of the result](figures/fig_qc_xiangrui.png)

qc.png of the run, with the MRI foreground through the transform (red) and the OCT specimen mask (cyan) in the fourth column. qc_montage.png, with four planes per axis, is [fig_qc_montage_xiangrui.png](figures/fig_qc_montage_xiangrui.png). In every plane the MRI outline follows the OCT specimen apart from the torn and folded cerebellar pieces and debris; the OCT mask takes in a margin of agarose in several planes (most in axis 1 at 8.32 mm and axis 2 at 6.37 mm). The folia correspond in the axis-0 planes at 5.92, 17.47 and 23.17 mm, in the axis-1 planes at 24.07 and 32.02 mm and in the axis-2 planes at 12.67 and 19.12 mm, and the round nucleus at the top of the axis-2 planes lies on its MRI counterpart.

![Smooth deformation](figures/fig_qc_deform_xiangrui.png)

qc_deform.png of the run: the MRI, the OCT through the affine, the OCT through the affine and the smooth field of §6, and the magnitude of that field, on three planes with the outline of the MRI foreground. The field is 0.22 mm in the median and 1.16 mm at most. It closes the one-sided gap between the OCT surface and the MRI boundary along one flank and leaves the torn cerebellum where the affine puts it, apart from one flap edge that it squeezes by about 10 % in two axial planes (docs/METHOD.md, "Two flaws were introduced").

At the MRI voxel size (21 MRI planes with MRI, OCT and a colour fusion on one grid) vessels, fissures and fibre-bundle edges of the brainstem agree to a median of about 0.3 mm. Two parts misfit: the torn cerebellar pieces (1 to 2 mm) and the sections at one end of the block, below about 12 mm along MRI axis 1, which are 10 to 14 % larger in plane than the MRI and misfit by up to about 2 mm (1.8 mm at the scalloped lower edge of the axis-0 planes) while vessels a few millimetres further in match within 0.1 to 0.3 mm. In a blinded comparison of the §4 and §5 poses on 15 MRI planes (A and B in random order, internal structure and boundaries judged separately by language-model image readers) the §5 pose was preferred 20 times, the §4 pose 3 times, with 7 ties.

## Main result

| | |
|---|---|
| raw-data frame check (Spearman) | 0.990 at the pose; axis flips <= 0.256; 2 mm shifts <= 0.472; pass |
| score S of the §4 pose = (2 polarity S_class + S_outline) / 3; S_class, S_outline; polarity | 0.2855; -0.1144, 0.6277; -1 |
| search top-1 / top-2 | 0.2855 / 0.2777 |
| fine-structure agreement F (§5), at the §4 pose -> refined; handedness | 0.0878 -> 0.1026; 1 |
| pose change of §5 (block-corner mean) | 2.36 mm |
| scale per OCT array axis | 0.977 / 0.974 / 0.985 |
| smooth deformation (§6): status; model; interior matches, supported boundary points used by the fit | applied; control lattice 5 mm, lambda 0.51; 502, 8402 |
| §6 held-out median error, interior + boundary: no deformation -> chosen lambda | 0.247 -> 0.155 mm + 0.249 -> 0.093 mm |
| §6 residuals affine -> deformed, measured again: interior matches; surface-edge offsets (within 0.3 mm) | 0.25 -> 0.12 mm; 0.29 -> 0.10 mm (51 -> 74 %) |
| §6 field over the MRI foreground, median / p95 / max; max strain | 0.22 / 0.59 / 1.16 mm; 0.15 |
| flags | none |
| pose vs the previous run, mean / corners mean / corners max (the mean is over the run's OCT specimen mask) | 0.00 / 0.00 / 0.00 mm |
| rim boundary agreement forward / reverse, method masks | 1.12 / 1.68 mm; previous 1.12 / 1.68 |
| OCT specimen mask | 19.24 cm3 |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 12.1 min, 5.2 GiB, 1.59 GiB |
| time per step (s) | mri 2, oct fine grid 153, oct mask 93, two class 2, search 35, refinement 349, fine-structure refinement 6, smooth deformation 23, outputs 62 |

The raw-data frame check passes. The pair has no labels, so none of these numbers is a success criterion.

## Ablations

Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). Pose change is against base (the method through the same driver), as the mean over the points of the base specimen mask and the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every variant, so it reflects the pose only. The smooth deformation (§6) leaves the pose alone, so no variant runs it: 'no §6' would be base with the identical pose, and its read-outs are the 'before' residuals in the Main result table above.

| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | the method | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2855 | 0.7192 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 419 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 19.98 / 41.96 | 23.89 / 42.92 / 66.13 | 0.0752 | 0.9255 | -1 | 0.994 / 0.994 / 1.000 | 2.61 / 2.11 | 29.52 | 407 |
| A0c | texture specimen mask with holes filled in 3-D only (method: in every array plane) | 0.99 / 2.19 | 0.11 / 0.17 / 0.34 | 0.2871 | 0.7202 | -1 | 0.971 / 0.974 / 0.986 | 1.11 / 1.65 | 16.88 | 399 |
| A1 | MRI flattening off | 0.10 / 0.18 | 0.00 / 0.00 / 0.00 | 0.2878 | 0.7171 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 396 |
| A2 | OCT flattening off | 1.07 / 2.41 | 0.00 / 0.00 / 0.00 | 0.2837 | 0.7232 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 404 |
| A4 | standardised intensity channels (z, -z) instead of two-class maps | 0.58 / 0.81 | 0.00 / 0.00 / 0.00 | 0.3024 | 0.7007 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 399 |
| A5+1 | polarity forced +1 | 19.77 / 41.13 | 20.15 / 42.27 / 46.87 | 0.2516 | 0.7560 | 1 | 0.992 / 0.995 / 0.994 | 1.79 / 2.51 | 19.24 | 405 |
| A5-1 | polarity forced -1 | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2855 | 0.7192 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 408 |
| A6 | no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15) | 15.87 / 38.73 | 14.81 / 35.71 / 46.35 | 0.5013 | 0.4987 | -1 | 0.556 / 0.507 / 0.567 | 4.42 / 2.29 | 19.24 | 463 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 15.70 / 29.55 | 17.25 / 30.42 / 47.82 | 0.3128 | 0.7035 | -1 | 0.996 / 0.993 / 0.994 | 1.51 / 2.23 | 19.24 | 465 |
| A9 | no outline term: S = 2 S_class / 3 in the search and the refinement | 19.65 / 41.69 | 19.87 / 40.92 / 63.67 | 0.0977 | 0.9032 | -1 | 0.990 / 0.997 / 0.994 | 2.28 / 2.64 | 19.24 | 308 |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch | 0.15 / 0.23 | 0.00 / 0.00 / 0.00 | 0.2747 | 0.7296 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 345 |
| A11 | simulated cut face: specimen mask removed beyond 70 % of its extent along OCT axis 1 (data kept as embedding) | 0.29 / 0.40 | 0.09 / 0.11 / 0.21 | 0.3488 | 0.6550 | -1 | 0.974 / 0.975 / 0.986 | 1.12 / 1.67 | 19.24 | 396 |
| A11b | the same cut face with the two-sided outline | 2.51 / 4.68 | 2.47 / 3.87 / 5.87 | 0.3223 | 0.6804 | -1 | 0.986 / 0.988 / 0.990 | 1.18 / 1.68 | 19.24 | 331 |
| A12 | no fine-structure refinement (§5): the pose of §4 | 0.00 / 0.00 | 1.50 / 2.36 / 3.63 | 0.2855 | 0.7192 | -1 | 0.994 / 0.966 / 0.970 | 1.22 / 1.66 | 19.24 | 395 |
| A6p | no penalty, clamp kept: lam 0, clamp 0.15 | 16.90 / 40.95 | 17.22 / 38.59 / 57.84 | 0.3381 | 0.6619 | -1 | 0.861 / 0.870 / 0.882 | 1.90 / 3.08 | 19.24 | 416 |
| A6c | no clamp, penalty kept: lam 2, clamp 1.0 | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2855 | 0.7192 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 399 |

Driver check: base through bench/ablate.py lies 0.00 mm (corners max 0.00 mm) from the CLI run.

Deletion rule: a step goes when removing it moves the pose by at most 0.5 mm (mean over the specimen-mask points; the §4 pose for steps of §1-4, the final pose for §5), changes no other metric beyond noise, and no test outside this pair shows it load bearing. Removing MRI flattening (A1, 0.10 mm), the scale clamp (A6c, 0.00 mm) or the one-sided outline (A10, 0.15 mm) stays within 0.5 mm, and the evidence for keeping each is in docs/METHOD.md after the ablation table. Removing per-plane hole filling (A0c, 0.99 mm), OCT flattening (A2, 1.07 mm), the two-class maps (A4, 0.58 mm), the scale prior (A6, 15.87 mm), the outline term (A9, 19.65 mm) or the fine-structure refinement (A12, 1.50 mm) moves the pose further.

### Removed steps

Copied from the first ablation run, whose base still had the section-stripe flat field and the ladder (and the earlier score, two-class maps under an overlap gate), so pose changes in these rows are against that base. Removing either step moved the pose by less than the 0.5 mm deletion threshold and both were deleted.

| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A3 | section-stripe flat field off | n/a / n/a | 0.07 / 0.15 / 0.24 | 0.1454 | 0.8576 | -1 | 1.000 / 0.968 / 0.984 | 1.88 / 1.24 | 16.88 | 147 |
| A7 | affine directly at the finest level from every search pose, no ladder | n/a / n/a | 0.00 / 0.00 / 0.00 | 0.1486 | 0.8544 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 | 121 |

The present base lies 12.21 mm (corners mean 29.38 mm, corners max 46.12 mm, rotation n/a deg) from the base of that run. Source: ablate/ablations.json.

## Runtime and memory of the ablation driver

| step | seconds | peak RAM (GiB) |
|---|---|---|
| OCT fine grid 728x1006x797 (streamed once) | 137 | 3.3 |
| prep intensity | 31 | 3.8 |
| prep texture | 87 | 5.0 |
| prep texture3d | 88 | 5.0 |
| prep mri | 2 | 0.7 |
| all variants and preprocessing | 6288 | 5.0 |

Peak GPU memory allocated by torch over the variants: 1.6 GiB.

<!-- reading: written by hand below this line; bench/report.py keeps it when it rewrites the file -->

## Reading

Every variant except A12 ends with §5 on its own best pose; the table gives the change of the pose after §4 against the base's §4 pose and of the final pose against the base. All distances below are block-corner means.

A specimen mask rather than the intensity foreground, the outline term, the polarity rule, the scale prior and the handedness decide this pair. With the intensity-threshold OCT mask (A0, 29.5 cm3 against 19.2 cm3), without the outline term (A9), with the polarity forced to +1 (A5+1) or without the scale prior (A6) the block turns over, 36 to 43 mm from the result, and the mirrored OCT (A8) lies 30 mm away; §5, which only refines, does not bring any of them back. Without the scale prior S rises to 0.5013 (§4 pose) while the final OCT shrinks to 0.51-0.57 of its length along all three axes. The mirrored pose has the lower §4 loss but the lower F (0.078 against 0.103) and its anatomy on the wrong side.

§5 (A12) moves the result by 2.36 mm (1.50 mm over the specimen, 5.3 degrees). OCT flattening off (A2, 2.41 mm after §4), holes filled in 3-D only (A0c, 2.19), standardised intensities (A4, 0.81) and the two-sided outline (A10, 0.23) all change the §4 pose, and §5 settles them to 0.00-0.17 mm. MRI flattening off (A1) moves the §4 pose by 0.18 mm and the final pose by 0.00 mm, within the deletion rule, and is kept on evidence from another specimen (docs/METHOD.md, after the ablation table).

With a cut face simulated by removing the specimen mask beyond 70 % of its extent along OCT axis 1 and keeping the data there as embedding, the method moves 0.40 mm after §4 and 0.11 mm after §5 (A11), while the two-sided outline moves 4.68 mm and still 3.87 mm after §5 (A11b): §5 does not replace the one-sided outline for a roughly cropped MRI.

The "previous run" in the table is release 1.0, and 1.1 gives the same affine: 0.003 mm at the block corners, which is float noise between two runs, and the same boundary agreement, since §1-5 did not change. A second §6 paragraph and the §6 ablations are in docs/METHOD.md. The section-stripe flat field and the refinement ladder were removed after the first ablation run: switching off the flat field moved that run's pose by 0.15 mm (A3), and one affine refinement from every search pose gave the ladder's pose to within 0.0015 mm (A7).
