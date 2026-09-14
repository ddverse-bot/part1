# Benchmark: Xiangrui's I58 brainstem pair

octreg 1.0 registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash 8f731954d824ccac). The pair has no labels, so every number here is label-free. The reference R5 is the pose of the earlier research pipeline (v1.1) converted into the header frames. It is a body-level pose, not ground truth. The numbers are read from the run's result.json and eval.json and from ablations.json (copies in docs/results/xiangrui_I58/). Commands: `bash bench/run_xiangrui.sh` (see bench/README.md).

## Visual result

The result is judged by visual inspection of the overlays (docs/METHOD.md, "Evaluation"). The numbers below support that judgement.

![Result and the best pose of the other handedness](figures/fig_handedness_xiangrui.png)

Planes through the centroid of the specimen mask, normal to each OCT array axis: the OCT, then for the result and for the best pose of the other handedness (ablation A8) the MRI through the transform (inverted inside its foreground, polarity -1) and a 2 mm checkerboard. The MRI outline of the result follows the OCT specimen in all three planes, and the cerebellar folia of the MRI lie on the folded folia pieces of the OCT, at the lower right of the axis-1 plane and at the upper right of the axis-2 plane. The mirrored pose fits the outline almost as well and even has the lower loss (L 0.7133 against 0.7296), but its folia lie at the upper right of the axis-1 plane and are missing from the upper right of the axis-2 plane.

![QC of the result](figures/fig_qc_xiangrui.png)

qc.png of the run, with the MRI foreground through the transform (red) and the OCT specimen mask (cyan) in the fourth column. qc_montage.png, with four planes per axis, is [fig_qc_montage_xiangrui.png](figures/fig_qc_montage_xiangrui.png). In every plane the two outlines agree apart from the torn and folded cerebellar pieces, and the folia also correspond in the axis-0 planes at 17.47 and 23.17 mm and in the axis-1 plane at 24.07 mm.

## Main result

| | |
|---|---|
| pose vs R5, mean (max) over the v1.1 specimen mask | 13.84 (36.70) mm |
| pose vs R5 at the block corners, mean (max) | 30.36 (45.02) mm |
| rotation vs R5 | n/a deg |
| raw-data frame check (Spearman) | 0.991 at the pose; axis flips <= 0.270; 2 mm shifts <= 0.480; pass |
| score S = (2 polarity S_class + S_outline) / 3; S_class, S_outline; polarity | 0.2747; -0.1175, 0.5891; -1 |
| search top-1 / top-2 | 0.2741 / 0.2721 |
| scale per OCT array axis | 1.007 / 0.971 / 0.970 |
| flags | none |
| pose vs the previous run, mean / corners mean / corners max | 12.20 / 30.01 / 46.01 mm |
| boundary agreement in result.json (median, mm) | 1.30 |
| rim boundary agreement forward / reverse, method masks | 1.26 / 1.70 mm; R5 3.04 / 3.53; previous 1.58 / 1.89 |
| the same with the v1.1 masks | 1.80 / 1.38 mm; R5 2.02 / 1.05; previous 1.83 / 1.14 |
| OCT specimen mask | 19.24 cm3, Dice 0.919 against the v1.1 mask (18.05 cm3) |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 9.6 min, 5.2 GB, 0.95 GB (wall clock 9.6 min, nvidia-smi peak 1.8 GB) |
| time per step (s) | mri 4, oct fine grid 194, oct mask 173, two class 2, search 30, refinement 133, outputs 36 |

The pose is not within the spec's 1 mm tolerance for R5 at the block corners, and the raw-data frame check passes. The reading at the end discusses this.

R5 in the header frames (T_R5 @ A_spr @ inv(A_hdr)) agrees with the stored header-frame export to 0.000000 mm.

## Ablations

Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). Pose change is against base (the method through the same driver), as the mean over the v1.1 specimen-mask points and the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every variant, so it reflects the pose only.

| variant | change | pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | the method | 0.00 / 0.00 / 0.00 | 45.02 | 0.2747 | 0.7296 | -1 | 1.007 / 0.970 / 0.970 | 1.26 / 1.70 | 19.24 (0.919) | 164 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 19.89 / 42.10 / 62.89 | 46.68 | 0.0749 | 0.9257 | -1 | 0.995 / 0.992 / 0.989 | 2.00 / 2.29 | 29.52 (0.759) | 187 |
| A0b | v1.1 rim-watershed specimen mask given as the OCT mask | 0.21 / 0.32 / 0.54 | 45.01 | 0.2536 | 0.7519 | -1 | 1.007 / 0.967 / 0.966 | 1.23 / 1.68 | 18.04 (1.000) | 163 |
| A0c | texture specimen mask with holes filled in 3-D only (method: in every array plane) | 0.77 / 1.69 / 2.17 | 45.66 | 0.2165 | 0.7902 | -1 | 1.004 / 0.962 / 0.964 | 1.18 / 1.64 | 16.88 (0.910) | 162 |
| A1 | MRI flattening off | 0.08 / 0.12 / 0.17 | 45.11 | 0.2771 | 0.7274 | -1 | 1.009 / 0.970 / 0.969 | 1.25 / 1.68 | 19.24 (0.919) | 165 |
| A2 | OCT flattening off | 0.06 / 0.10 / 0.15 | 44.94 | 0.2702 | 0.7341 | -1 | 1.004 / 0.971 / 0.969 | 1.25 / 1.69 | 19.24 (0.919) | 167 |
| A4 | standardised intensity channels (z, -z) instead of two-class maps | 0.51 / 0.74 / 1.14 | 45.24 | 0.2939 | 0.7090 | -1 | 1.012 / 0.978 / 0.977 | 1.25 / 1.74 | 19.24 (0.919) | 164 |
| A5+1 | polarity forced +1 | 20.60 / 47.66 / 56.56 | 58.77 | 0.2345 | 0.7677 | 1 | 0.989 / 0.984 / 0.977 | 1.50 / 2.05 | 19.24 (0.919) | 166 |
| A5-1 | polarity forced -1 | 0.00 / 0.00 / 0.00 | 45.02 | 0.2747 | 0.7296 | -1 | 1.007 / 0.970 / 0.970 | 1.26 / 1.70 | 19.24 (0.919) | 165 |
| A6 | no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15) | 18.27 / 44.49 / 54.28 | 63.48 | 0.3610 | 0.6390 | -1 | 0.913 / 0.825 / 0.428 | 1.37 / 1.13 | 19.24 (0.919) | 164 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 15.47 / 29.84 / 43.69 | 58.69 | 0.2955 | 0.7133 | -1 | 0.963 / 0.952 / 0.980 | 1.46 / 1.67 | 19.24 (0.919) | 162 |
| A9 | no outline term: the outline weight set to the specimen mask, so S_outline = 0 and S = 2 S_class / 3 | 3.51 / 6.62 / 9.07 | 47.24 | 0.0835 | 0.9181 | -1 | 1.022 / 0.993 / 0.992 | 1.44 / 1.77 | 19.24 (0.919) | 144 |

Driver check: base through bench/ablate.py lies 0.00 mm (corners max 0.00 mm) from the CLI run.

Deletion rule of the spec: a step goes when removing it moves the pose by at most 0.5 mm (mean over the specimen-mask points) and changes no other metric beyond noise. Removing MRI flattening (A1, 0.08 mm) or OCT flattening (A2, 0.06 mm) stays within 0.5 mm. Removing per-plane hole filling (A0c, 0.77 mm), the two-class maps (A4, 0.51 mm), the scale prior (A6, 18.27 mm) or the outline term (A9, 3.51 mm) moves the pose further.

### Removed steps

Copied from the first ablation run, whose base still had the section-stripe flat field and the ladder (and the earlier score, two-class maps under an overlap gate), so pose changes in these rows are against that base. Removing either step moved the pose by less than the 0.5 mm deletion threshold and both were deleted.

| variant | change | pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A3 | section-stripe flat field off | 0.07 / 0.15 / 0.24 | 16.23 | 0.1454 | 0.8576 | -1 | 1.000 / 0.968 / 0.984 | 1.88 / 1.24 | 16.88 (0.910) | 147 |
| A7 | affine directly at the finest level from every search pose, no ladder | 0.00 / 0.00 / 0.00 | 16.14 | 0.1486 | 0.8544 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 (0.910) | 121 |

The present base lies 12.24 mm (corners mean 30.09 mm, corners max 45.88 mm, rotation n/a deg) from the base of that run. Source: /root/autodl-tmp/bench_runs/xiangrui_I58/ablate/ablations.json.

## Runtime and memory of the ablation driver

| step | seconds | peak RAM of the process so far (GB) |
|---|---|---|
| OCT fine grid 728x1006x797 (streamed once) | 197 | 3.3 |
| prep intensity | 68 | 3.8 |
| prep texture | 158 | 5.0 |
| prep texture3d | 158 | 5.0 |
| prep v11mask | 88 | 5.5 |
| prep mri | 5 | 0.7 |
| all variants and preprocessing | 2662 | 5.5 |

Peak GPU memory allocated by torch over the variants: 1.2 GB. The table was assembled a second time from the cached variants (13 s) after the report format changed; the total above is that of the run that computed them.
<!-- reading: written by hand below this line; bench/report.py keeps it when it rewrites the file -->

## Reading

The outline term and the file-header handedness decide this pair. Without the outline term (A9) the pose moves by 6.6 mm and turns by 15 degrees. The mirrored OCT (A8) gives a pose 29.8 mm away with a lower loss, whose anatomy the overlays show on the wrong side, so handedness is taken from the headers and not from the score. Forcing the polarity to +1 (A5+1), dropping the scale prior (A6) or thresholding the OCT by intensity (A0, 29.5 cm3 against 19.2 cm3) turns the pose over, 42 to 48 mm away. Without the scale prior S rises to 0.3610 while the OCT is scaled to 0.43 of its length along one axis. All distances are block-corner means.

Holes filled in 3-D only (A0c) move the pose by 1.69 mm and lower S_outline from 0.5891 to 0.3986, since holes that reach a cut face then count as embedding. The watershed mask of the earlier pipeline (A0b) gives the same pose to 0.32 mm.

The two-class maps and the flattening change the pose only a little once the outline is in the score: 0.74 mm for standardised intensities (A4, 0.51 mm over the specimen points, just above the deletion threshold), and 0.12 and 0.10 mm for MRI and OCT flattening (A1, A2). Flattening stays because it keeps the pose stable when other parts change. In an ablation run of the same method without flattening (docs/results/xiangrui_I58/no_flattening/ablations.json), the base pose moved by only 0.24 mm, but A0b, A0c and A9 moved the pose by 42.3, 40.9 and 42.5 mm, against 0.32, 1.69 and 6.62 mm here.

R5, the pose of the earlier research pipeline, lies 30.4 mm away and has the other handedness in the header frames; it is not a criterion (spec deviation 8). Its outlines agree worse than the result's with the method masks (rim medians 3.04 / 3.53 mm against 1.26 / 1.70 mm). The "previous run" in the table is the first release run, with the two-class score alone, the overlap gate and mirrored orientations in the search; it lies 30.0 mm away and also has the other handedness.

The section-stripe flat field and the refinement ladder were removed after the first ablation run. Switching off the flat field moved that run's pose by 0.15 mm (A3), and one affine refinement from every search pose gave the ladder's pose to within 0.0015 mm (A7). The run needs 9 min 36 s and 5.2 GiB.
