# Benchmark: Xiangrui's I58 brainstem pair

octreg 1.0 registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash 3027a8e4b308fa91). The pair has no labels, so every number here is label-free. The reference is the v1.1 pose R5 converted into the header frames; it is a body-level pose (rim boundary agreement 2.02 / 1.05 mm, run-to-run spread about 1 mm at the block corners), not ground truth. Commands: `bash bench/run_xiangrui.sh` (see bench/README.md).

## Main result

| | |
|---|---|
| pose vs R5, mean (max) over the v1.1 specimen mask | 6.06 (12.60) mm |
| pose vs R5 at the block corners, mean (max) | 10.13 (16.14) mm |
| rotation vs R5 | 24.6 deg |
| raw-data frame check (Spearman) | 0.990 at the pose; axis flips <= 0.200; 2 mm shifts <= 0.413; pass |
| final score S, polarity | 0.1486, -1 |
| search top-1 / top-2 | 0.1659 / 0.1600 |
| scale per OCT array axis | 0.999 / 0.969 / 0.983 |
| overlap | 0.584 |
| flags | none |
| boundary agreement in result.json (median, mm) | 2.02 |
| boundary agreement, rim median forward / reverse | 1.88 / 1.23 mm |
| OCT specimen mask | 16.88 cm3, Dice 0.910 against the v1.1 mask (18.05 cm3) |
| registration time, peak RAM, peak GPU | 15.1 min, 8.3 GB, 1.6 GB |

The pose is not within the known run-to-run spread of R5 (about 1 mm at the block corners) and the raw-data frame check passes. By the spec this needs an explanation before release; the ablations below are the first place to look.

R5 in the header frames (T_R5 @ A_spr @ inv(A_hdr)) agrees with the stored header-frame export to 0.000000 mm.

## Ablations

Preprocessing once per OCT foreground source and destripe setting, then two-class maps, search and ladder per variant. Pose change is against base (the default method through the same driver) over the v1.1 specimen mask points; the boundary agreement uses the base masks for every variant, so it reflects the pose only.

| variant | change | pose change: mean / corners max (mm) | vs R5 corners max (mm) | S | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| base | the method | 0.00 / 0.00 | 16.14 | 0.1486 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 (0.910) | 153 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 21.08 / 62.79 | 61.82 | 0.1293 | -1 | 0.983 / 0.977 / 0.985 | 2.78 / 1.72 | 29.52 (0.759) | 150 |
| A0b | v1.1 rim-watershed specimen mask given as the OCT mask | 14.92 / 46.30 | 46.83 | 0.1650 | -1 | 0.987 / 0.987 / 0.975 | 2.32 / 1.56 | 18.04 (1.000) | 149 |
| A1 | MRI flattening off | 0.15 / 0.39 | 16.25 | 0.1513 | -1 | 0.998 / 0.967 / 0.983 | 1.89 / 1.23 | 16.88 (0.910) | 148 |
| A2 | OCT flattening off | 13.05 / 33.23 | 35.49 | 0.1570 | -1 | 0.996 / 0.983 / 0.988 | 2.04 / 1.32 | 16.88 (0.910) | 149 |
| A3 | section-stripe flat field off | 0.07 / 0.24 | 16.23 | 0.1454 | -1 | 1.000 / 0.968 / 0.984 | 1.88 / 1.24 | 16.88 (0.910) | 147 |
| A4 | intensity channels instead of two-class maps | 13.54 / 36.66 | 31.12 | 0.1800 | -1 | 0.998 / 0.979 / 0.982 | 2.78 / 1.79 | 16.88 (0.910) | 146 |
| A5+1 | polarity forced +1 | 11.08 / 19.13 | 19.07 | 0.1113 | 1 | 0.972 / 0.986 / 0.984 | 1.83 / 1.38 | 16.88 (0.910) | 149 |
| A5-1 | polarity forced -1 | 0.00 / 0.00 | 16.14 | 0.1486 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 (0.910) | 150 |
| A6 | no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15) | 6.12 / 16.30 | 17.54 | 0.2492 | -1 | 0.828 / 0.762 / 0.368 | 2.63 / 1.28 | 16.88 (0.910) | 150 |
| A7 | affine directly at the finest level from every search pose, no ladder | 0.00 / 0.00 | 16.14 | 0.1486 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 (0.910) | 121 |

Driver check: base through bench/ablate.py lies 0.00 mm (corners max 0.00 mm) from the CLI run.

Deletion rule (removal moves the pose by <= 0.5 mm and no other metric beyond noise): removing MRI flattening (A1, 0.15 mm), section-stripe flat field (A3, 0.07 mm), affine ladder (A7, 0.00 mm) stays within 0.5 mm, so these are deletion candidates once the other columns are checked. Removing OCT flattening (A2, 13.05 mm), scale prior (A6, 6.12 mm) moves the pose further, so these steps stay.

## Runtime and memory of the ablation driver

| step | seconds (destripe, pooling, writing) | foreground mask (s) | peak RAM of the process so far (GB) |
|---|---|---|---|
| prep intensity | 370 | 69 | 9.1 |
| prep texture | 365 | 154 | 9.1 |
| prep texture_nodestripe | 5 | 154 | 9.1 |
| prep v11mask | 365 | 84 | 9.1 |
| prep mri | 4 | n/a | 0.7 |
| OCT fine grid 728x1006x797 (streamed once) | 199 | | 5.5 |
| all variants and preprocessing | 3242 | | 9.1 |

A foreground mask is computed once per source and shared by the keys that use it (texture and texture_nodestripe).

Peak GPU memory allocated by torch over the variants: 1.0 GB.

## Notes on the server run (written by hand, 2026-09-14)

### Why the pose is 10 mm from R5

R5 is not an optimum of the 1.0 objective, and the 1.0 pose fits the outlines better than R5 does. All numbers below use the same masks (prep/texture) and the same objective as the main run.

| pose | S at 0.15 mm | overlap | rim fwd / rev (mm) | ladder started at this pose moves |
|---|---|---|---|---|
| main (1.0) | 0.1486 | 0.584 | 1.88 / 1.23 | 0 mm |
| R5 (v1.1) | 0.1309 | 0.467 | 2.80 / 1.37 | 5.7 mm, to S 0.139 |
| v1.1 mirror candidate | 0.1291 | 0.570 | 2.06 / 1.41 | 0.5 mm |

The main pose moves the block centre by 4.7 mm (4.3 mm along MRI y) and rotates it by 24.6 deg relative to R5. The v1.1 probe P8 had already found that a 4-7 mm shift along OCT y was the only rigid move that reduced R5's outline disagreement. R5 came from other masks and another objective, and it sat on the v1 overlap gate (search candidate overlap 0.5905 against a gate of 0.590, top-1 / top-2 0.1379 / 0.1372). The 1 mm spread quoted above was seed to seed within that pipeline, so it does not bound a change of masks or objective.

The pose is fragile. With the v1.1 watershed mask (A0b, Dice 0.91 against ours) the method lands 15 mm (46 mm at the corners) away, with the other handedness, at overlap 0.474 just above its tau of 0.461. The v1.1 mirror candidate, 30 mm from R5, is also a local optimum of this objective.

### Code fixes made during the run

1. MRI foreground. On the 0.15 mm grid the box-averaged background peak is so narrow that the tissue peak reaches only 4 % of its height (prominence 3.5 % < 5 %). The rule found no valley, flagged `foreground_no_valley:mri` and took the whole crop (41.2 cm3). Peaks are now found on the square root of the smoothed counts. That gives 13.8 cm3 (valley ratio 0.07), against v1's 13.3 cm3 at 0.08 mm. The 0.08 mm result does not change.
2. OCT texture mask. With a two-component log-GMM on the unsmoothed block field, the second component caught only the bright rim tail (weights 0.89 / 0.11). The mask was 4.13 cm3 with 7985 components, and run 1 came out 14 mm (corners mean) from R5. log F is now smoothed on the block grid (new `texture_smooth_mm` = 1.2) and thresholded by Otsu.

   Variants scored on the pooled fields (volume at 0.16 mm, Dice against the v1.1 mask):
   - GMM without smoothing: 4.2 cm3, Dice 0.37.
   - GMM with any smoothing: 1.4-26.9 cm3.
   - Otsu with smoothing sigma 0.48 / 0.8 / 1.0 / 1.2 / 1.6 / 2.0 mm: 13.0 / 15.4 / 16.3 / 17.0 / 18.4 / 18.7 cm3, Dice 0.82 / 0.89 / 0.90 / 0.91 / 0.92 / 0.91.

   The final mask is 16.9 cm3 with Dice 0.910. The synthetic tests use 0.32 mm, because their blocks are only a few mm across.
3. Overlap gate. With the fixed masks, tau = 0.8 x 0.822 = 0.658, but the best overlap of any orientation and translation is 0.644. The search refused the pair (run 2: "no admissible pose"). `overlap_rho` is now 0.6 (tau 0.493).
4. Ladder admissibility. With a lower tau and an ungated ladder, the lowest-L pose slid to overlap 0.442 (S 0.164, 30 mm from R5, the other handedness, rim 2.29 / 1.56 mm). The ladder now drops poses whose overlap falls below the search tau after each level. With that rule the result was the same pose (the main pose) for tau 0.45, 0.49, 0.515, 0.55 and 0.575. Without it, tau up to 0.55 gave the low-overlap pose.

All four changes were chosen on this one pair. They need confirmation on the next dataset.
