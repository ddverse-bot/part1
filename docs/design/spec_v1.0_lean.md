# octreg 1.0 (lean) — specification

Supersedes the method parts of `spec_v1.0.md` after the user's direction of 2026-09-14: a few concise, intuitive, effective
innovations; no redundant content; the MRI is given cropped around the block (no whole-brain global search); handedness is not
a topic for now; publication grade; release goes to the GitHub repository `oct-mri-registration`.
`spec_v1.0.md` stays as the design record (audit, evidence, judges). Where the two disagree, this file wins.

## Scope and assumptions

- Input: one OCT block volume (NIfTI with header, or TIFF/OME-TIFF/NPY plus voxel spacing) and one MRI volume (NIfTI) that is
  already cropped to a region containing the block (a position prior, as delivered by the imaging side; any margin is fine).
- Output: one affine transform from the OCT file frame to the MRI file frame, overlays and a short label-free report.
- Affine only. No labels, no landmarks, no dataset switches.

## The method in five steps

1. Grids. Read both volumes in their header frames (array frame for TIFF/NPY). Resample to an isotropic pyramid
   0.6 / 0.3 / 0.15 mm (base h = max(0.15 mm, coarsest MRI voxel edge); box average then trilinear). The OCT is read plane by
   plane and never held twice in memory.
2. Foreground. The same histogram-valley threshold on both modalities (the v1 rule), closing + fill holes, components >= 1 %
   of the foreground kept. If no valley exists the run continues with the fallback threshold and records flag
   `foreground_no_valley` (no silent fallback). A user mask can replace it.
3. Two-class maps (innovation 1). Flatten each volume by its local foreground mean (Gaussian sigma 10 mm), blur sigma h,
   Otsu threshold t on foreground values, p = sigmoid((I - t) / (0.25 std)). OCT channels u = (p, 1 - p) with weight = OCT
   mask; MRI channels v = (p M, (1 - p) M). Candidate preprocessing, kept only if its ablation shows an effect: per-plane
   median normalisation of the OCT along the detected sectioning axis (adjacent-plane NCC rule).
4. Orientation search (innovations 2 and 3a) at 0.6 mm. N = 8000 uniformly distributed rotations (fixed seed), each also
   mirrored (a plain x2 in the search, no handedness analysis). For each orientation the weighted NCC over all translations
   inside the MRI crop is one FFT. Swapping the OCT classes gives exactly -S, so the contrast polarity is the sign of the best
   score: keep argmax |S| per orientation and record its sign. Admissible translations need overlap >= tau =
   0.8 min(1, V_MRI / V_OCT) (floor 0.15, flagged). Keep the top K = 24 poses after non-maximum suppression (3 mm / 10 deg).
5. Affine ladder (innovation 3b). Refine the candidates with the polarity fixed by its sign:
   0.6 mm rigid -> similarity (keep 8), 0.3 mm rigid -> affine (keep 3), 0.15 mm affine (keep 1), Adam with cosine schedule,
   loss L = 1 - S + lambda (sum log_scale^2 + sum shear^2), lambda = 2, |log_scale|, |shear| <= 0.15 (absolute).
   The pose with the lowest L wins.

Report (descriptive, never a pass/fail gate): final S and polarity, search top-1/top-2 scores, per-axis scale, overlap,
flags, boundary agreement of the two foreground outlines when the OCT block has a real outline, runtime and memory.

## Innovation claims and the evidence each must show (ablations)

| claim | ablation | must show |
|---|---|---|
| I1 two-class maps | intensity channels instead (A4); MRI flattening off (A1); OCT flattening off (A2) | two-class needed; each flattening kept only if it matters |
| I2 polarity as sign | fixed polarity = +1 and = -1 (A5) | sign rule picks the better of the two in one pass |
| I3 search + prior ladder | no scale prior (A6); direct affine without ladder (A7) | prior prevents degenerate scales; ladder needed |
| sectioning normalisation | off (A3) | deleted if no effect on I46, I55 and the brainstem |

A step whose removal changes the pose by <= 0.5 mm and no metric beyond noise is deleted before release.

## Validation (current scope: Xiangrui's pair only)

User direction 2026-09-14: "要在我们服务器上Xiangrui原本给的数据上去跑成功，先不用跑别的数据". Other datasets come later.

- Pair: Xiangrui's I58 brainstem, the two original files on the server
  (`data/xiangrui/OCT_to_MRI/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz`, 1457x2013x1595 @ 20 um, and
  `I58_brainstem_MRI_cropped_to_OCT.nii.gz`, 343x489x495 @ 0.08 mm, a FreeSurfer crop of a whole-brain scan, 27x39x40 mm,
  i.e. already cropped to the block, so translation needs no global search; only the block orientation is searched).
- Success: `octreg register` runs end to end from these two files, the pose agrees with the reference R5
  (`work/runs/v11/xiangrui_I58bs/T_oct2mri.npy`, converted from the v1 pipeline frame to the header frame) to within the known
  run-to-run spread (about 1 mm at the block corners) or the difference is explained, and the exported transform passes the raw-data
  frame check (raw 20 um values through the header affine correlate with the exported overlay; axis-flip controls do not).
- Ablations A1-A7 on this pair decide which steps are kept (A8 does not apply: the crop is given). Label-free metrics only:
  pose distance to R5, boundary agreement of the foreground outlines, final score, runtime and memory.

## Code (target <= 1,000 package lines)

```
octreg/  params.py  io.py  geometry.py  preprocess.py  search.py  refine.py  register.py  cli.py  __init__.py  __main__.py
tests/   test_frames.py  test_preprocess.py  test_polarity_sign.py  test_search_synthetic.py  test_end_to_end.py   (CPU, < 3 min)
bench/   run_xiangrui.sh  evaluate.py  ablate.py  report.py   (Xiangrui pair only for now)
docs/    METHOD.md  BENCHMARK.md
README.md  pyproject.toml
```

CLI: `octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--device cuda|cpu]` and
`octreg apply --run OUT --moving X --reference Y -o Z [--inverse]`.
Outputs: `T_oct2mri.txt`, `T_mri2oct.txt`, `oct2mri.lta`, `oct2mri_itk.txt`, `oct_in_mri.nii.gz`, `mri_in_oct.nii.gz`,
`qc.png`, `result.json`.

## Removed (reachable at tag v1.1-archive, listed with numbers in docs/METHOD.md "What we tried")

Vascular channel (helps I46 only), fine stage (rejected on the brainstem, harmful on cortex), texture specimen mask, parser,
MIND, non-rigid, MI / surface objectives, restarts and MI landscapes as QC, handedness decisions and jackknife, compat
switches, whole-brain search as a claim.
