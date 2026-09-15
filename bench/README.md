# bench

Benchmark of octreg 1.0 on Xiangrui's I58 brainstem pair, and a generality check on the DANDI:000026 blocks. It lives in the repository but is not part of the package, and the method never reads anything here. The I58 pair has
no labels, so its metrics are label-free. Paths are those of the AutoDL server.

| file | what it does |
|---|---|
| `run_xiangrui.sh` | runs the steps below from `/root/autodl-tmp/octreg1`, detached-friendly, logs next to the run dir |
| `evaluate.py` | pose against the reference R5, OCT mask volume and Dice, boundary agreement, raw-data frame check |
| `ablate.py` | preprocessing once, then the variants base and A0-A12, one JSON table |
| `dandi.py` | DANDI:000026 blocks: MRI crop around the Broca-area labels, a block position of an earlier registration or a coarse whole-hemisphere location; §1-5 for both handednesses; label checks (`bench/results/dandi/summary.json`) |
| `report.py` | writes `bench/BENCHMARK.md` and the two figures in `bench/figures/` from the outputs |
| `compose_pair.py` | two `octreg qc` outputs of one run side by side (`bench/figures/fig_handedness_xiangrui.png`) |

## Running

```bash
cd /root/autodl-tmp/octreg1
setsid nohup bash bench/run_xiangrui.sh > /dev/null 2>&1 < /dev/null &                          # register + evaluate
STEPS="ablate evaluate report" setsid nohup bash bench/run_xiangrui.sh > /dev/null 2>&1 < /dev/null &   # then the rest
```

`register` is `python -m octreg register OCT MRI -o OUT` on the two original files. Outputs go to
`/root/autodl-tmp/bench_runs/xiangrui_I58/rel7/{main,main_logs}` and `rel6/ablate` (rel6/main has the identical transform), the run in bench/results/xiangrui_I58 (override with
`OUT`, `ABL`, `PREV`, `PREV_MAIN`, `DEVICE`, `CODE`). Every step writes
`NAME.log`, `NAME.time` (wall time, peak RSS) and `NAME.gpu_mib` (nvidia-smi samples) into `main_logs`, and a start and a done
line per step (and the last 5 log lines on failure) to `chain.log`. The script registers its process group in `/root/autodl-tmp/v11_dev/killable/octreg1_xiangrui.pgid` while it runs
and refuses to start when another registration job is running. The v1.1 run of this pair peaked at 32 GB of RAM in a 62 GB
container, so run nothing else heavy at the same time.

`python bench/evaluate.py --selftest` checks the frame conversion, the frame check, Dice and the boundary agreement on synthetic
arrays in a few seconds (CPU, no server files).

## What is measured

The reference is R5, the v1.1 pose of this pair. It maps the v1 pipeline OCT frame (layout SPR of the raw array,
`work/v11/xiangrui_I58bs/oct_native_affine.npy`) to the MRI header world, so in the header frames octreg uses it is
`T_ref = T_R5 @ A_spr @ inv(A_hdr)`, with `A_hdr` the OCT NIfTI header affine. `evaluate.py` checks this against the stored,
already verified header-frame export of R5. R5 is a body-level pose of the earlier pipeline, not ground truth and
not a success criterion.

- Pose: displacement of `T_oct2mri` against `T_ref`, mean and max over the v1.1 OCT specimen-mask points and at the 8 block corners,
  plus the rotation angle between the two. The same distance to an earlier run with `--previous` (the script passes `PREV_MAIN`,
  by default `rel4/main`, the release run before §5).
- OCT specimen mask: volume and Dice against the stored v1.1 mask (`oct150_mask.npy`, 18.05 cm3), compared through the header
  affines.
- Boundary agreement: OCT mask outline through the pose to the MRI foreground outline and back, median per face; "rim" leaves out
  the deep end of the block (raw axis 0, high index), which has no specimen rim. Cut faces of either field of view are excluded.
  It is computed with the method masks and again with the stored v1.1 masks (the MRI tissue mask cleaned as in v1.1 `qc_fine.py`,
  which reproduces the R5 rim medians of `qc_fine.py`, 2.02 / 1.05 mm, to 0.001 mm), each time also for R5 and the earlier run.
- Raw-data frame check: raw 20 um values streamed from the OCT `.nii.gz` through the header affine and the pose, against the exported
  `oct_in_mri.nii.gz` (Spearman, 7^3 voxel boxes). Controls flip each raw OCT axis or shift the pose by 2 mm. It passes when the pose
  gives at least 0.9 and every flip at most 0.3.

Masks come from `ablate.py` (`rel4/ablate/prep/texture/`), so `evaluate` reports mask and boundary metrics only after the ablate step.

## Ablations

| name | change |
|---|---|
| base | the method (default Params) through the ablation driver |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask |
| A0b | the v1.1 rim-watershed specimen mask as the OCT mask (also written to `prep/v11_mask.nii.gz` for `--oct-mask`) |
| A0c | the texture mask with holes filled in 3-D only instead of in every array plane |
| A1, A2 | MRI flattening off, OCT flattening off (`two_class(..., flatten=False)`) |
| A4 | standardised intensity channels (z, -z) instead of two-class maps |
| A5+1, A5-1 | polarity forced to +1 or -1 instead of the sign of the two-class score (`align(..., polarity=...)`) |
| A6 | no scale prior: lambda 0 and clamp 1.0 instead of 2 and 0.15 |
| A8 | the other handedness: the OCT world mirrored (z negated) before search and refinement |
| A9 | no outline term: S = 2 S_class / 3 in the search (patched `search.combined`) and in the refinement (outline weight = the specimen mask, so S_outline = 0) |
| A10 | two-sided outline of the previous release: OCT embedding over MRI tissue or outside the crop counted as a mismatch |
| A11, A11b | simulated cut face (specimen mask removed beyond 70 % of its extent along OCT axis 1, data kept as embedding), with the method and with the two-sided outline |
| A12 | no fine-structure refinement: the pose of §4 (every other variant ends with §5) |
| A3, A7 | removed steps (section-stripe flat field; rigid, similarity and affine ladder), rows copied from the first ablation run |

`Params` holds method constants only. Each variant is one explicit change made in the driver around the package functions the
method itself uses: an argument of `register.fine_mask`, `preprocess.two_class` or `register.align`, or a Params override (A6).
A0 takes its OCT mask from `preprocess.foreground`, A4 builds its channels in `ablate.py`, and A0c and A9 temporarily replace
`preprocess._fill_planes` (with the 3-D `ndimage.binary_fill_holes`) and `search.combined`. The method has no ablation switches. The OCT is
streamed once and each OCT mask source (texture, texture3d, intensity, v11mask) is computed once and cached under `ablate/prep/`. The
driver's base run is compared with the CLI run ("driver check", expected at float precision). Pose changes are measured
against base. The boundary agreement of every variant uses the base masks, so it reflects the pose only. A variant that makes
the method refuse is recorded as failed with the message. Finished variants are reused on a
rerun unless `--force` is given.

A3 and A7 were run in the first ablation run, with the earlier two-class score, the overlap gate and mirrored orientations in
the search, (`/root/autodl-tmp/bench_runs/xiangrui_I58/ablate`), against a base that still had
the flat field and the ladder. Neither moved the pose by more than 0.5 mm, so both steps were deleted. `ablate.py --previous`
copies their rows and reports how far the present base lies from that earlier base.

The deletion rule: a step whose removal moves the pose by at most 0.5 mm (mean over the specimen-mask points; the pose after §4
for steps of §1-4, the final pose for §5) and changes no metric beyond noise is deleted before release. MRI flattening meets it
and is kept as the one exception, so that both volumes pass through the same class map (see bench/BENCHMARK.md).
