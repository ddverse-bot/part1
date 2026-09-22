# bench

Benchmark of octreg 1.1 on Xiangrui's I58 brainstem pair, and a generality check on the DANDI:000026 Broca-area blocks. It
lives in the repository but is not part of the package, and the method never reads anything here. The I58 pair has no labels,
so its metrics are label-free.

The scripts find their files through `bench/paths.py`: a project root, which holds `bench_runs/`, and a data root, which holds
the two I58 files and the DANDI tree. `OCTREG_PROJECT_ROOT` and `OCTREG_DATA_ROOT` override the two. Without an override the
project root is `D:/Projects/oct-mri-registration`, and the data root is `<project root>/oct-mri-registration/data` where that
directory exists and `D:/Datasets/oct-mri-registration` otherwise. On any other machine set both variables. Nothing else here
is machine-specific.

| file | what it does |
|---|---|
| `run_xiangrui.py` | the runner for the I58 pair: the steps register, ablate, evaluate and report, their overrides and log files, on any OS |
| `paths.py` | the two machine roots and the input files derived from them |
| `evaluate.py` | one run measured against the data itself: boundary agreement, OCT mask volume, raw-data frame check, and the pose distance to an earlier octreg run |
| `ablate.py` | the preprocessing once, then the variants base and A0-A12, one JSON table, the §6 read-outs of base |
| `dandi.py` | the DANDI blocks: MRI crop, `octreg register`, the cortical-layer read-out and its figure, a markdown summary |
| `report.py` | writes `bench/BENCHMARK.md` and two figures in `bench/figures/` from the outputs |
| `compose_pair.py` | two `octreg qc` outputs of one run side by side (`bench/figures/fig_handedness_xiangrui.png`) |
| `fig_registration.py` | the registration figure: an MRI plane, the OCT placed on it by the run, and those two panels cut into 8 mm squares and interleaved |
| `BENCHMARK.md` | the I58 report, written by `report.py`, which keeps its hand-written "Visual result" section and its closing reading |
| `results/` | the JSON copied out of the run directories |
| `figures/` | the figures of `BENCHMARK.md` and of the registration figure |

## Running the I58 benchmark

`bench/run_xiangrui.py` runs the steps from the repository root, with the interpreter that runs it. The whole benchmark is
`register ablate evaluate report`, with `evaluate` run twice: the mask and boundary metrics need the preprocessing that the
ablate step writes.

```bash
export OUT=/new/dir/bench_runs/xiangrui_I58/myrun/main          # a directory that holds no run yet
export ABL=/new/dir/bench_runs/xiangrui_I58/myrun/ablate
STEPS="register evaluate" python bench/run_xiangrui.py          # the registration, then the metrics that need no masks
STEPS="ablate evaluate report" python bench/run_xiangrui.py     # the ablations, then all metrics, then BENCHMARK.md
```

PowerShell sets the same variables with `$env:OUT = "...\myrun\main"`. To detach: `setsid nohup ... &` on POSIX,
`Start-Process -WindowStyle Hidden python bench/run_xiangrui.py` on Windows.

`register` is `python -m octreg register OCT MRI -o OUT` on the two original files, with default parameters. The other
overrides are `PREV` (the `ablations.json` of the run that measured the removed steps A3 and A7, by default
`bench_runs/xiangrui_I58/ablate/ablations.json`, the first ablation run), `PREV_MAIN` (an earlier run for the pose distance,
by default `bench_runs/xiangrui_I58/rel7/main`, release 1.0), `DEVICE` (`cuda` or `cpu`) and `CODE` (the repository to run
from, by default the one this file is in). `OUT` and `ABL` default to the released run, `rel8/main` and `rel8/ablate`, so a
new run has to set both. Every step writes `NAME.log`, `NAME.time` (wall time, peak RSS) and `NAME.gpu_mib`
(nvidia-smi samples, when nvidia-smi is on the PATH) into `${OUT}_logs`, and a start and a done line per step (and the last 5
log lines on failure) into `chain.log` there.

Two rules. One registration job at a time: the runner refuses to start while another registration python is running. And
`register` and `ablate` refuse to write into their default `OUT` and `ABL` once those hold a run, so a new run needs a new
`OUT` and a new `ABL`. `report` has no such guard: it rewrites `bench/BENCHMARK.md` from whatever `OUT` and `ABL` name, which
without an override is the released run.

`python bench/evaluate.py --selftest` checks the frame check, the boundary agreement, the pose distance and the mask volume on
synthetic arrays in a few seconds, on the CPU and without any of the data.

The released run is `bench_runs/xiangrui_I58/rel8`. The report step copies its `result.json`, `eval.json` and
`ablations.json` into `bench/results/xiangrui_I58/` (`report.py --store`) and writes `BENCHMARK.md` and `bench/figures` from
them, so the document and the numbers it quotes are one command and never drift apart.

## The DANDI:000026 blocks

`bench/dandi.py` is the generality check of the method on cortex. Nine subjects have both an OCT block (OME-TIFF of the
scattering coefficient, no orientation in its header, voxel spacing from the BIDS sidecar) and an ex-vivo MRI of the hemisphere
it was cut from. Eight are used. sub-I56 is refused: the affine of every one of its label files disagrees with its array, so
only 30 to 72 % of each label lands inside the MRI and neither the crop nor the evaluation can be read from it.

```bash
python bench/dandi.py [SUBJ ...] [--steps crop register evaluate summary] [--device cuda] [--out DIR]
```

The default output root is `bench_runs/dandi`, one directory per subject.

The MRI is one EPIC `*_VFA.nii.gz` asset per subject, pinned in `MRI_FLIP` of `bench/dandi.py` and not chosen by a glob or by
flip angle, so every run reads the same file whatever else a download holds: `flip-2` for sub-I38, sub-I46, sub-I48 and
sub-I55, `flip-3` for sub-I58 and sub-I62, `flip-4` for sub-I57 and sub-I61. These are the assets the published runs used.
Their sidecars give no rule that would reproduce them (sub-I48 has no sidecar, and sub-I58's pinned `flip-3` has FlipAngle 10
against 30 for its `flip-1`), so the map is explicit.

- `crop`: the method assumes an MRI already cropped around the block, so the MRI is cut to the bounding box of the subject's own
  Broca-area label plus 5 mm and written to `OUT/SUBJ/mri_crop.nii.gz`. That label says which part of the hemisphere the block
  was taken from, which is what a user of the method knows. It is not the registration: it fixes neither the orientation nor the
  position inside the crop.
- `register`: `octreg register` on the block and the crop, the released entry point and the same one as for any other pair, so
  §1-6 run unchanged. An OME-TIFF carries no orientation, so §3-5 run for both handednesses and the higher fine-structure
  agreement F wins.
- `evaluate`: a number the registration never sees, written to `OUT/SUBJ/eval.json`. The subject's cortical-layer label divides
  the cortex into layers. Its two ends, the classes with the lowest and the highest median MRI intensity, are brought into the
  OCT by the pose, and the two-class map of §2 is read there. The separation |AUC - 0.5| of the two sets of values says whether
  the registered OCT tells the two ends of the cortex apart, and does not depend on which modality is bright where. The same
  number at the pose of §4 and at 10 random poses of the block inside the crop says what it is worth on that subject. A random
  pose that lands fewer than 500 labelled points inside the specimen mask, or fewer than 100 in a class, is not scored, so the
  median and the max are over 3 to 8 of those 10 poses, and `random_separation["n"]` of `eval.json` says how many.
  It also writes `qc_labels.png`, the registered OCT with the outlines of that label: where the pose is right they follow
  the bright and dark bands of the OCT.
- `summary`: prints one markdown row per subject that has a run: crop size, the scores of §4, the handedness, F, the scale per
  OCT axis, §6, and the layer separation next to the random poses, whose column carries the number of poses actually scored.

Visual inspection stays the evaluation (docs/METHOD.md): `qc.png`, `qc_montage.png` and `qc_labels.png` of every run are the
primary evidence and these numbers only support them.

Where the texture specimen mask of §1 keeps less than half of what the OCT measured, the block was registered again with the
OCT given as its own mask, which is the mask a block without embedding should have. The registration is the released command
with one more argument, since a mask file is read as its positive voxels, and the read-out is the evaluate step with a tag:

```bash
python -m octreg register OCT OUT/SUBJ/mri_crop.nii.gz -o OUT/SUBJ_allmeasured     --oct-spacing-um Z,Y,X --oct-mask OCT
python bench/dandi.py SUBJ --steps evaluate --tag _allmeasured --out OUT
```

The second command writes `OUT/SUBJ_allmeasured/eval.json`, the read-out of that run, taken over the mask the run was given.
The numbers docs/METHOD.md quotes for the diagnostic come from those files.

## What is measured

The I58 pair has no labels, so `evaluate.py` measures the run against the data itself. It reads the two input files, the run
directory, the masks of an `ablate.py` prep directory (`--masks`) and, for the comparison, a second octreg run (`--previous`).
Nothing else. `T = RUN/T_oct2mri.txt` maps the OCT header world to the MRI header world.

- Boundary agreement: the OCT specimen-mask outline through the pose to the MRI foreground outline and back, median per face;
  "rim" leaves out the deep end of the block (raw axis 0, high index), which has no specimen rim. Cut faces of either field of
  view are excluded. With `--previous` the earlier run's rim medians are measured under the same masks, so the two runs are
  compared on the same footing.
- OCT specimen mask: the volume of `oct_mask.nii.gz` in cm3.
- Pose distance to an earlier octreg run (`--previous`, the runner passes `PREV_MAIN`): mean and max displacement over the
  points of the run's own OCT specimen mask, the same at the 8 block corners, and the rotation angle between the two poses.
  Without `--masks` the mean is over the 8 corners of the OCT array plus a uniform sample of its grid instead.
- Raw-data frame check: raw 20 um values streamed from the OCT `.nii.gz` through the header affine and the pose, against the
  exported affine overlay `oct_in_mri_affine.nii.gz`, or `oct_in_mri.nii.gz` for a run before 1.1 (Spearman, 7^3 voxel boxes).
  Controls flip each raw OCT axis or shift the pose by 2 mm. It passes when the pose gives at least 0.9 and every flip at most
  0.3.

Masks come from `ablate.py` (`ABL/prep/texture/`), so `evaluate` reports the mask and boundary metrics only after the ablate
step. The pose distance and the frame check need no masks.

`ablate.py` reads the same two input files and its own runs, and nothing else. Its pose changes are measured over the points of
the base specimen mask and over the 8 corners of the OCT array.

## Ablations

| name | change |
|---|---|
| base | the method (default Params) through the ablation driver |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask |
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
| A3, A7 | removed steps (section-stripe flat field; rigid, similarity and affine ladder), rows copied from the first ablation run through `PREV`, see below |

`Params` holds method constants only. Each variant is one explicit change made in the driver around the package functions the
method itself uses: an argument of `register.fine_mask`, `preprocess.two_class` or `register.align`, or a Params override (A6).
A0 takes its OCT mask from `preprocess.foreground`, A4 builds its channels in `ablate.py`, and A0c, A9 and A10 temporarily
replace `preprocess._fill_planes` (with the 3-D `ndimage.binary_fill_holes`), `search.combined` and the outline of the search
and of the refinement. The method has no ablation switches. The OCT is streamed once and each OCT mask source (texture,
texture3d, intensity) is computed once and cached under `ABL/prep/`. The driver's base run is compared with the CLI run
("driver check", expected at float precision). Pose changes are measured against base. The boundary agreement of every variant
uses the base masks, so it reflects the pose only. A variant that makes the method refuse is recorded as failed with the
message. Finished variants are reused on a rerun unless `--force` is given.

The smooth deformation (§6) does not change the pose, so the driver runs it for base only and stores its read-outs (`deform` in
`variants/base/result.json` and in the base row of `ablations.json`, without the field). "No §6" (A14) is base with the
identical pose, so it is not a variant of its own: its read-outs are the residuals before the deformation, which the base row
holds next to those after it, and `report.py` prints both. The read-outs are the lattice spacing and the chosen membrane weight,
the interior matches and the supported boundary points of the fit, the held-out errors, and the residuals of both kinds (block
matches, surface-edge offsets) measured on the affine OCT and again on the warped one. The ablations of §6 itself (one kind of
evidence alone, the rim ridge in place of the edge, no support rule, no Huber re-weighting, other lattice spacings and strain
limits) are tabulated in docs/METHOD.md. They were run on the cached base grids of `ABL/prep` at the pose of release 1.0,
with development scripts that are not part of this repository. Since 1.1 a prep is checked against the Params fields that §1 reads
(`prep_hash` in `prep.json`), so new Params of later stages leave it valid, and the preps of 1.0 are accepted.

A3 and A7 were run in the first ablation run, with the earlier two-class score, the overlap gate and mirrored orientations in
the search (`bench_runs/xiangrui_I58/ablate/ablations.json`, the default `PREV`), against a base that still had the flat field
and the ladder. Neither moved the pose by more than 0.5 mm, so both steps were deleted. `ablate.py --previous` copies their rows
and reports how far the present base lies from that earlier base.

The rows appear only when `PREV` points at that first run, which is the default. A `--previous` file that holds neither step
makes `ablate.py` print a warning naming the variants it did find, so the rows cannot go missing unremarked.

The deletion rule: a step whose removal moves the pose by at most 0.5 mm (mean over the specimen-mask points; the pose after §4
for steps of §1-4, the final pose for §5) and changes no metric beyond noise is deleted before release, unless a test outside
this pair shows it load bearing. Two steps are kept that way, MRI flattening and the one-sided outline; the evidence for each is
in docs/METHOD.md after the ablation table.
