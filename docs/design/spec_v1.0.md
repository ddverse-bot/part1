

# method_spec

octreg 1.0: label-free OCT-to-MRI block registration

Base design: "minimal", which both judges ranked first. It takes the per-block jackknife decision rule, the scale-identifiability check, the status semantics and the prior-restricted search from "robust". It takes the boundary-QC rule, pre-registration, the baselines and the documentation plan from "publication".

Principle: each stage may change only what its evidence can see. The two ambiguities the data may not resolve, handedness and relative contrast polarity, are enumerated as discrete hypotheses and scored against a noise scale measured inside the specimen. Nothing in the method is specific to a dataset. Every constant is in mm, in units of h, or in units of a measured period, and lives in one frozen Params.

INPUTS
- OCT volume: NIfTI; or TIFF/OME-TIFF/NPY with spacing from OME PhysicalSize (length units only), a BIDS sidecar, or --oct-spacing-um.
- MRI: 3-D NIfTI.
- Optional: --oct-mask / --mri-mask (recorded as overrides); --init T.txt (a prior pose).
- No labels. No polarity, handedness, axis, embedding or vessel switches.

OUTPUTS (all transforms in the world frames of the input files)
- T_oct2mri.txt, T_mri2oct.txt, oct2mri.lta, oct2mri_itk.txt.
- hypotheses/{proper,mirrored}_{same,inverted}/T_oct2mri.txt: all four, always.
- oct_in_mri.nii.gz: MRI grid, block radius + 8 mm.
- mri_in_oct.nii.gz: OCT frame at h.
- qc.png.
- result.json (schema in code_spec).

STEP 0. Frames
- NIfTI: voxel-to-world from the sform, else the qform. Both codes 0 means frame 'array' and handedness 'unknown'.
- TIFF/NPY: world = diag(spacing) in array order (OME DimensionOrder, else zyx); handedness 'unknown'.
- Images are scaled by their foreground median and stored as float16. This replaces the 16384 gain and cap.
- Refuse, with a message:
  - no spacing source;
  - input not 3-D (a 4-D input lists its volumes; never take the first one silently);
  - fewer than 1000 foreground voxels at h in either image;
  - OCT foreground spans fewer than 8 voxels at 4h along any axis;
  - OCT foreground bounding radius larger than the MRI FOV half-diagonal ("inputs swapped?").

STEP 1. Grids
- h = max(0.15 mm, largest MRI voxel edge). Pyramid 4h / 2h / h, which is 0.6/0.3/0.15 mm on every current pair.
- One resampler for both images: per-axis box average over k = max(1, round(level/spacing)) voxels (as in v1), then trilinear onto an exactly isotropic grid aligned with that image's own array axes.
- The OCT is read plane by plane from memmap; it is never loaded whole.

STEP 2. Foreground threshold (the same rule for both modalities; v1 rule unchanged)
- Sample: at most 1/64 of positive finite values, clipped at p99.5; 256-bin histogram smoothed over 5 bins.
- Start from the rightmost peak (prominence >= 0.05 x max, distance >= 5 bins). Walk left to the first valley with depth ratio < 0.5 and threshold there.
- No such valley: threshold = min(p1, 0.5 x the first 3-class multi-Otsu cut), and set flag foreground_no_valley:<modality>. The run continues. I55 takes this path and is correct.
- A user mask replaces this step (source 'user').

STEP 3. Sectioning gain (OCT only; decided by the data; can be a recorded no-op)
- Detection grid: OCT pooled to >= 40 um per axis.
- Axis: for each array axis, high-pass the planes (subtract a Gaussian of 3 voxels) and take the median NCC over 40 adjacent-plane pairs, skipping the outer 5 %. Axis a* = lowest median; decisive iff it is < 0.8 x the second lowest.
- Period: P = FFT peak of the per-plane median profile m_k (voxels above the step-2 threshold) along a*, after a 2.4 mm running-mean detrend. Search 0.1-2 mm; accept iff the peak is >= 3 x the median of the spectrum.
- Correction: if a* is decisive and P is accepted, divide plane k at native resolution by g_k = m_k / runmean over 3P of m. Planes with fewer than 2000 foreground voxels get g = 1.
- Otherwise the step does nothing, and the reason and the numbers are recorded.
- Expected from earlier measurements: I46 axis 0, P = 0.39 mm, window about 100 slices, equal to v1; I56 window about 34 slices, equal to v1; brainstem axis 2, P = 0.30 mm.
- Pre-declared deletion rule: the step is removed before release if ablation A3 meets the acceptance criteria without it.

STEP 4. Foreground mask at h
- Apply the step-2 rule again on the gained, pooled image.
- Cleanup: closing with radius 2h, fill holes, keep every component >= 1 % of the foreground volume. There is no largest-component rule, so multi-piece blocks are allowed.

STEP 5. Two-class maps (one function, both modalities)
- Flatten: I <- I / max(G10*(I·M) / G10*M, 0.5·p5), with G10 a Gaussian of sigma 10 mm.
- Blur with sigma = h (mm).
- t = Otsu of foreground values clipped at p99.5; p = sigmoid((I − t) / (0.25·std)).
- OCT channels u = (p_O, 1 − p_O), NOT multiplied by the mask; weight w = M_O (sampled, so soft at the edges).
- MRI channels v = (p_M·M_M, (1 − p_M)·M_M). The foreground factor is kept on purpose: the second channel carries the specimen outline that places the brainstem.
- Coarser levels are box averages of the maps.
- Score S(T) = ½ Σ_c NCC_w(u_c, v_c∘T).
- Because NCC_w(1−u, v) = −NCC_w(u, v) for any weight, swapping the OCT classes gives exactly −S. Relative polarity is therefore the sign of one score map; there is no absolute polarity.

STEP 6. Global search at 4h
- Orientations: 8000 uniformly distributed rotations from a fixed seed (R[0] = I), each used as R and as R·diag(1,1,−1).
- For each orientation, sample u and w on an MRI-axis-aligned template covering the block's bounding sphere + 3 voxels. Compute masked NCC over all translations by FFT, with the MRI zero-padded by half a template and the local MRI variance floored at 0.02 x the foreground variance.
- Admissible translations: overlap Σ(w·M_M)/Σw >= tau, with tau = 0.8·min(1, V_M/V_O). If tau < 0.15, set tau = 0.15 and flag overlap_floor.
- Per orientation, keep argmax S (polarity 'same') and argmax −S (polarity 'inverted'). One pass yields all four hypotheses.
- Per hypothesis: NMS (a pose counts as the same if its centre is < 3 mm away AND its rotation differs by < 10 deg), keep the top 12.
- Decisiveness per hypothesis, reported and never gated: d = (s1 − s2) / (s1 − median of the per-orientation maxima).
- With --init: translations within 10 mm of the prior block centre; rotations within 30 deg of the prior's rotation part in each handedness branch (the mirrored branch composes it with the mirror); polarity stays free. Same code path.

STEP 7. Refinement per hypothesis (handedness fixed)
- Model: x_MRI = R·Sh·diag(e^ls)·[mirror]·(x_OCT − c) + t, with c the OCT foreground centroid.
- Penalised loss L = 1 − S + λ(Σls² + Σsh²), λ = 2; |ls|, |sh| <= 0.15 (scale 0.86-1.16). Clamps are absolute.
- Adam with a cosine schedule; learning rates 0.02 rad / 0.3 mm / 0.01 / 0.01. Return the best-L iterate and record both S and L.
- Ladder:
  - at 4h: rigid 120 then similarity 120 iterations; keep 4 by L;
  - at 2h, on an MRI crop of block radius + 6 mm: rigid 200 then affine 200; keep 2;
  - at h: affine 200 for both; keep 2.

STEP 8. Decisions
- Winner W = the hypothesis/pose with the lowest L. Hypotheses are always compared on the penalised objective the refiner optimises, never on raw NCC, because shrink poses exploit that gap.
- Competitors of W: (c1) best of the other polarity with the same handedness; (c2) best of the other handedness; (c3) the distinct runner-up inside W's hypothesis (mean mask-point displacement >= max(2 mm, 0.1·R_g), where R_g is the OCT foreground radius of gyration).
- Noise scale by paired block jackknife:
  - Split the OCT foreground at h into cubes of edge max(2 mm, (V_fg/27)^(1/3)); merge cubes holding < 25 % of the median point count into a neighbour. B = number of blocks.
  - For each competitor C, compute Δ = L_C − L_W. The prior term is constant under the jackknife.
  - Compute Δ_(−j) by leaving block j's points out of both NCCs; this is exact from per-block moment sums, with no re-optimisation. σ² = (B−1)/B·Σ(Δ_(−j) − mean)². z = Δ/σ.
  - A choice is decided iff z >= 3 and B >= 8; otherwise undecided, with the reason.
- Scale identifiability per OCT axis k: evaluate S at ls_k ± 0.05 and ± 0.10 with everything else fixed. Identifiable iff some offset gives a paired jackknife z >= 3. Flag scale_unidentifiable:k if not, and clamp_saturated:k if |ls_k| >= 0.14.
- Info fields, not flags: handedness_conflicts_headers (both frames physical and det T < 0); sectioning {axis, P, applied}.
- Descriptive boundary agreement: only if >= 50 % of OCT foreground boundary voxels lie >= 2h from the OCT FOV faces (a whole specimen with a real outline, not a cut face). Report median forward/reverse distances to the MRI foreground boundary. Never called accuracy.

STEP 9. Status (descriptive; never a correctness claim)
- 'ambiguous': any of c1-c3 undecided. All hypotheses/poses with z < 3 are listed in rank order next to the primary W.
- 'flagged': otherwise, if any of foreground_no_valley, oct_foreground_exceeds_mri (V_O > V_M), overlap_floor, clamp_saturated, scale_unidentifiable, nondefault_params fires.
- 'ok': otherwise.
- No gate ever uses absolute NCC, restart counts, raw top1/top2 margins or corpus-fitted volume constants. z >= 3 and B >= 8 are fixed a priori; only synthetic controls check them.

PARAMS (frozen, each with unit and provenance)
- h_min 0.15 mm
- sigma_flat 10 mm; flat floor 0.5·p5; sigmoid 0.25 std
- valley ratio 0.5; cleanup closing 2h, min component 1 %
- section: NCC ratio 0.8, SNR 3, window 3P, detrend 2.4 mm
- N_rot 8000, seed 0; K 12; NMS 3 mm / 10 deg
- rho 0.8, floor 0.15; variance floor 0.02
- λ 2, clamp 0.15
- ladder table
- jackknife z 3, B_min 8, block rule
- prior radius 10 mm, angle 30 deg

During migration only, Params.compat holds one v1-behaviour switch per changed step (see implementation_plan). These are removed at release.

FAILURE REPORTING
- Refusals give no transform.
- Low contrast, bath-heavy MRI and non-corresponding tissue are not hidden. They show up as flags, small d, or undecided competitors.
- The model is affine only. Tears and moved cerebellum remain a stated limitation.


# component_decisions

Format: component -> decision -> reason (evidence).

CORE PACKAGE
- common.py geometry, resampling, I/O -> KEEP, split into geometry.py and io.py.
  - Conventions verified end to end: export NCC 0.99999999998; header check Spearman 0.933 vs 0.09-0.25 for flipped axes.
  - Delete iso_grid_affine, crop_volume, load_nifti (no callers).
  - Merge the 3 pooling copies.
  - Replace transform_diff and polar_rotation comparison with pose_distance: I46 reported 129 deg against a true 2.73; 41 rotation fields are invalid.
- oct_slab_normalize (axis 0, 1.2 mm window) -> REPLACE with section gain on the detected axis and period (compat switch M2).
  - I58 sections along axis 2.
  - The window is 3.03 / 2.83 periods on I46 / I56 only by chance.
  - Never ablated, so it carries a deletion rule.
- float16 gain 16384 + CAP_F16 -> REPLACE with foreground-median scaling. Storage detail only; features are scale-invariant.
- histogram_tissue_threshold -> KEEP the rule unchanged for both modalities; replace the silent fallback with flag foreground_no_valley. Six MRIs have healthy valleys; I55 is correct on the fallback; I56, I61 and DANDI I58 fail open silently.
- oct_tissue_mask and the MRI raw mask -> MERGE into one cleanup in mm, no largest component (M8, regression-gated).
  - I57 is a two-piece block.
  - qc_fine had to re-clean the MRI mask.
- 0.5×Otsu raw slab threshold -> REMOVE; the step-2 threshold is used instead.
- layout_affine / --oct-layout SPR -> REPLACE with header affines, or an array frame marked handedness unknown (M1).
  - On I58 the layout frame has det −1 against the header (REPORT C.4).
  - This is why export_registration.py existed.
- features.otsu_two_class + otsu_two_class_lowmem -> MERGE into features.two_class.
  - Same 10 mm flattening on both sides; OCT-side flattening is M3.
  - OCT channels unmasked, MRI channels stay masked (M4).
  - MRI flattening evidence: I46 Dice 0.65/0.60 -> 0.91/0.93.
  - Two-class beat intensity, parser and MIND on I46.
- Frangi functions -> REMOVE from the package. Only bench keeps an evaluation-only vessel map. The per-chunk gamma striping bug goes with it.
- mind_ssc, parser_features, build_features -> REMOVE. MIND: 17.9 mm off, 3/8 restarts. MIND is re-implemented in bench only, as baseline B2.
- search.FFTSearcher -> KEEP.
  - Found I46/I55 in whole hemispheres; crop and whole gave the same pose.
  - Change: polarity from the sign of one map; per-hypothesis NMS/top-12 (M6); remove the duplicate defaults.
  - Mirror branch kept: without it I46 top1 falls 0.7328 -> 0.6848; 6/9 DANDI poses are mirrored.
- refine.py -> KEEP.
  - Absolute prior λ 2 / clamp 0.15 is load-bearing: prior costs 0.045-0.078 against a depth NCC range of 0.021; it stops the P5/P6 shrink basins.
  - Change: return both S and L; remove chan_w/subsample (vascular only).
- decide.py (new) -> ADD: block jackknife z, scale profile, selection by L.
  - Restarts are anti-correlated with correctness (I61, DANDI I58 12/12 and wrong; I46/I55 9/12 and correct).
  - Raw margins flag I55 (1.004), which is correct.
- Overlap gate -> KEEP as tau = 0.8·min(1, V_M/V_O) with floor 0.15. Reproduces 0.369 / 0.59 on the brainstem; never binds on DANDI (M9 checks the 0.85 -> 0.8 change).
- vascular.py -> ARCHIVE (tag v1.1-archive).
  - Helps I46 only (265 -> 167 um, with a wrong depth sidecar).
  - I55: 0.33 mm move, no labelled gain.
  - Vessel NCC <= 0.053 at 30-35 um.
  - Brainstem: harmful (2.65 mm, 91 % of wall time).
  - Re-entry bar: gain on >= 2 labelled vessel pairs. The I46 result stays reproducible from the tag (B5).
- fine.py -> ARCHIVE. Rejected in 13/13 I58 configurations. Accepted but harmful on I46 (ves_seg 167 -> 264 um, 7/11 regression rows fail) and I55 (GM Dice 0.930 -> 0.916).
- specimen.py (texture mask, auto rule, v1.1b options) -> ARCHIVE.
  - Keeps 0.28 of 0.68 cm3 on I46.
  - The auto rule can only fire on MRI crops.
  - The options change < 0.07 cm3.
  - Used only as a user-supplied mask in ablation A10 on the brainstem.
- destripe.py -> MERGE detect_stripe_axis (axis, period) into normalize.section_gain.
  - REMOVE destripe_inplace: its pose effect (0.35 mm / 1.4 deg) is within the seed spread (0.31 / 1.1); P11 changed NCC by <= 0.0004.
  - REMOVE section_phase_modulation (its gate never fired).
- evaluate.py -> MOVE to bench/evaluate.py. Dice on labelled voxels; label coding as a parameter.
- nonrigid.py, parser.py, synth.py -> ARCHIVE. Nonrigid: +0.015 Dice on I46 only. Parser: needs labels, crop Dice 0.44/0.70. gaussian_blur3d is rewritten in geometry.
- __init__.py -> REWRITE: register() API, version 1.0.0. The "structural parser" docstring was false.

SCRIPTS AND FLAGS
- prep_subject.py + register.py -> REPLACE with octreg.pipeline plus cli (49 + 27 flags -> 5 public options).
  - Remove: --features, --parser-dir, --ref-transform, --target-mm, --vessel-level-um, threshold overrides, all mask, destripe and vessel flags, --no-mirror, --oct-wm-bright (default 'no' is wrong on 7/9 DANDI plus the brainstem), --mri-wm-bright, --n-restarts, the 9 vascular flags, the 22 fine flags.
  - --n-rot, --topk, --min-overlap, --reg, clamps, --seed -> Params.
  - --crop-centre/--init-transform -> --init, which restricts the same search.
- map_labels_to_grid + annotation flags -> MOVE to bench/labels.py and FIX. 32-42 % of labels are lost on the seven 0.12 mm subjects.
- qc_fine.py -> Boundary agreement MERGES into qc.py under the whole-specimen rule. REMOVE the MI landscape (optima are run-away shrink poses, P6) and the z+ deep end.
- export_registration.py -> MERGE into io/pipeline outputs plus `octreg apply`.
- check_export_header.py -> KEEP as tests/test_frames.py (synthetic) and bench/check_frames.py (raw data, box from spacing).
- regress.py -> REPLACE with bench/compare.py + acceptance.toml.
- summarize.py, viz_result.py, fig_doc2.py -> MERGE into qc.py, bench/report.py and bench/figures.py.
- fig_doc, fig_bs, fig_handedness, viz_pose, viz_region, viz_vascular -> DELETE.
- run_xiangrui_i58.sh -> ARCHIVE as the R5 reproduction record. The other handedness now comes out of the search itself.
- run_subject.sh, run_all.sh -> REPLACE with bench/pairs/*.toml + bench/run.py.
- DELETE from the release tree (kept in the tag):
  - recovery drivers (recover_serial*, rx2, rx3, tail_fixups, chain_after_batch, restart_xiangrui_first, rerun_xiangrui_fixed, run_xiangrui.sh, build_inspection.sh);
  - v1.1 run plans;
  - parser-branch scripts;
  - run_demo_i46.py, run_ablations.sh, summarize_runs.py;
  - the I46 prep chain;
  - reference-pose forensics;
  - brainstem one-offs;
  - test_fine.py.
- I46 validation suite -> ARCHIVE. The paper cites archived numbers by tag. The oracle displacement uses the stored oracle T in bench/evaluate.py.
- selftest_search.py, test_specimen_mask.py, test_destripe.py -> REPLACE with synthetic tests.
- docs/v11 probe scripts, P11 patch -> ARCHIVE, never merged. Exclusion jumped 6.5-8.2 mm into a worse basin.

DUPLICATES RESOLVED
- Normalisation (5+ mechanisms) -> section gain + one flattening.
- Masks (>10) -> one foreground rule, plus erosions in mm inside bench only.
- Polarity (5) and handedness (4) -> one 4-hypothesis search + jackknife.
- Pose metrics -> pose_distance.
- QC/summary tools (7+) -> qc.py + bench/report.py.
- Helpers -> a single copy each in geometry.py.

DOCS
- README (Chinese, stale pointers) -> REWRITE in English with a short Chinese summary.
- HANDOFF.md, REPORT_I58_v11.md, probe JSONs, runs_summary.md -> docs/archive/ unchanged.
- figures/ -> regenerated from benchmark runs.


# code_spec

Release branch release/1.0 in /Users/yyttim/workspace/lsfm-project/octreg. Tag v1.1-archive is set at db2282e before any deletion. Everything removed stays reachable through the tag.

octreg/ (installed package, target <= 1,600 lines)
- params.py: frozen dataclass Params, with every constant as a field (unit and provenance in the docstring), plus a nested Compat dataclass (migration switches M1-M9, all defaulting to the release behaviour). Also Params.to_json(), from_json() and hash(), and diff_from_default().
- types.py: dataclasses.
  - Volume: reader (memmap or plane iterator), affine (4x4), spacing_mm, frame ('header'|'array'), handedness ('physical'|'unknown'), spacing_source, path, sha256.
  - Hypothesis: hand, polarity, T, S, L, log_scales, shears, overlap, search_top1, search_top2, decisiveness.
  - Decision: name, value, z, B, decided, reason.
  - Result: status, T_oct2mri, hypotheses, decisions, flags, info, timings.
- io.py:
  - load_volume(path, spacing_um=None, axes=None) -> Volume;
  - write_matrix, write_lta(T, src_geom, dst_geom), write_itk;
  - save_nifti;
  - write_json (numpy-safe);
  - sha256_file (streamed).
- geometry.py:
  - rotations(n, seed);
  - rodrigues; compose(params) and decompose(T) with the mirror handled;
  - sample_at_world (torch grid_sample);
  - resample(vol, affine, level_mm) -> (array, affine), using box average then trilinear;
  - gaussian(vol, sigma_mm, spacing);
  - pose_distance(T1, T2, points) -> {mean_mm, max_mm, rot_deg or None, handedness_differs};
  - radius_of_gyration; apply_transform(moving, reference, T, nearest).
- normalize.py:
  - foreground_threshold(sample) -> (t, info{valley_ratio, status});
  - section_gain(volume, t) -> (gain_per_plane, info{axis, ncc_by_axis, ratio, period_mm, snr, applied, reason});
  - foreground_mask(img_h, t, h, compat) -> (mask, info{volume_cm3, n_components});
  - flatten(img, mask, h, sigma_mm).
- features.py: two_class(img, mask, h, params) -> p (float16, GPU-chunked for whole hemispheres); oct_channels(p, mask) -> (u, w); mri_channels(p, mask) -> v; pyramid(x, levels).
- search.py: FFTSearcher(mri_v, mri_mask, A_M, oct_u, oct_w, A_O, level_mm, params, init=None).run() -> {hypothesis_name: [Hypothesis x K]} plus per-hypothesis decisiveness and n_admissible.
- refine.py: Refiner(mri pyramid, oct points and channels, hypothesis, params) with fit(T0, dof, iters, level) -> (T, S, L); ladder(hypotheses, pyramids, params) -> refined per hypothesis (2 per hypothesis at h).
- decide.py:
  - block_partition(points, V_fg);
  - paired_jackknife(T_W, T_C, prior_W, prior_C) -> (delta, sigma, z, B);
  - select(refined) -> (W, competitors, decisions);
  - scale_profile(W) -> per-axis {identifiable, z, saturated}.
- qc.py:
  - boundary_agreement(oct_mask, mri_mask, T) -> dict or None (whole-specimen rule);
  - qc_png(...): 3 planes (OCT, MRI through T, checkerboard, class contours) plus a bar panel of the 4 hypothesis L values with jackknife σ.
- pipeline.py:
  - register(oct, mri, out, oct_spacing_um=None, oct_mask=None, mri_mask=None, init=None, params=Params(), device='cuda') -> Result.
  - Runs steps 0-9 and sets status.
  - Caches steps 0-5 under out/cache, keyed by input sha256 + params hash, so downstream-only ablations reuse them.
  - Frees memory between stages and records seconds and peak RSS/GPU per stage.
- cli.py and __main__.py.

PUBLIC CLI (the whole surface)
- `octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--init T.txt] [--device cuda|cpu]`.
  - The hidden `--params FILE.json` is for bench ablations only and sets flag nondefault_params.
  - Prior radius and angle live in Params.
- `octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--nearest]`.
- `octreg qc OUT` re-renders the figure.

OUTPUT FILES
- T_oct2mri.txt, T_mri2oct.txt, oct2mri.lta, oct2mri_itk.txt.
- hypotheses/<hand>_<polarity>/T_oct2mri.txt (4 folders).
- oct_in_mri.nii.gz, mri_in_oct.nii.gz, qc.png, log.txt, cache/ (deletable).
- result.json keys:
  - octreg_version, git_hash, argv, params (+ diff), inputs (path, sha256, shape, spacing, spacing_source, frame, handedness);
  - levels_mm;
  - foreground {oct, mri: threshold, valley_ratio, status, volume_cm3, n_components, source};
  - sectioning {...};
  - search {n_orientations, seconds, per hypothesis: top1, top2, decisiveness, n_admissible};
  - hypotheses [4: S, L, T, log_scales, shears, overlap];
  - decisions [polarity, handedness, runner_up: delta, sigma, z, B, decided];
  - scale_axes; boundary (or null); flags []; info {handedness_conflicts_headers};
  - status; alternatives []; timings_s; peak_rss_gb; peak_gpu_gb.

bench/ (in the repository, not installed)
- pairs/*.toml, 13 entries: I38, I46, I48, I55, I56, I57, I58_flip10, I61, I62, xiangrui_I58bs, plus variants I58_flip30, I46_spacing_12_12_14, I55_spacing_12_12_12. File facts only:
  - raw paths, sha256 from the DANDI API or Xiangrui md5, dandiset version;
  - OCT spacing and its source;
  - annotation files (wholehemi, BA, MRI vessels, ves_seg with offsets 0 and 66 for I46);
  - label reference grid (the same-shape label file);
  - reference transforms (v1 structural, v1 final, R5, oracle, the authors' I46 LTA);
  - data-card notes.
  - No method parameters.
- verify_data.py: sha256 check.
- labels.py: label-frame repair. Pair label axes by array shape with the same-shape label file. For I57/I61, pick flips by label-on-MRI-foreground fraction. Map crops through label world.
- evaluate.py:
  - Dice on labelled voxels plus confusion matrix;
  - BA-inside fraction and centroid distance;
  - MRI manual vessels vs an evaluation-only OCT Frangi distance map (global gamma), median and f150, when >= 20 annotated points are inside;
  - I46 ves_seg at offsets 0 and 66;
  - oracle displacement; depth scale vs metadata;
  - pose_distance to references.
- null.py: 200 random admissible poses (100 uniform + 100 from rejected top-K) and ±3 mm shifts; gives percentiles.
- stage0_rescore.py: re-score the stored v1 transforms and apply the verdict rule.
- signoff.csv: human verdict per figure.
- run.py: strictly serial queue, RSS watchdog kill at 55 GB, outputs to /root/autodl-tmp/bench_runs/<git_hash>/<pair>[/<variant>]; never overwrites old work/ caches.
- compare.py + acceptance.toml: regression ladder and acceptance checks.
- ablations.toml: named Params/compat diffs and input variants.
- synth.py: known-transform benchmark.
- check_frames.py: raw-data frame test.
- competitor.py: brainstem 0.1358 mirrored-pose check.
- baselines/: features_intensity.py and features_mind.py (same search and refine), ants_crop.sh, authors_i46.py.
- report.py -> docs/benchmark.md tables; figures.py -> figures/.

tests/ (pytest, CPU, < 5 min, GitHub Actions)
- test_geometry.py: compose/decompose round trip including mirrors; pose_distance on a mirrored pair gives handedness_differs and a correct rotation for the proper part.
- test_frames.py: oblique det<0 NIfTI; exported T agrees with direct world sampling; LTA/ITK round trip; `apply` consistency; flipped-axis control.
- test_normalize.py: bimodal histogram gives ok, unimodal gives no_valley; stripes on a random axis and period (0.2-0.5 mm) are detected and the residual drops; stripe-free input gives a no-op; x100 gain gives the same output.
- test_features.py: flattening removes a linear bias; maps are scale-invariant.
- test_duality.py: swapping OCT channels gives −S to 1e-5, both in the FFT search and in the refiner.
- test_search_synthetic.py: 64^3 two-class random field; block with known rotation x handedness x polarity x ±10 % spacing error is recovered to < 0.5 mm with the right hypothesis.
- test_refine_grad.py: autograd matches finite differences.
- test_decide.py:
  - mirror-symmetric phantom gives handedness undecided;
  - asymmetric phantom gives decided;
  - fewer than 8 blocks gives undecided;
  - jackknife from moment sums equals brute-force recomputation.
- test_cli.py: tiny pair end to end; result.json schema validated.

ENVIRONMENT: pyproject.toml (torch, numpy, scipy, scikit-image, nibabel, tifffile, matplotlib, tomli); environment.lock.yml exported from the server octmri env; LICENSE; CITATION.cff; CHANGELOG.md. Deterministic rotation set; git hash stamped in outputs.


# validation_spec

All compute runs on the AutoDL server, one heavy job at a time through bench/run.py with the 55 GB watchdog. Outputs go to /root/autodl-tmp/bench_runs/<hash>/. Only markdown, JSON and PNG files under 20 MB come back to the Mac. Every acceptance criterion is committed in bench/acceptance.toml before the first run.

STAGE 0: honest reference verdicts (before any method run; about 3 h)
1. verify_data.py: sha256 of all 13 input sets, about 40 min.
2. labels.py: label-frame repair for the seven 0.12 mm subjects. stage0_rescore.py re-scores the stored v1 structural and final transforms of all 9 DANDI subjects (no re-registration), about 2 h.
3. Pre-declared verdict rule:
   - CORRECT if every available label metric (Dice_WM and Dice_GM on labelled voxels; BA-inside fraction; MRI-vessel median) is beyond the 99th percentile of the random-admissible-pose null in the right direction.
   - WRONG if every available metric lies inside the null [p5, p95].
   - Otherwise UNDECIDABLE.
   - Yuntian signs off each overlay in signoff.csv. A disagreement makes the pair UNDECIDABLE.
4. Output: the reference table (confirmed / wrong / undecidable). Every later criterion refers to it.
5. Checks to report: the header-only estimates (I56 block 0.4 mm from its BA box; I48 about 44 mm from its vessel box).

STAGE 1: CPU tests pass (tests/, < 5 min).

STAGE 2: regression ladder (attributable migration; about 10 h)
- R0 compat-all-v1: new package with all Compat switches set to v1, prepped from raw. Run on I46, I55, I38 and the brainstem (v1 intensity-mask structural target).
  - Pass: mean mask-point displacement <= 0.5 mm vs the stored v1 structural T (brainstem <= 1.0 mm), and the same hypothesis.
  - A failure here is an implementation bug and blocks everything after it.
- Then flip one switch per commit, each run on I46, I55 and the brainstem, reusing caches where the switch is downstream:
  - M1 header frames;
  - M2 section gain;
  - M3 OCT flattening;
  - M4 unmasked OCT channels + sign polarity;
  - M5 per-hypothesis retention + selection by L;
  - M6 h-rule levels, also on I38, I56, I62;
  - M7 symmetric foreground cleanup without largest component, also on I56 and I57;
  - M8 rho 0.8 instead of 0.85.
- Rung tolerance vs the previous rung: I46/I55 <= 0.5 mm and Dice_lab within 0.01; brainstem <= 1.0 mm; 0.12 mm subjects <= 1.0 mm.
- A failed rung reverts that switch to v1 behaviour, recorded in docs/decisions.md. Later rungs continue.

STAGE 3: release-candidate benchmark (13 runs, about 8 h)
- 10 core pairs plus 3 declared variants.
- Primary runs use sidecar spacing and the v1 MRI volume choice. Variants change exactly one input fact.
- Metrics:
  - Label-based (DANDI, after repair): Dice on labelled voxels plus confusion; BA-inside plus centroid distance; MRI vessels vs evaluation-only OCT vessel map; I46 ves_seg at offsets 0 and 66; I46 oracle displacement; depth scale vs metadata. Each with its null percentile and the ±3 mm control.
  - Label-free (all pairs): the 4 L values, z for polarity / handedness / runner-up, B, decisiveness, overlap, flags, scale axes, boundary agreement (brainstem only), status.
  - Cost: runtime per stage, peak RAM and GPU.
  - Frames: check_frames.py on the brainstem, Spearman at the pose vs each header-axis flip and a 2 mm shift.
  - Brainstem competitor: competitor.py refines from the REPORT C.4 mirrored 0.1358 pose (T from the C.4 run directory; if it is not stored, report the nearest surviving mirrored candidate within 5 mm). Reports L vs R5 with the arithmetic: if the 5.4 % shrink is per axis, the prior adds about 0.018 and R5 wins; if it is volumetric, about 0.002 and the competitor wins.
- QC calibration table: status × Stage-0 verdict (2×2). Descriptive, n <= 9; never used to tune anything.

STAGE 4: synthetic benchmark (bench/synth.py, about 5 h)
- Blocks cut from I46, I55 and brainstem MRI two-class maps.
- Degradations: block edge 5/10/20 mm; random pose, handedness and polarity; ±15 % spacing error per axis; bias; stripes on a random axis at 0.2-0.5 mm period; speckle; embedding shell. 50 trials per condition, search restricted to ±30 mm of truth.
- Controls:
  - mirror-symmetric blocks (symmetrised maps) must come out undecided for handedness;
  - featureless blocks (constant maps) must come out undecided for polarity;
  - asymmetric blocks must come out decided.
- The only use of synthetic data is to measure capture range and whether the decision rule is calibrated. No accuracy claims.

STAGE 5: ablations (config diffs; about 20 h)
- On {I46, I55, I38, I56, I62, brainstem}, all deletion candidates:
  - A1 OCT flattening off;
  - A2 MRI flattening off;
  - A3 section gain off;
  - A6 direct affine at h from the search pose (no ladder).
- A4/A5 proper-only and fixed polarity: free, read from the hypothesis table.
- A7: λ {0.5, 1, 4} × clamp {0.10, 0.30} on I46, I55, brainstem.
- A8: N_rot {2000, 4000, 16000}; search level 6h; K {4, 24}, on I46, I55, brainstem.
- A9: oracle foreground (whole-hemisphere label dilated 1 mm as the MRI mask) on I56, I57, I61, DANDI I58. Separates mask failures from objective failures.
- A10: brainstem with the archived texture mask supplied as --oct-mask.
- A11: jackknife sensitivity, target block count {8, 27, 64} and z {2, 3, 4}. Reported only.
- A12: rotation-set seeds 1-2 on I46, I55, brainstem (reproducibility spread).

STAGE 6: baselines (about 15 h)
- B1: intensity features in our search and refine, all pairs.
- B2: MIND-SSC features in our search and refine, all pairs.
- B3: ANTs antsAI plus antsRegistration Rigid→Affine (Mattes MI 0.6/0.3/0.15 mm, same foreground masks), on an MRI cropped to 30 mm around the reference block centre (a prior-given comparison). Run it next to octreg with --init on the same crop. Whole-hemisphere ANTs on I46 and I55 only.
- B4: the authors' I46 affine as-is (23 mm discrepancy reported).
- B5: v1.1 as shipped (stored archive runs, including I46 vascular ves_seg 167 um).

ACCEPTANCE (pre-declared; outcomes are required only where ground truth exists)
- R1 I46:
  - Dice_lab WM/GM >= Stage-0 v1 structural − 0.02;
  - pose <= 1.0 mm mean to v1 structural;
  - ves_seg median <= 1.1 × v1 structural (about 265-276 um);
  - chosen polarity and handedness equal the label-verified ones.
  - Whether they come out decided is reported. If undecided, the docs state that the rule is conservative; the rule is not retuned.
  - The 167 um vascular result is explicitly not a release target.
- R2 I55: Dice_lab >= v1 − 0.02; pose <= 1.0 mm; correct polarity and handedness.
- R3 all other DANDI pairs: no Stage-0 CORRECT pair becomes WRONG. Any pair that moves > 1.0 mm from v1 gets re-verdicted with the same rule.
- R4 brainstem:
  - mean displacement <= 1.5 mm to v1 structural or to R5 (the prep-variant spread is 10.6-20 deg);
  - all 4 hypotheses exported;
  - competitor check reported;
  - frame test Spearman >= 0.9 at the pose and <= 0.3 for every flip.
- R5 known failures (I48, I57, I61, DANDI I58 at 10 deg): no success required. All appear in the main table with their A9 analysis.
- R6 synthetic, non-symmetric blocks >= 10 mm: >= 95 % within 0.5 mm with the correct hypothesis; decided-and-wrong <= 2 %; symmetric/featureless controls >= 90 % undecided. If this fails, the z threshold may be changed using synthetic data only, and that is documented.
- R7: rerunning the tag from raw on the same GPU gives <= 0.01 mm max displacement and the same hypothesis on every pair.
- R8: per-pair registration time <= 1.2 × the v1 structural-stage time (both polarities); peak RAM <= 40 GB; GPU <= 24 GB; core benchmark <= 8 h.
- R9 deletion: if A1, A3 or A6 meets R1-R4, that step is deleted. Fewer steps wins ties.
- R10: zero per-pair Params diffs in the primary runs.
- R11: tests pass on CPU in < 5 min.

SERIAL ORDER AND TIME
| Step | Estimate |
|---|---|
| sha256 | 0.7 h |
| Stage 0 | 2 h |
| R0 | 3 h |
| M1-M8 | 7 h |
| Stage 3 | 8 h |
| Evaluation + frames + competitor | 2 h |
| Stage 4 | 5 h |
| Stage 5 | 20 h |
| Stage 6 | 15 h |
| Total | about 63 h serial |

Basis for the estimates:
- v1 prep took 8,530 s in total, including 60-800 s of label mapping per pair and 25-650 s of Frangi, both removed.
- v1 structural took 388-1,167 s per polarity; the new code runs one search pass plus 4 short ladders.

Two-week schedule:
| Days | Work |
|---|---|
| 1-2 | Stage 0 and package skeleton |
| 2-6 | Work packages 1-4 |
| 6-8 | Ladder |
| 8-9 | Stage 3 |
| 9-13 | Stages 4-6 |
| 12-14 | Docs, figures, tag v1.0.0 |


# docs_spec

All documentation is in English markdown, written plainly in the author's voice, with no process narrative. Every table and figure is generated from benchmark run directories by bench/report.py or bench/figures.py, and every caption carries the git hash of the run that produced it.

- README.md (1-2 screens)
  - What it does and does not claim: mm-level label-free affine placement; handedness and polarity reported with their evidence; no vessel-level or non-rigid accuracy.
  - Install; the register command and `octreg apply`; the output files.
  - How to read status and flags, with the next action for each (e.g. foreground_no_valley: inspect the overlay, supply --mri-mask; ambiguous: see alternatives/, decide with landmarks or header provenance).
  - Limitations in five sentences; how to reproduce the benchmark (`python -m bench.run --all`); citation.
  - A short Chinese summary at the end.
- docs/METHOD.md (the paper Methods section, about 1,800 words)
  - 2.1 Inputs and frames.
  - 2.2 Normalisation: foreground rule with its failure flag; sectioning gain; flattening.
  - 2.3 Two-class representation and objective, with a three-line derivation that swapping the OCT classes gives exactly −S, and why the MRI channels keep the foreground factor (outline).
  - 2.4 Exhaustive FFT search over orientation × handedness × polarity with the achievable-overlap rule.
  - 2.5 Prior-bounded affine ladder.
  - 2.6 Decisions: penalised-loss selection, paired block jackknife, scale identifiability.
  - 2.7 Reported quantities and why none is a correctness gate, with the counterexamples: restarts 12/12 on the wrong I61; NCC 0.50 wrong vs 0.28 correct; I55 margin 1.004.
  - Table 1: all parameters with value, unit, role and provenance (generated from params.py).
- docs/conventions.md: frames, handedness ('unknown' for header-less stacks), LTA/ITK, how to map labels with `octreg apply`.
- docs/statuses.md: every flag and status, what triggers it, what the user should do.
- docs/benchmark.md (generated)
  - Table 2: pairs (source, dandiset version, sha256, spacings and their sources, MRI flip angle, block volume and fraction of hemisphere, annotations).
  - Table 3: main results for all 13 runs, failures in the same table: Stage-0 verdict, label metrics with null percentiles, 4-hypothesis L, z values, flags, status, runtime, memory.
  - Table 4: regression ladder outcomes.
  - Table 5: ablations, including the deletion decisions.
  - Table 6: baselines.
  - Table 7: synthetic capture and decision calibration.
  - Table 8: status × verdict.
- docs/data_cards.md: one card per pair.
  - The I46 depth sidecar is 12 µm while the evidence suggests about 14; the declared variant tests it.
  - OME units are 'pixel' for I46/I58/I61/I62; I48/I56/I57 are plain TIFF; the I48 MRI sidecar is missing; I58 flip-1 is 30 deg, flip-3 10 deg.
  - Label-frame repair; ves_seg crop status.
  - Xiangrui pair provenance and the open header, landmark and crop questions.
- docs/negative_results.md (supplement, each with its numbers): fine stage; vascular channel (I46 265→167 um as n=1, silent at 30-35 µm, harmful on the brainstem, plus the variant outcome); texture specimen mask; parser and self-distillation; nonrigid; MIND and raw intensity; surface distance and MI objectives; exclusion and renormalisation; restarts, absolute NCC and boundary agreement as QC. Framed as "self-consistency is not accuracy".
- docs/decisions.md: component table (keep / merge / replace / remove / archive with evidence) and the migration switch outcomes, each linked to its v1.1-archive path.
- docs/archive/: HANDOFF.md, REPORT_I58_v11.md, probe JSONs, runs_summary.md, unchanged.
- CHANGELOG.md, CITATION.cff, LICENSE.

FIGURES (vector PDF + PNG, colour-blind-safe, scale bars)
1. Pipeline schematic with I55 thumbnails and the parameter of each step.
2. Representation on I46 and the brainstem (raw, flattened, p, channels); shows the outline-dominated brainstem channel.
3. Search and decisions: score distributions per hypothesis, the S/−S duality, per-pair L of the 4 hypotheses with jackknife σ and decided marks.
4. Overlay montage of all 13 runs, failures included, each marked with Stage-0 verdict and human sign-off.
5. Label accuracy vs the random-pose null (I46, I55, re-scored DANDI) with baselines.
6. Synthetic capture and accuracy vs block size and spacing error, plus decision calibration on symmetric controls.
7. Ablation and sensitivity heatmap.
8. Failure anatomy: I61, I56 and DANDI I58 foreground histograms; A9 oracle-mask outcome; I58 at 10 vs 30 deg; I57 two-piece block; brainstem handedness pair and competitor.
Supplementary: runtime and memory per stage.

REPRODUCIBILITY: pinned environment lock, sha256 manifest, dandiset version, git hash and argv in every result.json, and one command for the whole benchmark. Transforms and evaluation JSONs are published with the release.

LIMITATIONS (stated in METHOD and README)
- Affine only.
- Two-class contrast assumption.
- Foreground failures on bath-heavy MRI are flagged, not fixed.
- Physical handedness is unknowable for frame-less stacks.
- Defaults validated only on human cortex and brainstem (12-35 µm OCT, 0.08-0.15 mm MRI).
- Metadata errors beyond about 15 % saturate the clamp.
- Prior-free claims apply to DANDI whole hemispheres only; the brainstem MRI is already a crop.
- n <= 9 labelled pairs.


# implementation_plan

Owners are agents working on disjoint file sets. Interfaces are fixed in WP0 so WP1-WP3 and WP5 can run in parallel. Server compute stays strictly serial.

WP0 Skeleton and archive (owner A; must finish first, about 1 day)
- Files: git tag v1.1-archive at db2282e; branch release/1.0; git rm of all old octreg/*.py, scripts/*, results/runs_summary.md; move HANDOFF.md, docs/v11/* and the old README to docs/archive/; create pyproject.toml, environment.lock.yml, .github/workflows/tests.yml, octreg/params.py, octreg/types.py, README.md placeholder.
- params.py holds every constant from method_spec plus Compat switches M1-M8, and types.py holds the dataclasses and signatures from code_spec.
- Tests: tests/test_params.py (hash stable; diff_from_default).
- Done: the tag exists on the local repo and is pushed if a remote exists; CI runs an empty suite; the signatures are reviewed by the other owners.

WP1 Frames and geometry (owner B)
- Files: octreg/io.py, octreg/geometry.py, tests/test_frames.py, tests/test_geometry.py.
- Port from the tag with `git show v1.1-archive:octreg/common.py`.
- Compat M1: io.load_volume(layout='SPR') path.
- Done:
  - oblique det<0 round trip < 1e-6 mm;
  - mirrored pose_distance correct;
  - resample matches the v1 pooling on a 12 µm test array;
  - LTA readable by FreeSurfer conventions (lta_text parity with v1).

WP2 Normalisation and features (owner C)
- Files: octreg/normalize.py, octreg/features.py, tests/test_normalize.py, tests/test_features.py.
- Compat branches:
  - M2: v1 axis-0 slab window round(1200/sp);
  - M3: OCT flattening off;
  - M4: masked OCT channels;
  - M7: v1 OCT closing 2 iterations + largest component, MRI raw threshold.
- Done:
  - synthetic stripe detection and no-op tests pass;
  - foreground status tests pass;
  - on the server, the I46 detector reproduces axis 0 / 0.39 mm and the brainstem axis 2 / 0.30 mm (read-only-sized job: one plane sample).

WP3 Search, refine, decide (owner D)
- Files: octreg/search.py, octreg/refine.py, octreg/decide.py, tests/test_duality.py, tests/test_search_synthetic.py, tests/test_refine_grad.py, tests/test_decide.py.
- Until WP1 lands, work against the WP0 signatures with a local test fixture.
- Compat branches: M5 v1 joint top-24 and keep 8/3/3, selection as in v1; M6 v1 levels rounded to the MRI grid; M8 rho 0.85 cap.
- First task: read `git show v1.1-archive:scripts/register.py` lines 193-213 and record whether v1 compared data or penalised loss. This sets M5's compat behaviour.
- Done: duality to 1e-5; synthetic recovery; jackknife moment-sum equals brute force; symmetric phantom gives undecided.

WP4 Pipeline, QC, CLI (owner B after WP1)
- Files: octreg/pipeline.py, octreg/qc.py, octreg/cli.py, octreg/__init__.py, octreg/__main__.py, tests/test_cli.py.
- Done:
  - tiny CPU pair runs end to end;
  - result.json validates against the schema;
  - cache reuse is verified (second run skips steps 0-5);
  - per-stage timings and peak memory recorded;
  - `octreg apply` reproduces oct_in_mri.nii.gz to NCC > 0.9999.

WP5 Evaluation and Stage 0 (owner E; starts on day 1 in parallel)
- Files: bench/pairs/*.toml, bench/verify_data.py, bench/labels.py, bench/evaluate.py, bench/null.py, bench/stage0_rescore.py, bench/signoff.csv, tests/test_bench_labels.py (synthetic permuted and flipped headers).
- Uses numpy/scipy only until WP1 lands, then imports octreg.geometry.pose_distance.
- Done:
  - the sha256 manifest verifies;
  - the repaired mapping keeps >= 99 % of labels on tissue for I38 (vs 65.7 % before);
  - the Stage-0 table exists with null percentiles and human sign-off.
- Stage 0 is the first server job.

WP6 Benchmark runner, ladder, experiments, reports (owner F; after WP4 and WP5)
- Files: bench/run.py, bench/compare.py, bench/acceptance.toml, bench/ablations.toml, bench/synth.py, bench/check_frames.py, bench/competitor.py, bench/baselines/*, bench/report.py, bench/figures.py.
- acceptance.toml and ablations.toml are committed before any Stage-2 run (pre-registration).
- Done:
  - watchdog tested with an RSS stub;
  - R0 passes;
  - the ladder table is produced;
  - Stage 3-6 outputs are complete;
  - docs/benchmark.md and figures/ regenerate from outputs with one command.

WP7 Migration closure (owners C and D on their own files, plus A on params.py; after the ladder)
- Remove every Compat switch: hard-code the passing behaviour, or the v1 behaviour where a rung failed.
- Apply the R9 deletions (A1/A3/A6 outcomes).
- Delete compat tests.
- Done:
  - package <= 1,600 lines;
  - no Compat class;
  - tests green;
  - R7 reproducibility rerun on I46 and the brainstem passes at the release-candidate tag.

WP8 Documentation (owner G; drafting from day 6, finalised after Stage 6)
- Files: README.md, docs/METHOD.md, docs/conventions.md, docs/statuses.md, docs/data_cards.md, docs/negative_results.md, docs/decisions.md, CHANGELOG.md, CITATION.cff, LICENSE.
- Done:
  - every number in METHOD, README and negative_results traces to a result.json at the tag or to a v1.1-archive path;
  - Yuntian has reviewed the text;
  - tag v1.0.0.

DEPENDENCIES: WP0 → (WP1, WP2, WP3, WP5 in parallel) → WP4 → WP6 ladder → WP7 → WP6 Stages 3-6 → WP8 final.

RULES FOR ALL WORK PACKAGES
- No new constant enters params.py without a unit, a provenance, and either an a-priori justification or an ablation on >= 3 pairs.
- No per-pair Params.
- Never run two heavy server jobs at once.
- Never copy volumes or caches to the Mac; local scratch stays under 20 MB.
- Server disk cleanup of the old parser, probe and v11_dev directories only after the benchmark and with the user's approval.


# open_decisions

- Primary MRI volume for DANDI I58. The 10 deg flip-3 is the v1 row and the known low-contrast failure; the 30 deg flip-1 is equally far from 20 deg. The spec keeps 10 deg as primary and 30 deg as a declared variant. Alternative: a general rule 'use the EPIC volume with flip angle closest to the FLASH GM/WM contrast optimum' decided before the runs.
- Whether to download and add I45, I59 and I60 (about 9 GB, about 3 h of compute). I45 and I60 carry OCT ves_seg and are the only way to meet the vessel re-entry bar (a gain on at least 2 labelled vessel pairs). Without them, vessels stay in the negative results.
- What v1 register.py compared when choosing polarity: refined data loss or penalised loss. refine.py returns the best data loss. This decides M5's compat behaviour and how the v1 references are interpreted; WP3 checks it first.
- Where the REPORT C.4 mirrored 0.1358 brainstem pose is stored, and whether its 5.4 % shrink is per axis or volumetric. That changes whether penalised-loss selection keeps R5 (about 0.018 prior cost) or the competitor (about 0.002).
- What to ask Xiangrui for: 10-20 landmark pairs, the uncropped MRI or crop offsets, the optical-depth axis, and confirmation that the OCT NIfTI header orientation is physical. Without these the brainstem handedness stays undecidable and the pair gets label-free metrics only.
- What the paper's scientific endpoint is (PI question Q6): mm-level block placement, or sub-mm accuracy on cortex. This sets whether the vessel extension or a correct-metadata study is in scope for v1.0, or deferred.
- Fallback if the paired block jackknife fails the synthetic calibration (R6), for example because spatial correlation makes sigma too small. Options: recalibrate z on synthetic data only, use a larger block size, or report z without an ambiguous status. It must never be tuned on the real pairs.
- Baseline scope: whole-hemisphere ANTs on only I46/I55 versus all 9 DANDI pairs (cost, possible failure to converge), and whether to add NiftyReg reg_aladin or MIND+ConvexAdam for reviewers.
- Release venue and licence: keep the release/1.0 branch in the octreg repository or create a new public repository, choose a licence, and decide where to publish the benchmark transforms and evaluation JSONs (e.g. Zenodo with the pinned DANDI version).
- Whether the archived texture specimen recipe is offered as a documented bench script for agarose-embedded OCT (user-supplied mask path), depending on the A10 brainstem outcome.
- Handling of I57 and I61 label flips if label-on-tissue fraction does not separate the 8 flip variants clearly. Such a pair is marked UNDECIDABLE in Stage 0 rather than forcing a choice.
- Server cleanup after the benchmark (parser dirs about 30 G, probe_xr, v11_dev 38 G, the corrupt I58 backup). This needs the user's explicit approval.
