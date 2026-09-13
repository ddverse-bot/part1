# octreg v1.1 design record (2026-09-12/13)

Target: run the lab's own I58 brainstem pair (Xiangrui) end to end and make the method handle
agarose embedding, section-seam striping and non-cortical tissue, without changing results on the
cortical DANDI subjects.

- `spec_v11.json` — the binding specification (keys: spec, ownership, run_plan, acceptance, open_questions).
- `probe_P1_specimen_mask.json` — agarose vs tissue: intensity cannot separate them (old mask 28.9 cm3 vs 13.3 cm3 MRI tissue, 2.17x); winning recipe = min-over-axes directional texture at 0.04 mm -> GMM threshold -> rim watershed, 18.05 cm3 (1.36x).
- `probe_P2_similarity.json` — fine-stage landscape at 0.3 mm around the v1 pose: the OCT(mu_s) <-> MRI intensity relation is INVERTED (raw NCC -0.23, sharp V minimum at T); winner = sign-corrected 3 mm-flattened masked NCC (contrast 0.54); MIND/NGF useless at this scale.
- `probe_P3_stripes.json` — striping = planes of constant array axis 2 (x), period 7.5 vox @0.04 mm = 0.30 mm = 15 slices @20 um, sawtooth: array axis 2 is the physical sectioning axis (v1 assumed axis 0); destripe = normalised-convolution flat field along that axis; 3-D Frangi vessel channel not salvageable on this data.
- `probe_P4_code_map.json` — intervention map, data contract, fragilities.
- `judge_*.json` — the design panel's verdicts. `probe_scripts/` — the validated probe code the implementation ports. `figs/` — key evidence figures.
