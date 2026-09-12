#!/usr/bin/env bash
# Feature ablations on the crop scenario (same search/refinement machinery), sequential.
set -e
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh && conda activate octmri
for feat in parser otsu mind intensity; do
  python octreg/scripts/run_demo_i46.py --work work/i46 --parser work/parser_v1 --out work/runs/crop_${feat}_v2 --scenario crop --features $feat --n-rot 4000 --topk 24 --n-restarts 8 --ref-transform work/i46/ref_author_T.json > logs/run_crop_${feat}_v2.log 2>&1 || echo "FAILED $feat"
done
python octreg/scripts/run_demo_i46.py --work work/i46 --parser work/parser_v1 --out work/runs/crop_mixed_nomirror --scenario crop --features mixed --no-mirror --n-rot 4000 --topk 24 --n-restarts 8 --ref-transform work/i46/ref_author_T.json > logs/run_crop_mixed_nomirror.log 2>&1 || echo "FAILED nomirror"
python octreg/scripts/summarize_runs.py work/runs/crop_mixed_v2 work/runs/whole_mixed_v2 work/runs/crop_parser_v2 work/runs/crop_otsu_v2 work/runs/crop_mind_v2 work/runs/crop_intensity_v2 work/runs/crop_mixed_nomirror work/runs/crop_parser_v1 work/runs/crop_parser_vessel_v1 > work/runs/summary.md 2>&1
echo ABLATIONS_DONE
