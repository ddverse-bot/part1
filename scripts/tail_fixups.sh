#!/bin/bash
# after the dandi serial: redo any subject missing result.json (e.g. I57 output-OOM), then final summary
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
until grep -q "\[dandi serial done\]" logs/recover2.log 2>/dev/null; do sleep 180; done
export OMP_NUM_THREADS=8
for pair in "I57 flip-4" "I61 flip-4" "I48 flip-2" "I58 flip-3" "I38 flip-2" "I55 flip-2" "I56 flip-2" "I62 flip-3"; do
  set -- $pair; SUB=$1
  if [ ! -f work/runs/${SUB}_otsu/result.json ] && [ -f work/$SUB/octv_vessels.npy ]; then
    echo "[fixup register $SUB] $(date)"
    python octreg/scripts/register.py --work work/$SUB --out work/runs/${SUB}_otsu --oct-wm-bright auto > logs/reg_${SUB}_otsu.log 2>&1 || echo "[fixup $SUB FAILED]"
    python octreg/scripts/viz_result.py --work work/$SUB --run work/runs/${SUB}_otsu >> logs/reg_${SUB}_otsu.log 2>&1
  fi
done
python octreg/scripts/summarize.py --pattern "*_otsu" > logs/summarize.log 2>&1
cp -r work/runs/summary_all work/runs/summary_final 2>/dev/null
echo "[tail fixups done] $(date)"
