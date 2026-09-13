#!/bin/bash
# octreg v1.1b follow-up to run_v11_polish.sh (R15): score every fine-stage CANDIDATE pose of the labelled R12 runs with the manual labels,
# whether the gates accepted it or not, so the gate decision itself can be judged (did a rejection block a harmful or a helpful move?).
# Pure evaluation: register.py --init-transform <candidate> --vascular off --fine off skips the search and every refinement (polarity is
# evaluated, not optimised); the labelled metrics land in result.json evaluation.final.  Light GPU use, run after the polish chain.
# usage: bash octreg/scripts/run_v11_candeval.sh
set -u
cd /root/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
S=octreg/scripts; RUNS=work/runs/v11; CHAIN=logs/v11_chain_polish.log
while ! grep -q "run_v11_polish done" $CHAIN 2>/dev/null; do sleep 60; done
for SUB in I46 I55; do
  for D in a b; do R=$RUNS/${SUB}_v11_polish_$D
    [ -f $R/T_oct2mri_fine_candidate.npy ] || { echo "[R15 ${SUB}_$D skipped: no candidate]" | tee -a $CHAIN; continue; }
    echo "[R15_${SUB}_${D}_candeval start] $(date '+%F %T')" | tee -a $CHAIN
    python $S/register.py --work work/$SUB --out ${R}_candeval --oct-wm-bright auto --init-transform $R/T_oct2mri_fine_candidate.npy --vascular off --fine off > logs/v11_R15_${SUB}_${D}_candeval.log 2>&1
    echo "[R15_${SUB}_${D}_candeval done rc=$?] $(date '+%F %T')" | tee -a $CHAIN
  done
done
python - <<'PY' 2>&1 | tee -a $CHAIN
import json
from pathlib import Path
R = Path("work/runs/v11")
def m(p):
    e = json.load(open(p / "result.json"))["evaluation"]["final"]; g = e.get("gmwm", {}); o = (e.get("vessels_own") or {}).get("registered", {}); v = (e.get("vessels_vesseg") or {}).get("registered", {})
    return f"Dice WM {g.get('dice_WM', float('nan')):.4f} GM {g.get('dice_GM', float('nan')):.4f} | own {o.get('median_um', float('nan')):.1f} um f150 {o.get('frac_within_150um', float('nan')):.3f} | vesseg {v.get('median_um', float('nan')):.1f} um f150 {v.get('frac_within_150um', float('nan')):.3f} | stretch {[round(x, 3) for x in e.get('stretch_ijk', [])]}"
for sub in ("I46", "I55"):
    print(f"[R15 {sub}] v1.1 default     : {m(R / f'{sub}_v11')}")
    for d in ("a", "b"):
        p = R / f"{sub}_v11_polish_{d}"
        if (p / "result.json").exists():
            f = json.load(open(p / "result.json")).get("fine", {})
            print(f"[R15 {sub}] polish_{d} shipped : {m(p)}  (fine accepted={f.get('accepted')} {f.get('reasons')})")
        if (R / f"{sub}_v11_polish_{d}_candeval" / "result.json").exists(): print(f"[R15 {sub}] polish_{d} candidate: {m(R / f'{sub}_v11_polish_{d}_candeval')}")
PY
echo "[run_v11_candeval done] $(date)" | tee -a $CHAIN
