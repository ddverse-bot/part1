#!/bin/bash
# octreg v1.1b run plan, conservative-polish test (REPORT_I58_v11 section 6 "What R10 means" and section 7): the fine stage
# rerun as a fixed-point-set polish with small restarts, first on the labelled cortex pair I46 (the calibration set: it must
# leave regress.py's fine tolerances intact against the v1.1 default run), then on I58 from the structural pose P_R5.
# Strictly serial, one GPU job at a time; a failed step is logged and the chain continues (no `set -e`).
# usage: bash octreg/scripts/run_v11_polish.sh [R12 R13 R14]
#   R12  labelled-cortex calibration (GPU ~25 min each): I46 and I55, a = affine polish, b = rigid polish; each vs work/runs/v11/<SUB>_v11 with --tol fine
#   R13  I58 polish from P_R5 (GPU ~20 min each, --fine-verify full) with the same two variants + FOV rind exclusion, then qc_fine
#   R14  summary table
set -u
cd /root/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
S=octreg/scripts; RUNS=work/runs/v11
mkdir -p logs $RUNS
CHAIN=logs/v11_chain_polish.log
step() {  # step NAME cmd...  -> logs/v11_NAME.log; never aborts the chain
  local name=$1; shift
  echo "[$name start] $(date '+%F %T')  $*" | tee -a $CHAIN
  "$@" > logs/v11_$name.log 2>&1; local rc=$?
  echo "[$name done rc=$rc] $(date '+%F %T')" | tee -a $CHAIN
  [ $rc -ne 0 ] && tail -5 logs/v11_$name.log | tee -a $CHAIN
  return 0
}
need_run() { if [ ! -f "$2/result.json" ]; then echo "[$1 skipped: no $2/result.json]" | tee -a $CHAIN; return 1; fi; return 0; }
fine_report() {  # fine_report RUN_DIR: the fine gate numbers of a run, appended to the chain log
  python - "$1" <<'PY' 2>&1 | tee -a $CHAIN
import json, sys
r = json.load(open(sys.argv[1] + "/result.json")); f = r.get("fine") or {}; rs = f.get("restarts") or {}
print(f"[fine {sys.argv[1]}] accepted={f.get('accepted')} {f.get('reasons')}: U_mm {f.get('U_mm')}, moved {(f.get('vs_start') or {}).get('block_corner_mean_mm')} mm (block), "
      f"rotated {(f.get('vs_start') or {}).get('delta_rotation_deg')} deg, restarts {rs.get('n_converged')}/{rs.get('n')} (tol {rs.get('tol_mm')} mm, within {rs.get('n_within_mm')}), "
      f"options {f.get('options')}, levels {[(e.get('level'), e.get('sign'), e.get('sim')) for e in f.get('levels', [])]}")
PY
}
# the conservative polish: fixed OCT point set and frozen tissue weight (P7), restarts 1 mm / 2 deg (the measured rigid capture radius)
POLISH="--fine on --fine-fixed-mask on --fine-fixed-weight on --fine-restart-mm 1 --fine-restart-deg 2"
STEPS=${*:-"R12 R13 R14"}
for R in $STEPS; do case $R in
R12)  # labelled-cortex calibration on I46 and I55: a = affine DOF (v1.1 schedule), b = rigid DOF; each vs <SUB>_v11 with regress.py --tol fine
  # Reading rule (REPORT_I58_v11 section 6): a variant is harmless only if, on BOTH subjects, it is either rejected by its own gates or accepted with
  # every accuracy row (F2-F5: pose move, Dice, vessel medians / f150, depth stretch) inside the fine tolerances; F6 (wall time) is not an accuracy row.
  for SUB in I46 I55; do
    for V in a:affine b:rigid; do D=${V%%:*}; DOF=${V##*:}
      step R12_${SUB}_polish_$D  python $S/register.py --work work/$SUB --out $RUNS/${SUB}_v11_polish_$D --oct-wm-bright auto $POLISH --fine-dof $DOF
      need_run R12_${SUB}_$D $RUNS/${SUB}_v11_polish_$D && fine_report $RUNS/${SUB}_v11_polish_$D
      need_run R12_${SUB}_${D}_regress $RUNS/${SUB}_v11 && need_run R12_${SUB}_${D}_regress $RUNS/${SUB}_v11_polish_$D && \
        step R12_${SUB}_polish_${D}_regress python $S/regress.py $RUNS/${SUB}_v11 $RUNS/${SUB}_v11_polish_$D --tol fine
    done
  done ;;
R13)  # I58 from P_R5 (structural pose of xiangrui_I58bs), vascular off, FOV rind 1 mm excluded, full verification, then qc_fine
  T0=$RUNS/xiangrui_I58bs/T_oct2mri_structural.npy
  for V in a:affine b:rigid; do D=${V%%:*}; DOF=${V##*:}
    step R13_I58_polish_$D python $S/register.py --work work/v11/xiangrui_I58bs --out $RUNS/I58_polish_$D --oct-wm-bright auto --init-transform $T0 --vascular off \
         $POLISH --fine-dof $DOF --fine-exclude-fov-mm 1.0 --fine-verify full
    need_run R13_I58_$D $RUNS/I58_polish_$D && fine_report $RUNS/I58_polish_$D
    step R13_I58_polish_${D}_qc python $S/qc_fine.py --work work/v11/xiangrui_I58bs --run $RUNS/I58_polish_$D
  done ;;
R14) step R14_summary python $S/summarize.py --runs $RUNS --out $RUNS/summary ;;
*) echo "unknown step $R" | tee -a $CHAIN ;;
esac; done
echo "[run_v11_polish done] $(date)" | tee -a $CHAIN
