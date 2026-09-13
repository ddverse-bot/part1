#!/bin/bash
# octreg v1.1 run plan, DANDI regression part (spec run_plan R8-R10 + R11): v1 snapshot baseline vs v1.1 defaults on I46 / I55,
# then the fine-gate test on I46.  Strictly serial, one job at a time; a failed step is logged and the chain continues (no `set -e`).
# usage: bash octreg/scripts/run_v11_regress.sh [R8 R9 R10 R11]      env V1_DIR = the v1 snapshot tree (default ../oct-mri-registration_v1)
#
# PREREQUISITE (R8, done by the main loop, not by this script): the v1 snapshot = the code at git commit 7ef90e5, laid out like the
# live tree ($V1_DIR/octreg/scripts/register.py + $V1_DIR/octreg/octreg/*.py).  From the local git repo:
#   git -C /Users/yyttim/workspace/lsfm-project/octreg archive 7ef90e5 | ssh -p 49018 root@connect.westb.seetacloud.com \
#       "mkdir -p /root/autodl-tmp/oct-mri-registration_v1/octreg && tar -x -C /root/autodl-tmp/oct-mri-registration_v1/octreg"
# Without it R8 is skipped, R9's regress steps are skipped (no reference), and 5.3 cannot be evaluated.
#
# Vascular gate on DANDI: v1's accepted I46 solution drops the structural [WM,GM] NCC by 0.027 (I55: 0.004); the v1.1 default
# --vascular-struct-drop is 0.05 (spec said 0.03: a 0.003 margin).  The R9 gate step prints the measured margin from result.json;
# an I46 REJECTION in R9 is a gate-calibration event (re-examine the threshold with the numbers), not a code regression.
set -u
cd /root/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
S=octreg/scripts; RUNS=work/runs/v11; V1_DIR=${V1_DIR:-/root/autodl-tmp/oct-mri-registration_v1}
mkdir -p logs $RUNS
CHAIN=logs/v11_chain_regress.log
step() {  # step NAME cmd...  -> logs/v11_NAME.log; never aborts the chain
  local name=$1; shift
  echo "[$name start] $(date '+%F %T')  $*" | tee -a $CHAIN
  "$@" > logs/v11_$name.log 2>&1; local rc=$?
  echo "[$name done rc=$rc] $(date '+%F %T')" | tee -a $CHAIN
  [ $rc -ne 0 ] && tail -5 logs/v11_$name.log | tee -a $CHAIN
  return 0
}
need_run() {  # need_run STEP DIR: true iff DIR/result.json exists, else log a skip line
  if [ ! -f "$2/result.json" ]; then echo "[$1 skipped: no $2/result.json]" | tee -a $CHAIN; return 1; fi; return 0
}
gate_report() {  # gate_report RUN_DIR: the vascular / fine gate numbers of a run (margins to the thresholds), appended to the chain log
  python - "$1" <<'EOF' 2>&1 | tee -a $CHAIN
import json, sys
r = json.load(open(sys.argv[1] + "/result.json")); v = r.get("vascular") or {}; f = r.get("fine") or {}; a = r.get("args", {})
if "accepted" in v:
    print(f"[gate {sys.argv[1]}] vascular accepted={v['accepted']} {v.get('reasons')}: vessel NCC {v['ncc_channels'][2]:.4f} (min {a.get('vascular_min_ncc')}), "
          f"structural NCC {v['struct_ncc_before']:.4f} -> {v['struct_ncc_after']:.4f} drop {v.get('struct_drop', v['struct_ncc_before'] - v['struct_ncc_after']):.4f} "
          f"(max {a.get('vascular_struct_drop')}, margin {v.get('struct_drop_margin', float('nan')):+.4f}), moved {v['vs_structural']['corner_mean_mm']:.3f} mm (7 mm) / "
          f"{(v.get('vs_structural_block') or {}).get('block_corner_mean_mm')} mm (block corners, max {a.get('vascular_max_move')})")
else: print(f"[gate {sys.argv[1]}] vascular (no gate recorded): mode {v.get('mode')}, ncc_channels {v.get('ncc_channels')}, moved {(v.get('vs_structural') or {}).get('corner_mean_mm')} mm (7 mm)")
if f: print(f"[gate {sys.argv[1]}] fine accepted={f.get('accepted')} {f.get('reasons')}: U_mm {f.get('U_mm')}, moved {(f.get('vs_start') or {}).get('block_corner_mean_mm')} mm (block), rotated {(f.get('vs_start') or {}).get('delta_rotation_deg')} deg")
EOF
}
STEPS=${*:-"R8 R9 R10 R11"}
for R in $STEPS; do case $R in
R8)  # GPU ~85 min: DANDI baseline on the v1 snapshot (stored runs predate the current code -> re-measure)
  if [ ! -f $V1_DIR/octreg/scripts/register.py ]; then
    echo "[R8 skipped: v1 snapshot not found at $V1_DIR/octreg/scripts/register.py -- create it from git commit 7ef90e5 (see the header of this script) or set V1_DIR; without it R9 has no reference and 5.3 cannot be evaluated]" | tee -a $CHAIN; continue; fi
  for SUB in I46 I55; do
    step R8_${SUB}_v1snap  bash -c "cd $V1_DIR && python octreg/scripts/register.py --work $PWD/work/$SUB --out $PWD/$RUNS/${SUB}_v1snap --oct-wm-bright auto"
    step R8_${SUB}_v1snap_viz bash -c "cd $V1_DIR && python octreg/scripts/viz_result.py --work $PWD/work/$SUB --run $PWD/$RUNS/${SUB}_v1snap"
  done ;;
R9)  # GPU ~85 min: DANDI with v1.1 defaults (prep untouched, no re-prep) + gate margins + regression check against the v1 snapshot
  for SUB in I46 I55; do
    step R9_${SUB}_v11     python $S/register.py --work work/$SUB --out $RUNS/${SUB}_v11 --oct-wm-bright auto
    need_run R9_${SUB}_gate $RUNS/${SUB}_v11 && gate_report $RUNS/${SUB}_v11
    step R9_${SUB}_v11_viz python $S/viz_result.py --work work/$SUB --run $RUNS/${SUB}_v11
    need_run R9_${SUB}_regress $RUNS/${SUB}_v1snap && need_run R9_${SUB}_regress $RUNS/${SUB}_v11 && step R9_${SUB}_regress python $S/regress.py $RUNS/${SUB}_v1snap $RUNS/${SUB}_v11
  done ;;
R10) # GPU ~50 min: DANDI fine-gate test (I46; I55 if time allows)
  for SUB in I46 I55; do
    step R10_${SUB}_fine    python $S/register.py --work work/$SUB --out $RUNS/${SUB}_v11_fine --oct-wm-bright auto --fine on
    need_run R10_${SUB}_gate $RUNS/${SUB}_v11_fine && gate_report $RUNS/${SUB}_v11_fine
    need_run R10_${SUB}_regress $RUNS/${SUB}_v11 && need_run R10_${SUB}_regress $RUNS/${SUB}_v11_fine && step R10_${SUB}_regress python $S/regress.py $RUNS/${SUB}_v11 $RUNS/${SUB}_v11_fine --tol fine
  done ;;
R11) # CPU 5 min: summary table
  step R11_summary       python $S/summarize.py --runs $RUNS --out $RUNS/summary ;;
*) echo "unknown step $R" | tee -a $CHAIN ;;
esac; done
echo "[run_v11_regress done] $(date)" | tee -a $CHAIN
