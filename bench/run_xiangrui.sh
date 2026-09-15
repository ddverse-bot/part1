#!/bin/bash
# octreg 1.0 on Xiangrui's I58 brainstem pair (AutoDL server), from the two original files.
#
#   cd /root/autodl-tmp/octreg1 && setsid nohup bash bench/run_xiangrui.sh > /dev/null 2>&1 < /dev/null &
#   STEPS="register ablate evaluate report" setsid nohup bash bench/run_xiangrui.sh > /dev/null 2>&1 < /dev/null &
#
# STEPS   in order; default "register evaluate". The whole benchmark is "register ablate evaluate report".
#           register  python -m octreg register OCT MRI -o OUT
#           ablate    bench/ablate.py: preprocessing once, variants A0-A11b -> ABL/ablations.json (removed steps A3, A7 from PREV)
#           evaluate  bench/evaluate.py OUT: pose to R5 and to PREV_MAIN, boundary with the v1.1 masks, raw-data frame check;
#                     mask and boundary metrics with the method masks once ABL/prep/texture exists
#           report    bench/report.py -> bench/BENCHMARK.md, bench/figures/fig_qc_xiangrui.png and fig_ablation.png
# OUT     run dir (default /root/autodl-tmp/bench_runs/xiangrui_I58/rel4/main, the run in bench/results); logs go to ${OUT}_logs
# ABL     ablation dir (default /root/autodl-tmp/bench_runs/xiangrui_I58/rel4/ablate); DEVICE cuda | cpu (default cuda)
# PREV    ablations.json files of the runs that measured the removed steps (space-separated): A3 and A7
#         (default BENCH/ablate/ablations.json; missing files are skipped)
# PREV_MAIN  an earlier CLI run dir for evaluate's pose distance (default /root/autodl-tmp/bench_runs/xiangrui_I58/rel3/main,
#            the previous release run, with the two-sided outline)
# Every step writes NAME.log, NAME.time (wall time and peak RSS, GNU time wording) and NAME.gpu_mib (nvidia-smi every 5 s);
# chain.log has a start and a done line per step (and the last 5 log lines on failure). One heavy job at a time (62 GB
# container): refuses to start while another registration runs.
set -u
CODE=${CODE:-/root/autodl-tmp/octreg1}
DATA=/root/autodl-tmp/oct-mri-registration/data/xiangrui/OCT_to_MRI
OCT=$DATA/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz
MRI=$DATA/I58_brainstem_MRI_cropped_to_OCT.nii.gz
BENCH=/root/autodl-tmp/bench_runs/xiangrui_I58
OUT=${OUT:-$BENCH/rel4/main}
ABL=${ABL:-$BENCH/rel4/ablate}
PREV=${PREV:-$BENCH/ablate/ablations.json}
PREV_MAIN=${PREV_MAIN:-$BENCH/rel3/main}
LOGS=${OUT}_logs
STEPS=${STEPS:-register evaluate}
DEVICE=${DEVICE:-cuda}
KILL=/root/autodl-tmp/v11_dev/killable/octreg1_xiangrui.pgid

mkdir -p "$LOGS" "$(dirname "$KILL")"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/chain.log"; }
busy=$(ps -eo args= | grep -E "(^|/)python[0-9.]* .*(-m octreg|bench/(ablate|evaluate)\.py|prep_subject\.py|register\.py|qc_fine\.py)" | grep -vc "grep")
[ "$busy" -gt 0 ] && { say "another registration python is running ($busy): not starting"; exit 1; }
for f in "$OCT" "$MRI"; do [ -f "$f" ] || { say "missing input $f"; exit 1; }; done
ps -o pgid= -p $$ | tr -d ' ' > "$KILL"
trap 'rm -f "$KILL"' EXIT
source /root/miniconda3/etc/profile.d/conda.sh && conda activate octmri || { say "conda env octmri not available"; exit 1; }
export OMP_NUM_THREADS=8
cd "$CODE" || { say "no code copy at $CODE"; exit 1; }

step() {    # step NAME CMD...: output to NAME.log, wall time and peak RSS to NAME.time, GPU memory samples to NAME.gpu_mib
  local name=$1; shift
  say "$name start: $*"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$LOGS/$name.gpu_mib" 2> /dev/null &
  local smi=$!
  python - "$LOGS/$name.time" "$@" > "$LOGS/$name.log" 2>&1 <<'PY'
import resource, subprocess, sys, time
t0 = time.time()
rc = subprocess.call(sys.argv[2:])
m, s = divmod(time.time() - t0, 60)
with open(sys.argv[1], "w") as f:
    f.write(f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {int(m)}:{s:05.2f}\n"
            f"\tMaximum resident set size (kbytes): {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}\n\tExit status: {rc}\n")
sys.exit(rc)
PY
  local rc=$?
  kill "$smi" 2> /dev/null
  say "$name done rc=$rc, $(grep -h 'Elapsed' "$LOGS/$name.time" 2> /dev/null | sed 's/.*: //') (m:ss)"
  [ $rc -eq 0 ] || tail -5 "$LOGS/$name.log" | tee -a "$LOGS/chain.log"
  return $rc
}

for s in $STEPS; do
  case $s in
    register) step register python -m octreg register "$OCT" "$MRI" -o "$OUT" --device "$DEVICE" || exit 1 ;;
    ablate)   MAIN=(); [ -f "$OUT/T_oct2mri.txt" ] && MAIN=(--main "$OUT")
              for f in $PREV; do [ -f "$f" ] && MAIN+=(--previous "$f"); done
              step ablate python bench/ablate.py --out "$ABL" ${MAIN[@]+"${MAIN[@]}"} --device "$DEVICE" || exit 1 ;;
    evaluate) ARGS=(); [ -f "$PREV_MAIN/T_oct2mri.txt" ] && ARGS=(--previous "$PREV_MAIN")
              if [ -f "$ABL/prep/texture/oct_mask.nii.gz" ]; then ARGS+=(--masks "$ABL/prep/texture")
              else say "evaluate: no $ABL/prep/texture yet, so no metrics with the method masks (run the ablate step first)"; fi
              step evaluate python bench/evaluate.py "$OUT" ${ARGS[@]+"${ARGS[@]}"} || exit 1 ;;
    report)   step report python bench/report.py --main "$OUT" --ablate "$ABL" --logs "$LOGS" -o bench/BENCHMARK.md --figures bench/figures || exit 1 ;;
    *)        say "unknown step '$s' (register | ablate | evaluate | report)"; exit 1 ;;
  esac
done
say "all steps done: $STEPS"
