#!/bin/bash
# octreg v1.1 run plan, I58 part (spec run_plan R0-R7 + R11): strictly serial, one heavy job at a time, logs under logs/v11_*.log.
# A failed step is logged (exit code in logs/v11_chain_i58.log) and the chain continues — no `set -e` on purpose.
# usage: bash octreg/scripts/run_v11_i58.sh [R0 R1 ...]   (default: all steps in order)
set -u
cd /root/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
S=octreg/scripts; W=work/xiangrui_I58bs; RUNS=work/runs/v11; V11=work/v11; OLD=work/runs/xiangrui_I58bs_novasc
DATA=data/xiangrui/OCT_to_MRI
mkdir -p logs $RUNS $V11/tests
CHAIN=logs/v11_chain_i58.log
step() {  # step NAME cmd...  -> logs/v11_NAME.log; never aborts the chain
  local name=$1; shift
  echo "[$name start] $(date '+%F %T')  $*" | tee -a $CHAIN
  "$@" > logs/v11_$name.log 2>&1; local rc=$?
  echo "[$name done rc=$rc] $(date '+%F %T')" | tee -a $CHAIN
  [ $rc -ne 0 ] && tail -5 logs/v11_$name.log | tee -a $CHAIN
  return 0
}
STEPS=${*:-"R0 R1 R2 R3 R4 R5 R6 R7 R11"}
for R in $STEPS; do case $R in
R0)  # CPU-only unit tests (~25 min): mask port, destripe, fine-stage sanity (P2 reproduction)
  step R0_mask      python $S/test_specimen_mask.py --work $W --ref work/probe_xr/mask/final_mask150.npy --out $V11/tests/mask
  step R0_destripe  python $S/test_destripe.py --work $W --out $V11/tests/destripe
  step R0_mask_I46  python $S/test_specimen_mask.py --work work/I46 --out $V11/tests/mask_I46
  step R0_destripe_I46 python $S/test_destripe.py --work work/I46 --out $V11/tests/destripe_I46
  step R0_fine_cpu  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python $S/test_fine.py --work $W --T $OLD/T_oct2mri.npy --out $V11/tests/fine_cpu --levels 0.3 --iters 5 --restarts 0 --verify basic ;;
R1)  # GPU ~15 min: fine-only on the OLD prep from the OLD pose (go/no-go for the P2 hypothesis)
  step R1_register  python $S/register.py --work $W --out $RUNS/I58_fine_oldprep --init-transform $OLD/T_oct2mri.npy --oct-wm-bright auto --vascular off --fine on --fine-verify full
  step R1_qc        python $S/qc_fine.py --work $W --run $RUNS/I58_fine_oldprep
  step R1_qc_v1pose python $S/qc_fine.py --work $W --run $OLD --out $RUNS/qc_v1pose ;;
R2)  # GPU+CPU ~20 min: smoke prep from the cached 40 um octv (full v1.1 prep path without the 13 GB read)
  mkdir -p $V11/I58bs_s40; for f in mri.npy mri_affine.npy mri_tissue.npy; do [ -e $V11/I58bs_s40/$f ] || ln -s $PWD/$W/$f $V11/I58bs_s40/$f; done
  python -c "import json; d=json.load(open('$W/prep.json')); json.dump({'mri': d['mri']}, open('$V11/I58bs_s40/prep.json','w'), indent=1)"
  step R2_prep_s40  python $S/prep_subject.py --work $V11/I58bs_s40 --skip-mri --mri $W/mri.npy --oct $W/octv.npy --oct-spacing-um 40,40,40 --oct-mask auto --destripe on --skip-vessels ;;
R3)  # GPU ~20 min: register on the smoke prep
  step R3_register  python $S/register.py --work $V11/I58bs_s40 --out $RUNS/I58bs_s40 --oct-wm-bright auto --fine on --fine-verify full
  step R3_viz       python $S/viz_result.py --work $V11/I58bs_s40 --run $RUNS/I58bs_s40
  step R3_qc        python $S/qc_fine.py --work $V11/I58bs_s40 --run $RUNS/I58bs_s40 ;;
R4)  # GPU+CPU 35-60 min: native I58 prep (once; keep the dir)
  step R4_prep      python $S/prep_subject.py --work $V11/xiangrui_I58bs --mri $DATA/I58_brainstem_MRI_cropped_to_OCT.nii.gz --oct $DATA/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz --oct-mask auto --destripe on --skip-vessels ;;
R5)  # GPU ~20 min: THE REPORTED RUN (no prior, label-free)
  step R5_register  python $S/register.py --work $V11/xiangrui_I58bs --out $RUNS/xiangrui_I58bs --oct-wm-bright auto --fine on --fine-verify full
  step R5_viz       python $S/viz_result.py --work $V11/xiangrui_I58bs --run $RUNS/xiangrui_I58bs
  step R5_qc        python $S/qc_fine.py --work $V11/xiangrui_I58bs --run $RUNS/xiangrui_I58bs ;;
R6)  # GPU ~20 min: destripe ablation (register.py never loads octv): oct150.npy -> oct150_nodestripe.npy
  mkdir -p $V11/xiangrui_I58bs_nods
  for f in $V11/xiangrui_I58bs/*; do b=$(basename $f); [ "$b" = oct150.npy ] && continue; [ -e $V11/xiangrui_I58bs_nods/$b ] || ln -s $PWD/$f $V11/xiangrui_I58bs_nods/$b; done
  [ -e $V11/xiangrui_I58bs_nods/oct150.npy ] || ln -s $PWD/$V11/xiangrui_I58bs/oct150_nodestripe.npy $V11/xiangrui_I58bs_nods/oct150.npy
  step R6_register  python $S/register.py --work $V11/xiangrui_I58bs_nods --out $RUNS/xiangrui_I58bs_nods --oct-wm-bright auto --fine on
  step R6_qc        python $S/qc_fine.py --work $V11/xiangrui_I58bs_nods --run $RUNS/xiangrui_I58bs_nods ;;
R7)  # GPU ~30 min: start independence (prior from the v1 pose) and seed
  step R7_fromold   python $S/register.py --work $V11/xiangrui_I58bs --out $RUNS/I58_fromold --init-transform $OLD/T_oct2mri.npy --oct-wm-bright yes --fine on
  step R7_seed1     python $S/register.py --work $V11/xiangrui_I58bs --out $RUNS/I58_seed1 --oct-wm-bright auto --fine on --seed 1 ;;
R11) # CPU 5 min: summary table + inspection bundle
  step R11_summary  python $S/summarize.py --runs $RUNS --out $RUNS/summary
  mkdir -p results_inspection_v11
  for d in $RUNS/*/; do n=$(basename $d); mkdir -p results_inspection_v11/$n; cp $d/result.json $d/qc_*.png $d/T_oct2mri*.npy results_inspection_v11/$n/ 2>/dev/null; done
  for d in $V11/*/; do n=$(basename $d); [ -f $d/prep.json ] && mkdir -p results_inspection_v11/prep_$n && cp $d/prep.json $d/oct150_mask_qc.png results_inspection_v11/prep_$n/ 2>/dev/null; done
  echo "[R11 inspection bundle] $(date)" | tee -a $CHAIN ;;
*) echo "unknown step $R" | tee -a $CHAIN ;;
esac; done
echo "[run_v11_i58 done] $(date)" | tee -a $CHAIN
