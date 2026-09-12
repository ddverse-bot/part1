#!/bin/bash
# stop everything, re-prep xiangrui with the unified threshold (skip MRI: unchanged), re-register, then dandi serial
cd ~/autodl-tmp/oct-mri-registration
for pid in $(pgrep -f recover_serial); do for c in $(pgrep -P $pid); do pkill -P $c; kill $c 2>/dev/null; done; kill $pid 2>/dev/null; done
for pat in restart_xiangrui_first run_xiangrui.sh run_subject.sh; do for pid in $(pgrep -f $pat); do for c in $(pgrep -P $pid); do kill $c 2>/dev/null; done; kill $pid 2>/dev/null; done; done
for pid in $(pgrep -f "prep_subject.py"); do kill $pid 2>/dev/null; done
for pid in $(pgrep -f "register.py"); do kill $pid 2>/dev/null; done
sleep 2
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
D=data/xiangrui/OCT_to_MRI
echo "[rx] prep (skip-mri) $(date)"
python octreg/scripts/prep_subject.py --work work/xiangrui_I58bs --mri $D/I58_brainstem_MRI_cropped_to_OCT.nii.gz --oct $D/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz --skip-mri > logs/prep_xiangrui.log 2>&1 || { echo "[rx] prep FAILED"; tail -6 logs/prep_xiangrui.log; }
echo "[rx] register $(date)"
python octreg/scripts/register.py --work work/xiangrui_I58bs --out work/runs/xiangrui_I58bs --oct-wm-bright auto --min-overlap 0.5 > logs/reg_xiangrui.log 2>&1 || echo "[rx] register FAILED"
python octreg/scripts/viz_result.py --work work/xiangrui_I58bs --run work/runs/xiangrui_I58bs >> logs/reg_xiangrui.log 2>&1
python octreg/scripts/diag_xiangrui.py > logs/diag_xiangrui.log 2>&1
echo "[rx] xiangrui done $(date)"
(nohup setsid bash octreg/scripts/recover_serial_dandi.sh > logs/recover2.log 2>&1 < /dev/null &)
echo "[rx] dandi serial launched $(date)"
