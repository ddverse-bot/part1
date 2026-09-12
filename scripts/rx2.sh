#!/bin/bash
cd ~/autodl-tmp/oct-mri-registration
for pat in rerun_xiangrui_fixed run_xiangrui.sh recover_serial run_subject.sh; do for pid in $(pgrep -f $pat); do for c in $(pgrep -P $pid); do kill $c 2>/dev/null; done; kill $pid 2>/dev/null; done; done
for pid in $(pgrep -f "register.py"); do kill $pid 2>/dev/null; done
for pid in $(pgrep -f "prep_subject.py"); do kill $pid 2>/dev/null; done
sleep 2
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
echo "[rx2] register $(date)"
python octreg/scripts/register.py --work work/xiangrui_I58bs --out work/runs/xiangrui_I58bs --oct-wm-bright auto > logs/reg_xiangrui.log 2>&1 || echo "[rx2] register FAILED"
python octreg/scripts/viz_result.py --work work/xiangrui_I58bs --run work/runs/xiangrui_I58bs >> logs/reg_xiangrui.log 2>&1
echo "[rx2] done $(date)"
(nohup setsid bash octreg/scripts/recover_serial_dandi.sh > logs/recover2.log 2>&1 < /dev/null &)
echo "[rx2] dandi serial launched $(date)"
