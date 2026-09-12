#!/bin/bash
cd ~/autodl-tmp/oct-mri-registration
for pat in recover_serial run_subject.sh rx2.sh; do for pid in $(pgrep -f $pat); do for c in $(pgrep -P $pid); do kill $c 2>/dev/null; done; kill $pid 2>/dev/null; done; done
for pid in $(pgrep -f "register.py"); do kill $pid 2>/dev/null; done
for pid in $(pgrep -f "prep_subject.py"); do kill $pid 2>/dev/null; done
sleep 2
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
echo "[rx3] register (vascular off) $(date)"
python octreg/scripts/register.py --work work/xiangrui_I58bs --out work/runs/xiangrui_I58bs_novasc --oct-wm-bright auto --vascular off > logs/reg_xiangrui_nv.log 2>&1 || echo "[rx3] FAILED"
python octreg/scripts/viz_result.py --work work/xiangrui_I58bs --run work/runs/xiangrui_I58bs_novasc >> logs/reg_xiangrui_nv.log 2>&1
echo "[rx3] done $(date)"
(nohup setsid bash octreg/scripts/recover_serial_dandi.sh > logs/recover2.log 2>&1 < /dev/null &)
echo "[rx3] dandi serial launched $(date)"
