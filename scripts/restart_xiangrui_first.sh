#!/bin/bash
# stop the serial DANDI branch, run the xiangrui pair ALONE, then restart the serial remainder
cd ~/autodl-tmp/oct-mri-registration
for pid in $(pgrep -f recover_serial.sh); do for c in $(pgrep -P $pid); do pkill -P $c; kill $c 2>/dev/null; done; kill $pid 2>/dev/null; done
sleep 2
for pat in run_subject.sh run_xiangrui.sh; do for pid in $(pgrep -f $pat); do kill $pid 2>/dev/null; done; done
for pid in $(pgrep -f "prep_subject.py"); do kill $pid 2>/dev/null; done
for pid in $(pgrep -f "register.py"); do kill $pid 2>/dev/null; done
sleep 2
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
echo "[xf] running xiangrui alone $(date)"
bash octreg/scripts/run_xiangrui.sh > logs/run_xiangrui_main.log 2>&1
echo "[xf] xiangrui rc=$? $(date)"
(nohup setsid bash octreg/scripts/recover_serial_dandi.sh > logs/recover2.log 2>&1 < /dev/null &)
echo "[xf] dandi serial relaunched $(date)"
