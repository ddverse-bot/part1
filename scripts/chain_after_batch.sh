#!/bin/bash
# After run_all finishes: re-run I38 (inf fix); run the Xiangrui pair as soon as its OCT md5-verifies; summarize.
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
until grep -q "\[run_all done\]" logs/run_all.log 2>/dev/null; do sleep 120; done
echo "[chain] batch done $(date)"
rm -rf work/I38 work/I55 work/I56
bash octreg/scripts/run_subject.sh I38 flip-2 >> logs/run_all.log 2>&1
bash octreg/scripts/run_subject.sh I55 flip-2 >> logs/run_all.log 2>&1
bash octreg/scripts/run_subject.sh I56 flip-2 >> logs/run_all.log 2>&1
echo "[chain] I38 done $(date)"
X=data/xiangrui/OCT_to_MRI/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz
# ready only after the pusher reassembled AND verified (it removes the segs dir on success)
for i in $(seq 1 300); do
  [ ! -d data/xiangrui/OCT_to_MRI/segs ] && [ "$(stat -c %s "$X" 2>/dev/null)" = "13025490607" ] && break; sleep 120
done
if [ -f work/runs/xiangrui_I58bs/result.json ]; then
  echo "[chain] xiangrui already done, skipping $(date)"
elif [ ! -d data/xiangrui/OCT_to_MRI/segs ] && [ "$(stat -c %s "$X" 2>/dev/null)" = "13025490607" ]; then
  bash octreg/scripts/run_xiangrui.sh > logs/run_xiangrui.log 2>&1
  echo "[chain] xiangrui done $(date)"
else
  echo "[chain] xiangrui OCT not complete, skipped $(date)"
fi
OMP_NUM_THREADS=8 python octreg/scripts/summarize.py --pattern "I*" > logs/summarize.log 2>&1
echo "[chain] all done $(date)"
