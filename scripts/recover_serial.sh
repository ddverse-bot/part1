#!/bin/bash
# One-shot recovery: kill everything, run the Xiangrui pair ALONE, then the remaining subjects sequentially.
cd ~/autodl-tmp/oct-mri-registration
pkill -f run_all.sh; pkill -f run_subject.sh; pkill -f chain_after; pkill -f register.py; pkill -f prep_subject.py; pkill -f parse_mri.py; pkill -x aria2c; sleep 3
echo "[recover] cleaned $(date)"
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
bash octreg/scripts/run_xiangrui.sh > logs/run_xiangrui_main.log 2>&1
echo "[recover] xiangrui finished $(date)"
rm -rf work/I38 work/I55 work/I56
for pair in "I38 flip-2" "I55 flip-2" "I56 flip-2" "I62 flip-3" "I57 flip-4" "I61 flip-4" "I48 flip-2" "I58 flip-3"; do
  set -- $pair; SUB=$1; FLIP=$2
  [ -f work/runs/${SUB}_parser/result.json ] && { echo "[skip $SUB done]"; continue; }
  D=data/costantini/dandi-000026
  MRI=$D/derivatives/EPIC/sub-$SUB/ses-MRI/anat/sub-${SUB}_ses-MRI_${FLIP}_VFA.nii.gz
  OCT=$D/sub-$SUB/ses-OCT/micr/sub-${SUB}_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff
  if [ -f "$MRI.aria2" ] || [ -f "$OCT.aria2" ] || [ ! -f "$MRI" ] || [ ! -f "$OCT" ]; then echo "[skip $SUB: download incomplete]"; continue; fi
  bash octreg/scripts/run_subject.sh $SUB $FLIP
done
OMP_NUM_THREADS=8 python octreg/scripts/summarize.py --pattern "I*" > logs/summarize.log 2>&1
echo "[recover] all done $(date)"
