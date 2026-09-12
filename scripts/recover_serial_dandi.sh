#!/bin/bash
# DANDI-only serial remainder + summary (no xiangrui here)
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
for pair in "I38 flip-2" "I55 flip-2" "I56 flip-2" "I62 flip-3" "I57 flip-4" "I61 flip-4" "I48 flip-2" "I58 flip-3"; do
  set -- $pair; SUB=$1; FLIP=$2
  [ -f work/runs/${SUB}_otsu/result.json ] && { echo "[skip $SUB done]"; continue; }
  D=data/costantini/dandi-000026
  MRI=$D/derivatives/EPIC/sub-$SUB/ses-MRI/anat/sub-${SUB}_ses-MRI_${FLIP}_VFA.nii.gz
  OCT=$D/sub-$SUB/ses-OCT/micr/sub-${SUB}_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff
  if [ -f "$MRI.aria2" ] || [ -f "$OCT.aria2" ] || [ ! -f "$MRI" ] || [ ! -f "$OCT" ]; then echo "[skip $SUB: download incomplete]"; continue; fi
  bash octreg/scripts/run_subject.sh $SUB $FLIP
done
OMP_NUM_THREADS=8 python octreg/scripts/summarize.py --pattern "*" > logs/summarize.log 2>&1
echo "[dandi serial done] $(date)"
