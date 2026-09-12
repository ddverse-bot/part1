#!/bin/bash
# Sequential multi-subject driver (one heavy job at a time: the instance froze once with two running).
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
rm -f work/I46/octv_vessels.npy            # re-prep I46 with the final prep code (global Frangi gamma)
for pair in "I46 flip-2" "I38 flip-2" "I55 flip-2" "I56 flip-2" "I62 flip-3" "I57 flip-4" "I61 flip-4" "I48 flip-2" "I58 flip-3"; do
  set -- $pair; SUB=$1; FLIP=$2
  D=data/costantini/dandi-000026
  MRI=$D/derivatives/EPIC/sub-$SUB/ses-MRI/anat/sub-${SUB}_ses-MRI_${FLIP}_VFA.nii.gz
  OCT=$D/sub-$SUB/ses-OCT/micr/sub-${SUB}_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff
  if [ -f "$MRI.aria2" ] || [ -f "$OCT.aria2" ] || [ ! -f "$MRI" ] || [ ! -f "$OCT" ]; then echo "[skip $SUB: download incomplete] $(date)"; continue; fi
  bash octreg/scripts/run_subject.sh $SUB $FLIP
done
python octreg/scripts/summarize.py > logs/summarize.log 2>&1
echo "[run_all done] $(date)"
