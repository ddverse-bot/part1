#!/bin/bash
# usage: run_subject.sh SUB FLIP [extra prep args]   — prep (if needed) + otsu whole run + parser whole run (+ viz)
set -u
SUB=$1; FLIP=$2; shift 2
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
D=data/costantini/dandi-000026
MRI=$D/derivatives/EPIC/sub-$SUB/ses-MRI/anat/sub-${SUB}_ses-MRI_${FLIP}_VFA.nii.gz
OCT=$D/sub-$SUB/ses-OCT/micr/sub-${SUB}_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff
L=$D/derivatives/Labels/sub-$SUB/ses-MRI/anat/sub-${SUB}_ses-MRI_space-EPIC_label
VS=$D/derivatives/sub-$SUB/ses-OCT/vessel/ves_seg.tif
ARGS="--work work/$SUB --mri $MRI --oct $OCT --labels-wholehemi ${L}-infrasupra_seg-wholehemi_dseg.nii.gz --labels-ba ${L}-infrasupra_dseg.nii.gz --mri-vessels ${L}-vessels_dseg.nii.gz"
[ -f "$VS" ] && ARGS="$ARGS --oct-vesseg $VS"
if [ ! -f work/$SUB/octv_vessels.npy ] || [ ! -f work/$SUB/mri_tissue.npy ]; then
  echo "[prep $SUB] $(date)"; python octreg/scripts/prep_subject.py $ARGS "$@" > logs/prep_$SUB.log 2>&1 || { echo "[prep $SUB FAILED]"; tail -5 logs/prep_$SUB.log; exit 1; }
fi
echo "[register otsu $SUB] $(date)"
python octreg/scripts/register.py --work work/$SUB --out work/runs/${SUB}_otsu --oct-wm-bright auto > logs/reg_${SUB}_otsu.log 2>&1 || echo "[otsu $SUB FAILED]"
python octreg/scripts/viz_result.py --work work/$SUB --run work/runs/${SUB}_otsu >> logs/reg_${SUB}_otsu.log 2>&1
echo "[done $SUB] $(date)"
