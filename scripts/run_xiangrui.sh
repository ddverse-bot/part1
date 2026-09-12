#!/bin/bash
# Xiangrui I58 brainstem pair: prep + register (crop mode: MRI is already cropped to the OCT) + QC figure.
cd ~/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
D=data/xiangrui/OCT_to_MRI
echo "[prep xiangrui] $(date)"
python octreg/scripts/prep_subject.py --work work/xiangrui_I58bs \
  --mri $D/I58_brainstem_MRI_cropped_to_OCT.nii.gz \
  --oct $D/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz > logs/prep_xiangrui.log 2>&1 || { echo "[prep FAILED]"; tail -6 logs/prep_xiangrui.log; exit 1; }
echo "[register xiangrui] $(date)"
python octreg/scripts/register.py --work work/xiangrui_I58bs --out work/runs/xiangrui_I58bs --oct-wm-bright auto --min-overlap 0.5 > logs/reg_xiangrui.log 2>&1 || { echo "[register FAILED]"; tail -8 logs/reg_xiangrui.log; exit 1; }
python octreg/scripts/viz_result.py --work work/xiangrui_I58bs --run work/runs/xiangrui_I58bs >> logs/reg_xiangrui.log 2>&1
echo "[done xiangrui] $(date)"
