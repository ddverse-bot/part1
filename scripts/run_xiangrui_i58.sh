#!/bin/bash
# One command for Xiangrui's I58 brainstem pair (AutoDL server), label-free, final settings of 2026-09-14:
#   prep (texture specimen mask, axis-detected destripe)  ->  register (FFT search + rigid/similarity/affine, no fine stage)
#   ->  QC (viz + qc_fine)  ->  export in the ORIGINAL NIfTI frames (4x4 + FreeSurfer LTA + overlays)  ->  summary.json
# The handedness of the OCT relative to the MRI is not decided by the data (REPORT_I58_v11 Appendix C.4): by default the script
# therefore also registers the other handedness (mirrored OCT frame, --no-mirror) and exports both candidates side by side, so a
# person can decide by eye in freeview.  Set HANDEDNESS=search to run only the pipeline's own choice.
# Why no fine stage: on I58 every fine configuration was rejected by its own gates, and on labelled cortex an accepted fine stage made
# the labelled metrics worse (REPORT sections 3.4, 6, C.3).
# usage: bash octreg/scripts/run_xiangrui_i58.sh            env: W (prep dir), R (run dir), DATA, HANDEDNESS=both|search, REUSE_PREP=0|1
# Memory: the native prep holds the 18.7 GB float32 OCT plus a 9.4 GB float16 copy (container limit 62 GB): run nothing else meanwhile.
set -u
cd /root/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
S=octreg/scripts; DATA=${DATA:-data/xiangrui/OCT_to_MRI}
OCT=$DATA/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz; MRI=$DATA/I58_brainstem_MRI_cropped_to_OCT.nii.gz
W=${W:-work/xiangrui_I58_final}; R=${R:-work/runs/xiangrui_I58_final}; HANDEDNESS=${HANDEDNESS:-both}; REUSE_PREP=${REUSE_PREP:-0}
mkdir -p logs $R; CHAIN=logs/xiangrui_I58_final_chain.log
step() { local name=$1; shift; echo "[$name start] $(date '+%F %T')  $*" | tee -a $CHAIN; "$@" > logs/xiangrui_I58_final_$name.log 2>&1; local rc=$?
         echo "[$name done rc=$rc] $(date '+%F %T')" | tee -a $CHAIN; [ $rc -ne 0 ] && tail -5 logs/xiangrui_I58_final_$name.log | tee -a $CHAIN; return $rc; }
busy=$(ps -eo args | grep -E "python .*(prep_subject|register|test_fine|qc_fine|test_specimen_mask)\.py" | grep -v grep | wc -l)
[ "$busy" -gt 0 ] && { echo "another registration job is running ($busy); refusing to start (memory limit 62 GB)" | tee -a $CHAIN; exit 1; }
[ -f $OCT ] && [ -f $MRI ] || { echo "input NIfTI files not found under $DATA" | tee -a $CHAIN; exit 1; }

# 1. prep
if [ "$REUSE_PREP" = 1 ] && [ -f $W/oct150.npy ] && [ -f $W/prep.json ]; then echo "[prep reused] $W" | tee -a $CHAIN
else step prep python $S/prep_subject.py --work $W --mri $MRI --oct $OCT --oct-mask auto --destripe on --skip-vessels || exit 1; fi
# 2. register (the pipeline's own handedness choice) + QC + export
step register python $S/register.py --work $W --out $R --oct-wm-bright auto || exit 1
step viz      python $S/viz_result.py --work $W --run $R
step qc       python $S/qc_fine.py --work $W --run $R
step export   python $S/export_registration.py --work $W --run $R --oct-nifti $OCT --mri-nifti $MRI --out $R/export
# 3. the other handedness: OCT frame flipped along array axis 2, search restricted to proper poses in that frame
if [ "$HANDEDNESS" = both ]; then
  WM=${W}_mirror; RM=${R}_mirror; mkdir -p $WM
  for f in $W/*; do b=$(basename $f); [ "$b" = oct150_affine.npy ] && continue; ln -sfn $PWD/$f $WM/$b; done
  python -c "
import numpy as np; A = np.load('$W/oct150_affine.npy'); n = np.load('$W/oct150.npy', mmap_mode='r').shape
F = np.eye(4); F[2, 2] = -1; F[2, 3] = n[2] - 1; np.save('$WM/oct150_affine.npy', A @ F); np.save('$WM/flip_voxel.npy', F)" || exit 1
  step register_mirror python $S/register.py --work $WM --out $RM --oct-wm-bright auto --no-mirror
  step qc_mirror       python $S/qc_fine.py --work $WM --run $RM
  python -c "
import numpy as np; A = np.load('$W/oct150_affine.npy'); F = np.load('$WM/flip_voxel.npy'); T = np.load('$RM/T_oct2mri.npy')
np.save('$RM/T_oct2mri_in_prep_frame.npy', T @ A @ F @ np.linalg.inv(A))"
  step export_mirror   python $S/export_registration.py --work $W --run $RM --T $RM/T_oct2mri_in_prep_frame.npy --oct-nifti $OCT --mri-nifti $MRI --out $RM/export
  step fig_handedness  python $S/fig_handedness.py $MRI $R/export/oct_in_mri.nii.gz $RM/export/oct_in_mri.nii.gz $R/handedness_side_by_side.png
fi
# 4. summary
python - "$W" "$R" "$HANDEDNESS" <<'PY' 2>&1 | tee -a $CHAIN
import json, sys, numpy as np
W, R, H = sys.argv[1:4]
def load(p):
    try: return json.load(open(p))
    except Exception: return {}
def qc(run):
    b = (load(f"{run}/qc_fine.json").get("boundary") or {}).get("final") or {}; fw, rv = b.get("forward_oct_to_mri") or {}, b.get("reverse_mri_to_oct") or {}
    return {"rim_median_fwd_mm": (fw.get("rim_faces") or {}).get("median_mm"), "rim_median_rev_mm": (rv.get("rim_faces") or {}).get("median_mm"),
            "per_face_fwd_mm": {k: (v or {}).get("median_mm") for k, v in (fw.get("per_face") or {}).items()}}
def cand(run, T_key):
    r = load(f"{run}/result.json"); e = load(f"{run}/export/export.json"); s = r.get("search") or {}
    return {"run": run, "search_top1_top2": [s.get("top1"), s.get("top2")], "structural_ncc": (r.get("refine") or {}).get("final_ncc"),
            "structural_restarts": (r.get("evaluation") or {}).get("restarts", {}).get("n_converged"), "stretch_ijk": (r.get("final_transform") or {}).get("stretch_ijk"),
            "det_in_prep_frame": (r.get("final_transform") or {}).get("det"), "det_in_nifti_frames": e.get("det"), "export": f"{run}/export", "T_octnii_to_mrinii": e.get("T_octnii_to_mrinii"),
            "qc": qc(run), "export_check_vs_register": (e.get("checks") or {}).get("vs_register_oct_in_mri_region")}
out = {"prep": W, "prep_oct": (load(f"{W}/prep.json").get("oct") or {}).get("mask_mode"), "candidate_pipeline_choice": cand(R, "T_oct2mri")}
if H == "both": out["candidate_other_handedness"] = cand(R + "_mirror", "T_oct2mri_in_prep_frame"); out["figure"] = f"{R}/handedness_side_by_side.png"
out["reading"] = ("Body-level pose only (boundary agreement ~1.5-2 mm). The two handedness candidates trade outline agreement against internal class "
                  "correlation; decide by eye in freeview (README.md in each export dir) or with a few landmark pairs.")
json.dump(out, open(f"{R}/summary.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str)[:3000])
PY
echo "[run_xiangrui_i58 done] $(date)" | tee -a $CHAIN
