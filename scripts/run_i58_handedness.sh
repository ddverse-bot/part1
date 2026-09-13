#!/bin/bash
# Handedness test on Xiangrui's I58 pair (2026-09-14). The OCT NIfTI header (LPI, det < 0) says the shipped pose P_R5 (proper in the
# prep's SPR layout frame) is a MIRROR image in header coordinates; the search preferred proper 0.1379 vs mirrored 0.1282 but several
# anomalies (folia on the opposite side of the body, fine stage never converging at P_R5, a mirrored pose scoring 0.1358 > 0.1241 at
# 0.15 mm in P11) are also what a wrong handedness would produce.  Test: register in a mirrored OCT frame (oct150_affine flipped along
# array axis 2 about the block centre) with --no-mirror, so the search and every refinement see only the other handedness, then QC
# and the rigid fine polish exactly as for P_R5.  Criteria are written to logs before any number exists.  GPU, strictly serial.
set -u
cd /root/autodl-tmp/oct-mri-registration
source /root/miniconda3/etc/profile.d/conda.sh; conda activate octmri
export OMP_NUM_THREADS=8
S=octreg/scripts; W0=work/v11/xiangrui_I58bs; WM=work/v11/xiangrui_I58bs_mirror; RUNS=work/runs/v11; CHAIN=logs/v11_chain_handedness.log
step() { local name=$1; shift; echo "[$name start] $(date '+%F %T')  $*" | tee -a $CHAIN; "$@" > logs/v11_$name.log 2>&1; local rc=$?; echo "[$name done rc=$rc] $(date '+%F %T')" | tee -a $CHAIN; [ $rc -ne 0 ] && tail -5 logs/v11_$name.log | tee -a $CHAIN; return 0; }
cat >> $CHAIN <<'CRIT'
[criteria, declared before any result] H_mirror (the true handedness is the mirrored one) is SUPPORTED only if all of:
 (1) structural refined NCC of the mirrored run >= 0.1241 + 0.005 (P_R5 0.1241);
 (2) qc_fine rim median fwd AND rev both lower than P_R5 (2.018 / 1.054 mm) by >= 0.2 mm, and no rim face worse by > 0.5 mm fwd;
 (3) the rigid fine polish from the mirrored structural pose has >= 5/8 restarts within 0.3 mm AND U_mm <= 3.0 AND block-corner move <= 3.0 mm
     (never achieved at P_R5: U 4.04, move 4.85 mm).
 REJECTED if (1) is below 0.1241 - 0.005 AND (2) is worse (either direction higher by >= 0.2 mm). Otherwise INCONCLUSIVE.
CRIT
mkdir -p $WM
for f in $W0/*; do b=$(basename $f); [ "$b" = oct150_affine.npy ] && continue; [ -e $WM/$b ] || ln -s $PWD/$f $WM/$b; done
python - <<'PY' 2>&1 | tee -a $CHAIN
import numpy as np
W0, WM = "work/v11/xiangrui_I58bs", "work/v11/xiangrui_I58bs_mirror"
A = np.load(f"{W0}/oct150_affine.npy"); n = np.load(f"{W0}/oct150.npy", mmap_mode="r").shape
F = np.eye(4); F[2, 2] = -1; F[2, 3] = n[2] - 1                     # voxel k -> n2-1-k
np.save(f"{WM}/oct150_affine.npy", A @ F); np.save(f"{WM}/flip_voxel.npy", F)
print(f"[mirror frame] oct150_affine flipped along array axis 2 (n={n[2]}); det {np.linalg.det(A[:3,:3]):.3e} -> {np.linalg.det((A @ F)[:3,:3]):.3e}")
PY
step H1_register   python $S/register.py --work $WM --out $RUNS/I58_mirror --oct-wm-bright auto --no-mirror
step H1_qc         python $S/qc_fine.py --work $WM --run $RUNS/I58_mirror
step H2_polish     python $S/register.py --work $WM --out $RUNS/I58_mirror_polish --oct-wm-bright auto --init-transform $RUNS/I58_mirror/T_oct2mri_structural.npy --vascular off \
                   --fine on --fine-fixed-mask on --fine-restart-mm 1 --fine-restart-deg 2 --fine-dof rigid --fine-exclude-fov-mm 1.0 --fine-verify full
step H2_qc         python $S/qc_fine.py --work $WM --run $RUNS/I58_mirror_polish
python - <<'PY' 2>&1 | tee -a $CHAIN
import json, numpy as np
R = "work/runs/v11"; W0, WM = "work/v11/xiangrui_I58bs", "work/v11/xiangrui_I58bs_mirror"
def j(p):
    try: return json.load(open(p))
    except Exception as e: return {"_err": str(e)}
m, p5 = j(f"{R}/I58_mirror/result.json"), j(f"{R}/xiangrui_I58bs/result.json"); pol = j(f"{R}/I58_mirror_polish/result.json")
qm, q5 = j(f"{R}/I58_mirror/qc_fine.json"), j(f"{R}/xiangrui_I58bs/qc_fine.json")
print("[H summary] search top1/top2 mirrored-frame", m.get("search_top1_top2") or [m.get("search", {}).get("top1"), m.get("search", {}).get("top2")], "| P_R5", [p5.get("search", {}).get("top1"), p5.get("search", {}).get("top2")])
print("[H summary] refined NCC mirrored", (m.get("refine") or {}).get("final_ncc"), "| P_R5", (p5.get("refine") or {}).get("final_ncc"), "| mirrored stretch", (m.get("final_transform") or {}).get("stretch_ijk"))
def bq(q):
    b = (q.get("boundary") or {}).get("final") or {}; fw, rv = b.get("forward_oct_to_mri") or {}, b.get("reverse_mri_to_oct") or {}
    r3 = lambda x: None if x is None else round(float(x), 3)
    return {"rim_fwd": r3((fw.get("rim_faces") or {}).get("median_mm")), "rim_rev": r3((rv.get("rim_faces") or {}).get("median_mm")), "deep_fwd": r3((fw.get("deep_end") or {}).get("median_mm")),
            "faces_fwd": {k: r3((v or {}).get("median_mm")) for k, v in (fw.get("per_face") or {}).items()}, "mi_argmax": {k: (v or {}).get("argmax") for k, v in (((q.get("mi_landscape") or {}).get("final") or {}).get("per_axis") or {}).items()}}
print("[H summary] qc mirrored", bq(qm)); print("[H summary] qc P_R5   ", bq(q5))
f = pol.get("fine") or {}
print("[H summary] rigid polish from the mirrored pose: accepted", f.get("accepted"), f.get("reasons"), "U", f.get("U_mm"), "move", (f.get("vs_start") or {}).get("block_corner_mean_mm"),
      "rot", (f.get("vs_start") or {}).get("delta_rotation_deg"), "restarts", (f.get("restarts") or {}).get("n_converged"), (f.get("restarts") or {}).get("n_within_mm"), "guard", [(g.get("level"), g.get("frac_neg")) for g in f.get("guard", [])])
try:
    A = np.load(f"{W0}/oct150_affine.npy"); F = np.load(f"{WM}/flip_voxel.npy"); Tm = np.load(f"{R}/I58_mirror/T_oct2mri.npy")
    T_orig = Tm @ A @ F @ np.linalg.inv(A); np.save(f"{R}/I58_mirror/T_oct2mri_in_original_frame.npy", T_orig)
    P5 = np.load(f"{R}/xiangrui_I58bs/T_oct2mri.npy"); n = np.load(f"{W0}/oct150.npy", mmap_mode="r").shape
    C = np.array([[i, j, k, 1.0] for i in (0, n[0]-1) for j in (0, n[1]-1) for k in (0, n[2]-1)]) @ A.T
    print("[H summary] mirrored pose in the original frame: det", round(float(np.linalg.det(T_orig[:3,:3])), 4), "block-corner distance to P_R5", round(float(np.linalg.norm((C @ T_orig.T - C @ P5.T)[:, :3], axis=1).mean()), 2), "mm")
except Exception as e: print("[H summary] frame conversion failed:", e)
PY
echo "[run_i58_handedness done] $(date)" | tee -a $CHAIN
