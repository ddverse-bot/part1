#!/usr/bin/env python3
"""Collect work/runs/<subject>_<features>[...]/result.json into one table (markdown + json).
v1.1: columns vasc_accepted, fine_accepted, fine_moved_mm, U_mm, ice_mm, fine_restarts, mask_mode, destripe; search top1/top2 may be None (--init-transform)."""
import argparse, json, glob
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=Path, default=Path("work/runs")); ap.add_argument("--out", type=Path, default=Path("work/runs/summary_all"))
ap.add_argument("--pattern", default="*")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(glob.glob(str(a.runs / a.pattern / "result.json"))):
    r = json.load(open(f)); run = Path(f).parent.name
    ev = r.get("evaluation", {}); fin = ev.get("final", {}) or {}; st = ev.get("structural") or {}
    vas = r.get("vascular") or {}; fi = r.get("fine") or {}; ps = r.get("prep_summary") or {}
    def ves(e, tag):
        x = e.get(f"vessels_{tag}", {}).get("registered", {}); c = e.get(f"vessels_{tag}", {}).get("random_shift_control", {})
        return (x.get("median_um"), x.get("frac_within_150um"), c.get("median_um_mean"))
    row = {"run": run, "subject": r["args"].get("work", "").split("/")[-1], "features": r["args"].get("features"),
           "search_top1": (r.get("search") or {}).get("top1"), "search_top2": (r.get("search") or {}).get("top2"), "oct_wm_bright": r.get("oct_wm_bright"),
           "mirror": r.get("final_transform", {}).get("mirror"), "centre": [round(x, 1) for x in r.get("final_transform", {}).get("block_centre_mri_mm", [])],
           "ncc_struct": r.get("refine", {}).get("final_ncc"), "restarts": f"{ev.get('restarts', {}).get('n_converged')}/{ev.get('restarts', {}).get('n')}",
           "vasc_restarts": (vas.get("restarts") or {}).get("n_within_0.3mm"), "vasc_moved_mm": (vas.get("vs_structural") or {}).get("corner_mean_mm"), "vasc_accepted": vas.get("accepted", vas.get("mode")),
           "stretch_struct": vas.get("stretch_ijk_structural") or fi.get("stretch_ijk_before"), "stretch_final": r.get("final_transform", {}).get("stretch_ijk"),
           "fine_accepted": fi.get("accepted"), "fine_moved_mm": (fi.get("vs_start") or {}).get("corner_mean_mm"), "fine_moved_block_mm": (fi.get("vs_start") or {}).get("block_corner_mean_mm"),
           "fine_rot_deg": (fi.get("vs_start") or {}).get("delta_rotation_deg"), "U_mm": fi.get("U_mm"), "ice_mm": (fi.get("ice") or {}).get("mean_mm"),
           "fine_restarts": f"{(fi.get('restarts') or {}).get('n_converged')}/{(fi.get('restarts') or {}).get('n')}" if fi.get("restarts") else None,
           "fine_sign": fi.get("sign_per_level"), "mask_mode": ps.get("mask_mode"), "destripe": ps.get("destripe_applied"),
           "ves_own_struct": ves(st, "own"), "ves_own_final": ves(fin, "own"), "ves_vesseg_struct": ves(st, "vesseg"), "ves_vesseg_final": ves(fin, "vesseg"),
           "dice_struct": (st.get("gmwm", {}).get("dice_WM"), st.get("gmwm", {}).get("dice_GM")), "dice_final": (fin.get("gmwm", {}).get("dice_WM"), fin.get("gmwm", {}).get("dice_GM")),
           "total_s": r.get("total_seconds")}
    rows.append(row)
json.dump(rows, open(a.out / "summary.json", "w"), indent=1, default=float)
def f2(x, n=2): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{n}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
def fv(v): return "—" if not v or v[0] is None else f"{v[0]:.0f} / {v[1]:.2f} (ctrl {v[2]:.0f})"
def fd(v): return "—" if not v or v[0] is None else f"{v[0]:.3f} / {v[1]:.3f}"
lines = ["| run | search top1 / top2 | mirror | OCT WM bright | centre (mm) | restarts | vasc. restarts | vasc. moved (mm) | vasc. accepted | fine accepted | fine moved (mm, 7 mm cube / block corners) | fine rot (deg) | U (mm) | ICE (mm) | fine restarts | mask | destripe | stretch ijk struct → final | vessels own: struct → final (median µm / f150) | vessels ves_seg: struct → final | Dice WM/GM struct → final | time (s) |", "|" + "---|" * 22]
for r in rows:   # vasc. restarts is an integer count: printed raw as in v1 (not through f2)
    lines.append(f"| {r['run']} | {f2(r['search_top1'],3)} / {f2(r['search_top2'],3)} | {r['mirror']} | {r['oct_wm_bright']} | {r['centre']} | {r['restarts']} | {r['vasc_restarts']} | {f2(r['vasc_moved_mm'])} | {f2(r['vasc_accepted'])} | {f2(r['fine_accepted'])} | {f2(r['fine_moved_mm'])} / {f2(r['fine_moved_block_mm'])} | {f2(r['fine_rot_deg'])} | {f2(r['U_mm'])} | {f2(r['ice_mm'])} | {f2(r['fine_restarts'])} | {f2(r['mask_mode'])} | {f2(r['destripe'])} | {r['stretch_struct']} → {r['stretch_final']} | {fv(r['ves_own_struct'])} → {fv(r['ves_own_final'])} | {fv(r['ves_vesseg_struct'])} → {fv(r['ves_vesseg_final'])} | {fd(r['dice_struct'])} → {fd(r['dice_final'])} | {f2(r['total_s'],0)} |")
(a.out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
