#!/usr/bin/env python3
"""Collect work/runs/<subject>_<features>[...]/result.json into one table (markdown + json)."""
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
    def ves(e, tag):
        x = e.get(f"vessels_{tag}", {}).get("registered", {}); c = e.get(f"vessels_{tag}", {}).get("random_shift_control", {})
        return (x.get("median_um"), x.get("frac_within_150um"), c.get("median_um_mean"))
    row = {"run": run, "subject": r["args"].get("work", "").split("/")[-1], "features": r["args"].get("features"),
           "search_top1": r.get("search", {}).get("top1"), "search_top2": r.get("search", {}).get("top2"), "oct_wm_bright": r.get("oct_wm_bright"),
           "mirror": r.get("final_transform", {}).get("mirror"), "centre": [round(x, 1) for x in r.get("final_transform", {}).get("block_centre_mri_mm", [])],
           "ncc_struct": r.get("refine", {}).get("final_ncc"), "restarts": f"{ev.get('restarts', {}).get('n_converged')}/{ev.get('restarts', {}).get('n')}",
           "vasc_restarts": r.get("vascular", {}).get("restarts", {}).get("n_within_0.3mm"), "vasc_moved_mm": r.get("vascular", {}).get("vs_structural", {}).get("corner_mean_mm"),
           "stretch_struct": r.get("vascular", {}).get("stretch_ijk_structural"), "stretch_final": r.get("final_transform", {}).get("stretch_ijk"),
           "ves_own_struct": ves(st, "own"), "ves_own_final": ves(fin, "own"), "ves_vesseg_struct": ves(st, "vesseg"), "ves_vesseg_final": ves(fin, "vesseg"),
           "dice_struct": (st.get("gmwm", {}).get("dice_WM"), st.get("gmwm", {}).get("dice_GM")), "dice_final": (fin.get("gmwm", {}).get("dice_WM"), fin.get("gmwm", {}).get("dice_GM")),
           "total_s": r.get("total_seconds")}
    rows.append(row)
json.dump(rows, open(a.out / "summary.json", "w"), indent=1, default=float)
def f2(x, n=2): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
def fv(v): return "—" if not v or v[0] is None else f"{v[0]:.0f} / {v[1]:.2f} (ctrl {v[2]:.0f})"
def fd(v): return "—" if not v or v[0] is None else f"{v[0]:.3f} / {v[1]:.3f}"
lines = ["| run | search top1 / top2 | mirror | OCT WM bright | centre (mm) | restarts | vasc. restarts | vasc. moved (mm) | stretch ijk struct → final | vessels own: struct → final (median µm / f150) | vessels ves_seg: struct → final | Dice WM/GM struct → final | time (s) |", "|" + "---|" * 13]
for r in rows:
    lines.append(f"| {r['run']} | {f2(r['search_top1'],3)} / {f2(r['search_top2'],3)} | {r['mirror']} | {r['oct_wm_bright']} | {r['centre']} | {r['restarts']} | {r['vasc_restarts']} | {f2(r['vasc_moved_mm'])} | {r['stretch_struct']} → {r['stretch_final']} | {fv(r['ves_own_struct'])} → {fv(r['ves_own_final'])} | {fv(r['ves_vesseg_struct'])} → {fv(r['ves_vesseg_final'])} | {fd(r['dice_struct'])} → {fd(r['dice_final'])} | {f2(r['total_s'],0)} |")
(a.out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
