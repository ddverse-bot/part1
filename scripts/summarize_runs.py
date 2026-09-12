#!/usr/bin/env python3
"""Collect result.json of several runs into one table (markdown + json)."""
import json, sys
from pathlib import Path
import numpy as np
runs = [Path(p) for p in sys.argv[1:]]
ref = None
rows = []
for r in runs:
    f = r / "result.json"
    if not f.exists():
        rows.append({"run": r.name, "status": "no result.json"}); continue
    j = json.load(open(f)); ev = j["evaluation"]; ft = j["final_transform"]
    row = {"run": r.name, "scenario": j["args"]["scenario"], "features": j["args"]["features"], "refine_features": j["args"].get("refine_features") or j["args"]["features"],
           "search_top1": round(j["search"]["top1"], 3), "search_top2": round(j["search"]["top2"], 3), "search_s": round(j["search"]["seconds"]),
           "final_ncc": round(j["refine"]["final_ncc"], 3), "mirror": ft.get("mirror"), "centre": [round(x, 1) for x in ft["block_centre_mri_mm"]],
           "scales": [round(x, 3) for x in ft["column_scales"]],
           "dice_WM_otsu": round(ev["gmwm_otsu_classes"]["dice_WM"], 3), "dice_GM_otsu": round(ev["gmwm_otsu_classes"]["dice_GM"], 3),
           "dice_WM_parser": round(ev["gmwm_parser_classes"]["dice_WM"], 3), "dice_GM_parser": round(ev["gmwm_parser_classes"]["dice_GM"], 3),
           "vessel_median_um": (lambda v: round(v) if isinstance(v, (int, float)) else "n/a")(ev.get("vessels_zoff0", {}).get("registered", {}).get("median_um")),
           "vessel_f300": (lambda v: round(v, 2) if isinstance(v, (int, float)) else "n/a")(ev.get("vessels_zoff0", {}).get("registered", {}).get("frac_within_300um")),
           "vessels_inside": ev.get("vessels_zoff0", {}).get("registered", {}).get("n_inside"),
           "restarts": f"{ev['restarts']['n_converged_to_solution']}/{ev['restarts']['n']}", "total_s": round(j["total_seconds"])}
    rows.append(row)
cols = ["run", "scenario", "features", "refine_features", "search_top1", "search_top2", "final_ncc", "mirror", "centre", "scales", "dice_WM_otsu", "dice_GM_otsu", "dice_WM_parser", "dice_GM_parser", "vessel_median_um", "vessel_f300", "vessels_inside", "restarts", "search_s", "total_s"]
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
json.dump(rows, open("runs_summary.json", "w"), indent=1)
