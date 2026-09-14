#!/usr/bin/env python3
"""Write docs/BENCHMARK.md for Xiangrui's I58 brainstem pair from the bench outputs (formatting only, no computation).

    python bench/report.py --main RUN --ablate ABL [--logs RUN_logs] [-o docs/BENCHMARK.md]

RUN: the CLI run (result.json; eval.json from bench/evaluate.py). ABL: bench/ablate.py output (ablations.json).
LOGS: bench/run_xiangrui.sh step logs (register.time from GNU time -v, register.gpu_mib from nvidia-smi). Missing values print n/a.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DELETION_MM = 0.5            # spec: a step whose removal moves the pose by <= 0.5 mm (and no metric beyond noise) is deleted
STEP_OF = {"A1": "MRI flattening", "A2": "OCT flattening", "A3": "section-stripe flat field", "A6": "scale prior", "A7": "affine ladder"}


def load(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}


def num(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def gnu_time(path):
    """(wall seconds, peak RSS GB) from a GNU time -v report."""
    t = Path(path).read_text() if path and Path(path).exists() else ""
    wall = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", t)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    secs = sum(float(p) * 60 ** i for i, p in enumerate(reversed(wall.group(1).split(":")))) if wall else None
    return secs, int(rss.group(1)) / 1024 ** 2 if rss else None


def gpu_peak_gb(path):
    vals = [float(v) for v in Path(path).read_text().split()] if path and Path(path).exists() else []
    return max(vals) / 1024 if vals else None


def main_section(run, logs):
    res, ev = load(run / "result.json"), load(run / "eval.json")
    pose, srch = res.get("pose", {}), res.get("search", {})
    p, fc, b, mk = ev.get("pose_to_R5", {}), ev.get("frame_check", {}), ev.get("boundary", {}), ev.get("mask", {})
    wall, rss = gnu_time(logs / "register.time" if logs else None)
    gpu = gpu_peak_gb(logs / "register.gpu_mib" if logs else None)
    shifts = [v["spearman"] for k, v in fc.get("variants", {}).items() if k.startswith("shift") and v["spearman"] is not None]
    shift_max = max(shifts) if shifts else None
    rim = b.get("rim_median_mm") or [None, None]
    rows = [
        ("pose vs R5, mean (max) over the v1.1 specimen mask", f"{num(p.get('mean_mm'))} ({num(p.get('max_mm'))}) mm"),
        ("pose vs R5 at the block corners, mean (max)", f"{num(p.get('corners_mean_mm'))} ({num(p.get('corners_max_mm'))}) mm"),
        ("rotation vs R5", f"{num(p.get('rotation_deg'), 1)} deg"),
        ("raw-data frame check (Spearman)", f"{num(fc.get('spearman_export'), 3)} at the pose; axis flips <= "
                                            f"{num(fc.get('max_flip_spearman'), 3)}; 2 mm shifts <= {num(shift_max, 3)}; "
                                            f"{('pass' if fc['ok'] else 'fail') if fc else 'n/a'}"),
        ("final score S, polarity", f"{num(pose.get('S'), 4)}, {pose.get('polarity', 'n/a')}"),
        ("search top-1 / top-2", f"{num(srch.get('top1'), 4)} / {num(srch.get('top2'), 4)}"),
        ("scale per OCT array axis", " / ".join(f"{v:.3f}" for v in pose["scale_per_oct_axis"]) if pose else "n/a"),
        ("overlap", num(pose.get("overlap"), 3)),
        ("flags", "n/a" if "flags" not in res else ", ".join(res["flags"]) or "none"),
        ("boundary agreement in result.json (median, mm)", num(res.get("boundary_mm"))),
        ("boundary agreement, rim median forward / reverse", f"{num(rim[0])} / {num(rim[1])} mm"),
        ("OCT specimen mask", f"{num(mk.get('volume_cm3'))} cm3, Dice {num(mk.get('dice'), 3)} against the v1.1 mask "
                              f"({num(mk.get('reference_cm3'))} cm3)"),
        ("registration time, peak RAM, peak GPU", f"{num(wall / 60 if wall else None, 1)} min, {num(rss, 1)} GB, {num(gpu, 1)} GB"),
    ]
    ok_pose = p.get("corners_max_mm") is not None and p["corners_max_mm"] <= 1.0
    frame = ("passes" if fc["ok"] else "does not pass") if fc else "was not run"
    if not ev:
        verdict = "bench/evaluate.py has not been run on this run yet."
    else:
        verdict = (f"The pose is {'within' if ok_pose else 'not within'} the known run-to-run spread of R5 (about 1 mm at the block "
                   f"corners) and the raw-data frame check {frame}.")
        if not (ok_pose and fc.get("ok")):
            verdict += " By the spec this needs an explanation before release; the ablations below are the first place to look."
    ref = ev.get("reference_check", {})
    check = (f"R5 in the header frames (T_R5 @ A_spr @ inv(A_hdr)) agrees with the stored header-frame export to "
             f"{num(ref.get('formula_vs_stored_export_max_mm'), 6)} mm.")
    return ["## Main result", "", "| | |", "|---|---|"] + [f"| {a} | {v} |" for a, v in rows] + ["", verdict, "", check, ""]


def ablation_section(abl):
    if not abl:
        return ["## Ablations", "", "Not run yet (bench/ablate.py).", ""]
    V = abl["variants"]
    head = ["## Ablations", "",
            "Preprocessing once per OCT foreground source and destripe setting, then two-class maps, search and ladder per variant. "
            "Pose change is against base (the default method through the same driver) over the v1.1 specimen mask points; the boundary "
            "agreement uses the base masks for every variant, so it reflects the pose only.", "",
            "| variant | change | pose change: mean / corners max (mm) | vs R5 corners max (mm) | S | polarity | scale "
            "| rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for n, r in V.items():
        if "error" in r:
            rows.append(f"| {n} | {r['change']} | failed: {r['error']} | | | | | | {num(r['mask']['volume_cm3'])} "
                        f"({num(r['mask']['dice'], 3)}) | {num(r['seconds'], 0)} |")
            continue
        rim = r["boundary"]["rim_median_mm"]
        rows.append(f"| {n} | {r['change']} | {num(r['pose_to_base']['mean_mm'])} / {num(r['pose_to_base']['corners_max_mm'])} | "
                    f"{num(r['pose_to_R5']['corners_max_mm'])} | {num(r['S'], 4)} | {r['polarity']} | "
                    f"{' / '.join(f'{s:.3f}' for s in r['scales'])} | {num(rim[0])} / {num(rim[1])} | "
                    f"{num(r['mask']['volume_cm3'])} ({num(r['mask']['dice'], 3)}) | {num(r['seconds'], 0)} |")
    tail = [""]
    dc = abl.get("driver_check")
    if dc:
        tail += [f"Driver check: base through bench/ablate.py lies {num(dc['base_vs_main']['mean_mm'])} mm (corners max "
                 f"{num(dc['base_vs_main']['corners_max_mm'])} mm) from the CLI run.", ""]
    done = [n for n in STEP_OF if n in V and "error" not in V[n]]
    small = [f"{STEP_OF[n]} ({n}, {num(V[n]['pose_to_base']['mean_mm'])} mm)" for n in done if V[n]["within_0.5mm_of_base"]]
    large = [f"{STEP_OF[n]} ({n}, {num(V[n]['pose_to_base']['mean_mm'])} mm)" for n in done if not V[n]["within_0.5mm_of_base"]]
    if done:
        tail += [(f"Deletion rule (removal moves the pose by <= {DELETION_MM} mm and no other metric beyond noise): "
                 + (f"removing {', '.join(small)} stays within {DELETION_MM} mm, so these are deletion candidates once the other columns "
                    "are checked. " if small else "")
                 + (f"Removing {', '.join(large)} moves the pose further, so these steps stay." if large else "")).strip(), ""]
    return head + rows + tail


def runtime_section(abl):
    if not abl:
        return []
    out = ["## Runtime and memory of the ablation driver", "",
           "| step | seconds (destripe, pooling, writing) | foreground mask (s) | peak RAM of the process so far (GB) |",
           "|---|---|---|---|"]
    for k, p in abl["prep"].items():
        out.append(f"| prep {k} | {num(p.get('seconds'), 0)} | {num(p.get('mask_seconds'), 0)} | {num(p.get('peak_rss_gb'), 1)} |")
    grid = next((p["grid"] for p in abl["prep"].values() if "grid" in p), None)
    if grid:
        out.append(f"| OCT fine grid {'x'.join(map(str, grid['fine_shape']))} (streamed once) | {num(grid['seconds'], 0)} | | "
                   f"{num(grid['peak_rss_gb'], 1)} |")
    gpu = [r["gpu_peak_gb"] for r in abl["variants"].values() if r.get("gpu_peak_gb") is not None]
    out += [f"| all variants and preprocessing | {num(abl.get('seconds'), 0)} | | {num(abl.get('peak_rss_gb'), 1)} |", "",
            "A foreground mask is computed once per source and shared by the keys that use it (texture and texture_nodestripe).", "",
            f"Peak GPU memory allocated by torch over the variants: {num(max(gpu) if gpu else None, 1)} GB.", ""]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ablate", type=Path, default=None)
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("docs/BENCHMARK.md"))
    a = ap.parse_args()
    abl = load(a.ablate / "ablations.json") if a.ablate else {}
    phash = load(a.main / "result.json").get("params_hash") or abl.get("params_hash")
    intro = ["# Benchmark: Xiangrui's I58 brainstem pair", "",
             "octreg 1.0 registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at "
             "0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters"
             + (f" (Params hash {phash})" if phash else "") + ". "
             "The pair has no labels, so every number here is label-free. The reference is the v1.1 pose R5 converted into the header "
             "frames; it is a body-level pose (rim boundary agreement 2.02 / 1.05 mm, run-to-run spread about 1 mm at the block corners), "
             "not ground truth. Commands: `bash bench/run_xiangrui.sh` (see bench/README.md).", ""]
    text = "\n".join(intro + main_section(a.main, a.logs) + ablation_section(abl) + runtime_section(abl))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
