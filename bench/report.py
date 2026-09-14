#!/usr/bin/env python3
"""Write docs/BENCHMARK.md for Xiangrui's I58 brainstem pair from the bench outputs (formatting only, no computation).

    python bench/report.py --main RUN --ablate ABL [--logs RUN_logs] [-o docs/BENCHMARK.md] [--figures docs/figures]

RUN: the CLI run (result.json; eval.json from bench/evaluate.py). ABL: bench/ablate.py output (ablations.json).
LOGS (optional): bench/run_xiangrui.sh step logs, adding the wall clock (register.time) and the nvidia-smi peak (register.gpu_mib)
to the in-process time and memory of result.json. Missing values print n/a. The reading after the READING marker in the old
output is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DELETION_MM = 0.5            # spec: a step whose removal moves the pose by <= 0.5 mm (and no metric beyond noise) is deleted
STEP_OF = {"A1": "MRI flattening", "A2": "OCT flattening", "A6": "the scale prior"}
READING = "<!-- reading: written by hand below this line; bench/report.py keeps it when it rewrites the file -->"


def either(items):
    """'a', 'a or b', 'a, b or c'."""
    return " or ".join([", ".join(items[:-1]), items[-1]] if len(items) > 1 else items)


def load(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}


def num(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def wall_seconds(path):
    """Wall-clock seconds from a GNU time -v report, or None."""
    t = Path(path).read_text() if path and Path(path).exists() else ""
    wall = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", t)
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(wall.group(1).split(":")))) if wall else None


def gpu_peak_gb(path):
    vals = [float(v) for v in Path(path).read_text().split()] if path and Path(path).exists() else []
    return max(vals) / 1024 if vals else None


def rims(b):
    """Rim medians forward / reverse at the pose, then those of R5 and the previous run under the same masks."""
    pair = lambda r: f"{num(r[0])} / {num(r[1])}" if r else "n/a"
    other = "".join(f"; {n} {pair(r)}" for n, r in b.get("rim_median_mm_of", {}).items())
    return f"{pair(b.get('rim_median_mm'))} mm{other}" if b.get("rim_median_mm") else "n/a"


def main_section(run, logs):
    res, ev = load(run / "result.json"), load(run / "eval.json")
    pose, srch = res.get("pose", {}), res.get("search", {})
    p, fc, b, mk = ev.get("pose_to_R5", {}), ev.get("frame_check", {}), ev.get("boundary", {}), ev.get("mask", {})
    wall = wall_seconds(logs / "register.time" if logs else None)
    gpu = gpu_peak_gb(logs / "register.gpu_mib" if logs else None)
    sec = res.get("seconds", {})
    steps = ", ".join((f"search {num(srch.get('seconds'), 0)}, refinement {num(res['refine']['seconds'], 0)}"
                       if k == "search_refine" and "refine" in res else f"{k.replace('_', ' ')} {num(v, 0)}")
                      for k, v in sec.items() if k != "total")
    shifts = [v["spearman"] for k, v in fc.get("variants", {}).items() if k.startswith("shift") and v["spearman"] is not None]
    shift_max = max(shifts) if shifts else None
    pv = ev.get("pose_to_previous", {})
    rows = [
        ("pose vs R5, mean (max) over the v1.1 specimen mask", f"{num(p.get('mean_mm'))} ({num(p.get('max_mm'))}) mm"),
        ("pose vs R5 at the block corners, mean (max)", f"{num(p.get('corners_mean_mm'))} ({num(p.get('corners_max_mm'))}) mm"),
        ("rotation vs R5", f"{num(p.get('rotation_deg'), 1)} deg"),
        ("raw-data frame check (Spearman)", f"{num(fc.get('spearman_export'), 3)} at the pose; axis flips <= "
                                            f"{num(fc.get('max_flip_spearman'), 3)}; 2 mm shifts <= {num(shift_max, 3)}; "
                                            f"{('pass' if fc['ok'] else 'fail') if fc else 'n/a'}"),
        ("score S = (2 polarity S_class + S_outline) / 3; S_class, S_outline; polarity",
         f"{num(pose.get('S'), 4)}; {num(pose.get('S_class'), 4)}, {num(pose.get('S_outline'), 4)}; {pose.get('polarity', 'n/a')}"),
        ("search top-1 / top-2", f"{num(srch.get('top1'), 4)} / {num(srch.get('top2'), 4)}"),
        ("scale per OCT array axis", " / ".join(f"{v:.3f}" for v in pose["scale_per_oct_axis"]) if pose else "n/a"),
        ("flags", "n/a" if "flags" not in res else ", ".join(res["flags"]) or "none"),
        ("pose vs the previous run, mean / corners mean / corners max",
         f"{num(pv.get('mean_mm'))} / {num(pv.get('corners_mean_mm'))} / {num(pv.get('corners_max_mm'))} mm" if pv else "n/a"),
        ("boundary agreement in result.json (median, mm)", num(res.get("boundary_mm"))),
        ("rim boundary agreement forward / reverse, method masks", rims(b)),
        ("the same with the v1.1 masks", rims(ev.get("boundary_v11_masks", {}))),
        ("OCT specimen mask", f"{num(mk.get('volume_cm3'))} cm3, Dice {num(mk.get('dice'), 3)} against the v1.1 mask "
                              f"({num(mk.get('reference_cm3'))} cm3)"),
        ("registration time in process, peak RAM, peak GPU memory allocated by torch",
         f"{num(sec['total'] / 60 if 'total' in sec else None, 1)} min, {num(res.get('peak_rss_gb'), 1)} GB, "
         f"{num(res.get('gpu_peak_gb'), 2)} GB" + (f" (wall clock {num(wall / 60, 1)} min, nvidia-smi peak {num(gpu, 1)} GB)" if wall else "")),
        ("time per step (s)", steps or "n/a"),
    ]
    ok_pose = p.get("corners_max_mm") is not None and p["corners_max_mm"] <= 1.0
    frame = ("passes" if fc["ok"] else "does not pass") if fc else "was not run"
    if not ev:
        verdict = "bench/evaluate.py has not been run on this run yet."
    else:
        verdict = (f"The pose is {'within' if ok_pose else 'not within'} the spec's 1 mm tolerance for R5 at the block corners, and the "
                   f"raw-data frame check {frame}.")
        if not (ok_pose and fc.get("ok")):
            verdict += " The reading at the end discusses this."
    ref = ev.get("reference_check", {})
    check = (f"R5 in the header frames (T_R5 @ A_spr @ inv(A_hdr)) agrees with the stored header-frame export to "
             f"{num(ref.get('formula_vs_stored_export_max_mm'), 6)} mm.")
    return ["## Main result", "", "| | |", "|---|---|"] + [f"| {a} | {v} |" for a, v in rows] + ["", verdict, "", check, ""]


def ablation_row(n, r):
    if "error" in r:
        return (f"| {n} | {r['change']} | failed: {r['error']} | | | | | | | {num(r['mask']['volume_cm3'])} "
                f"({num(r['mask']['dice'], 3)}) | {num(r['seconds'], 0)} |")
    rim = r["boundary"]["rim_median_mm"]
    d = r["pose_to_base"]
    return (f"| {n} | {r['change']} | {num(d['mean_mm'])} / {num(d['corners_mean_mm'])} / {num(d['corners_max_mm'])} | "
            f"{num(r['pose_to_R5']['corners_max_mm'])} | {num(r['S'], 4)} | {num(r.get('L'), 4)} | {r['polarity']} | "
            f"{' / '.join(f'{s:.3f}' for s in r['scales'])} | {num(rim[0])} / {num(rim[1])} | "
            f"{num(r['mask']['volume_cm3'])} ({num(r['mask']['dice'], 3)}) | {num(r['seconds'], 0)} |")


def ablation_section(abl):
    if not abl:
        return ["## Ablations", "", "Not run yet (bench/ablate.py).", ""]
    V = abl["variants"]
    cols = ("| variant | change | pose change: mean / corners mean / corners max (mm) | vs R5 corners max (mm) | S | L | polarity | scale "
            "| rim fwd / rev (mm) | OCT mask cm3 (Dice) | time (s) |")
    head = ["## Ablations", "",
            "Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). "
            "Pose change is against base (the method through the same driver), as the mean over the v1.1 specimen-mask points and "
            "the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every variant, "
            "so it reflects the pose only.", "",
            cols, "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = [ablation_row(n, r) for n, r in V.items()]
    tail = [""]
    dc = abl.get("driver_check")
    if dc:
        tail += [f"Driver check: base through bench/ablate.py lies {num(dc['base_vs_main']['mean_mm'])} mm (corners max "
                 f"{num(dc['base_vs_main']['corners_max_mm'])} mm) from the CLI run.", ""]
    done = [n for n in STEP_OF if n in V and "error" not in V[n]]
    small = [f"{STEP_OF[n]} ({n}, {num(V[n]['pose_to_base']['mean_mm'])} mm)" for n in done if V[n]["within_0.5mm_of_base"]]
    large = [f"{STEP_OF[n]} ({n}, {num(V[n]['pose_to_base']['mean_mm'])} mm)" for n in done if not V[n]["within_0.5mm_of_base"]]
    if done:
        tail += [(f"Deletion rule of the spec: a step goes when removing it moves the pose by at most {DELETION_MM} mm (mean over the "
                  "specimen-mask points) and changes no other metric beyond noise. "
                  + (f"Removing {either(small)} stays within {DELETION_MM} mm. " if small else "")
                  + (f"Removing {either(large)} moves the pose further." if large else "")).strip(), ""]
    rm = abl.get("removed_steps")
    if rm and rm.get("rows"):
        pb = rm["present_base_vs_previous_base"]
        tail += ["### Removed steps", "", rm["note"], "", cols, "|---|---|---|---|---|---|---|---|---|---|---|"]
        tail += [ablation_row(n, r) for n, r in rm["rows"].items()]
        tail += ["", f"The present base lies {num(pb['mean_mm'])} mm (corners mean {num(pb['corners_mean_mm'])} mm, corners max "
                 f"{num(pb['corners_max_mm'])} mm, rotation "
                 f"{num(pb['rotation_deg'], 2)} deg) from the base of that run. Source: {rm['source']}.", ""]
    return head + rows + tail


def runtime_section(abl):
    if not abl:
        return []
    out = ["## Runtime and memory of the ablation driver", "",
           "| step | seconds | peak RAM of the process so far (GB) |", "|---|---|---|"]
    grid = next((p["grid"] for p in abl["prep"].values() if "grid" in p), None)
    if grid:
        out.append(f"| OCT fine grid {'x'.join(map(str, grid['fine_shape']))} (streamed once) | {num(grid['seconds'], 0)} | "
                   f"{num(grid['peak_rss_gb'], 1)} |")
    for k, p in abl["prep"].items():
        out.append(f"| prep {k} | {num(p.get('mask_seconds', p.get('seconds')), 0)} | {num(p.get('peak_rss_gb'), 1)} |")
    gpu = [r["gpu_peak_gb"] for r in abl["variants"].values() if r.get("gpu_peak_gb") is not None]
    out += [f"| all variants and preprocessing | {num(abl.get('seconds'), 0)} | {num(abl.get('peak_rss_gb'), 1)} |", "",
            f"Peak GPU memory allocated by torch over the variants: {num(max(gpu) if gpu else None, 1)} GB.", ""]
    return out


SHORT = {"A0": "intensity OCT mask", "A0b": "v1.1 watershed OCT mask", "A0c": "holes filled in 3-D only", "A8": "other handedness",
         "A1": "MRI flattening off", "A2": "OCT flattening off",
         "A3": "destripe off, ladder kept", "A4": "intensity channels", "A5+1": "polarity forced +1", "A5-1": "polarity forced -1",
         "A6": "no scale prior", "A7": "ladder off, destripe kept"}


def figures(run, abl, out):
    """out/fig_qc_xiangrui.png: RUN/qc.png as 8-bit grey without the white margin. out/fig_ablation.png: block-corner mean pose change of every variant
    against the final pose (variants: against base; removed steps: against the present base), log scale."""
    from matplotlib.figure import Figure
    from PIL import Image, ImageOps
    out.mkdir(parents=True, exist_ok=True)
    qc = Image.open(run / "qc.png").convert("L")
    x0, y0, x1, y1 = ImageOps.invert(qc).getbbox()
    qc.crop((max(x0 - 8, 0), max(y0 - 8, 0), min(x1 + 8, qc.width), min(y1 + 8, qc.height))).save(out / "fig_qc_xiangrui.png", optimize=True)
    rows = [(n, r.get("pose_to_base", {}).get("corners_mean_mm"), r.get("error"), False) for n, r in abl["variants"].items() if n != "base"]
    rows += [(n, r["pose_to_present_base"]["corners_mean_mm"], None, True) for n, r in abl.get("removed_steps", {}).get("rows", {}).items()]
    rows.sort(key=lambda r: math.inf if r[1] is None else r[1])
    lo, ink, grey = 1e-3, "#0b0b0b", "#52514e"
    fig = Figure(figsize=(7.6, 0.34 * len(rows) + 1.4))
    ax = fig.add_axes([0.39, 0.2, 0.58, 0.74])
    for i, (n, d, err, removed) in enumerate(rows):
        if d is not None and d > lo:
            ax.barh(i, d - lo, left=lo, height=0.6, color="#d6d5d0" if removed else "#2a78d6", hatch="///" if removed else None,
                    edgecolor=grey if removed else "#2a78d6", linewidth=0)
        label = (f"failed: {err.split(':')[-1].strip()[:40]}" if err else "same pose" if d == 0 else
                 f"< {lo:g} mm" if d < lo else f"{d:.2g} mm")
        ax.text(max(d or lo, lo) * 1.2, i, label, va="center", fontsize=8, color=grey,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
    ax.axvline(0.5, color=ink, linestyle="--", linewidth=1, zorder=1)
    ax.text(0.5 * 1.1, len(rows) - 0.4, "0.5 mm deletion threshold", fontsize=8, color=ink, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(lo, 400)
    ax.set_ylim(-0.6, len(rows) + 0.2)
    ax.set_yticks(range(len(rows)), [f"{n}  {SHORT.get(n, n)}" + ("  (first run)" if rm else "") for n, _, _, rm in rows], fontsize=8.5)
    ax.set_xlabel("pose change against the final pose, block-corner mean (mm)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, colors=grey)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.text(0.02, 0.02, "Hatched: the two steps removed from the method, as variants of the first ablation run (one step off, the other\n"
             "still on), measured against the final pose, which has neither.", fontsize=7, color=grey)
    fig.savefig(out / "fig_ablation.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ablate", type=Path, default=None)
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("docs/BENCHMARK.md"))
    ap.add_argument("--figures", type=Path, default=None, help="also write fig_qc_xiangrui.png and fig_ablation.png into this dir")
    a = ap.parse_args()
    abl = load(a.ablate / "ablations.json") if a.ablate else {}
    phash = load(a.main / "result.json").get("params_hash") or abl.get("params_hash")
    intro = ["# Benchmark: Xiangrui's I58 brainstem pair", "",
             "octreg 1.0 registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at "
             "0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters"
             + (f" (Params hash {phash})" if phash else "") + ". "
             "The pair has no labels, so every number here is label-free. The reference R5 is the pose of the earlier research "
             "pipeline (v1.1) converted into the header frames. It is a body-level pose, not ground truth. The numbers are read from "
             "the run's result.json and eval.json and from ablations.json (copies in docs/results/xiangrui_I58/). "
             "Commands: `bash bench/run_xiangrui.sh` (see bench/README.md).", ""]
    text = "\n".join(intro + main_section(a.main, a.logs) + ablation_section(abl) + runtime_section(abl))
    old = a.out.read_text() if a.out.exists() else ""
    if READING in old:                                   # the hand-written reading at the end survives a rewrite
        text += "\n" + READING + old.split(READING, 1)[1]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f"wrote {a.out}")
    if a.figures and abl:
        figures(a.main, abl, a.figures)
        print(f"wrote {a.figures}/fig_qc_xiangrui.png, fig_ablation.png")


if __name__ == "__main__":
    main()
