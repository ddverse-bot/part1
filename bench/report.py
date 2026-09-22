#!/usr/bin/env python3
"""Write bench/BENCHMARK.md for Xiangrui's I58 brainstem pair from the bench outputs (formatting only, no computation).

    python bench/report.py --main RUN --ablate ABL [--logs RUN_logs] [-o bench/BENCHMARK.md] [--figures bench/figures]
        [--store bench/results/xiangrui_I58]

RUN: the CLI run (result.json; eval.json from bench/evaluate.py). ABL: bench/ablate.py output (ablations.json).
LOGS (optional): bench/run_xiangrui.py step logs, adding the wall clock (register.time) and the nvidia-smi peak (register.gpu_mib)
to the in-process time and memory of result.json. Missing values print n/a. Two hand-written parts of the old output are kept:
the '## Visual result' section before '## Main result', and the reading after the READING marker.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DELETION_MM = 0.5            # a step whose removal moves the pose by <= 0.5 mm (and no metric beyond noise) is deleted
STEP_OF = {"A0c": "per-plane hole filling", "A1": "MRI flattening", "A2": "OCT flattening", "A4": "the two-class maps",
           "A6": "the scale prior", "A6c": "the scale clamp", "A9": "the outline term", "A10": "the one-sided outline",
           "A12": "the fine-structure refinement"}
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
    """Rim medians forward / reverse at the pose, then those of any other pose (the previous run) under the same masks."""
    pair = lambda r: f"{num(r[0])} / {num(r[1])}" if r else "n/a"
    other = "".join(f"; {n} {pair(r)}" for n, r in (b or {}).get("rim_median_mm_of", {}).items())
    return f"{pair(b.get('rim_median_mm'))} mm{other}" if (b or {}).get("rim_median_mm") else "n/a"


def arrow(pair, scale=1.0, nd=2):
    """'before -> after' of a [before, after] read-out."""
    return " -> ".join(num(None if x is None else scale * x, nd) for x in (pair or [None, None]))


def deform_rows(d):
    """Table rows of the smooth deformation (§6) read-outs (result.json 'deform'); none when the run has no such block."""
    if not d:
        return []
    r, f, cv = d.get("residual", {}), d.get("field", {}), d.get("cv", {})
    model = f"control lattice {num(d.get('grid_mm'), 0)} mm, lambda {num(d.get('lam'), 2)}" if d.get("status") == "applied" else "no field"
    return [("smooth deformation (§6): status; model; interior matches, supported boundary points used by the fit",
             f"{d.get('status', 'n/a')}; {model}; {d.get('n_interior', 'n/a')}, {d.get('n_boundary', 'n/a')}"),
            ("§6 held-out median error, interior + boundary: no deformation -> chosen lambda",
             f"{arrow([cv['none'][0], cv['chosen'][0]], nd=3)} mm + {arrow([cv['none'][1], cv['chosen'][1]], nd=3)} mm"
             if cv.get("none") and cv.get("chosen") else "n/a"),
            ("§6 residuals affine -> deformed, measured again: interior matches; surface-edge offsets (within 0.3 mm)",
             f"{arrow(r.get('interior_mm'))} mm; {arrow(r.get('boundary_mm'))} mm ({arrow(r.get('boundary_within_0.3mm'), 100, 0)} %)"),
            ("§6 field over the MRI foreground, median / p95 / max; max strain",
             f"{num(f.get('median_mm'))} / {num(f.get('p95_mm'))} / {num(f.get('max_mm'))} mm; {num(d.get('max_strain'))}")]


def main_section(run, logs):
    res, ev = load(run / "result.json"), load(run / "eval.json")
    pose, srch = res.get("pose", {}), res.get("search", {})
    fc, b, mk = ev.get("frame_check", {}), ev.get("boundary", {}), ev.get("mask", {})
    wall = wall_seconds(logs / "register.time" if logs else None)
    gpu = gpu_peak_gb(logs / "register.gpu_mib" if logs else None)
    sec = res.get("seconds", {})
    ngf_s = res.get("ngf", {}).get("seconds")
    steps = ", ".join((f"search {num(srch.get('seconds'), 0)}, refinement {num(res['refine']['seconds'], 0)}"
                       + (f", fine-structure refinement {num(ngf_s, 0)}" if ngf_s is not None else "")
                       if k == "search_refine_ngf" and "refine" in res else
                       f"smooth deformation {num(v, 0)}" if k == "deform" else f"{k.replace('_', ' ')} {num(v, 0)}")
                      for k, v in sec.items() if k != "total")
    shifts = [v["spearman"] for k, v in fc.get("variants", {}).items() if k.startswith("shift") and v["spearman"] is not None]
    shift_max = max(shifts) if shifts else None
    pv = ev.get("pose_to_previous", {})
    rows = [
        ("raw-data frame check (Spearman)", f"{num(fc.get('spearman_export'), 3)} at the pose; axis flips <= "
                                            f"{num(fc.get('max_flip_spearman'), 3)}; 2 mm shifts <= {num(shift_max, 3)}; "
                                            f"{('pass' if fc['ok'] else 'fail') if fc else 'n/a'}"),
        ("score S of the §4 pose = (2 polarity S_class + S_outline) / 3; S_class, S_outline; polarity",
         f"{num(pose.get('S'), 4)}; {num(pose.get('S_class'), 4)}, {num(pose.get('S_outline'), 4)}; {pose.get('polarity', 'n/a')}"),
        ("search top-1 / top-2", f"{num(srch.get('top1'), 4)} / {num(srch.get('top2'), 4)}"),
        ("fine-structure agreement F (§5), at the §4 pose -> refined; handedness",
         f"{num(pose.get('NGF_start'), 4)} -> {num(pose.get('NGF'), 4)}; {pose.get('handedness', 'n/a')}"),
        ("pose change of §5 (block-corner mean)", f"{num(pose.get('ngf_shift_mm'))} mm"),
        ("scale per OCT array axis", " / ".join(f"{v:.3f}" for v in pose["scale_per_oct_axis"]) if pose else "n/a"),
        *deform_rows(res.get("deform")),
        ("flags", "n/a" if "flags" not in res else ", ".join(res["flags"]) or "none"),
        ("pose vs the previous run, mean / corners mean / corners max"
         + (f" (the mean is over {pv['points']})" if pv.get("points") else ""),
         f"{num(pv.get('mean_mm'))} / {num(pv.get('corners_mean_mm'))} / {num(pv.get('corners_max_mm'))} mm" if pv else "n/a"),
        ("rim boundary agreement forward / reverse, method masks", rims(b)),
        ("OCT specimen mask", f"{num(mk.get('volume_cm3'))} cm3"),
        ("registration time in process, peak RAM, peak GPU memory allocated by torch",
         f"{num(sec['total'] / 60 if 'total' in sec else None, 1)} min, {num(res.get('peak_rss_gb'), 1)} GiB, "
         f"{num(res.get('gpu_peak_gb'), 2)} GiB" + (f" (wall clock {num(wall / 60, 1)} min, nvidia-smi peak {num(gpu, 1)} GiB)" if wall else "")),
        ("time per step (s)", steps or "n/a"),
    ]
    frame = ("passes" if fc["ok"] else "does not pass") if fc else "was not run"
    verdict = ("bench/evaluate.py has not been run on this run yet." if not ev else
               f"The raw-data frame check {frame}. The pair has no labels, so none of these numbers is a success criterion.")
    return ["## Main result", "", "| | |", "|---|---|"] + [f"| {a} | {v} |" for a, v in rows] + ["", verdict, ""]


def ablation_row(n, r):
    """One row of the ablation table. Every read-out is optional, so a table written by another release still prints."""
    vol = num((r.get("mask") or {}).get("volume_cm3"))
    if "error" in r:
        return f"| {n} | {r['change']} | failed: {r['error']} | | | | | | | {vol} | {num(r.get('seconds'), 0)} |"
    rim = (r.get("boundary") or {}).get("rim_median_mm") or [None, None]
    d, d4 = r.get("pose_to_base", {}), r.get("pose_to_base_s4", {})
    return (f"| {n} | {r['change']} | {num(d4.get('mean_mm'))} / {num(d4.get('corners_mean_mm'))} | "
            f"{num(d.get('mean_mm'))} / {num(d.get('corners_mean_mm'))} / {num(d.get('corners_max_mm'))} | "
            f"{num(r.get('S'), 4)} | {num(r.get('L'), 4)} | "
            f"{r.get('polarity', 'n/a')} | {' / '.join(f'{s:.3f}' for s in r.get('scales', [])) or 'n/a'} | "
            f"{num(rim[0])} / {num(rim[1])} | {vol} | {num(r.get('seconds'), 0)} |")


def ablation_section(abl):
    if not abl:
        return ["## Ablations", "", "Not run yet (bench/ablate.py).", ""]
    V = abl["variants"]
    cols = ("| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) "
            "| S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 | time (s) |")
    rule = "|---|---|---|---|---|---|---|---|---|---|---|"
    head = ["## Ablations", "",
            "Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). "
            "Pose change is against base (the method through the same driver), as the mean over the points of the base specimen "
            "mask and the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every "
            "variant, so it reflects the pose only. The smooth deformation (§6) leaves the pose alone, so no variant runs it: "
            "'no §6' would be base with the identical pose, and its read-outs are the 'before' residuals in the Main result "
            "table above.", "",
            cols, rule]
    rows = [ablation_row(n, r) for n, r in V.items()]
    tail = [""]
    dc = abl.get("driver_check")
    if dc:
        tail += [f"Driver check: base through bench/ablate.py lies {num(dc['base_vs_main']['mean_mm'])} mm (corners max "
                 f"{num(dc['base_vs_main']['corners_max_mm'])} mm) from the CLI run.", ""]
    done = [n for n in STEP_OF if n in V and "error" not in V[n]]
    key = lambda n: "pose_to_base" if n == "A12" else "pose_to_base_s4"            # steps of §1-4 act through the §4 pose
    small = [f"{STEP_OF[n]} ({n}, {num(V[n][key(n)]['mean_mm'])} mm)" for n in done if V[n][key(n)]["mean_mm"] <= DELETION_MM]
    large = [f"{STEP_OF[n]} ({n}, {num(V[n][key(n)]['mean_mm'])} mm)" for n in done if V[n][key(n)]["mean_mm"] > DELETION_MM]
    if done:
        tail += [(f"Deletion rule: a step goes when removing it moves the pose by at most {DELETION_MM} mm (mean over the "
                  "specimen-mask points; the §4 pose for steps of §1-4, the final pose for §5), changes no other metric beyond "
                  "noise, and no test outside this pair shows it load bearing. "
                  + (f"Removing {either(small)} stays within {DELETION_MM} mm, and the evidence for keeping each is in "
                     "docs/METHOD.md after the ablation table. " if small else "")
                  + (f"Removing {either(large)} moves the pose further." if large else "")).strip(), ""]
    groups = [g for g in abl.get("removed_steps", []) if g.get("rows")]
    if groups:
        tail += ["### Removed steps", ""]
    for g in groups:
        pb = g["present_base_vs_previous_base"]
        tail += [g["note"], "", cols, rule]
        tail += [ablation_row(n, r) for n, r in g["rows"].items()]
        tail += ["", f"The present base lies {num(pb['mean_mm'])} mm (corners mean {num(pb['corners_mean_mm'])} mm, corners max "
                 f"{num(pb['corners_max_mm'])} mm, rotation {num(pb['rotation_deg'], 2)} deg) from the base of that run. "
                 f"Source: {Path(g['source']).parent.name}/{Path(g['source']).name}.", ""]
    return head + rows + tail


def runtime_section(abl):
    if not abl:
        return []
    out = ["## Runtime and memory of the ablation driver", "",
           "| step | seconds | peak RAM (GiB) |", "|---|---|---|"]
    grid = next((p["grid"] for p in abl["prep"].values() if "grid" in p), None)
    if grid:
        out.append(f"| OCT fine grid {'x'.join(map(str, grid['fine_shape']))} (streamed once) | {num(grid['seconds'], 0)} | "
                   f"{num(grid['peak_rss_gb'], 1)} |")
    for k, p in abl["prep"].items():
        out.append(f"| prep {k} | {num(p.get('mask_seconds', p.get('seconds')), 0)} | {num(p.get('peak_rss_gb'), 1)} |")
    gpu = [r["gpu_peak_gb"] for r in abl["variants"].values() if r.get("gpu_peak_gb") is not None]
    out += [f"| all variants and preprocessing | {num(abl.get('seconds'), 0)} | {num(abl.get('peak_rss_gb'), 1)} |", "",
            f"Peak GPU memory allocated by torch over the variants: {num(max(gpu) if gpu else None, 1)} GiB.", ""]
    return out


def figures(run, out):
    """The three figures bench/BENCHMARK.md shows, from the run's own qc images: out/fig_qc_xiangrui.png is RUN/qc.png
    without the white margin, outline colours kept, and out/fig_qc_montage_xiangrui.png and out/fig_qc_deform_xiangrui.png are
    RUN/qc_montage.png and RUN/qc_deform.png as they are. The ablation distances are in the table, which needs no picture."""
    import shutil
    from PIL import Image, ImageOps
    out.mkdir(parents=True, exist_ok=True)
    qc = Image.open(run / "qc.png").convert("RGB")
    x0, y0, x1, y1 = ImageOps.invert(qc.convert("L")).getbbox()
    qc.crop((max(x0 - 8, 0), max(y0 - 8, 0), min(x1 + 8, qc.width), min(y1 + 8, qc.height))).save(out / "fig_qc_xiangrui.png", optimize=True)
    for src, dst in (("qc_montage.png", "fig_qc_montage_xiangrui.png"), ("qc_deform.png", "fig_qc_deform_xiangrui.png")):
        if (run / src).exists():
            shutil.copyfile(run / src, out / dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ablate", type=Path, default=None)
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("bench/BENCHMARK.md"))
    ap.add_argument("--figures", type=Path, default=None, help="also write the qc figures BENCHMARK.md shows into this dir")
    ap.add_argument("--store", type=Path, default=None,
                    help="also copy the run's result.json and eval.json and the ablations.json into this dir "
                         "(bench/results/xiangrui_I58), so the numbers the document quotes travel with it")
    a = ap.parse_args()
    abl = load(a.ablate / "ablations.json") if a.ablate else {}
    phash = load(a.main / "result.json").get("params_hash") or abl.get("params_hash")
    intro = ["# Benchmark: Xiangrui's I58 brainstem pair", "",
             "octreg registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI; MRI crop 343x489x495 at "
             "0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters"
             + (f" (Params hash {phash})" if phash else "") + ". "
             "The pair has no labels, so every number here is label-free. "
             "The numbers are read from the run's result.json and eval.json and from ablations.json (copies in bench/results/xiangrui_I58/). "
             "Commands: `python bench/run_xiangrui.py` (see bench/README.md).", ""]
    old = a.out.read_text(encoding="utf-8") if a.out.exists() else ""
    visual = re.search(r"^## Visual result\n.*?(?=^## Main result)", old, re.S | re.M)       # hand-written, kept
    text = "\n".join(intro + ([visual.group(0).rstrip("\n"), ""] if visual else []) + main_section(a.main, a.logs)
                     + ablation_section(abl) + runtime_section(abl))
    if READING in old:                                   # the hand-written reading at the end survives a rewrite
        text += "\n" + READING + old.split(READING, 1)[1]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(text.encode("utf-8"))             # UTF-8 and LF on every OS
    print(f"wrote {a.out}")
    if a.figures:
        figures(a.main, a.figures)
        print(f"wrote the figures of {a.out.name} into {a.figures}")
    if a.store:
        import shutil
        a.store.mkdir(parents=True, exist_ok=True)
        for src in [a.main / "result.json", a.main / "eval.json"] + ([a.ablate / "ablations.json"] if a.ablate else []):
            if src.exists():
                shutil.copyfile(src, a.store / src.name)
                print(f"stored {a.store / src.name}")


if __name__ == "__main__":
    main()
