#!/usr/bin/env python3
"""DANDI regression check between two register.py runs (octreg v1.1 acceptance 5.3 / 5.4):

    python scripts/regress.py work/runs/v11/I46_v1snap work/runs/v11/I46_v11                 # 5.3 table (--tol default)
    python scripts/regress.py work/runs/v11/I46_v11    work/runs/v11/I46_v11_fine --tol fine  # 5.4 fine-gate table

REF_RUN and NEW_RUN are run directories (result.json, T_oct2mri.npy); the two --work dirs are read from result.json args.work
(relative paths are resolved against the cwd, then against the run dir's ancestors; --work-ref/--work-new override).  Prints one
PASS/FAIL/n-a row per criterion and exits 1 on any FAIL.  Tolerances are the hard-coded table below; override single entries
with --set KEY=VALUE (e.g. --set G4.corner_mean_mm=0.5).  A criterion whose inputs are absent in BOTH runs is 'n/a' (not a
failure); absent in only one of them is a FAIL unless the row says otherwise."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.evaluate import transform_diff

TOL = {  # 5.3 DANDI regression (v1 snapshot vs v1.1 defaults, same prep)
    "G1.mask_ratio_max": 1.5,                 # auto mask rule must pick intensity: V_int / V_mri <= 1.5 (from prep.json)
    "G2.search_top": 0.005, "G2.final_ncc": 0.005, "G2.restarts_converged": 1,
    "G3.vascular_moved_mm": 0.1, "G3.vascular_restarts": 1, "G3.min_vessel_ncc": 0.02,
    "G4.corner_mean_mm": 0.3, "G4.rotation_deg": 0.5, "G4.centre_mm": 0.3, "G4.stretch": 0.01,
    "G5.dice": 0.005, "G5.vessel_median_um": 10.0, "G5.frac150": 0.02,
    "G6.wall_factor": 1.1,
}
TOL_FINE = {  # 5.4 DANDI fine-gate test (REF = v1.1 defaults, NEW = --fine on): applies only when NEW accepted the fine pose
    "F.corner_mean_mm": 0.5, "F.dice_drop": 0.01, "F.vessel_median_increase_um": 15.0, "F.frac150_drop": 0.03,
    "F.stretch_depth": 0.02, "F.fine_seconds": 600.0, "F.U_vs_residual_factor": 0.8,
}


def get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d


def resolve_work(run: Path, res: dict, override: Path | None) -> Path | None:
    if override is not None: return override
    w = get(res, "args.work")
    if w is None: return None
    p = Path(w)
    if p.is_absolute() and p.exists(): return p
    cands = [Path.cwd() / p] + [anc / p for anc in run.resolve().parents]
    for c in cands:
        if (c / "prep.json").exists(): return c
    return Path.cwd() / p


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return h.hexdigest()


def prep_volumes(prep: dict, A_oct: np.ndarray | None):
    """(V_int cm3 of the intensity/ chosen 0.15 mm mask, V_mri cm3) from prep.json (None when absent)."""
    o = prep.get("oct", {}); m = prep.get("mri", {})
    vox = float(np.linalg.norm(A_oct[:3, :3], axis=0).mean()) if A_oct is not None else 0.15
    frac = o.get("tissue_frac150_intensity", o.get("tissue_frac150")); shp = o.get("shape150")
    V_int = float(frac) * float(np.prod(shp)) * vox ** 3 / 1000.0 if (frac is not None and shp) else None
    V_mri = o.get("V_mri_cm3")
    if V_mri is None and m.get("tissue_fraction") is not None and m.get("shape") and m.get("voxel_mm"):
        V_mri = float(m["tissue_fraction"]) * float(np.prod(m["shape"])) * float(np.prod(m["voxel_mm"])) / 1000.0
    return V_int, V_mri


class Table:
    def __init__(self): self.rows = []; self.n_fail = 0
    def row(self, cid, desc, ok, ref=None, new=None, delta=None, tol=None, note=""):
        """ok: True/False/None (None = n/a)."""
        st = "n/a " if ok is None else ("PASS" if ok else "FAIL")
        if ok is False: self.n_fail += 1
        self.rows.append((st, cid, desc, ref, new, delta, tol, note))
    def show(self):
        f = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) else str(v))
        print(f"{'':4s} {'id':5s} {'criterion':52s} {'ref':>12s} {'new':>12s} {'delta':>10s} {'tol':>8s}  note")
        for st, cid, desc, ref, new, delta, tol, note in self.rows:
            print(f"{st:4s} {cid:5s} {desc[:52]:52s} {f(ref):>12s} {f(new):>12s} {f(delta):>10s} {f(tol):>8s}  {note}")
        print(f"{self.n_fail} FAIL of {len(self.rows)} rows" + (" -> REGRESSION" if self.n_fail else " -> unchanged within tolerance"))
    def as_json(self):
        return [{"status": r[0].strip(), "id": r[1], "criterion": r[2], "ref": r[3], "new": r[4], "delta": r[5], "tol": r[6], "note": r[7]} for r in self.rows]


def both(a, b):
    """Convenience for 'n/a if absent in both, FAIL if absent in one'."""
    if a is None and b is None: return None
    if a is None or b is None: return False
    return True


def default_table(T: Table, ref, new, ref_run: Path, new_run: Path, work_ref: Path | None, work_new: Path | None, tol: dict):
    # ---------------------------------------------------------------- G1 prep untouched / mask rule
    if work_ref is not None and work_new is not None and (work_ref / "prep.json").exists() and (work_new / "prep.json").exists():
        h0, h1 = md5(work_ref / "prep.json"), md5(work_new / "prep.json")
        T.row("G1a", "prep.json identical for both runs (md5)", h0 == h1, h0[:8], h1[:8], note=f"{work_ref} | {work_new}")
        mt = os.path.getmtime(work_new / "prep.json"); starts = []
        for run, r in ((ref_run, ref), (new_run, new)):
            if (run / "result.json").exists() and r.get("total_seconds") is not None: starts.append(os.path.getmtime(run / "result.json") - float(r["total_seconds"]))
        if starts: T.row("G1b", "prep.json not modified after the runs started (mtime)", mt <= min(starts) + 1.0, note=f"prep mtime - earliest run start = {mt - min(starts):+.0f} s")
        else: T.row("G1b", "prep.json not modified after the runs started (mtime)", None)
        prep_new = json.load(open(work_new / "prep.json")); A_oct = np.load(work_new / "oct150_affine.npy") if (work_new / "oct150_affine.npy").exists() else None
        V_int, V_mri = prep_volumes(prep_new, A_oct)
        if V_int is not None and V_mri is not None and V_mri > 0:
            T.row("G1c", "auto mask rule picks intensity: V_int/V_mri <= 1.5", V_int / V_mri <= tol["G1.mask_ratio_max"], V_int, V_mri, V_int / V_mri, tol["G1.mask_ratio_max"], "ref col = V_int cm3, new col = V_mri cm3, delta = ratio (prep.json)")
        else: T.row("G1c", "auto mask rule picks intensity: V_int/V_mri <= 1.5", None, note="volumes not derivable from prep.json")
    else:
        T.row("G1a", "prep.json identical for both runs (md5)", None, note="work dir(s) not found; use --work-ref/--work-new")
    mm = get(new, "prep_summary.mask_mode")
    T.row("G1d", "new run logs prep_summary.mask_mode null (no texture mask)", mm is None, None, mm, note="v1.1 register logs prep_summary; legacy prep -> null")
    Vo, Vm = get(new, "overlap_gate.V_oct_mask_cm3"), get(new, "overlap_gate.V_mri_tissue_cm3")
    if Vo is not None and Vm:
        T.row("G1e", "no over-inclusive-mask warning: V_oct_mask <= 1.5 V_mri", Vo <= 1.5 * Vm, get(ref, "overlap_gate.V_oct_mask_cm3"), Vo, Vo / Vm, 1.5, "ratio shown in delta")
    # ---------------------------------------------------------------- G2 structural stage
    m0, m1 = get(ref, "final_transform.mirror"), get(new, "final_transform.mirror")
    T.row("G2a", "same handedness (final_transform.mirror)", m0 == m1 if both(m0, m1) else both(m0, m1), m0, m1)
    p0, p1 = get(ref, "oct_wm_bright"), get(new, "oct_wm_bright")
    T.row("G2b", "same oct_wm_bright", p0 == p1 if both(p0, p1) else both(p0, p1), p0, p1)
    for k in ("top1", "top2"):
        a, b = get(ref, f"search.{k}"), get(new, f"search.{k}")
        if a is None or b is None: T.row("G2c", f"search {k} |delta| <= {tol['G2.search_top']}", None, a, b, note="None (search skipped: --init-transform)")
        else: T.row("G2c", f"search {k} |delta| <= {tol['G2.search_top']}", abs(a - b) <= tol["G2.search_top"], a, b, b - a, tol["G2.search_top"])
    a, b = get(ref, "refine.final_ncc"), get(new, "refine.final_ncc")
    T.row("G2d", f"refine.final_ncc |delta| <= {tol['G2.final_ncc']}", (abs(a - b) <= tol["G2.final_ncc"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G2.final_ncc"])
    a, b = get(ref, "evaluation.restarts.n_converged"), get(new, "evaluation.restarts.n_converged")
    T.row("G2e", f"structural restarts n_converged within +-{tol['G2.restarts_converged']}", (abs(a - b) <= tol["G2.restarts_converged"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G2.restarts_converged"])
    # ---------------------------------------------------------------- G3 vascular stage
    vr, vn = ref.get("vascular", {}), new.get("vascular", {})
    if vn.get("mode") in ("off", "skipped_section_artefact") and vr.get("mode") in ("off", "skipped_section_artefact"):
        T.row("G3a", "vascular.accepted == true", None, vr.get("mode"), vn.get("mode"), note="vascular off/skipped in both")
    else:
        acc = vn.get("accepted")
        T.row("G3a", "vascular.accepted == true (new run)", acc is True, "n/a (v1)" if "accepted" not in vr else vr.get("accepted"), acc, note="" if "accepted" in vn else "key missing: legacy register or vascular not run")
        nc0, nc1 = (vr.get("ncc_channels") or [None] * 3)[-1] if vr.get("ncc_channels") else None, (vn.get("ncc_channels") or [None] * 3)[-1] if vn.get("ncc_channels") else None
        T.row("G3b", f"vessel-channel NCC >= {tol['G3.min_vessel_ncc']} (gate input, reported)", (nc1 >= tol["G3.min_vessel_ncc"]) if nc1 is not None else None, nc0, nc1, None, tol["G3.min_vessel_ncc"])
        a, b = get(vr, "vs_structural.corner_mean_mm"), get(vn, "vs_structural.corner_mean_mm")
        T.row("G3c", f"vascular moved (corner mean) |delta| <= {tol['G3.vascular_moved_mm']} mm", (abs(a - b) <= tol["G3.vascular_moved_mm"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G3.vascular_moved_mm"])
        a, b = get(vr, "restarts.n_within_0.3mm"), get(vn, "restarts.n_within_0.3mm")
        T.row("G3d", f"vascular restarts n_within_0.3mm within +-{tol['G3.vascular_restarts']}", (abs(a - b) <= tol["G3.vascular_restarts"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G3.vascular_restarts"])
    # ---------------------------------------------------------------- G4 final pose
    T0 = load_T(ref_run, ref); T1 = load_T(new_run, new); c_o = block_centre(work_new, new)
    if T0 is not None and T1 is not None and c_o is not None:
        d = transform_diff(T1, T0, c_o)
        T.row("G4a", f"final pose corner_mean_mm <= {tol['G4.corner_mean_mm']}", d["corner_mean_mm"] <= tol["G4.corner_mean_mm"], None, None, d["corner_mean_mm"], tol["G4.corner_mean_mm"], f"corner max {d['corner_max_mm']:.3f}")
        T.row("G4b", f"final pose rotation_deg <= {tol['G4.rotation_deg']}", d["rotation_deg"] <= tol["G4.rotation_deg"], None, None, d["rotation_deg"], tol["G4.rotation_deg"])
        T.row("G4c", f"final pose centre_mm <= {tol['G4.centre_mm']}", d["centre_mm"] <= tol["G4.centre_mm"], None, None, d["centre_mm"], tol["G4.centre_mm"])
    else: T.row("G4a", "final pose transform_diff", False, note="T_oct2mri.npy or block centre unavailable")
    s0, s1 = get(ref, "final_transform.stretch_ijk"), get(new, "final_transform.stretch_ijk")
    if s0 and s1:
        dd = [abs(x - y) for x, y in zip(s0, s1)]
        T.row("G4d", f"stretch_ijk |delta| <= {tol['G4.stretch']} per axis", max(dd) <= tol["G4.stretch"], s0, s1, max(dd), tol["G4.stretch"])
    else: T.row("G4d", "stretch_ijk |delta|", both(s0, s1), s0, s1)
    # ---------------------------------------------------------------- G5 label metrics (evaluation only)
    e0, e1 = get(ref, "evaluation.final", {}) or {}, get(new, "evaluation.final", {}) or {}
    for k in ("dice_WM", "dice_GM"):
        a, b = get(e0, f"gmwm.{k}"), get(e1, f"gmwm.{k}")
        T.row("G5a", f"{k} |delta| <= {tol['G5.dice']}", (abs(a - b) <= tol["G5.dice"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G5.dice"])
    for tag in ("own", "vesseg"):
        a, b = get(e0, f"vessels_{tag}.registered.median_um"), get(e1, f"vessels_{tag}.registered.median_um")
        T.row("G5b", f"vessels_{tag} registered median |delta| <= {tol['G5.vessel_median_um']:g} um", (abs(a - b) <= tol["G5.vessel_median_um"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G5.vessel_median_um"])
        a, b = get(e0, f"vessels_{tag}.registered.frac_within_150um"), get(e1, f"vessels_{tag}.registered.frac_within_150um")
        T.row("G5c", f"vessels_{tag} frac_within_150um |delta| <= {tol['G5.frac150']}", (abs(a - b) <= tol["G5.frac150"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["G5.frac150"])
    # ---------------------------------------------------------------- G6 wall time
    a, b = ref.get("total_seconds"), new.get("total_seconds")
    T.row("G6", f"wall time <= {tol['G6.wall_factor']} x ref", (b <= tol["G6.wall_factor"] * a) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b / a, tol["G6.wall_factor"], "delta = new/ref")


def fine_table(T: Table, ref, new, ref_run: Path, new_run: Path, work_new: Path | None, tol: dict):
    fine = new.get("fine") or {}
    if not fine: T.row("F0", "new run has a fine stage (result.json fine.*)", False, note="no 'fine' key"); return
    acc = fine.get("accepted")
    T.row("F0", "fine stage ran (accepted or rejected with reasons)", acc is not None, None, acc, note=f"reasons {fine.get('reasons')}; U_mm {fine.get('U_mm')}; sign {fine.get('sign_per_level')}")
    a = fine.get("seconds"); T.row("F6", f"fine stage wall <= {tol['F.fine_seconds']:g} s", (a <= tol["F.fine_seconds"]) if a is not None else None, None, a, None, tol["F.fine_seconds"])
    if acc is not True:
        T.row("F1", "fine rejected -> shipped pose equals the pre-fine pose (5.4 'either' branch)", True, note="fine.accepted == false; label criteria not applicable")
        T0 = load_T(ref_run, ref); T1 = load_T(new_run, new); c_o = block_centre(work_new, new)
        if T0 is not None and T1 is not None and c_o is not None: d = transform_diff(T1, T0, c_o); T.row("F1b", "rejected: T_final(new) == T_final(ref)", d["corner_mean_mm"] <= 1e-3, None, None, d["corner_mean_mm"], 1e-3, "corner mean mm")
        return
    T0 = load_T(ref_run, ref); T1 = load_T(new_run, new); c_o = block_centre(work_new, new)
    if T0 is not None and T1 is not None and c_o is not None:
        d = transform_diff(T1, T0, c_o); T.row("F1", f"corner_mean_mm(T_final_fine, T_final_v11) <= {tol['F.corner_mean_mm']}", d["corner_mean_mm"] <= tol["F.corner_mean_mm"], None, None, d["corner_mean_mm"], tol["F.corner_mean_mm"])
    e0, e1 = get(ref, "evaluation.final", {}) or {}, get(new, "evaluation.final", {}) or {}
    for k in ("dice_WM", "dice_GM"):
        a, b = get(e0, f"gmwm.{k}"), get(e1, f"gmwm.{k}")
        T.row("F2", f"{k} change >= -{tol['F.dice_drop']}", (b - a >= -tol["F.dice_drop"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, -tol["F.dice_drop"])
    for tag in ("own", "vesseg"):
        a, b = get(e0, f"vessels_{tag}.registered.median_um"), get(e1, f"vessels_{tag}.registered.median_um")
        T.row("F3", f"vessels_{tag} median change <= +{tol['F.vessel_median_increase_um']:g} um", (b - a <= tol["F.vessel_median_increase_um"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, tol["F.vessel_median_increase_um"])
        a, b = get(e0, f"vessels_{tag}.registered.frac_within_150um"), get(e1, f"vessels_{tag}.registered.frac_within_150um")
        T.row("F4", f"vessels_{tag} f150 change >= -{tol['F.frac150_drop']}", (b - a >= -tol["F.frac150_drop"]) if both(a, b) else both(a, b), a, b, None if not both(a, b) else b - a, -tol["F.frac150_drop"])
    s0, s1 = get(ref, "final_transform.stretch_ijk"), get(new, "final_transform.stretch_ijk")
    if s0 and s1: T.row("F5", f"depth stretch (ijk[0]) within +-{tol['F.stretch_depth']} of ref", abs(s1[0] - s0[0]) <= tol["F.stretch_depth"], s0[0], s1[0], s1[0] - s0[0], tol["F.stretch_depth"])
    U = fine.get("U_mm"); T.row("F7", "G4-calibration (reported): U_mm vs labelled residual 0.35-0.48 mm", None, None, U, note=f"needs U >= {tol['F.U_vs_residual_factor']} x residual; residual from results/final/summary/vessel_channel_refine.json")


def load_T(run: Path, res: dict):
    if (run / "T_oct2mri.npy").exists(): return np.load(run / "T_oct2mri.npy")
    t = get(res, "final_transform.T_oct2mri_world"); return np.array(t, float) if t is not None else None


def block_centre(work: Path | None, res: dict):
    """OCT block centre in OCT world (needs oct150_affine + shape150) -> c_o for transform_diff."""
    if work is None or not (work / "oct150_affine.npy").exists(): return None
    A = np.load(work / "oct150_affine.npy"); prep = json.load(open(work / "prep.json")) if (work / "prep.json").exists() else {}
    shp = get(prep, "oct.shape150")
    if shp is None and (work / "oct150_mask.npy").exists(): shp = np.load(work / "oct150_mask.npy", mmap_mode="r").shape
    if shp is None: return None
    return (A @ np.r_[(np.array(shp, float) - 1) / 2.0, 1.0])[:3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_run", type=Path); ap.add_argument("new_run", type=Path)
    ap.add_argument("--tol", choices=["default", "fine"], default="default"); ap.add_argument("--set", action="append", default=[], help="KEY=VALUE tolerance override")
    ap.add_argument("--work-ref", type=Path, default=None); ap.add_argument("--work-new", type=Path, default=None); ap.add_argument("--json", type=Path, default=None, help="write the table as json")
    a = ap.parse_args()
    tol = dict(TOL if a.tol == "default" else TOL_FINE)
    for s in a.set:
        k, v = s.split("=", 1)
        if k not in tol: sys.exit(f"unknown tolerance key {k}; known: {sorted(tol)}")
        tol[k] = type(tol[k])(float(v)) if not isinstance(tol[k], bool) else v.lower() in ("1", "true")
    ref = json.load(open(a.ref_run / "result.json")); new = json.load(open(a.new_run / "result.json"))
    work_ref = resolve_work(a.ref_run, ref, a.work_ref); work_new = resolve_work(a.new_run, new, a.work_new)
    print(f"regress [{a.tol}]  REF {a.ref_run} (work {work_ref})  NEW {a.new_run} (work {work_new})")
    T = Table()
    if a.tol == "default": default_table(T, ref, new, a.ref_run, a.new_run, work_ref, work_new, tol)
    else: fine_table(T, ref, new, a.ref_run, a.new_run, work_new, tol)
    T.show()
    if a.json: a.json.parent.mkdir(parents=True, exist_ok=True); a.json.write_text(json.dumps({"tol": tol, "ref_run": str(a.ref_run), "new_run": str(a.new_run), "rows": T.as_json(), "n_fail": T.n_fail}, indent=1, default=str))
    return 1 if T.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
