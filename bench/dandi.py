#!/usr/bin/env python3
"""octreg on the DANDI:000026 Broca-area blocks (Costantini et al. 2023): the generality check of the method on cortex.

    python bench/dandi.py [SUBJ ...] [--steps crop register evaluate summary] [--device cuda] [--out DIR] [--tag TAG]

Nine subjects have both an OCT block (OME-TIFF of the scattering coefficient, no orientation in its header, voxel spacing
from the BIDS sidecar) and an ex-vivo MRI of the hemisphere it was cut from, with labels the registration never sees. Eight of
them are used. In sub-I56 the affine of every label file disagrees with its array, so only 30 to 72 % of each label lands
inside the MRI and neither the crop nor the evaluation can be read from it. That is a defect of the dataset derivative, and
crop() refuses the subject rather than working on a clipped box.

crop      The method assumes an MRI already cropped around a region that contains the block, so the MRI is cut to the box of
          the subject's own Broca-area label plus MARGIN_MM, and every side is grown to at least the longest side of the OCT
          array, because that label marks the cortical ribbon and is a thin sheet while the block reaches below it. The result
          goes to OUT/SUBJ/mri_crop.nii.gz. The label says which part of the hemisphere the block was taken from and how big
          the block is, which is what a user of the method knows. It is not the registration: it fixes neither the
          orientation nor the position inside the crop.
register  octreg.register.register on the block and the crop, the same entry point as for any other pair. The OCT frame has no
          orientation, so §3-5 run for both handednesses and the higher NGF F wins (docs/METHOD.md §5).
evaluate  A number the registration never sees, written to OUT/SUBJ/eval.json. The subject's cortical-layer label divides the
          cortex into layers. Its two ends, the classes with the lowest and the highest median MRI intensity, are brought into
          the OCT by the pose, and the two-class map of §2 is read there. The separation |AUC - 0.5| of the two sets of values
          says whether the registered OCT tells the two ends of the cortex apart. It does not depend on which modality is
          bright where. The labels are an anatomical segmentation the method never reads. The same number at the pose of §4
          and at the scorable ones of N_RANDOM random poses of the block inside the crop says what it is worth on this
          subject (random_separation["n"] of eval.json counts them). It also writes
          qc_labels.png, the registered OCT with the outlines of that label: where the pose is right they follow the bright
          and dark bands of the OCT. --tag reads OUT/SUBJ+TAG instead of OUT/SUBJ, for a variant run of the same subject such
          as the `_allmeasured` diagnostic of docs/METHOD.md: the crop and the MRI stay the subject's own and the OCT mask is
          the one that run was given.
summary   One markdown row per subject and the same as JSON in bench/results/dandi/summary.json. It refuses to write unless
          every subject has a run under OUT, so a partial run cannot overwrite the released table.

Visual inspection stays the evaluation (docs/METHOD.md): qc.png and qc_montage.png of every run are the primary evidence and
the numbers here only support them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg import geometry as G, io                                          # noqa: E402
from octreg.params import Params                                              # noqa: E402
from octreg.register import register                                          # noqa: E402

from paths import BENCH_RUNS, DANDI                                           # noqa: E402  (bench/paths.py)

OUT = BENCH_RUNS / "dandi"
MARGIN_MM = 5.0            # the crop is the Broca-area label box plus this
COLOURS = ("#ff4040", "#ffd000", "#40ffff", "#ff40ff")                 # qc_labels.png, one per cortical-layer label value
N_RANDOM = 10              # random poses of the block inside the crop, drawn for the scale of the layer separation; the ones
                           # with too few labelled points inside the specimen mask are not scored (random_separation["n"])
SUBJECTS = ("I38", "I46", "I48", "I55", "I57", "I58", "I61", "I62")     # I56 is left out, see the module docstring
MRI_FLIP = {"I38": 2, "I46": 2, "I48": 2, "I55": 2, "I57": 4, "I58": 3, "I61": 4, "I62": 3}   # the EPIC VFA asset per subject


def files(s):
    """The subject's OCT, MRI and Broca-area label.

    The MRI is sub-<s>_ses-MRI_flip-<MRI_FLIP[s]>_VFA.nii.gz: one asset pinned per subject, the one every published run used
    (bench/README.md lists them). It is not picked by a glob, so it does not depend on which assets a download happens to
    contain, and not by flip angle, since the sidecars do not give one consistent choice (sub-I48 has no sidecar at all).
    Raises FileNotFoundError naming the missing path rather than failing inside a step."""
    d = DANDI / f"derivatives/EPIC/sub-{s}/ses-MRI/anat"
    lab = DANDI / f"derivatives/Labels/sub-{s}/ses-MRI/anat"
    f = {"oct": DANDI / f"sub-{s}/ses-OCT/micr/sub-{s}_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff",
         "mri": d / f"sub-{s}_ses-MRI_flip-{MRI_FLIP[s]}_VFA.nii.gz",
         "area": lab / f"sub-{s}_ses-MRI_space-EPIC_label-infrasupra_dseg.nii.gz"}
    if not DANDI.is_dir():
        raise FileNotFoundError(f"no DANDI:000026 tree at {DANDI}: set OCTREG_DATA_ROOT (see bench/paths.py)")
    missing = [str(p) for p in f.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"sub-{s}: missing {', '.join(missing)} (data root: OCTREG_DATA_ROOT, see bench/paths.py)")
    return f


def spacing_um(s):
    """(z, y, x) um from the BIDS sidecar of the OCT (PixelSize is (x, y, z), um unless PixelSizeUnits says otherwise)."""
    p = files(s)["oct"].with_suffix("").with_suffix(".json")
    d = json.loads(p.read_text())
    unit = {"um": 1.0, "mm": 1000.0}[d.get("PixelSizeUnits", "um")]
    return tuple(float(x) * unit for x in d["PixelSize"][::-1])


def box(affine, shape):
    """The eight corners of a grid in world mm."""
    return G.apply_affine(affine, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                            for k in (0, shape[2] - 1)], float))


def crop(s):
    """MRI cut to the Broca-area label box + MARGIN_MM, streamed plane by plane -> OUT/s/mri_crop.nii.gz."""
    import nibabel as nib
    f, out = files(s), OUT / s
    out.mkdir(parents=True, exist_ok=True)
    vm = io.load_volume(f["mri"])
    lab = nib.load(str(f["area"]))
    idx = np.argwhere(np.asarray(lab.dataobj) > 0)
    all_ijk = G.apply_affine(np.linalg.inv(vm.affine) @ lab.affine, idx[::97].astype(float))
    inside = ((all_ijk >= 0) & (all_ijk < np.array(vm.shape))).all(1).mean()
    if inside < 0.99:                                      # the label file must describe the same volume as the MRI
        raise ValueError(f"{s}: only {100 * inside:.0f} % of {f['area'].name} lies inside {f['mri'].name}; the affine of that "
                         "label file disagrees with its array, so the block region cannot be read from it")
    pts = G.apply_affine(lab.affine, np.array([[x, y, z] for x in idx[:, 0][[0, -1]] for y in (idx[:, 1].min(), idx[:, 1].max())
                                               for z in (idx[:, 2].min(), idx[:, 2].max())], float))
    ijk = G.apply_affine(np.linalg.inv(vm.affine), pts)
    vo = io.load_volume(f["oct"], spacing_um(s))                   # the crop has to contain the block, and the label of the
    side = float(np.max(np.array(vo.shape) * np.linalg.norm(vo.affine[:3, :3], axis=0)))   # cortical ribbon is a thin sheet,
    grow = np.maximum(side - (ijk.max(0) - ijk.min(0)) * vm.spacing_mm, 0) / 2             # so every side is grown to at
    lo = np.maximum(np.floor(ijk.min(0) - (MARGIN_MM + grow) / vm.spacing_mm), 0).astype(int)     # least the longest side of
    hi = np.minimum(np.ceil(ijk.max(0) + (MARGIN_MM + grow) / vm.spacing_mm) + 1, vm.shape).astype(int)   # the OCT array
    arr = np.zeros(tuple(hi - lo), np.float32)
    for k, plane in io.iter_planes(vm):
        if lo[2] <= k < hi[2]:
            arr[:, :, k - lo[2]] = plane[lo[0]:hi[0], lo[1]:hi[1]]
    A = vm.affine.copy()
    A[:3, 3] = vm.affine[:3, :3] @ lo + vm.affine[:3, 3]
    io.save_nifti(arr, A, out / "mri_crop.nii.gz")
    rec = {"subject": s, "mri": str(f["mri"]), "label": str(f["area"]), "margin_mm": MARGIN_MM, "oct_longest_side_mm": side,
           "shape": list(arr.shape), "mm": (np.array(arr.shape) * vm.spacing_mm).round(1).tolist()}
    io.write_json(rec, out / "crop.json")
    print(json.dumps(rec), flush=True)


def layer_auc(oct_p, mask_o, A_o, T, pts, hi):
    """Does the registered OCT tell the two ends of the cortical-layer label apart? The labelled MRI points are brought into the
    OCT by the pose T (OCT world -> MRI world) and kept inside the specimen mask, and the two-class map of §2 is read there.
    -> (auc, separation, points used) or None with too few points. auc: area under the ROC curve of the map values of the
    brighter class `hi` against those of the darker one. separation = |auc - 0.5|, which does not depend on the contrast
    polarity: 0 = the pose tells the two ends apart no better than chance."""
    from scipy import ndimage
    ijk = G.apply_affine(np.linalg.inv(A_o) @ np.linalg.inv(np.asarray(T, float)), pts).T
    inside = ndimage.map_coordinates(mask_o.astype(np.float32), ijk, order=1, cval=0.0) > 0.99
    if inside.sum() < 500:
        return None
    p = ndimage.map_coordinates(oct_p, ijk[:, inside], order=1, cval=0.0)
    a, b = p[hi[inside]], p[~hi[inside]]
    if min(len(a), len(b)) < 100:
        return None
    sb = np.sort(b)
    auc = float((np.searchsorted(sb, a, "left") + np.searchsorted(sb, a, "right")).mean() / (2 * len(sb)))
    return auc, abs(auc - 0.5), int(inside.sum())


def evaluate(s, tag=""):
    """The layer separation at the pose of the run, at the pose of §4 and at N_RANDOM random poses -> OUT/s+tag/eval.json. A
    random pose that lands too few labelled points inside the specimen mask is not scored, so random_separation["n"] is how
    many of the N_RANDOM poses the median and the max are over.

    tag evaluates a variant run of the same subject, OUT/s+tag, such as the `_allmeasured` diagnostic of docs/METHOD.md: the
    crop and the MRI stay the subject's own, and the OCT mask is the one that run was given, so the read-out is taken over
    the specimen the run itself believed in."""
    import nibabel as nib
    from scipy import ndimage
    from octreg import preprocess as pp
    from octreg.register import prepare_mri, prepare_oct
    f, out, base, P = files(s), OUT / (s + tag), OUT / s, Params()
    res = json.loads((out / "result.json").read_text())
    given = res["inputs"]["oct_mask"]
    fine, A_f = G.resample_iso(io.load_volume(f["oct"], spacing_um(s)), P.fine_mm)
    (oct_h, mask_o, A_o), _, _ = prepare_oct(fine, A_f, P,
                                             io.load_volume(given, spacing_um(s)) if given else None)
    measured_cm3 = float(np.count_nonzero(fine)) * P.fine_mm ** 3 / 1e3        # what the OCT actually measured
    del fine
    oct_p = pp.two_class(oct_h, mask_o, P.base_mm, P)                             # the OCT map of §2
    vm = io.load_volume(base / "mri_crop.nii.gz")
    (mri_h, fg_m, A_m), _ = prepare_mri(vm)
    lab = nib.load(str(f["area"]))
    d = np.asarray(lab.dataobj)
    idx = np.argwhere(d > 0)
    rng = np.random.default_rng(0)
    idx = idx[rng.permutation(len(idx))[:200000]]
    val, pts = d[tuple(idx.T)], G.apply_affine(lab.affine, idx.astype(float))
    inten = ndimage.map_coordinates(mri_h, G.apply_affine(np.linalg.inv(A_m), pts).T, order=1, cval=0.0)
    med = {int(v): float(np.median(inten[val == v])) for v in np.unique(val)}     # the two ends of the label, by MRI intensity
    lo_v, hi_v = min(med, key=med.get), max(med, key=med.get)
    keep = (val == lo_v) | (val == hi_v)
    pts, hi = pts[keep], val[keep] == hi_v
    poses = {"final": np.array(res["T_oct2mri"]), "s4": np.array(res["pose"]["T_before_ngf"])}
    idx = np.argwhere(mask_o)                                                  # does the crop satisfy the input assumption?
    block = np.sort((idx.max(0) - idx.min(0) + 1) * P.base_mm)[::-1]
    crop = np.sort(np.array(json.loads((base / "crop.json").read_text())["mm"]))[::-1]
    rec = {"subject": s, "run": s + tag, "oct_mask_given": given,
           "measured_cm3": measured_cm3, "specimen_mask_cm3": float(np.count_nonzero(mask_o)) * P.base_mm ** 3 / 1e3,
           "block_mm": block.round(1).tolist(), "crop_fits_block": bool((crop >= block).all()),
           "mri_median_per_label": med, "label_ends": [lo_v, hi_v], "n_label_points": int(keep.sum()),
           "poses": {k: layer_auc(oct_p, mask_o, A_o, T, pts, hi) for k, T in poses.items()}}
    c = box(A_o, oct_h.shape).mean(0)                                             # the block dropped anywhere in the crop
    lo, hi_c = box(A_m, fg_m.shape).min(0), box(A_m, fg_m.shape).max(0)
    rand = []
    for _ in range(N_RANDOM):
        R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        R *= np.sign(np.linalg.det(R))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, rng.uniform(lo, hi_c) - R @ c
        rand.append(layer_auc(oct_p, mask_o, A_o, T, pts, hi))
    sep = [r[1] for r in rand if r]
    rec["random_separation"] = {"n": len(sep), "median": float(np.median(sep)), "max": float(np.max(sep))} if sep else None
    rec["random_poses"] = rand
    io.write_json(rec, out / "eval.json")
    label_figure(out, lab, d, s, (lo_v, hi_v), base)
    print(json.dumps({k: v for k, v in rec.items() if k != "random_poses"}), flush=True)


def label_figure(out, lab, d, subject="", ends=(), base=None):
    """OUT/SUBJ/qc_labels.png: the registered OCT with the outlines of the MRI cortical-layer label, three planes per axis.
    Where the pose is right the outlines follow the bright and dark bands of the OCT. The outlines are cut to the measured
    OCT, since outside it there is nothing to follow, and the two label values the separation is read over (ends) are named
    in the legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib.lines import Line2D
    mri = nib.load(str((base or out) / "mri_crop.nii.gz"))     # the crop belongs to the subject, not to the run
    o = np.asarray(nib.load(str(out / "oct_in_mri.nii.gz")).dataobj, np.float32)
    L = G.resample_to(d.astype(np.float32), lab.affine, o.shape, mri.affine, np.eye(4), order=0)
    idx = np.argwhere(o > 0)
    lo, hi = idx.min(0), idx.max(0)
    values = sorted(int(v) for v in np.unique(L[(L > 0) & (o > 0)]))
    fig, axs = plt.subplots(3, 3, figsize=(21, 21), facecolor="k")
    for a, (ax, frac) in zip(axs.ravel(), [(ax, f) for ax in range(3) for f in (0.3, 0.5, 0.7)]):
        sl = [slice(None)] * 3
        sl[ax] = int(lo[ax] + frac * (hi[ax] - lo[ax]))
        sl = tuple(sl)
        O, Ls = o[sl], L[sl]
        a.set_axis_off()
        if (O > 0).sum() < 50:
            continue
        r, c = (np.flatnonzero(x) for x in ((O > 0).any(1), (O > 0).any(0)))       # zoom to the block plus 10 voxels
        r0, r1, c0, c1 = max(r[0] - 10, 0), r[-1] + 10, max(c[0] - 10, 0), c[-1] + 10
        O, Ls = O[r0:r1, c0:c1], Ls[r0:r1, c0:c1]
        v1, v2 = np.percentile(O[O > 0], [1, 99])
        a.imshow(np.where(O > 0, O, np.nan), cmap="gray", vmin=v1, vmax=v2, interpolation="nearest")
        for val, col in zip(values, COLOURS):
            keep = np.where(O > 0, (Ls == val).astype(float), np.nan)     # nan, not 0, outside the measured OCT: there is
            if np.nanmax(keep, initial=0.0) > 0:                          # nothing to follow there and no outline is drawn
                a.contour(keep, [0.5], colors=col, linewidths=1.2)
        a.set_title(f"axis {ax} at {sl[ax]}", color="w", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=c, label=f"label {v}" + " (end)" * (v in ends))
                        for v, c in zip(values, COLOURS)], loc="upper left", ncol=len(values), frameon=False,
               labelcolor="w", fontsize=12)
    fig.suptitle(f"sub-{subject}: registered OCT with the MRI cortical-layer label; where the pose is right the outlines "
                 f"follow the bands of the OCT", color="w", fontsize=13, y=0.995)
    plt.subplots_adjust(0, 0, 1, 0.94, 0.01, 0.03)
    fig.savefig(out / "qc_labels.png", dpi=45, facecolor="k")
    plt.close(fig)


def run(s, device="cuda"):
    """octreg register on the block and the crop -> OUT/s (the standard outputs and result.json)."""
    f, out, t0 = files(s), OUT / s, time.time()
    res = register(f["oct"], out / "mri_crop.nii.gz", out, oct_spacing_um=spacing_um(s), device=device)
    print(f"{s}: {time.time() - t0:.0f} s, {json.dumps({k: res['pose'][k] for k in ('S', 'handedness', 'NGF')})}", flush=True)


def summary():
    """One markdown row per subject from the runs that exist, and the same as JSON in bench/results/dandi/summary.json. The
    random column gives how many of the N_RANDOM poses could be scored (random_separation["n"]), not N_RANDOM."""
    rows = {}
    num = lambda x, n=3: "" if x is None else f"{x:.{n}f}"
    print("| subject | crop (mm) | block (mm) | crop fits | specimen mask of the measured OCT | S | S_class | S_outline | hand | NGF | scale per OCT axis | §6 "
          "| layer separation | random |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SUBJECTS:
        if not (OUT / s / "result.json").exists():
            continue
        res = json.loads((OUT / s / "result.json").read_text())
        p, d = res["pose"], res.get("deform", {})
        e = json.loads((OUT / s / "eval.json").read_text()) if (OUT / s / "eval.json").exists() else {}
        v, rnd = (e.get("poses") or {}).get("final"), e.get("random_separation")
        mm = json.loads((OUT / s / "crop.json").read_text())["mm"]
        sp = np.array(spacing_um(s))[::-1] / 1000.0                    # how much of what the OCT measured the mask keeps
        arr_cm3 = float(np.prod(np.array(res["inputs"]["oct"]["shape"]) * sp) / 1e3)
        fo, meas = res["foreground"]["oct"], e.get("measured_cm3")
        fits = "" if e.get("crop_fits_block") is None else ("yes" if e["crop_fits_block"] else "NO")
        n = fo["n_components"]
        mask = (f"{fo['volume_cm3']:.1f} of {meas:.1f} cm3 measured ({100 * fo['volume_cm3'] / meas:.0f} %), "
                f"{n} part{'' if n == 1 else 's'}" if meas else f"{fo['volume_cm3']:.1f} cm3, {n} part{'' if n == 1 else 's'}")
        random = f"{rnd['median']:.3f} (max {rnd['max']:.3f}, n {rnd['n']} of {N_RANDOM})" if rnd else ""
        deform = d.get("status", "")
        if d.get("lam") is not None:
            deform += f", lam {d['lam']:.2f}, field {d['field']['median_mm']:.2f} / {d['field']['max_mm']:.2f} mm"
        rows[s] = {"crop_mm": mm, "oct_array_cm3": arr_cm3, "oct_measured_cm3": meas, "specimen_mask": fo,
                   "block_mm": e.get("block_mm"), "crop_fits_block": e.get("crop_fits_block"), "pose": {k: p[k] for k in ("S", "S_class", "S_outline", "L", "polarity", "handedness",
                                                              "NGF_start", "NGF", "NGF_other_handedness", "scale_per_oct_axis")},
                   "flags": res["flags"], "deform": {k: d.get(k) for k in ("status", "n_interior", "n_boundary")},
                   "layer_separation": v[1] if v else None, "random_separation": rnd, "seconds": res["seconds"]["total"]}
        blk = ' x '.join(f'{x:.0f}' for x in e['block_mm']) if e.get('block_mm') else ''
        print(f"| {s} | {' x '.join(f'{x:.0f}' for x in sorted(mm, reverse=True))} | {blk} | {fits} | {mask} | {num(p['S'], 4)} | {num(p['S_class'], 4)} | {num(p['S_outline'], 4)} "
              f"| {p['handedness']:+d} | {num(p['NGF_start'], 4)} -> {num(p['NGF'], 4)} "
              f"| {' / '.join(f'{x:.3f}' for x in p['scale_per_oct_axis'])} | {deform} "
              f"| {num(v[1], 3) if v else ''} | {random if rnd else ''} |")
    store = Path(__file__).resolve().parent / "results/dandi/summary.json"
    if set(rows) != set(SUBJECTS):                     # a partial run must not overwrite the released table in the repository
        missing = ", ".join(sorted(set(SUBJECTS) - set(rows)))
        raise SystemExit(f"{store} not written: no run under {OUT} for {missing}. Give --out the root that holds them.")
    store.parent.mkdir(parents=True, exist_ok=True)
    io.write_json({"margin_mm": MARGIN_MM, "n_random_poses_drawn": N_RANDOM,        # scored per subject: random_separation.n
                   "subjects": rows}, store)
    print(f"wrote {store}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*", default=[], help=f"one or more of {', '.join(SUBJECTS)} (default: all)")
    ap.add_argument("--steps", nargs="+", default=["crop", "register", "evaluate"],
                    choices=["crop", "register", "evaluate", "summary"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help=f"output root (default {OUT})")
    ap.add_argument("--tag", default="", help="evaluate OUT/SUBJ+TAG instead of OUT/SUBJ, e.g. _allmeasured (evaluate only)")
    a = ap.parse_args()
    if a.tag and set(a.steps) - {"evaluate"}:
        ap.error("--tag applies to the evaluate step only")
    if a.out:
        globals()["OUT"] = Path(a.out)
    subjects = a.subjects or list(SUBJECTS)
    if set(subjects) - set(SUBJECTS):
        ap.error(f"unknown subjects {sorted(set(subjects) - set(SUBJECTS))}; known: {', '.join(SUBJECTS)}")
    for step in a.steps:
        if step == "summary":
            summary()
            continue
        for s in subjects:
            print(f"=== {s} {step}", flush=True)
            run(s, a.device) if step == "register" else crop(s) if step == "crop" else evaluate(s, a.tag)


if __name__ == "__main__":
    main()
