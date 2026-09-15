#!/usr/bin/env python3
"""octreg on the DANDI:000026 Broca-area OCT blocks (Costantini et al. 2023): generality check of the method on cortex.

    python bench/dandi.py locate SUBJ    # no location prior: coarse §1-4 on the whole MRI (base 0.3 mm, search 1.2 mm)
    python bench/dandi.py crop SUBJ [--mode v1|locate|labels]   # MRI crop: box of the block where v1 or locate put it + MARGIN_MM
    python bench/dandi.py run SUBJ       # §1-5 for both handednesses of the OCT array frame (the OME-TIFF has no orientation)
    python bench/dandi.py evaluate SUBJ  # labels the registration never sees, for the §4 and §5 pose of each handedness

Inputs per subject (SUBJECTS): the OCT OME-TIFF with the spacing of its sidecar (I46: depth 14 um, see NOTES), the EPIC MRI
(flip angle with GM/WM contrast) and, for evaluation only, the MRI vessel and cortical-layer labels and the v1 preparation
(work/SUBJ: EDT maps of OCT vessel segmentations, OCT WM/GM classes). All outputs go to OUT/SUBJ/.

run writes run.json: for handedness +1 (array frame) and -1 (array frame with world z negated) the best §4 pose (S, S_class,
S_outline, L) and the §5 pose (F start and end), T_oct2mri of both in the OCT array world, the grids of §1 (base.npz) and the
OCT box-averaged to the MRI voxel size for figures (overlay.npz).

evaluate writes eval.json: per pose, MRI vessel label voxels inside the block -> OCT -> distance to the nearest OCT vessel
(median um, fraction within 150 um; own Frangi segmentation and, where present, the dataset's ves_seg), Dice of OCT WM/GM
classes against the MRI labels, and the distance to the v1 pose (block corners), whose visual verdict is in V1_VERDICT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg import geometry as G, io, preprocess as pp                        # noqa: E402
from octreg.ngf import refine_ngf                                              # noqa: E402
from octreg.params import Params                                               # noqa: E402
from octreg.register import align, prepare_mri, prepare_oct                   # noqa: E402

ROOT = Path("/root/autodl-tmp/oct-mri-registration")
DANDI = ROOT / "data/costantini/dandi-000026"
OUT = Path("/root/autodl-tmp/bench_runs/dandi")
MARGIN_MM = 12.0
MIRROR = np.diag([1.0, 1.0, -1.0, 1.0])


def subj(s, flip, spacing_zyx, labels=True):
    return {"oct": DANDI / f"sub-{s}/ses-OCT/micr/sub-{s}_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff",
            "mri": DANDI / f"derivatives/EPIC/sub-{s}/ses-MRI/anat/sub-{s}_ses-MRI_flip-{flip}_VFA.nii.gz",
            "labels_ba": DANDI / f"derivatives/Labels/sub-{s}/ses-MRI/anat/sub-{s}_ses-MRI_space-EPIC_label-infrasupra_dseg.nii.gz",
            "spacing_um": spacing_zyx, "labels": labels}


SUBJECTS = {
    "I46": subj("I46", 2, (14.0, 12.0, 12.0)),
    "I55": subj("I55", 2, (14.0, 12.0, 12.0)),
    "I38": subj("I38", 2, (35.0, 30.0, 30.0)),
    "I48": subj("I48", 2, (35.0, 30.0, 30.0)),
    "I56": subj("I56", 2, (35.0, 30.0, 30.0)),
    "I62": subj("I62", 3, (35.0, 30.0, 30.0)),
    "I57": subj("I57", 4, (35.0, 30.0, 30.0)),
    "I61": subj("I61", 4, (35.0, 30.0, 30.0)),
    "I38s": {**subj("I38", 2, (28.3, 28.7, 28.2)), "work": "I38"},     # I38 with the voxel spacing implied by its v1 pose
    "I38t": {**subj("I38", 2, (28.3, 28.7, 28.2)), "work": "I38", "margin_mm": 3.0},     # I38s with a crop close to the block
    "I45": {**subj("I45", 3, (35.0, 30.0, 30.0), labels=False),
            "mri": DANDI / "sub-I45/ses-MRI/anat/sub-I45_ses-MRI_echo-1_flip-3_VFA.nii.gz"},        # raw 20-degree echo, no EPIC
}
V1_VERDICT = {"I46": "correct", "I55": "correct", "I38": "correct", "I56": "correct", "I62": "correct", "I48": "uncertain",
              "I57": "wrong", "I61": "wrong"}
NOTES = {"I46": "sidecar PixelSize 12/12/12 um; depth 14 um as in I55 (the v1 run measured a depth stretch of 1.18-1.22)",
         "I38s": "I38 with the spacing implied by the v1 pose (sidecar 30/30/35 um; v1 scales 0.94/0.96/0.81)"}
V1_VERDICT["I38s"] = V1_VERDICT["I38t"] = "correct"


def wk(s):
    """Name of the v1 preparation and run of subject s."""
    return SUBJECTS[s].get("work", s)


def block_corners_mm(vo):
    return G.apply_affine(vo.affine, np.array([[i, j, k] for i in (0, vo.shape[0] - 1) for j in (0, vo.shape[1] - 1)
                                               for k in (0, vo.shape[2] - 1)], float))


def crop(s, mode="v1"):
    """MRI crop streamed from the MRI NIfTI -> OUT/s/mri_crop.nii.gz: the box of the OCT block placed by the v1 pose (a location
    prior only; the orientation is not used) or by locate (mode 'locate'), or of the Broca-area labels (mode 'labels'), + MARGIN_MM."""
    import nibabel as nib
    cfg, out = SUBJECTS[s], OUT / s
    out.mkdir(parents=True, exist_ok=True)
    vm = io.load_volume(cfg["mri"])
    if mode == "labels":
        lab = nib.load(str(cfg["labels_ba"]))
        idx = np.argwhere(np.asarray(lab.dataobj) > 0)
        pts = G.apply_affine(lab.affine, np.array([[x, y, z] for x in (idx[:, 0].min(), idx[:, 0].max())
                                                   for y in (idx[:, 1].min(), idx[:, 1].max()) for z in (idx[:, 2].min(), idx[:, 2].max())], float))
    else:
        vo = io.load_volume(cfg["oct"], cfg["spacing_um"])
        T = (np.load(ROOT / f"work/runs/{wk(s)}_otsu/T_oct2mri.npy") @ _v1_to_octreg(s, vo) if mode == "v1"
             else np.array(json.loads((out / "locate.json").read_text())["T"]))
        pts = G.apply_affine(T, block_corners_mm(vo))
    ijk = G.apply_affine(np.linalg.inv(vm.affine), pts)
    sp = vm.spacing_mm
    margin = cfg.get("margin_mm", MARGIN_MM)
    lo = np.maximum(np.floor(ijk.min(0) - margin / sp), 0).astype(int)
    hi = np.minimum(np.ceil(ijk.max(0) + margin / sp) + 1, vm.shape).astype(int)
    arr = np.zeros(tuple(hi - lo), np.float32)
    for k, plane in io.iter_planes(vm):
        if lo[2] <= k < hi[2]:
            arr[:, :, k - lo[2]] = plane[lo[0]:hi[0], lo[1]:hi[1]]
    A = vm.affine.copy()
    A[:3, 3] = vm.affine[:3, :3] @ lo + vm.affine[:3, 3]
    io.save_nifti(arr, A, out / "mri_crop.nii.gz")
    rec = {"subject": s, "mode": mode, "lo_ijk": lo.tolist(), "hi_ijk": hi.tolist(), "shape": arr.shape, "mm": (arr.shape * sp).tolist(),
           "margin_mm": margin}
    (out / "crop.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


def locate(s, device="cuda"):
    """Location prior for a subject without one: §1-4 on the whole MRI at base 0.3 mm / search 1.2 mm, both handednesses; the
    best S gives the block position (OUT/s/locate.json). Used only to cut the rough crop of the test."""
    cfg, out, t0 = SUBJECTS[s], OUT / s, time.time()
    out.mkdir(parents=True, exist_ok=True)
    P = Params.from_dict({"base_mm": 0.3, "search_mm": 1.2})
    vo, vm = io.load_volume(cfg["oct"], cfg["spacing_um"]), io.load_volume(cfg["mri"])
    (mri_h, mask_m, A_m), fg_m = prepare_mri(vm, P)
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    (oct_h, mask_o, A_o), valid_o, fg_o = prepare_oct(fine, A_f, P)
    del fine
    u, w = pp.oct_channels(pp.two_class(oct_h, mask_o, P.base_mm, P), mask_o)
    v = pp.mri_channels(pp.two_class(mri_h, mask_m, P.base_mm, P), mask_m)
    best = None
    for hand, F in ((1, np.eye(4)), (-1, MIRROR)):
        poses, info = align((u, w, valid_o, F @ A_o), (v, mask_m, A_m), P, device)
        p = poses[0]
        print(s, "locate hand", hand, "S", p["S"], "L", p["L"], flush=True)
        if best is None or p["L"] < best[0]:
            best = (p["L"], hand, (p["T"] @ F).tolist(), p["S"])
    rec = {"subject": s, "L": best[0], "hand": best[1], "T": best[2], "S": best[3], "mri_foreground": fg_m, "seconds": time.time() - t0}
    (out / "locate.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "T"}), flush=True)


def run(s, device="cuda"):
    cfg, out, P, t0 = SUBJECTS[s], OUT / s, Params(), time.time()
    vo, vm = io.load_volume(cfg["oct"], cfg["spacing_um"]), io.load_volume(out / "mri_crop.nii.gz")
    (mri_h, mask_m, A_m), fg_m = prepare_mri(vm, P)
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay = G.pool_iso(fine, A_f, P.fine_mm, max(float(vm.spacing_mm.min()), P.fine_mm))
    (oct_h, mask_o, A_o), valid_o, fg_o = prepare_oct(fine, A_f, P)
    del fine
    np.savez(out / "base.npz", oct_h=oct_h, mask_o=mask_o, valid_o=valid_o, A_o=A_o, mri_h=mri_h, mask_m=mask_m, A_m=A_m)
    np.savez(out / "overlay.npz", arr=overlay[0], affine=overlay[1])
    t_prep = time.time() - t0
    u, w = pp.oct_channels(pp.two_class(oct_h, mask_o, P.base_mm, P), mask_o)
    v = pp.mri_channels(pp.two_class(mri_h, mask_m, P.base_mm, P), mask_m)
    rec = {"subject": s, "inputs": {k: str(x) for k, x in cfg.items()}, "note": NOTES.get(s), "foreground": {"oct": fg_o, "mri": fg_m},
           "oct_affine": vo.affine.tolist(), "prep_seconds": t_prep, "hands": {}}
    for hand, F in ((1, np.eye(4)), (-1, MIRROR)):
        t1 = time.time()
        poses, info = align((u, w, valid_o, F @ A_o), (v, mask_m, A_m), P, device)
        best = poses[0]
        T5, ngf = refine_ngf((mri_h, mask_m, A_m), (oct_h, mask_o, F @ A_o), best["T"], P, device)
        rec["hands"][str(hand)] = {
            "S": best["S"], "S_class": best["S_class"], "S_outline": best["S_outline"], "L": best["L"], "polarity": best["polarity"],
            "search": {k: info["search"].get(k) for k in ("top1", "top2")}, "ngf": ngf,
            "T4": (best["T"] @ F).tolist(), "T5": (T5 @ F).tolist(), "seconds": time.time() - t1}
        print(s, "hand", hand, json.dumps({k: v for k, v in rec["hands"][str(hand)].items() if k not in ("T4", "T5")}), flush=True)
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
    rec["seconds"] = time.time() - t0
    (out / "run.json").write_text(json.dumps(rec, indent=1))


def _v1_to_octreg(s, vo):
    """C: OCT world of octreg (array frame, this spacing) -> OCT world of the v1 preparation (layout SPR)."""
    A_nat = np.load(ROOT / f"work/{wk(s)}/oct_native_affine.npy")
    M = np.zeros((4, 4))                                      # octreg index (x, y, z) = native numpy axes (2, 1, 0)
    M[3, 3] = 1.0
    M[:3, :3] = vo.affine[:3, :3] @ np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)     # native (a0, a1, a2) -> octreg world
    return A_nat @ np.linalg.inv(M)


def evaluate(s):
    cfg, out = SUBJECTS[s], OUT / s
    rec = json.loads((out / "run.json").read_text())
    vo = io.load_volume(cfg["oct"], cfg["spacing_um"])
    W = ROOT / f"work/{wk(s)}"
    C = _v1_to_octreg(s, vo)
    A_mri = np.load(W / "mri_affine.npy")
    ves = np.argwhere(np.load(W / "mri_vessels.npy", mmap_mode="r"))
    pts_m = G.apply_affine(A_mri, ves.astype(float))
    labels4 = np.load(W / "labels4.npy", mmap_mode="r")
    cls_path = ROOT / f"work/runs/{wk(s)}_otsu/oct_class150.npy"
    oct_class = np.load(cls_path) if cls_path.exists() else None
    A150 = np.load(W / "oct150_affine.npy")
    mask150 = np.load(W / "oct150_mask.npy") if (W / "oct150_mask.npy").exists() else None
    if oct_class is not None and oct_class.shape != np.load(W / "oct150.npy", mmap_mode="r").shape:
        oct_class = None                                                 # classes on another grid than oct150_affine
    dists = {k: (np.load(W / f"oct_ves_dist48_{k}.npy"), np.load(W / f"oct_ves_dist48_{k}_affine.npy"))
             for k in ("own", "vesseg") if (W / f"oct_ves_dist48_{k}.npy").exists()}
    T_v1 = np.load(ROOT / f"work/runs/{wk(s)}_otsu/T_oct2mri.npy")          # v1 OCT world -> MRI world
    T_v1_oct = T_v1 @ C
    corners = G.apply_affine(vo.affine, np.array([[i, j, k] for i in (0, vo.shape[0] - 1) for j in (0, vo.shape[1] - 1)
                                                  for k in (0, vo.shape[2] - 1)], float))

    def vessel_stats(T, dist, A_d):
        p_v1 = G.apply_affine(C @ np.linalg.inv(T), pts_m)          # MRI -> octreg OCT world -> v1 OCT world
        ijk = G.apply_affine(np.linalg.inv(A_d), p_v1)
        inside = np.all((ijk >= 0) & (ijk <= np.array(dist.shape) - 1), axis=1)
        if inside.sum() < 20:
            return {"n_inside": int(inside.sum())}
        from scipy import ndimage
        d = ndimage.map_coordinates(dist, ijk[inside].T, order=1)
        return {"n_inside": int(inside.sum()), "median_um": float(np.median(d)), "frac_within_150um": float((d <= 150).mean())}

    def dice(T):
        if oct_class is None:
            return None
        m = (oct_class > 0) if mask150 is None else (mask150 & (oct_class > 0))
        idx = np.argwhere(m)
        p_m = G.apply_affine(T @ np.linalg.inv(C) @ A150, idx.astype(float))
        ijk = np.rint(G.apply_affine(np.linalg.inv(A_mri), p_m)).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(labels4.shape)), axis=1)
        lab = np.zeros(len(idx), np.int64)
        lab[ok] = labels4[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        lab_c = np.where(lab == 1, 1, np.where(np.isin(lab, (2, 3)), 2, 0))
        oc = oct_class[tuple(idx.T)]
        on = lab_c > 0
        res = {"frac_on_labels": float(on.mean())}
        for c, name in ((1, "WM"), (2, "GM")):
            a, b = (oc == c) & on, lab_c == c
            res[f"dice_{name}"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        return res

    ev = {"subject": s, "v1_verdict": V1_VERDICT.get(s), "det_T_v1_octreg_frame": float(np.linalg.det(T_v1_oct[:3, :3])), "poses": {}}
    cand = {f"h{h}_{st}": np.array(rec["hands"][h][f"T{st[-1]}"]) for h in ("1", "-1") for st in ("s4", "s5")}
    cand["v1"] = T_v1_oct
    for name, T in cand.items():
        d = np.linalg.norm(G.apply_affine(T, corners) - G.apply_affine(T_v1_oct, corners), axis=1)
        ev["poses"][name] = {"to_v1_corners_mean_mm": float(d.mean()), "det": float(np.linalg.det(T[:3, :3])),
                             "vessels": {k: vessel_stats(T, *dv) for k, dv in dists.items()}, "gm_wm": dice(T)}
        print(s, name, json.dumps(ev["poses"][name]), flush=True)
    (out / "eval.json").write_text(json.dumps(ev, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("crop", "locate", "run", "evaluate"))
    ap.add_argument("--mode", default="v1", choices=("v1", "locate", "labels"))
    ap.add_argument("subjects", nargs="+")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for s in a.subjects:
        {"crop": lambda x: crop(x, a.mode), "locate": lambda x: locate(x, a.device), "run": lambda x: run(x, a.device),
         "evaluate": evaluate}[a.step](s)


if __name__ == "__main__":
    main()
