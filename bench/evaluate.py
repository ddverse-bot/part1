#!/usr/bin/env python3
"""Label-free evaluation of an octreg run on Xiangrui's I58 brainstem pair (bench only; the method never sees any of this).

    python bench/evaluate.py RUN [--masks DIR] [--previous OLD_RUN] [--no-frame-check] [-o RUN/eval.json]
    python bench/evaluate.py --selftest                     # synthetic, CPU, a few seconds

Everything here comes from the two input files of bench/paths.py (OCT_I58, MRI_I58), the run directory RUN and, for the
comparison read-outs, a second octreg run directory. T = RUN/T_oct2mri.txt maps the OCT header world to the MRI header world
(mm). DIR is a prep directory of bench/ablate.py, which holds the masks the method itself built: oct_mask.nii.gz (OCT specimen),
oct_valid.nii.gz (non-zero OCT data) and mri_mask.nii.gz (MRI foreground). Without --masks there is no mask and no boundary
read-out.

mask              Volume of the run's own OCT specimen mask (DIR/oct_mask.nii.gz) in cm3.
boundary          Outline agreement of that mask and the MRI foreground, in both directions. OCT mask boundary voxels (at least
                  3 voxels from the grid faces, on valid OCT data) go through T to their distance to the MRI foreground
                  boundary, and MRI boundary voxels go through inv(T) to their distance to the OCT mask boundary. Points within
                  1 mm of the MRI grid faces are dropped. Medians per outward-normal class along the raw OCT array axes. 'rim'
                  leaves out the deep end a0+, which has no specimen rim. With --previous, the rim medians of OLD_RUN under the
                  same masks as well, which compares two octreg runs on the same footing.
pose_to_previous  Distance between T and OLD_RUN/T_oct2mri.txt (--previous). Mean and max displacement over the points of the
                  run's own OCT specimen mask, or, without --masks, over the 8 corners of the OCT array plus a uniform sample of
                  its grid. The same at the 8 corners on their own, and the rotation angle of inv(T_previous) @ T.
frame_check       Random voxels of the affine overlay (RUN/oct_in_mri_affine.nii.gz, or oct_in_mri.nii.gz of a run before 1.1,
                  which had no deformation) go through inv(T) and inv(A_hdr) to raw OCT voxels, whose 20 um values (box^3 mean)
                  are streamed from the .nii.gz. Spearman against the exported values. Controls: T composed with a flip of each
                  raw OCT axis about the block centre, and 2 mm shifts along each MRI world axis. ok iff rho >= 0.9 and every
                  flip gives rho <= 0.3.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

from paths import MRI_I58 as MRI, OCT_I58 as OCT          # bench/paths.py: OCTREG_PROJECT_ROOT, OCTREG_DATA_ROOT

MASK_FILES = ("oct_mask.nii.gz", "oct_valid.nii.gz", "mri_mask.nii.gz")
DEEP_END = "a0+"                                        # raw OCT axis 0, high index: the rim-less deep end
FACE_VOX, FOV_MM = 3, 1.0


def apply(T, pts):
    return pts @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def load_T(path):
    path = Path(path)
    return np.asarray(np.loadtxt(path), float).reshape(4, 4)


def load_mask(path):
    img = nib.load(str(path))
    return np.asarray(img.dataobj) > 0, np.asarray(img.affine, float)


def oct_header(oct_path=OCT):
    """Affine and shape of the OCT NIfTI header, without reading the voxels."""
    img = nib.load(str(oct_path))
    return np.asarray(img.affine, float), tuple(int(s) for s in img.shape[:3])


def spacing(A):
    return np.linalg.norm(np.asarray(A)[:3, :3], axis=0)


def grid_points(mask, A, n_max=200_000):
    """World points of a mask on a regular stride (at most about n_max)."""
    s = max(1, int(np.ceil((mask.sum() / n_max) ** (1 / 3))))
    return apply(A, np.argwhere(mask[::s, ::s, ::s]).astype(float) * s)


def corners(shape, A):
    return apply(A, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float))


def volume_points(shape, A, n_max=200_000):
    """World points of a whole volume: its 8 corners plus a uniform sample of its grid (at most about n_max)."""
    shape = np.asarray(shape, int)
    s = max(1, int(np.ceil((float(shape.prod()) / n_max) ** (1 / 3))))
    g = np.indices(tuple(-(-shape // s))).reshape(3, -1).T.astype(float) * s
    return np.vstack([corners(shape, A), apply(A, g)])


def pose(T, T_other, pts, corner_pts):
    """Displacement of T against T_other (mm) over points and block corners; rotation angle of inv(T_other) @ T (deg)."""
    d = np.linalg.norm(apply(T, pts) - apply(T_other, pts), axis=1)
    dc = np.linalg.norm(apply(T, corner_pts) - apply(T_other, corner_pts), axis=1)
    D = (np.linalg.inv(T_other) @ T)[:3, :3]
    U, _, Vt = np.linalg.svd(D)
    proper = bool(np.linalg.det(D) > 0)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(U @ Vt) - 1) / 2, -1, 1)))) if proper else None
    return {"mean_mm": float(d.mean()), "max_mm": float(d.max()), "corners_mean_mm": float(dc.mean()),
            "corners_max_mm": float(dc.max()), "rotation_deg": ang, "same_handedness": proper}


def volume_cm3(mask, A):
    return float(mask.sum() * abs(np.linalg.det(A[:3, :3])) / 1000.0)


def _faces(n):
    """Face class of normals given in raw OCT voxel axes: 'a<axis><sign>' of the dominant component."""
    ax = np.abs(n).argmax(1)
    s = np.sign(n[np.arange(len(n)), ax])
    return np.array([f"a{a}{'+' if q > 0 else '-'}" if q else "" for a, q in zip(ax, s)])


def _stats(d):
    if not d.size:
        return {"n": 0, "median_mm": None, "p90_mm": None}
    return {"n": int(d.size), "median_mm": float(np.median(d)), "p90_mm": float(np.percentile(d, 90))}


def _agreement(d, cls):
    return {"all": _stats(d[cls != ""]), "rim": _stats(d[(cls != DEEP_END) & (cls != "")]), "deep_end": _stats(d[cls == DEEP_END]),
            "per_face_median_mm": {c: float(np.median(d[cls == c])) for c in sorted(set(cls.tolist())) if c}}


def _boundary(mask, A, keep):
    """Boundary voxels of mask (kept where keep): (boundary array, world points, outward normals as world covectors)."""
    b = mask & ~ndimage.binary_erosion(mask) & keep
    idx = np.argwhere(b)
    g = np.stack(np.gradient(ndimage.gaussian_filter(mask.astype(np.float32), 2.0)), -1)[tuple(idx.T)]
    return b, apply(A, idx.astype(float)), -g @ np.linalg.inv(A[:3, :3])


class Boundary:
    """Outline agreement of an OCT specimen mask and an MRI foreground through a pose; distance maps are built once."""

    def __init__(self, oct_mask, A_o, oct_valid, mri_mask, A_m, A_raw):
        self.A_o, self.A_m, self.A_raw = A_o, A_m, A_raw
        self.shape_o, self.shape_m = np.array(oct_mask.shape), np.array(mri_mask.shape)
        self.valid = ndimage.binary_erosion(oct_valid, iterations=2, border_value=1)
        keep = np.zeros(oct_mask.shape, bool)
        keep[FACE_VOX:-FACE_VOX, FACE_VOX:-FACE_VOX, FACE_VOX:-FACE_VOX] = True
        b, self.po, n = _boundary(oct_mask, A_o, keep & self.valid)
        self.cls_o = _faces(n @ A_raw[:3, :3])
        self.edt_o = ndimage.distance_transform_edt(~b, sampling=spacing(A_o)).astype(np.float32)
        self.margin = FOV_MM / spacing(A_m)
        bm = mri_mask & ~ndimage.binary_erosion(mri_mask)
        self.edt_m = ndimage.distance_transform_edt(~bm, sampling=spacing(A_m)).astype(np.float32)
        lo, hi = np.ceil(self.margin).astype(int), np.floor(self.shape_m - 1 - self.margin).astype(int) + 1
        keep_m = np.zeros(mri_mask.shape, bool)
        keep_m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        _, self.pm, self.nm = _boundary(mri_mask, A_m, keep_m)
        self.n_oct, self.n_mri = len(self.po), len(self.pm)

    def __call__(self, T):
        T = np.asarray(T, float)
        ijk = apply(np.linalg.inv(self.A_m) @ T, self.po)
        ok = np.all((ijk >= self.margin) & (ijk <= self.shape_m - 1 - self.margin), 1)
        fwd = _agreement(ndimage.map_coordinates(self.edt_m, ijk[ok].T, order=1), self.cls_o[ok])
        fwd["n_outside_mri"] = int((~ok).sum())
        ijk = apply(np.linalg.inv(self.A_o) @ np.linalg.inv(T), self.pm)
        ok = np.all((ijk >= FACE_VOX) & (ijk <= self.shape_o - 1 - FACE_VOX), 1)
        sel = np.flatnonzero(ok)
        sel = sel[self.valid[tuple(np.rint(ijk[sel]).astype(int).T)]]
        rev = _agreement(ndimage.map_coordinates(self.edt_o, ijk[sel].T, order=1), _faces(self.nm[sel] @ T[:3, :3] @ self.A_raw[:3, :3]))
        rev["frac_mri_boundary_inside_oct"] = float(len(sel) / max(1, self.n_mri))
        return {"rim_median_mm": [fwd["rim"]["median_mm"], rev["rim"]["median_mm"]], "forward_oct_to_mri": fwd, "reverse_mri_to_oct": rev,
                "n_boundary_voxels": {"oct": self.n_oct, "mri": self.n_mri}}


def frame_check(T, oct_in_mri, oct_path=OCT, n=40_000, box=7, seed=0):
    """Raw OCT values through T and the OCT header vs the exported overlay (Spearman), with flip and shift controls."""
    img = nib.load(str(oct_in_mri))
    overlay = np.asarray(img.dataobj, np.float32)
    oimg = nib.load(str(oct_path))
    A_o, shp = np.asarray(oimg.affine, float), np.array(oimg.shape[:3])
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(np.isfinite(overlay) & (overlay != 0))
    pick = np.unravel_index(rng.choice(cand, size=min(n, cand.size), replace=False), overlay.shape)
    exported = overlay[pick]
    P = apply(np.asarray(img.affine, float), np.stack(pick, 1).astype(float))
    del overlay
    variants = {"export": T}
    c = (shp - 1) / 2.0
    for k in range(3):
        F = np.eye(4)
        F[k, k], F[k, 3] = -1.0, 2 * c[k]
        variants[f"flip_axis{k}"] = T @ A_o @ F @ np.linalg.inv(A_o)
    for k, name in enumerate("xyz"):
        S = np.eye(4)
        S[k, 3] = 2.0
        variants[f"shift_2mm_{name}"] = S @ T
    h, names = box // 2, list(variants)
    rows = []                                            # (variant, point, i, j, k) of every raw voxel box needed
    for vi, name in enumerate(names):
        v = np.rint(apply(np.linalg.inv(A_o) @ np.linalg.inv(variants[name]), P)).astype(int)
        ok = np.flatnonzero(np.all((v >= h) & (v < shp - h), 1))
        rows.append(np.c_[np.full(len(ok), vi), ok, v[ok]])
    rows = np.concatenate(rows)
    rows = rows[np.argsort(rows[:, 4], kind="stable")]
    starts = np.searchsorted(rows[:, 4], np.arange(shp[2] + 1))
    raw = np.full((len(names), len(P)), np.nan)
    dt, off = oimg.header.get_data_dtype(), int(oimg.dataobj.offset)   # nibabel resets vox_offset in .gz headers
    slope, inter = float(oimg.dataobj.slope), float(oimg.dataobj.inter)
    nbytes, buf = int(shp[0] * shp[1]) * dt.itemsize, []
    with (gzip.open if str(oct_path).endswith(".gz") else open)(oct_path, "rb") as f:
        f.read(off)
        for k in range(int(shp[2])):
            buf = (buf + [np.frombuffer(f.read(nbytes), dt).reshape((shp[0], shp[1]), order="F")])[-box:]
            r = rows[starts[k - h]:starts[k - h + 1]] if k >= h else rows[:0]
            if len(buf) < box or not len(r):
                continue
            S = np.zeros((shp[0] + 1, shp[1] + 1))
            S[1:, 1:] = np.cumsum(np.cumsum(np.sum(buf, 0, dtype=np.float64), 0), 1)
            i0, i1, j0, j1 = r[:, 2] - h, r[:, 2] + h + 1, r[:, 3] - h, r[:, 3] + h + 1
            raw[r[:, 0], r[:, 1]] = (S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]) / box ** 3 * slope + inter
    res = {}
    for vi, name in enumerate(names):
        m = np.isfinite(raw[vi]) & (raw[vi] > 0)
        res[name] = {"n": int(m.sum()), "spearman": float(spearmanr(raw[vi][m], exported[m])[0]) if m.sum() > 100 else None}
    rho = res["export"]["spearman"]
    flips = [res[f"flip_axis{k}"]["spearman"] for k in range(3)]
    ok = rho is not None and rho >= 0.9 and all(x is None or x <= 0.3 for x in flips)
    return {"ok": bool(ok), "spearman_export": rho, "max_flip_spearman": max((x for x in flips if x is not None), default=None),
            "variants": res, "box": box, "n_points": int(len(P))}


def outline(bnd, T, others):
    """Boundary agreement at T plus the rim medians of the other poses {name: T} under the same masks."""
    return {**bnd(T), "rim_median_mm_of": {n: bnd(T2)["rim_median_mm"] for n, T2 in others.items()}}


def evaluate(run, masks=None, frame=True, previous=None):
    run = Path(run)
    masks = Path(masks) if masks else run
    T = load_T(run / "T_oct2mri.txt")
    A_hdr, shape = oct_header(OCT)
    out = {"run": str(run), "oct": str(OCT), "mri": str(MRI), "T_oct2mri": T.tolist()}
    others = {"previous": load_T(Path(previous) / "T_oct2mri.txt")} if previous else {}
    missing = [f for f in MASK_FILES if not (masks / f).exists()]
    if not missing:
        om, A_o = load_mask(masks / "oct_mask.nii.gz")
        mm, A_m = load_mask(masks / "mri_mask.nii.gz")
        valid, A_v = load_mask(masks / "oct_valid.nii.gz")
        if not np.allclose(A_v, A_o):
            raise ValueError("oct_valid.nii.gz and oct_mask.nii.gz are not on the same grid")
        out["mask"] = {"source": str(masks), "volume_cm3": volume_cm3(om, A_o)}
        pts, pts_of = grid_points(om, A_o), "the run's OCT specimen mask"
        out["boundary"] = outline(Boundary(om, A_o, valid, mm, A_m, A_hdr), T, others)
    else:
        out["mask"] = out["boundary"] = {"available": False, "reason": f"missing in {masks}: {', '.join(missing)}"}
        pts, pts_of = volume_points(shape, A_hdr), "the OCT array corners and a uniform sample of its grid"
    if previous:
        out["pose_to_previous"] = {"run": str(previous), "points": pts_of, "n_points": int(len(pts)),
                                   **pose(T, others["previous"], pts, corners(shape, A_hdr))}
    if frame:
        affine_overlay = run / "oct_in_mri_affine.nii.gz"       # since 1.1 oct_in_mri.nii.gz also goes through the field of §6
        out["frame_check"] = frame_check(T, affine_overlay if affine_overlay.exists() else run / "oct_in_mri.nii.gz", OCT)
    return out


def selftest():
    """Synthetic check of the frame check, the boundary agreement, the pose distance and the mask volume."""
    rng = np.random.default_rng(1)
    shape, sp = (40, 48, 36), 0.125                      # binary-exact spacings: NIfTI affines are float32
    V = (ndimage.gaussian_filter(rng.random(shape), 2.0) + 0.1).astype(np.float32)
    A_hdr = np.array([[-sp, 0, 0, 5.0], [0, -sp, 0, 3.0], [0, 0, -sp, -2.0], [0, 0, 0, 1]])   # header 'LPI'
    a = rng.normal(size=3) * 0.6
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T, th = np.eye(4), np.linalg.norm(a)
    T[:3, :3] = (np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K) @ np.diag([1.03, 0.98, 1.0])
    T[:3, 3] = [10.0, -4.0, 7.0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nib.save(nib.Nifti1Image(V, A_hdr), str(tmp / "oct.nii.gz"))
        assert oct_header(tmp / "oct.nii.gz")[1] == shape and np.allclose(oct_header(tmp / "oct.nii.gz")[0], A_hdr)
        # an MRI grid around the block, in a frame of its own
        A_m = np.array([[0, 0.1875, 0, 0], [0, 0, -0.1875, 0], [-0.1875, 0, 0, 0], [0, 0, 0, 1.0]])   # MRI header 'RIA'-like
        A_m[:3, 3] = np.round((apply(T @ A_hdr, (np.array(shape) - 1)[None] / 2.0)[0] - A_m[:3, :3] @ np.full(3, 23.5)) * 64) / 64
        Pm = apply(A_m, np.indices((48, 48, 48)).reshape(3, -1).T.astype(float))
        raw_ijk = apply(np.linalg.inv(A_hdr) @ np.linalg.inv(T), Pm)                       # MRI voxels -> raw OCT voxels
        # exported overlay = raw 3^3 box means at those voxels; the frame check must pass and its flips must fail
        v = np.rint(raw_ijk).astype(int)
        ok = np.all((v >= 1) & (v < np.array(shape) - 1), 1)
        Vb = ndimage.uniform_filter(V, 3)
        overlay = np.zeros(len(Pm), np.float32)
        overlay[ok] = Vb[tuple(v[ok].T)]
        nib.save(nib.Nifti1Image(overlay.reshape(48, 48, 48), A_m), str(tmp / "oct_in_mri.nii.gz"))
        fc = frame_check(T, tmp / "oct_in_mri.nii.gz", tmp / "oct.nii.gz", n=5000, box=3)
        assert fc["ok"] and fc["spearman_export"] > 0.999, fc
        # boundary agreement of an ellipsoid and its exact image through T: ~0 at T, about the shift under a 1 mm shift
        g = np.indices(shape).transpose(1, 2, 3, 0) - (np.array(shape) - 1) / 2.0
        ell = ((g / [12, 14, 10]) ** 2).sum(-1) <= 1
        mm = np.zeros(len(Pm), bool)
        mm[ok] = ell[tuple(v[ok].T)]
        B = Boundary(ell, A_hdr, np.ones(shape, bool), mm.reshape(48, 48, 48), A_m, A_hdr)
        top = np.argmax(apply(np.linalg.inv(A_hdr), B.po)[:, 0])      # the outline voxel furthest along raw axis 0 faces a0+
        assert B.cls_o[top] == "a0+"
        assert set(B(T)["reverse_mri_to_oct"]["per_face_median_mm"]) == {f"a{i}{s}" for i in range(3) for s in "+-"}
        S = np.eye(4)
        S[0, 3] = 1.0
        b0, b1 = outline(B, T, {"shifted": S @ T}), B(S @ T)
        assert max(b0["rim_median_mm"]) < 0.15 and min(b1["rim_median_mm"]) > 0.3, (b0["rim_median_mm"], b1["rim_median_mm"])
        assert b0["rim_median_mm_of"]["shifted"] == b1["rim_median_mm"]
        # the two point sets of pose_to_previous: zero against itself, the shift under a 1 mm shift
        cor = corners(shape, A_hdr)
        vp = volume_points(shape, A_hdr)
        assert len(vp) == 8 + int(np.prod(shape)) and np.allclose(vp[:8], cor)             # 40x48x36 is far below n_max
        for pts in (grid_points(ell, A_hdr), vp):
            assert pose(T, T, pts, cor)["max_mm"] < 1e-9
            d = pose(S @ T, T, pts, cor)
            assert abs(d["mean_mm"] - 1.0) < 1e-9 and abs(d["corners_max_mm"] - 1.0) < 1e-9 and d["same_handedness"]
        # mask volume: the ellipsoid against 4/3 pi abc voxels
        assert abs(volume_cm3(ell, A_hdr) / (4 / 3 * np.pi * 12 * 14 * 10 * sp ** 3 / 1000) - 1) < 0.02
    print(json.dumps({"selftest": "ok", "frame_check": {k: fc[k] for k in ("spearman_export", "max_flip_spearman")},
                      "boundary_rim_median_mm": {"at_T": b0["rim_median_mm"], "shift_1mm": b1["rim_median_mm"]},
                      "mask_volume_cm3": volume_cm3(ell, A_hdr)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path)
    ap.add_argument("--masks", type=Path, default=None, help="dir with oct_mask / oct_valid / mri_mask .nii.gz (default: RUN)")
    ap.add_argument("--previous", type=Path, default=None, help="an earlier run dir (T_oct2mri.txt) to measure this one against")
    ap.add_argument("--no-frame-check", action="store_true")
    ap.add_argument("-o", "--out", type=Path, default=None, help="default RUN/eval.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.run is None:
        ap.error("RUN is required")
    out = evaluate(a.run, a.masks, frame=not a.no_frame_check, previous=a.previous)
    with open(a.out or a.run / "eval.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1))
    summary = {k: out[k] for k in ("pose_to_previous",) if k in out}
    if "rim_median_mm" in out["boundary"]:
        summary["boundary"] = {k: out["boundary"][k] for k in ("rim_median_mm", "rim_median_mm_of")}
    if "frame_check" in out:
        summary["frame_check"] = {k: out["frame_check"][k] for k in ("ok", "spearman_export", "max_flip_spearman")}
    print(json.dumps(summary or {"mask": out["mask"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
