#!/usr/bin/env python3
"""Label-free evaluation of an octreg run on Xiangrui's I58 brainstem pair (bench only; the method never sees any of this).

    python bench/evaluate.py RUN [--masks DIR] [--no-frame-check] [-o RUN/eval.json]
    python bench/evaluate.py --selftest                     # synthetic, CPU, a few seconds

T = RUN/T_oct2mri.txt maps the OCT header world to the MRI header world (mm).
pose      Distance to the v1.1 reference pose R5. R5 maps the v1 pipeline OCT frame A_spr (layout 'SPR' of the raw array) to the
          MRI header world, so in the OCT header frame it is T_ref = T_R5 @ A_spr @ inv(A_hdr), with A_hdr the OCT NIfTI header
          affine of the same raw array. Mean and max displacement over the v1.1 OCT specimen-mask points, the same at the 8 block
          corners, and the rotation angle of inv(T_ref) @ T. T_ref is also compared with the stored header-frame export of R5.
mask      Volume of the run's OCT specimen mask (DIR/oct_mask.nii.gz) and its Dice against the stored v1.1 mask (18.05 cm3),
          counted on the v1.1 grid through the header affines.
boundary  The boundary part of v1.1 qc_fine.py, simplified: OCT mask boundary voxels (>= 3 voxels from the grid faces, on non-zero
          OCT data, DIR/oct_valid.nii.gz) through T -> distance to the MRI foreground boundary (DIR/mri_mask.nii.gz; points within
          1 mm of the MRI grid faces are dropped), and MRI boundary voxels through inv(T) -> distance to the OCT boundary. Medians
          per outward-normal class along the raw OCT array axes; 'rim' leaves out the deep end a0+, which has no specimen rim.
frame     Port of v1.1 check_export_header.py: random voxels of RUN/oct_in_mri.nii.gz go through inv(T) and inv(A_hdr) to raw OCT
          voxels, whose 20 um values (box^3 mean) are streamed from the .nii.gz; Spearman against the exported values. Controls:
          T composed with a flip of each raw OCT axis about the block centre, and 2 mm shifts along each MRI world axis.
          ok iff rho >= 0.9 and every flip gives rho <= 0.3 (spec_v1.0 R4).
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

ROOT = Path("/root/autodl-tmp/oct-mri-registration")
OCT = ROOT / "data/xiangrui/OCT_to_MRI/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz"
MRI = ROOT / "data/xiangrui/OCT_to_MRI/I58_brainstem_MRI_cropped_to_OCT.nii.gz"
V11 = ROOT / "work/v11/xiangrui_I58bs"                  # oct_native_affine.npy (A_spr), oct150_mask.npy, oct150_affine.npy
R5 = ROOT / "work/runs/v11/xiangrui_I58bs/T_oct2mri.npy"
R5_EXPORT = ROOT / "work/runs/xiangrui_I58_final/export/T_octnii_to_mrinii.npy"
DEEP_END = "a0+"                                        # raw OCT axis 0, high index: the rim-less deep end (v1.1 probe P1)
FACE_VOX, FOV_MM = 3, 1.0


def apply(T, pts):
    return pts @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def load_T(path):
    path = Path(path)
    return np.asarray(np.load(path) if path.suffix == ".npy" else np.loadtxt(path), float).reshape(4, 4)


def load_mask(path):
    img = nib.load(str(path))
    return np.asarray(img.dataobj) > 0, np.asarray(img.affine, float)


def spacing(A):
    return np.linalg.norm(np.asarray(A)[:3, :3], axis=0)


def grid_points(mask, A, n_max=200_000):
    """World points of a mask on a regular stride (at most about n_max)."""
    s = max(1, int(np.ceil((mask.sum() / n_max) ** (1 / 3))))
    return apply(A, np.argwhere(mask[::s, ::s, ::s]).astype(float) * s)


def corners(shape, A):
    return apply(A, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float))


def reference(oct_path=OCT, v11=V11, r5=R5):
    """R5 and the v1.1 specimen mask in the OCT header frame: dict(T_ref, A_hdr, shape, mask, A_mask, signed_permutation)."""
    img = nib.load(str(oct_path))
    A_hdr = np.asarray(img.affine, float)
    to_hdr = A_hdr @ np.linalg.inv(np.load(Path(v11) / "oct_native_affine.npy").astype(float))   # v1 pipeline world -> header world
    Q = np.abs(to_hdr[:3, :3])
    return {"T_ref": load_T(r5) @ np.linalg.inv(to_hdr), "A_hdr": A_hdr, "shape": tuple(int(s) for s in img.shape[:3]),
            "mask": np.load(Path(v11) / "oct150_mask.npy").astype(bool),
            "A_mask": to_hdr @ np.load(Path(v11) / "oct150_affine.npy").astype(float),
            "signed_permutation": bool(np.allclose(np.sort(Q, 1), [[0, 0, 1]] * 3, atol=1e-6) and np.allclose(Q.sum(0), 1, atol=1e-6))}


def pose(T, T_ref, pts, corner_pts):
    """Displacement of T against T_ref (mm) over points and block corners; rotation angle of inv(T_ref) @ T (deg)."""
    d = np.linalg.norm(apply(T, pts) - apply(T_ref, pts), axis=1)
    dc = np.linalg.norm(apply(T, corner_pts) - apply(T_ref, corner_pts), axis=1)
    D = (np.linalg.inv(T_ref) @ T)[:3, :3]
    U, _, Vt = np.linalg.svd(D)
    proper = bool(np.linalg.det(D) > 0)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(U @ Vt) - 1) / 2, -1, 1)))) if proper else None
    return {"mean_mm": float(d.mean()), "max_mm": float(d.max()), "corners_mean_mm": float(dc.mean()),
            "corners_max_mm": float(dc.max()), "rotation_deg": ang, "same_handedness": proper}


def mask_agreement(mask, A, ref_mask, A_ref):
    """Volumes (cm3) of mask (grid A) and ref_mask (grid A_ref), and Dice counted on the reference grid (nearest voxel)."""
    M = np.linalg.inv(A) @ A_ref
    jk = np.indices(ref_mask.shape[1:]).reshape(2, -1).T
    inter = on = 0
    for i in range(ref_mask.shape[0]):
        idx = np.rint(np.c_[np.full(len(jk), i), jk] @ M[:3, :3].T + M[:3, 3]).astype(int)
        ok = np.all((idx >= 0) & (idx < mask.shape), 1)
        v = np.zeros(len(jk), bool)
        v[ok] = mask[tuple(idx[ok].T)]
        on += int(v.sum())
        inter += int((v & ref_mask[i].ravel()).sum())
    return {"volume_cm3": volume_cm3(mask, A), "reference_cm3": volume_cm3(ref_mask, A_ref),
            "dice": 2.0 * inter / max(1, on + int(ref_mask.sum()))}


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


def evaluate(run, masks=None, frame=True):
    run = Path(run)
    masks = Path(masks) if masks else run
    T = load_T(run / "T_oct2mri.txt")
    ref = reference(OCT, V11, R5)
    pts, cor = grid_points(ref["mask"], ref["A_mask"]), corners(ref["shape"], ref["A_hdr"])
    out = {"run": str(run), "T_oct2mri": T.tolist(), "T_ref": ref["T_ref"].tolist(), "pose_to_R5": pose(T, ref["T_ref"], pts, cor),
           "reference_check": {"frames_signed_permutation": ref["signed_permutation"],
                               "formula_vs_stored_export_max_mm": pose(load_T(R5_EXPORT), ref["T_ref"], pts, cor)["max_mm"]}}
    missing = [f for f in ("oct_mask.nii.gz", "oct_valid.nii.gz", "mri_mask.nii.gz") if not (masks / f).exists()]
    if not missing:
        om, A_o = load_mask(masks / "oct_mask.nii.gz")
        mm, A_m = load_mask(masks / "mri_mask.nii.gz")
        valid, A_v = load_mask(masks / "oct_valid.nii.gz")
        if not np.allclose(A_v, A_o):
            raise ValueError("oct_valid.nii.gz and oct_mask.nii.gz are not on the same grid")
        out["mask"] = mask_agreement(om, A_o, ref["mask"], ref["A_mask"])
        out["boundary"] = Boundary(om, A_o, valid, mm, A_m, ref["A_hdr"])(T)
    else:
        out["mask"] = out["boundary"] = {"available": False, "reason": f"missing in {masks}: {', '.join(missing)}"}
    if frame:
        out["frame_check"] = frame_check(T, run / "oct_in_mri.nii.gz", OCT)
    return out


def selftest():
    """Synthetic check of the header-frame conversion, the mask conversion, the frame check, Dice and boundary agreement."""
    rng = np.random.default_rng(1)
    shape, sp = (40, 48, 36), 0.125                      # binary-exact spacings: NIfTI affines are float32
    V = (ndimage.gaussian_filter(rng.random(shape), 2.0) + 0.1).astype(np.float32)
    A_hdr = np.array([[-sp, 0, 0, 5.0], [0, -sp, 0, 3.0], [0, 0, -sp, -2.0], [0, 0, 0, 1]])   # header 'LPI'
    A_spr = np.array([[0, 0, sp, 0], [0, -sp, 0, 0], [sp, 0, 0, 0], [0, 0, 0, 1.0]])         # v1 layout 'SPR'
    a = rng.normal(size=3) * 0.6
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T_r5, th = np.eye(4), np.linalg.norm(a)
    T_r5[:3, :3] = (np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K) @ np.diag([1.03, 0.98, 1.0])
    T_r5[:3, 3] = [10.0, -4.0, 7.0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nib.save(nib.Nifti1Image(V, A_hdr), str(tmp / "oct.nii.gz"))
        m_raw = V > np.median(V)
        m150 = m_raw.reshape(20, 2, 24, 2, 18, 2).mean((1, 3, 5)) > 0.5                                # 2x2x2 pooling
        A150 = A_spr.copy()
        A150[:3, 3] += A_spr[:3, :3] @ np.full(3, 0.5)
        A150[:3, :3] *= 2
        np.save(tmp / "oct_native_affine.npy", A_spr), np.save(tmp / "oct150_mask.npy", m150), np.save(tmp / "oct150_affine.npy", A150)
        np.save(tmp / "T_r5.npy", T_r5)
        ref = reference(tmp / "oct.nii.gz", tmp, tmp / "T_r5.npy")
        T = ref["T_ref"]
        # an MRI grid around the block; its voxels land on the same raw voxel through the v1 frame and through the header frame
        A_m = np.array([[0, 0.1875, 0, 0], [0, 0, -0.1875, 0], [-0.1875, 0, 0, 0], [0, 0, 0, 1.0]])   # MRI header 'RIA'-like
        A_m[:3, 3] = np.round((apply(T_r5 @ A_spr, (np.array(shape) - 1)[None] / 2.0)[0] - A_m[:3, :3] @ np.full(3, 23.5)) * 64) / 64
        Pm = apply(A_m, np.indices((48, 48, 48)).reshape(3, -1).T.astype(float))
        via_v1 = apply(np.linalg.inv(A_spr) @ np.linalg.inv(T_r5), Pm)
        assert np.allclose(apply(np.linalg.inv(A_hdr) @ np.linalg.inv(T), Pm), via_v1, atol=1e-6), "header-frame pose formula"
        assert not np.allclose(apply(np.linalg.inv(A_hdr) @ np.linalg.inv(T_r5), Pm), via_v1, atol=1.0), "control has no power"
        c150 = np.argwhere(m150).astype(float)
        assert np.allclose(apply(np.linalg.inv(A_hdr) @ ref["A_mask"], c150), apply(np.linalg.inv(A_spr) @ A150, c150)), "mask affine"
        assert ref["signed_permutation"]
        # exported overlay = raw 3^3 box means at the header-frame voxels; the frame check must pass and its flips must fail
        v = np.rint(via_v1).astype(int)
        ok = np.all((v >= 1) & (v < np.array(shape) - 1), 1)
        Vb = ndimage.uniform_filter(V, 3)
        overlay = np.zeros(len(Pm), np.float32)
        overlay[ok] = Vb[tuple(v[ok].T)]
        nib.save(nib.Nifti1Image(overlay.reshape(48, 48, 48), A_m), str(tmp / "oct_in_mri.nii.gz"))
        fc = frame_check(T, tmp / "oct_in_mri.nii.gz", tmp / "oct.nii.gz", n=5000, box=3)
        assert fc["ok"] and fc["spearman_export"] > 0.999, fc
        # Dice: the same world mask stored flipped along axis 1 is identical; the pooled v1.1-style mask is close
        F = np.eye(4)
        F[1, 1], F[1, 3] = -1, shape[1] - 1
        assert abs(mask_agreement(m_raw[:, ::-1], A_hdr @ F, m_raw, A_hdr)["dice"] - 1) < 1e-12
        assert mask_agreement(m_raw, A_hdr, ref["mask"], ref["A_mask"])["dice"] > 0.8
        # boundary agreement of an ellipsoid and its exact image through T: ~0 at T, about the shift under a 1 mm shift
        g = np.indices(shape).transpose(1, 2, 3, 0) - (np.array(shape) - 1) / 2.0
        ell = ((g / [12, 14, 10]) ** 2).sum(-1) <= 1
        vm = np.rint(apply(np.linalg.inv(A_hdr) @ np.linalg.inv(T), Pm)).astype(int)
        okm = np.all((vm >= 0) & (vm < shape), 1)
        mm = np.zeros(len(Pm), bool)
        mm[okm] = ell[tuple(vm[okm].T)]
        B = Boundary(ell, A_hdr, np.ones(shape, bool), mm.reshape(48, 48, 48), A_m, A_hdr)
        top = np.argmax(apply(np.linalg.inv(A_hdr), B.po)[:, 0])      # the outline voxel furthest along raw axis 0 faces a0+
        assert B.cls_o[top] == "a0+"
        assert set(B(T)["reverse_mri_to_oct"]["per_face_median_mm"]) == {f"a{i}{s}" for i in range(3) for s in "+-"}
        b0, S = B(T), np.eye(4)
        S[0, 3] = 1.0
        b1 = B(S @ T)
        assert max(b0["rim_median_mm"]) < 0.15 and min(b1["rim_median_mm"]) > 0.3, (b0["rim_median_mm"], b1["rim_median_mm"])
        assert pose(T, T, grid_points(ell, A_hdr), corners(shape, A_hdr))["max_mm"] < 1e-9
    print(json.dumps({"selftest": "ok", "frame_check": {k: fc[k] for k in ("spearman_export", "max_flip_spearman")},
                      "boundary_rim_median_mm": {"at_T": b0["rim_median_mm"], "shift_1mm": b1["rim_median_mm"]}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path)
    ap.add_argument("--masks", type=Path, default=None, help="dir with oct_mask / oct_valid / mri_mask .nii.gz (default: RUN)")
    ap.add_argument("--no-frame-check", action="store_true")
    ap.add_argument("-o", "--out", type=Path, default=None, help="default RUN/eval.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.run is None:
        ap.error("RUN is required")
    out = evaluate(a.run, a.masks, frame=not a.no_frame_check)
    (a.out or a.run / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("pose_to_R5", "reference_check")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
