#!/usr/bin/env python3
"""Label-free QC of one registration run with intensity-independent and sign-free measures (octreg v1.1, spec 3.2):

    python scripts/qc_fine.py --work W --run R [--T PATH] [--out DIR] [--skip-landscape]

Q1 boundary agreement (geometry only): OCT specimen-mask boundary voxels (oct150_mask & ~erosion, >= 3 voxels from the box
   faces and >= 2 from the black tiles) mapped through T -> distance (mm, EDT) to the MRI tissue boundary; per face class of
   the outward normal (z-, z+, y-, y+, x-, x+ in OCT array axes; z+ = the rim-less deep end, reported separately from the rim
   faces); and the reverse direction (MRI tissue boundary voxels through inv T -> distance to the OCT mask boundary).  The
   MRI tissue mask (prep: threshold only) is cleaned first (closing 2 vox + fill holes + components >= 1 mm3): on I58 55 % of
   its raw boundary voxels are interior speckle holes, which would put an MRI 'boundary' within 0.3 mm of every point.
Q2 MI32 landscape at T on the 0.15 mm level (raw intensities, 32-bin joint histogram, bin edges fixed at T): translations
   +-0.25/0.5/1/2 mm along the OCT array axes, rotations +-1/2/5 deg through the block centre, log-scales +-3/6 %;
   parabolic argmax offset and unimodality per axis.
Q3 figures: qc_fine.png (5 depth planes x {OCT, MRI via T, checkerboard with 0.9 mm checks, Canny edges of the MRI over the
   OCT}; MRI tissue boundary via the final T solid / via the pre-fine T dashed; OCT mask contour) and qc_fine_landscape.png.
Everything is computed for the final T and, when run/T_oct2mri_prefine.npy exists, for the pre-fine pose (boundary numbers
also for T_oct2mri_fine_candidate.npy).  Reads only prep outputs and run outputs; imports only octreg.common/evaluate.
Writes qc_fine.json (+ the two PNGs) into --out (default: the run dir)."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import DEVICE, apply_affine, resample_to_grid, sample_at_world, to_t, voxel_size, world_bbox_to_voxel, write_json
from octreg.evaluate import transform_diff

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--T", type=Path, default=None, help="4x4 .npy/.json OCT world -> MRI world (default run/T_oct2mri.npy)"); ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--checker-mm", type=float, default=0.9); ap.add_argument("--erode-mm", type=float, default=1.3, help="MRI tissue erosion of the MI specimen mask")
ap.add_argument("--planes", type=str, default="0.2,0.35,0.5,0.65,0.8"); ap.add_argument("--skip-landscape", action="store_true"); ap.add_argument("--pad-mm", type=float, default=4.0)
a = ap.parse_args(); out = a.out or a.run; out.mkdir(parents=True, exist_ok=True); t_start = time.time()
torch.set_num_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "4")))))
def say(*x): print(*x, f"[{time.time()-t_start:.0f}s]", flush=True)
AXN = ["z", "y", "x"]; DEEP_END = "z+"                                   # OCT array axes (Z=depth, Y, X); z+ = rim-less deep end (P1 caveat 3)

# ------------------------------------------------------------------ load
oct150 = np.load(a.work / "oct150.npy").astype(np.float32); oct_mask = np.load(a.work / "oct150_mask.npy").astype(bool); A_oct = np.load(a.work / "oct150_affine.npy")
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); mri_tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
shape_o = np.array(oct150.shape); vox_o = float(voxel_size(A_oct).mean()); vox_m = voxel_size(A_mri)
c_o = (A_oct @ np.r_[(shape_o - 1) / 2.0, 1.0])[:3]
def load_T(p): return np.load(p) if p.suffix == ".npy" else np.array(json.load(open(p)), float)
Ts = {}; T_files = {}
T_files["final"] = a.T if a.T is not None else a.run / "T_oct2mri.npy"; Ts["final"] = load_T(T_files["final"])
for nm in ("prefine", "candidate"):
    f = a.run / {"prefine": "T_oct2mri_prefine.npy", "candidate": "T_oct2mri_fine_candidate.npy"}[nm]
    if f.exists():
        Tn = np.load(f)
        if nm == "candidate" and np.allclose(Tn, Ts["final"]): continue                # accepted fine pose: candidate == final
        Ts[nm] = Tn; T_files[nm] = f
res = json.load(open(a.run / "result.json")) if (a.run / "result.json").exists() else {}
say(f"OCT {tuple(shape_o)} @ {vox_o:.3f} mm, mask {oct_mask.mean():.3f}; MRI {mri.shape} @ {np.round(vox_m, 3).tolist()} mm; poses {list(Ts)}")

# ------------------------------------------------------------------ MRI box around the mapped block (all poses), padded
corn = np.array([[i, j, k] for i in (0, shape_o[0] - 1) for j in (0, shape_o[1] - 1) for k in (0, shape_o[2] - 1)], float)
cw = np.concatenate([(T @ np.c_[(A_oct @ np.c_[corn, np.ones(8)].T).T[:, :3], np.ones(8)].T).T[:, :3] for T in Ts.values()])
blo, bhi = world_bbox_to_voxel(A_mri, mri.shape, cw.min(0) - a.pad_mm, cw.max(0) + a.pad_mm)
A_box = A_mri.copy(); A_box[:3, 3] = (A_mri @ np.r_[blo, 1.0])[:3]
mri_box = np.asarray(mri[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]]).astype(np.float32); tis_box = np.asarray(mri_tissue[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]]).astype(bool)
shape_m = np.array(mri_box.shape); say(f"MRI box {blo.tolist()}..{bhi.tolist()} = {tuple(shape_m)} voxels, tissue {tis_box.mean():.3f}")
def ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]; return (z * z + y * y + x * x) <= r * r
def clean_mask(m, vox, min_cm3=1e-3, close_vox=2):
    """closing + fill holes + drop components < min_cm3 (speckle holes and islands of a threshold-only tissue mask)."""
    c = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=ball(close_vox)))
    lab, n = ndimage.label(c); sz = np.bincount(lab.ravel()); keep = np.where(sz * float(np.prod(vox)) / 1000.0 >= min_cm3)[0]
    return np.isin(lab, keep[keep > 0]) if n > 1 else c
n_raw_bnd = int((tis_box & ~ndimage.binary_erosion(tis_box)).sum()); tis_box = clean_mask(tis_box, vox_m)
say(f"MRI tissue mask cleaned (closing 2 vox + fill + cc >= 1 mm3): tissue {tis_box.mean():.3f}, boundary voxels {n_raw_bnd} -> {int((tis_box & ~ndimage.binary_erosion(tis_box)).sum())}")
mri_t = to_t(mri_box)[None]; tisf_t = to_t(tis_box.astype(np.float32))[None]; A_box_t = to_t(A_box); A_oct_t = to_t(A_oct); Ainv_mri = np.linalg.inv(A_mri)

# ------------------------------------------------------------------ Q1 boundary agreement
valid = oct150 > 0; valid2 = ndimage.binary_erosion(valid, iterations=2)                    # black tiles are missing data (P0.6)
bnd_o_all = oct_mask & ~ndimage.binary_erosion(oct_mask)
FACE = 3                                                                                    # box-face margin (vox): 3 rejects the layer scipy's closing erodes at a cut face
face_ok = np.zeros(oct150.shape, bool); face_ok[FACE:-FACE, FACE:-FACE, FACE:-FACE] = True
B = bnd_o_all & valid2 & face_ok
sm_o = ndimage.gaussian_filter(oct_mask.astype(np.float32), 2.0)
bnd_m = tis_box & ~ndimage.binary_erosion(tis_box)
sm_m = ndimage.gaussian_filter(tis_box.astype(np.float32), 2.0)
# MRI boundary voxels within 1 mm of the MRI field-of-view faces are crop artefacts (tissue truncated by the FOV), not tissue boundary
gidx = np.argwhere(bnd_m) + blo; fov_m = ((gidx >= np.ceil(1.0 / vox_m)) & (gidx <= np.array(mri.shape) - 1 - np.ceil(1.0 / vox_m))).all(1)
bnd_m_use = np.zeros_like(bnd_m); bnd_m_use[tuple((gidx[fov_m] - blo).T)] = True
say(f"OCT mask boundary {int(bnd_o_all.sum())} voxels -> {int(B.sum())} usable (>= {FACE} vox from faces, >= 2 from black tiles); MRI tissue boundary {int(bnd_m.sum())} -> {int(bnd_m_use.sum())} usable (>= 1 mm from the MRI FOV faces)")
if B.sum() < 1000: say(f"WARNING: only {int(B.sum())} usable OCT boundary voxels - the OCT mask has no specimen rim inside the box (intensity mask?); Q1 is not meaningful")
d_m = ndimage.distance_transform_edt(~bnd_m, sampling=tuple(vox_m)).astype(np.float32); d_m_t = to_t(d_m)[None]; del d_m
d_o = ndimage.distance_transform_edt(~B, sampling=(vox_o,) * 3).astype(np.float32) if B.sum() >= 1000 else None; d_o_t = to_t(d_o)[None] if d_o is not None else None; del d_o
say("EDTs done")

def grad_at(sm, idx):
    """central differences of a smoothed mask at voxel indices -> [N,3] (points INTO the mask; outward normal = -grad)."""
    g = np.zeros((len(idx), 3), np.float32); n = np.array(sm.shape)
    for ax in range(3):
        ip = idx.copy(); ip[:, ax] = np.minimum(ip[:, ax] + 1, n[ax] - 1); im = idx.copy(); im[:, ax] = np.maximum(im[:, ax] - 1, 0)
        g[:, ax] = (sm[ip[:, 0], ip[:, 1], ip[:, 2]] - sm[im[:, 0], im[:, 1], im[:, 2]]) / 2.0
    return g

def face_class(n):
    """dominant OCT array axis and sign of outward normals [N,3] -> array of 'z-','z+',...; '' for zero normals."""
    mag = np.abs(n); ax = mag.argmax(1); s = np.sign(n[np.arange(len(n)), ax]); ok = mag.max(1) > 1e-6
    return np.array([f"{AXN[q]}{'+' if sq > 0 else '-'}" if o else "" for q, sq, o in zip(ax, s, ok)])

def dstats(d):
    d = np.asarray(d, np.float64)
    if d.size == 0: return {"n": 0, "median_mm": None, "p75_mm": None, "p90_mm": None, "mean_mm": None, "frac_within_0.5mm": None}
    return {"n": int(d.size), "median_mm": float(np.median(d)), "p75_mm": float(np.percentile(d, 75)), "p90_mm": float(np.percentile(d, 90)), "mean_mm": float(d.mean()), "frac_within_0.5mm": float((d <= 0.5).mean())}

def agreement(d, cls):
    per = {c: dstats(d[cls == c]) for c in sorted(set(cls.tolist())) if c}
    rim = cls != DEEP_END; rim &= cls != ""
    return {"overall": dstats(d[cls != ""]), "rim_faces": dstats(d[rim]), "deep_end": dstats(d[cls == DEEP_END]), "per_face": per, "deep_end_class": DEEP_END}

def sample_np(vol_t, A_t, pts_np, chunk=4_000_000):
    outv = np.empty(len(pts_np), np.float32)
    if len(pts_np) == 0: return outv
    for s in range(0, len(pts_np), chunk): outv[s:s + chunk] = sample_at_world(vol_t, A_t, to_t(pts_np[s:s + chunk]))[0].cpu().numpy()
    return outv

idx_B = np.argwhere(B); p_o_B = (A_oct @ np.c_[idx_B, np.ones(len(idx_B))].T).T[:, :3]; cls_B = face_class(-grad_at(sm_o, idx_B))
idx_M = np.argwhere(bnd_m_use); p_m_M = (A_box @ np.c_[idx_M, np.ones(len(idx_M))].T).T[:, :3]; nvox_M = -grad_at(sm_m, idx_M)
n_w_M = nvox_M @ np.linalg.inv(A_box[:3, :3])                                              # normals: voxel -> world by L^-T  (rows: n @ L^-1)
margin = 1.0 / vox_m
boundary = {}
for nm, T in Ts.items():
    # forward: OCT mask boundary -> MRI tissue boundary distance
    p_m = (T @ np.c_[p_o_B, np.ones(len(p_o_B))].T).T[:, :3]; ijk = (Ainv_mri @ np.c_[p_m, np.ones(len(p_m))].T).T[:, :3]
    fov = ((ijk >= margin) & (ijk <= np.array(mri.shape) - 1 - margin)).all(1)
    fwd = agreement(sample_np(d_m_t, A_box_t, p_m[fov]), cls_B[fov]); fwd["n_excluded_fov_margin"] = int((~fov).sum())
    # reverse: MRI tissue boundary -> OCT mask boundary distance (points inside the OCT box, 2 voxels off the faces, on valid OCT)
    Tinv = np.linalg.inv(T); p_o = (Tinv @ np.c_[p_m_M, np.ones(len(p_m_M))].T).T[:, :3]; ijk_o = (np.linalg.inv(A_oct) @ np.c_[p_o, np.ones(len(p_o))].T).T[:, :3]
    inside = ((ijk_o >= FACE) & (ijk_o <= shape_o - 1 - FACE)).all(1); r = np.rint(ijk_o[inside]).astype(int)
    keep = inside.copy(); keep[inside] = valid2[r[:, 0], r[:, 1], r[:, 2]]
    n_o = (n_w_M[keep] @ T[:3, :3]) @ A_oct[:3, :3]                                           # world -> OCT world (inv T: normals by T^T) -> OCT voxel axes (by A^T)
    rev = agreement(sample_np(d_o_t, A_oct_t, p_o[keep]), face_class(n_o)) if d_o_t is not None else None
    if rev is not None: rev["n_mri_boundary_inside_oct"] = int(keep.sum()); rev["frac_mri_boundary_inside_oct"] = float(keep.mean())
    boundary[nm] = {"forward_oct_to_mri": fwd, "reverse_mri_to_oct": rev, "n_usable_oct_boundary": int(B.sum()), "n_usable_mri_boundary": int(bnd_m_use.sum())}
    f_ = fwd["rim_faces"]; r_ = rev["rim_faces"] if rev else {}; fm = lambda v: "n/a" if v is None else f"{v:.3f}"
    say(f"boundary [{nm}] rim median fwd {fm(f_['median_mm'])} / rev {fm(r_.get('median_mm'))} mm (p75 {fm(f_['p75_mm'])} / {fm(r_.get('p75_mm'))}); "
        f"deep end fwd {fm(fwd['deep_end']['median_mm'])} / rev {fm(rev['deep_end']['median_mm'] if rev else None)}; per face fwd " + " ".join(f"{c}:{v['median_mm']:.2f}({v['n']})" for c, v in fwd["per_face"].items()))
if "prefine" in boundary and "final" in boundary:
    inc = {c: (boundary["final"]["forward_oct_to_mri"]["per_face"].get(c, {}).get("median_mm") or 0) - (v["median_mm"] or 0) for c, v in boundary["prefine"]["forward_oct_to_mri"]["per_face"].items() if c != DEEP_END and v["median_mm"] is not None}
    boundary["prefine_to_final_rim_median_increase_mm"] = {"per_face_forward": inc, "max": max(inc.values()) if inc else None, "F7_limit": 0.1}
    say("prefine -> final rim-face median change (fwd, mm):", {k: round(v, 3) for k, v in inc.items()})

# ------------------------------------------------------------------ Q2 MI32 landscape
def mi32(ia, ib, m, bins=32):
    h = torch.bincount((ia[m] * bins + ib[m]), minlength=bins * bins).double().view(bins, bins); p = h / h.sum(); pa = p.sum(1); pb = p.sum(0); nz = p > 0
    return float((p[nz] * torch.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
def to_bins(v, lo, hi, bins=32): return ((v - lo) / (hi - lo + 1e-12) * bins).long().clamp(0, bins - 1)
u_axes = [A_oct[:3, k] / np.linalg.norm(A_oct[:3, k]) for k in range(3)]
def rodrigues(u, deg):
    th = math.radians(deg); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]); return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
def perturb(T, kind, ax, amt):
    """OCT-world perturbation applied before T: translation (mm) along / rotation (deg) about / log-scale along OCT array axis ax."""
    P = np.eye(4); u = u_axes[ax]
    if kind == "trans": P[:3, 3] = amt * u
    elif kind == "rot": R = rodrigues(u, amt); P[:3, :3] = R; P[:3, 3] = c_o - R @ c_o
    else: M = np.eye(3) + (math.exp(amt) - 1.0) * np.outer(u, u); P[:3, :3] = M; P[:3, 3] = c_o - M @ c_o
    return T @ P
OFFS = {"trans": [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2], "rot": [-5, -2, -1, 0, 1, 2, 5], "scale": [-0.06, -0.03, 0, 0.03, 0.06]}
def peak(xs, ys):
    i = int(np.argmax(ys)); pk = None
    if 0 < i < len(xs) - 1:
        c = np.polyfit(np.array(xs[i - 1:i + 2], float), np.array(ys[i - 1:i + 2]), 2)
        if c[0] < 0: pk = float(np.clip(-c[1] / (2 * c[0]), xs[i - 1], xs[i + 1]))
    interior_max = [j for j in range(1, len(xs) - 1) if ys[j] > ys[j - 1] and ys[j] > ys[j + 1]]
    return {"argmax_offset": float(xs[i]), "argmax_parabolic": pk, "unimodal": len(interior_max) <= 1 and (not interior_max or interior_max[0] == i), "value_at_T": float(ys[xs.index(0)]), "max_value": float(ys[i]), "offsets": list(xs), "values": [float(v) for v in ys]}
# erode_ball as in fine.py: pad one background voxel so the MRI-box / FOV faces erode like binary_erosion does (else a rind of
# erode_mm at every cut face would stay in m_spec and Q2 would be evaluated on a different mask than the fine stage's MI)
te = (ndimage.distance_transform_edt(np.pad(tis_box, 1), sampling=tuple(vox_m))[1:-1, 1:-1, 1:-1] >= a.erode_mm).astype(np.float32); te_t = to_t(te)[None]; del te
landscape = {}
if not a.skip_landscape:
    for nm in [k for k in ("final", "prefine") if k in Ts]:
        T = Ts[nm]; T_t = to_t(T); t1 = time.time()
        m_spec = (resample_to_grid(te_t, A_box_t, A_oct_t, tuple(shape_o), T_grid_to_vol=T_t)[0] > 0.5).cpu().numpy() & oct_mask & valid2
        idx = np.argwhere(m_spec); pts = to_t((A_oct @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]); I_o = to_t(oct150[m_spec])
        def sample(Tp):
            pm = apply_affine(to_t(Tp), pts); return sample_at_world(mri_t, A_box_t, pm)[0], sample_at_world(tisf_t, A_box_t, pm)[0] > 0.5
        im0, tis0 = sample(T)
        ra = np.percentile(I_o[tis0].cpu().numpy(), [0.5, 99.5]); rb = np.percentile(im0[tis0].cpu().numpy(), [0.5, 99.5]); ia = to_bins(I_o, *ra)
        def value(Tp):
            im, tis = sample(Tp); return mi32(ia, to_bins(im, *rb), tis)
        v0 = value(T); L = {"mask_voxels": int(m_spec.sum()), "mask_cm3": float(m_spec.sum()) * vox_o ** 3 / 1000, "value_at_T": v0, "bins": {"oct": ra.tolist(), "mri": rb.tolist()}}
        for kind, offs in OFFS.items():
            for ax in range(3):
                ys = [v0 if o == 0 else value(perturb(T, kind, ax, o)) for o in offs]
                L[f"{kind}_{AXN[ax]}"] = peak(offs, ys)
        L["seconds"] = time.time() - t1; landscape[nm] = L
        say(f"MI32 landscape [{nm}] mask {L['mask_cm3']:.2f} cm3, MI at T {v0:.4f}; argmax (parabolic|grid): " +
            " ".join(f"{k}:{(v['argmax_parabolic'] if v['argmax_parabolic'] is not None else v['argmax_offset']):+.2f}{'' if v['unimodal'] else '*'}" for k, v in L.items() if isinstance(v, dict) and "unimodal" in v) + "  (* = not unimodal)")

# ------------------------------------------------------------------ outputs: json first
Q = {"work": str(a.work), "run": str(a.run), "T_used": {k: {"file": str(v), "T": Ts[k].tolist()} for k, v in T_files.items()},
     "boundary": boundary, "mi_landscape": landscape or None, "settings": {"erode_mm": a.erode_mm, "checker_mm": a.checker_mm, "face_margin_vox": FACE, "valid_erosion_vox": 2, "fov_margin_mm": 1.0, "deep_end_class": DEEP_END, "mri_box_ijk": [blo.tolist(), bhi.tolist()], "mri_mask_cleanup": "closing 2 vox + fill holes + components >= 1 mm3", "mi_mask": "erode_ball(cleaned MRI tissue, erode_mm; box faces = background as in fine.py) via T & oct150_mask & (oct150 > 0 eroded 2), intersected with MRI tissue via the perturbed pose"},
     "pose_diffs": {f"{p}_vs_{q}": transform_diff(Ts[p], Ts[q], c_o) for p in Ts for q in Ts if p < q}, "fine": {k: res.get("fine", {}).get(k) for k in ("accepted", "reasons", "U_mm", "sign_per_level")} if res.get("fine") else None}
write_json(Q, out / "qc_fine.json"); say("wrote", out / "qc_fine.json")

# ------------------------------------------------------------------ Q3 figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.feature import canny
T_t = to_t(Ts["final"])
mri_in_oct = resample_to_grid(mri_t, A_box_t, A_oct_t, tuple(shape_o), T_grid_to_vol=T_t)[0].cpu().numpy()
tis_in = {nm: (resample_to_grid(tisf_t, A_box_t, A_oct_t, tuple(shape_o), T_grid_to_vol=to_t(T))[0] > 0.5).cpu().numpy() for nm, T in Ts.items() if nm in ("final", "prefine")}
D = int(shape_o[0]); fr = [float(v) for v in a.planes.split(",")]; zs = [min(D - 1, int(round(f * (D - 1)))) for f in fr]
vo = np.percentile(oct150[oct_mask], 99) if oct_mask.any() else oct150.max(); ml, mh = np.percentile(mri_in_oct[oct_mask], [1, 99]) if oct_mask.any() else (0, 1)
k = max(1, int(round(a.checker_mm / vox_o))); ii, jj = np.indices(oct150.shape[1:]); cb = ((ii // k) + (jj // k)) % 2
Hh, Ww = oct150.shape[1:]; fig, ax = plt.subplots(len(zs), 4, figsize=(4 * 4.4, len(zs) * 4.4 * Hh / Ww + 0.8))
def contours(axi, z):
    axi.contour(oct_mask[z], levels=[0.5], colors="w", linewidths=0.5, alpha=0.9)
    if "prefine" in tis_in: axi.contour(tis_in["prefine"][z], levels=[0.5], colors="orange", linewidths=0.9, linestyles="dashed")
    axi.contour(tis_in["final"][z], levels=[0.5], colors="cyan", linewidths=0.9)
for r, z in enumerate(zs):
    o = np.clip(oct150[z] / vo, 0, 1); m = np.clip((mri_in_oct[z] - ml) / (mh - ml + 1e-6), 0, 1)
    ax[r, 0].imshow(o, cmap="gray"); ax[r, 1].imshow(m, cmap="gray"); ax[r, 2].imshow(np.where(cb == 0, o, m), cmap="gray")
    ed = canny(m, sigma=1.5) & ndimage.binary_dilation(tis_in["final"][z], iterations=2)
    rgb = np.repeat(o[..., None], 3, -1); rgb[ed] = (1.0, 0.15, 0.15); ax[r, 3].imshow(rgb)
    for c in range(4): contours(ax[r, c], z); ax[r, c].axis("off")
    ax[r, 0].text(3, 12, f"z = {z}  ({fr[r]:.2f} D, {z * vox_o:.1f} mm)", color="yellow", fontsize=9, bbox=dict(facecolor="black", alpha=0.5, pad=1.5))
for c, t in enumerate(["OCT (0.15 mm)", "MRI via final T", f"checkerboard ({k} vox = {k * vox_o:.2f} mm)", "Canny edges of MRI (red) over OCT"]): ax[0, c].set_title(t, fontsize=10)
bf = boundary["final"]["forward_oct_to_mri"]; br = boundary["final"]["reverse_mri_to_oct"] or {"rim_faces": {}, "deep_end": {}}
fine = res.get("fine") or {}
fm2 = lambda v: "n/a" if v is None else f"{v:.2f}"
sup = (f"{a.work.name} / {a.run.name}: boundary agreement (final, {int(B.sum())} usable OCT rim voxels) rim median fwd {fm2(bf['rim_faces']['median_mm'])} / rev {fm2(br['rim_faces'].get('median_mm'))} mm, "
       f"deep end {bf['deep_end']['median_mm'] if bf['deep_end']['median_mm'] is None else round(bf['deep_end']['median_mm'], 2)} / {br['deep_end'].get('median_mm') if br['deep_end'].get('median_mm') is None else round(br['deep_end']['median_mm'], 2)} mm")
if "prefine" in boundary: sup += f"\nprefine rim median fwd {fm2(boundary['prefine']['forward_oct_to_mri']['rim_faces']['median_mm'])} mm (dashed orange = MRI tissue via prefine; cyan = via final; white = OCT mask)" + (f"; fine accepted {fine.get('accepted')} U {fine.get('U_mm')}" if fine else "")
else: sup += "\ncyan = MRI tissue boundary via T; white = OCT mask (no pre-fine pose in this run)"
if landscape.get("final"): sup += "\nMI32 argmax offsets (final): " + " ".join(f"{kk}:{(v['argmax_parabolic'] if v['argmax_parabolic'] is not None else v['argmax_offset']):+.2f}{'' if v['unimodal'] else '*'}" for kk, v in landscape["final"].items() if isinstance(v, dict) and "unimodal" in v)
fig.suptitle(sup, fontsize=9.5); plt.tight_layout(rect=(0, 0, 1, 0.965)); plt.savefig(out / "qc_fine.png", dpi=110); plt.close(fig); say("wrote", out / "qc_fine.png")
if landscape:
    fig, ax = plt.subplots(3, 3, figsize=(15, 11)); units = {"trans": "mm", "rot": "deg", "scale": "log-scale"}
    for r, kind in enumerate(("trans", "rot", "scale")):
        for c in range(3):
            key = f"{kind}_{AXN[c]}"; axi = ax[r, c]
            for nm, ls in (("final", "-"), ("prefine", "--")):
                if nm not in landscape: continue
                L = landscape[nm][key]; axi.plot(L["offsets"], L["values"], ls, marker="o" if nm == "final" else "s", ms=3.5, lw=1.4, label=f"{nm}: argmax {(L['argmax_parabolic'] if L['argmax_parabolic'] is not None else L['argmax_offset']):+.2f} {'unimodal' if L['unimodal'] else 'NOT unimodal'}")
            axi.axvline(0, color="k", lw=0.6, alpha=0.6); axi.grid(alpha=0.3); axi.legend(fontsize=8); axi.set_ylabel("MI32 (higher = better)", fontsize=8)
            axi.set_title({"trans": f"translate along OCT {AXN[c]} [{units[kind]}]", "rot": f"rotate about OCT {AXN[c]} through c_o [{units[kind]}]", "scale": f"log-scale along OCT {AXN[c]}"}[kind], fontsize=10)
    fig.suptitle(f"{a.work.name} / {a.run.name}: MI32 landscapes at T (0.15 mm; specimen mask = eroded MRI tissue via T & OCT mask & valid); solid = final, dashed = prefine", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(out / "qc_fine_landscape.png", dpi=100); plt.close(fig); say("wrote", out / "qc_fine_landscape.png")
Q["seconds"] = time.time() - t_start; write_json(Q, out / "qc_fine.json"); say("done")
