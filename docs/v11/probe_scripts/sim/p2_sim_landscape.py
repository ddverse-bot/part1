#!/usr/bin/env python
"""Probe P2: fine-stage similarity landscape around the current pose T (CPU only, ~0.3 mm).

OCT is the fixed image (oct150 pooled x2 -> 0.30 mm); the MRI (pooled x4 -> 0.32 mm) is resampled into the OCT grid
through T' = T @ P, P a translation / rotation in OCT world about the OCT block centre.  Everything is evaluated inside
a specimen mask = (eroded MRI tissue mapped through T) & oct150_mask, optionally intersected with the MOVING MRI tissue
mask at T' ("overlap" mask) so that leaving-the-specimen does not trivially punish every similarity.
"""
import os, sys, time, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage

BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration")
sys.path.insert(0, f"{BASE}/octreg")
from octreg.common import to_t, apply_affine, resample_to_grid, avg_pool_iso  # noqa: E402

torch.set_num_threads(4)
W = f"{BASE}/work/xiangrui_I58bs"; RUN = f"{BASE}/work/runs/xiangrui_I58bs_novasc"; OUT = f"{BASE}/work/probe_xr/sim"
os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/landscape_log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

ERODE_MM = 1.3          # erosion of the MRI tissue mask before mapping it into OCT space
SIG_BIAS_MM = (1.3, 3.0)  # Gaussian sigma of the bias-flattening window (1.3 mm sigma ~ 3 mm FWHM), and a wide variant
SIG_GRAD_VOX = 1.0      # 0.3 mm Gaussian before gradients
TRANS = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
ROT = [-10, -5, -2, -1, 0, 1, 2, 5, 10]

# ----------------------------------------------------------------------------------------------- load & pool
t0 = time.time()
A_oct = np.load(f"{W}/oct150_affine.npy"); A_mri = np.load(f"{W}/mri_affine.npy"); T = np.load(f"{RUN}/T_oct2mri.npy")
oct150 = np.load(f"{W}/oct150.npy"); oct_shape_full = oct150.shape
oct_p, A_op = avg_pool_iso(torch.from_numpy(oct150.astype(np.float32))[None], A_oct, 2)
octm_p, _ = avg_pool_iso(torch.from_numpy(np.load(f"{W}/oct150_mask.npy").astype(np.float32))[None], A_oct, 2)
del oct150
mri_p, A_mp = avg_pool_iso(torch.from_numpy(np.asarray(np.load(f"{W}/mri.npy", mmap_mode="r")).astype(np.float32))[None], A_mri, 4)
mrit_p, _ = avg_pool_iso(torch.from_numpy(np.load(f"{W}/mri_tissue.npy").astype(np.float32))[None], A_mri, 4)
oct_p, octm_p, mri_p, mrit_p = oct_p[0], octm_p[0], mri_p[0], mrit_p[0]
vox_o = float(np.linalg.norm(A_op[:3, 0])); vox_m = float(np.linalg.norm(A_mp[:3, 0]))
shp = tuple(oct_p.shape)
P(f"pooled: oct {shp} @ {vox_o:.3f} mm, mri {tuple(mri_p.shape)} @ {vox_m:.3f} mm  ({time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------------------------------- filters
def gauss1d(sigma):
    r = int(math.ceil(3 * sigma)); x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2); return k / k.sum()

def gauss3d(x, sigma):
    """x [D,H,W] -> separable Gaussian, zero padding."""
    k = gauss1d(sigma); r = (k.numel() - 1) // 2; y = x[None, None]
    for ax in range(3):
        s = [1, 1, 1, 1, 1]; s[2 + ax] = k.numel(); p = [0, 0, 0]; p[ax] = r
        y = F.conv3d(y, k.view(s), padding=tuple(p))
    return y[0, 0]

def masked_local_mean(x, m, sigma):
    return gauss3d(x * m, sigma) / (gauss3d(m, sigma) + 1e-6)

def flatten(x, m, sigma_vox):
    lm = masked_local_mean(x, m, sigma_vox); eps = 1e-3 * float(x[m > 0.5].mean())
    return x / (lm + eps)

def grad3(x):
    xp = F.pad(x[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    gz = (xp[2:, 1:-1, 1:-1] - xp[:-2, 1:-1, 1:-1]) / 2; gy = (xp[1:-1, 2:, 1:-1] - xp[1:-1, :-2, 1:-1]) / 2
    gx = (xp[1:-1, 1:-1, 2:] - xp[1:-1, 1:-1, :-2]) / 2
    return torch.stack([gz, gy, gx])

def ncc(a, b, m):
    a = a[m]; b = b[m]; a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12))

def lcc(a, b, m, w, eps=1e-2, signed=True):
    """a, b globally z-scored; local (box w) correlation with masked local statistics, averaged over mask voxels whose
    window is at least half inside the mask."""
    mf = m.float(); box = lambda x: F.avg_pool3d(x[None, None], w, stride=1, padding=w // 2, count_include_pad=True)[0, 0]
    n = box(mf) + 1e-6; am = a * mf; bm = b * mf
    mu_a = box(am) / n; mu_b = box(bm) / n
    va = (box(am * a) / n - mu_a ** 2).clamp(min=0); vb = (box(bm * b) / n - mu_b ** 2).clamp(min=0)
    cov = box(am * b) / n - mu_a * mu_b
    l = cov / torch.sqrt(va * vb + eps) if signed else cov ** 2 / (va * vb + eps)
    valid = m & (n > 0.5)
    return float(l[valid].mean())

def ngf(ga, gb, m, eta=1.0, comps=(0, 1, 2)):
    ga = ga[list(comps)]; gb = gb[list(comps)]; na = ga.norm(dim=0); nb = gb.norm(dim=0)
    ea = eta * na[m].mean(); eb = eta * nb[m].mean()
    ua = ga / torch.sqrt(na ** 2 + ea ** 2); ub = gb / torch.sqrt(nb ** 2 + eb ** 2)
    return float((((ua * ub).sum(0)) ** 2)[m].mean())

def _mind_filters():
    six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]], dtype=torch.long)
    d2 = torch.cdist(six.float()[None], six.float()[None])[0] ** 2
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    sel = (x > y).reshape(-1) & (d2.reshape(-1).round() == 2)
    i1 = six.unsqueeze(1).repeat(1, 6, 1).reshape(-1, 3)[sel]; i2 = six.unsqueeze(0).repeat(6, 1, 1).reshape(-1, 3)[sel]
    m1 = torch.zeros(12, 1, 3, 3, 3); m2 = torch.zeros(12, 1, 3, 3, 3); ar = torch.arange(12) * 27
    m1.view(-1)[ar + i1[:, 0] * 9 + i1[:, 1] * 3 + i1[:, 2]] = 1; m2.view(-1)[ar + i2[:, 0] * 9 + i2[:, 1] * 3 + i2[:, 2]] = 1
    return m1, m2
MIND_F = _mind_filters()

def mind_ssc(img, radius=1, dilation=1):
    m1, m2 = MIND_F; xp = F.pad(img[None, None], (dilation,) * 6, mode="replicate")
    d = (F.conv3d(xp, m1, dilation=dilation) - F.conv3d(xp, m2, dilation=dilation)) ** 2
    ssd = F.avg_pool3d(F.pad(d, (radius,) * 6, mode="replicate"), 2 * radius + 1, stride=1)
    mind = ssd - ssd.min(1, keepdim=True)[0]; var = mind.mean(1, keepdim=True)
    var = var.clamp(float(var.mean()) * 0.001, float(var.mean()) * 1000)
    return torch.exp(-mind / var)[0]                                                    # [12,D,H,W]

def mind_dist(ma, mb, m):
    return float(((ma - mb) ** 2).mean(0)[m].mean())

def mi(a, b, m, ra, rb, bins=32):
    a = a[m]; b = b[m]
    ia = ((a - ra[0]) / (ra[1] - ra[0]) * bins).long().clamp(0, bins - 1); ib = ((b - rb[0]) / (rb[1] - rb[0]) * bins).long().clamp(0, bins - 1)
    h = torch.bincount(ia * bins + ib, minlength=bins * bins).float().view(bins, bins); p = h / h.sum()
    pa = p.sum(1); pb = p.sum(0); nz = p > 0
    mi_ = float((p[nz] * torch.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    ha = float(-(pa[pa > 0] * torch.log(pa[pa > 0])).sum()); hb = float(-(pb[pb > 0] * torch.log(pb[pb > 0])).sum())
    hab = float(-(p[nz] * torch.log(p[nz])).sum())
    return mi_, (ha + hb) / hab

# ----------------------------------------------------------------------------------------------- MRI-space prep
mt = (mrit_p > 0.5).numpy(); r = int(round(ERODE_MM / vox_m)); zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
ball = (zz ** 2 + yy ** 2 + xx ** 2) <= r ** 2
mt_er = ndimage.binary_erosion(mt, structure=ball)
P(f"MRI tissue {mt.sum()*vox_m**3/1000:.2f} cm3 -> eroded by {r} vox ({r*vox_m:.2f} mm): {mt_er.sum()*vox_m**3/1000:.2f} cm3")
mtf = torch.from_numpy(mt.astype(np.float32))
mri_flat = {s: flatten(mri_p, mtf, s / vox_m) for s in SIG_BIAS_MM}
mov = torch.stack([mri_p, mrit_p, torch.from_numpy(mt_er.astype(np.float32)), mri_flat[SIG_BIAS_MM[0]], mri_flat[SIG_BIAS_MM[1]]])
A_mp_t, A_op_t = to_t(A_mp, device="cpu"), to_t(A_op, device="cpu")

def resample(Tp):
    return resample_to_grid(mov, A_mp_t, A_op_t, shp, T_grid_to_vol=to_t(Tp, device="cpu"))

# ----------------------------------------------------------------------------------------------- specimen mask at T
res0 = resample(T)
spec = (res0[2] > 0.5) & (octm_p > 0.5)
spec_np = spec.numpy(); lab, n = ndimage.label(spec_np)
if n > 1:
    sizes = ndimage.sum(spec_np, lab, index=np.arange(1, n + 1)); keep = np.argsort(sizes)[::-1]
    P("spec components (cm3):", [round(float(s) * vox_o ** 3 / 1000, 3) for s in sizes[keep][:6]])
P(f"specimen mask: {spec.sum().item()*vox_o**3/1000:.2f} cm3 ({spec.float().mean().item()*100:.1f}% of block); old oct150_mask {(octm_p>0.5).sum().item()*vox_o**3/1000:.2f} cm3")
specf = spec.float()
oct_flat = {s: flatten(oct_p, specf, s / vox_o) for s in SIG_BIAS_MM}
# fixed (pose independent) OCT-side quantities
def zs(x, m): v = x[m]; return (x - v.mean()) / (v.std() + 1e-6)
oct_z = zs(oct_p, spec)
oct_g = grad3(gauss3d(oct_flat[SIG_BIAS_MM[0]], SIG_GRAD_VOX))
oct_mind = mind_ssc(oct_p)
ra = tuple(np.percentile(oct_p[spec].numpy(), [0.5, 99.5]).tolist())
rb = tuple(np.percentile(res0[0][spec].numpy(), [0.5, 99.5]).tolist())
raf = tuple(np.percentile(oct_flat[SIG_BIAS_MM[0]][spec].numpy(), [0.5, 99.5]).tolist())
rbf = tuple(np.percentile(res0[3][spec].numpy(), [0.5, 99.5]).tolist())
P(f"hist ranges oct {ra}, mri {rb}")

# ----------------------------------------------------------------------------------------------- similarity bundle
TIMES = {}
def timed(name, fn, *a):
    t = time.perf_counter(); v = fn(*a); TIMES.setdefault(name, []).append(time.perf_counter() - t); return v

def evaluate(Tp):
    t = time.perf_counter(); res = resample(Tp); TIMES.setdefault("resample", []).append(time.perf_counter() - t)
    mri_r, mrit_r, mfl1, mfl2 = res[0], res[1], res[3], res[4]
    masks = {"overlap": spec & (mrit_r > 0.5), "fixed": spec}
    t = time.perf_counter(); mri_g = grad3(gauss3d(mfl1, SIG_GRAD_VOX)); TIMES.setdefault("prep_grad", []).append(time.perf_counter() - t)
    t = time.perf_counter(); mri_mind = mind_ssc(mri_r); TIMES.setdefault("prep_mind", []).append(time.perf_counter() - t)
    out = {}
    for mk, m in masks.items():
        mri_z = zs(mri_r, m); o = {}
        o["S0_ncc_raw"] = timed("S0_ncc_raw", ncc, oct_p, mri_r, m)
        o["S1_ncc_flat1.3"] = timed("S1_ncc_flat", ncc, oct_flat[1.3], mfl1, m)
        o["S1_ncc_flat3.0"] = timed("S1_ncc_flat", ncc, oct_flat[3.0], mfl2, m)
        o["S2_lcc5"] = timed("S2_lcc5", lcc, oct_z, mri_z, m, 5)
        o["S2_lcc9"] = timed("S2_lcc9", lcc, oct_z, mri_z, m, 9)
        o["S2_lcc9_sq"] = timed("S2_lcc9", lcc, oct_z, mri_z, m, 9, 1e-2, False)
        o["S3_ngf"] = timed("S3_ngf", ngf, oct_g, mri_g, m, 1.0)
        o["S3_ngf_eta0.3"] = timed("S3_ngf", ngf, oct_g, mri_g, m, 0.3)
        o["S3_ngf_inplane"] = timed("S3_ngf", ngf, oct_g, mri_g, m, 1.0, (1, 2))
        o["S4_negmind"] = -timed("S4_mind_dist", mind_dist, oct_mind, mri_mind, m)
        o["S5_gmag"] = timed("S5_gmag", ncc, oct_g.norm(dim=0), mri_g.norm(dim=0), m)
        o["S5_gmag_inplane"] = timed("S5_gmag", ncc, oct_g[1:].norm(dim=0), mri_g[1:].norm(dim=0), m)
        o["S5_gmag_sqrt"] = timed("S5_gmag", ncc, oct_g.norm(dim=0).sqrt(), mri_g.norm(dim=0).sqrt(), m)
        v, nmi_ = timed("S6_mi", mi, oct_p, mri_r, m, ra, rb); o["S6_mi"] = v; o["S6_nmi"] = nmi_
        v, nmi_ = timed("S6_mi", mi, oct_flat[1.3], mfl1, m, raf, rbf); o["S6_mi_flat"] = v
        o["mask_cm3"] = float(m.sum()) * vox_o ** 3 / 1000
        out[mk] = o
    return out, res

# ----------------------------------------------------------------------------------------------- perturbations
c_o = (A_oct @ np.r_[(np.array(oct_shape_full) - 1) / 2.0, 1.0])[:3]
axes_u = [A_oct[:3, k] / np.linalg.norm(A_oct[:3, k]) for k in range(3)]
AX_NAMES = ["depth(z)", "y", "x"]
def rodrigues(u, deg):
    th = math.radians(deg); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
def perturbed(kind, ax, amt):
    Pm = np.eye(4)
    if kind == "trans": Pm[:3, 3] = amt * axes_u[ax]
    else: R = rodrigues(axes_u[ax], amt); Pm[:3, :3] = R; Pm[:3, 3] = c_o - R @ c_o
    return T @ Pm
P("OCT axis world directions:", [np.round(u, 3).tolist() for u in axes_u], "block centre", np.round(c_o, 2).tolist())

results = {}   # results[kind][ax][amt] = {mask: {sim: v}}
t_all = time.time()
val0, _ = evaluate(T)
for kind, amts in (("trans", TRANS), ("rot", ROT)):
    results[kind] = {}
    for ax in range(3):
        results[kind][ax] = {}
        for amt in amts:
            if amt == 0: results[kind][ax][amt] = val0; continue
            v, _ = evaluate(perturbed(kind, ax, amt)); results[kind][ax][amt] = v
        P(f"{kind} axis {AX_NAMES[ax]} done ({time.time()-t_all:.0f}s)  S1 overlap: " +
          " ".join(f"{a}:{results[kind][ax][a]['overlap']['S1_ncc_flat1.3']:.3f}" for a in amts))
P("eval time per pose (s): " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in TIMES.items()))

# ----------------------------------------------------------------------------------------------- analysis
SIMS = [k for k in val0["overlap"] if k != "mask_cm3"]
summary = {}
for mk in ("overlap", "fixed"):
    summary[mk] = {}
    for s in SIMS:
        v0 = val0[mk][s]; prof = {}; best = (v0, "T", 0.0); shoulder = []; local_coarse = 0; local_fine = 0; peaks = {}
        for kind, amts in (("trans", TRANS), ("rot", ROT)):
            for ax in range(3):
                vals = [results[kind][ax][a][mk][s] for a in amts]; key = f"{kind}_{AX_NAMES[ax]}"; prof[key] = vals
                for a, v in zip(amts, vals):
                    if v > best[0]: best = (v, key, a)
                    if abs(a) == (4 if kind == "trans" else 10): shoulder.append(v)
                d = dict(zip(amts, vals)); c1 = (1, 2) if kind == "trans" else (2, 5); c0 = (0.5, 1) if kind == "trans" else (1, 2)
                local_coarse += int(v0 >= d[c1[0]] and v0 >= d[-c1[0]]); local_fine += int(v0 >= d[c0[0]] and v0 >= d[-c0[0]])
                i = int(np.argmax(vals)); pk = None
                if 0 < i < len(amts) - 1:
                    a3 = np.array(amts[i - 1:i + 2], float); v3 = np.array(vals[i - 1:i + 2]); c = np.polyfit(a3, v3, 2)
                    if c[0] < 0: pk = float(np.clip(-c[1] / (2 * c[0]), a3[0], a3[2]))
                peaks[key] = {"argmax": amts[i], "parabolic": pk}
        contrast = (best[0] - float(np.mean(shoulder))) / (abs(best[0]) + 1e-12)
        summary[mk][s] = {"value_at_T": v0, "best_value": best[0], "best_axis": best[1], "best_offset": best[2],
                          "shoulder_mean": float(np.mean(shoulder)), "contrast": contrast,
                          "n_axes_local_max_1mm_2deg": local_coarse, "n_axes_local_max_0.5mm_1deg": local_fine,
                          "peak_per_axis": peaks, "profiles": prof}
json.dump({"T": T.tolist(), "trans_mm": TRANS, "rot_deg": ROT, "axis_names": AX_NAMES, "axis_world_dirs": [u.tolist() for u in axes_u],
           "block_centre_world": c_o.tolist(), "erode_mm": ERODE_MM, "sigma_bias_mm": SIG_BIAS_MM, "vox_oct_mm": vox_o, "vox_mri_mm": vox_m,
           "mask_cm3_at_T": val0["overlap"]["mask_cm3"], "times_s": {k: float(np.mean(v)) for k, v in TIMES.items()},
           "summary": summary}, open(f"{OUT}/landscape.json", "w"), indent=1)

P("\n=== SUMMARY (mask=overlap) ===")
P(f"{'sim':18s} {'at T':>9s} {'best':>9s} {'where':>22s} {'contrast':>9s} {'locmax6(1mm/2deg)':>18s} {'locmax6(.5mm/1deg)':>18s}")
for mk in ("overlap", "fixed"):
    P(f"--- mask {mk}")
    for s in SIMS:
        r_ = summary[mk][s]
        P(f"{s:18s} {r_['value_at_T']:9.4f} {r_['best_value']:9.4f} {r_['best_axis']+' '+str(r_['best_offset']):>22s} {r_['contrast']:9.3f} {r_['n_axes_local_max_1mm_2deg']:18d} {r_['n_axes_local_max_0.5mm_1deg']:18d}")
P("\n=== per-axis parabolic peak offsets (mask=overlap; mm for trans, deg for rot) ===")
keys = [f"{k}_{n}" for k in ("trans", "rot") for n in AX_NAMES]
P(f"{'sim':18s} " + " ".join(f"{k:>14s}" for k in keys))
for s in SIMS:
    P(f"{s:18s} " + " ".join(f"{(summary['overlap'][s]['peak_per_axis'][k]['parabolic'] if summary['overlap'][s]['peak_per_axis'][k]['parabolic'] is not None else float('nan')):14.2f}" for k in keys))

# ----------------------------------------------------------------------------------------------- plots
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [("S1 NCC flat", ["S1_ncc_flat1.3", "S1_ncc_flat3.0", "S0_ncc_raw"]), ("S2 LCC", ["S2_lcc5", "S2_lcc9", "S2_lcc9_sq"]),
        ("S3 NGF", ["S3_ngf", "S3_ngf_eta0.3", "S3_ngf_inplane"]), ("S4 -MIND MSD", ["S4_negmind"]),
        ("S5 NCC |grad|", ["S5_gmag", "S5_gmag_inplane", "S5_gmag_sqrt"]), ("S6 MI", ["S6_mi", "S6_mi_flat"])]
cols = [("trans", ax, f"translate {AX_NAMES[ax]} [mm]") for ax in range(3)] + [("rot", ax, f"rotate about {AX_NAMES[ax]} [deg]") for ax in range(3)]
fig, axs = plt.subplots(len(rows), 6, figsize=(20, 15))
colors = ["C0", "C1", "C2"]
for r_, (rname, sims) in enumerate(rows):
    for c_, (kind, ax, title) in enumerate(cols):
        a = axs[r_, c_]; amts = TRANS if kind == "trans" else ROT
        for j, s in enumerate(sims):
            for mk, ls in (("overlap", "-"), ("fixed", ":")):
                vals = summary[mk][s]["profiles"][f"{kind}_{AX_NAMES[ax]}"]
                a.plot(amts, vals, ls, color=colors[j], marker="o" if mk == "overlap" else None, ms=3, lw=1.3 if mk == "overlap" else 0.9,
                       label=f"{s} ({mk})" if c_ == 0 else None)
        a.axvline(0, color="k", lw=0.6, alpha=0.5); a.grid(alpha=0.3)
        if r_ == 0: a.set_title(title, fontsize=10)
        if c_ == 0: a.set_ylabel(rname, fontsize=10); a.legend(fontsize=6.5, loc="best")
fig.suptitle(f"P2 similarity landscape around T (0.3 mm, specimen mask {val0['overlap']['mask_cm3']:.1f} cm3; solid=overlap mask, dotted=fixed mask). "
             f"Higher is better for all rows.", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(f"{OUT}/landscape.png", dpi=90); plt.close()

# QC figure: masks and images at T
mri_r0 = res0[0].numpy(); mfl0 = res0[3].numpy(); of = oct_flat[1.3].numpy(); op = oct_p.numpy(); sp = spec.numpy(); mt_r = (res0[1] > 0.5).numpy()
om = (octm_p > 0.5).numpy(); D = shp[0]; sl = [int(D * 0.2), D // 2, int(D * 0.8)]
fig, axs = plt.subplots(3, 5, figsize=(22, 13))
vo = np.percentile(op[sp], 99); ml, mh = np.percentile(mri_r0[sp], [1, 99])
for i, z in enumerate(sl):
    axs[i, 0].imshow(np.clip(op[z] / vo, 0, 1), cmap="gray"); axs[i, 0].contour(sp[z], [0.5], colors="lime", linewidths=0.7); axs[i, 0].contour(om[z], [0.5], colors="w", linewidths=0.4)
    axs[i, 0].set_title(f"OCT 0.3mm z={z}: specimen mask (green), old mask (white)", fontsize=8)
    axs[i, 1].imshow(np.clip(of[z], 0.4, 1.6), cmap="gray"); axs[i, 1].set_title("OCT bias-flattened (sigma 1.3 mm)", fontsize=8)
    axs[i, 2].imshow(np.clip((mri_r0[z] - ml) / (mh - ml), 0, 1), cmap="gray"); axs[i, 2].contour(sp[z], [0.5], colors="lime", linewidths=0.7); axs[i, 2].contour(mt_r[z], [0.5], colors="y", linewidths=0.5)
    axs[i, 2].set_title("MRI via T (mask green, MRI tissue yellow)", fontsize=8)
    axs[i, 3].imshow(np.clip(mfl0[z], 0.4, 1.6), cmap="gray"); axs[i, 3].set_title("MRI flattened via T", fontsize=8)
    ov = np.zeros(sp[z].shape + (3,)); ov[..., 1] = np.clip(of[z] - 0.4, 0, 1.2) / 1.2 * sp[z]; ov[..., 0] = np.clip(mfl0[z] - 0.4, 0, 1.2) / 1.2 * sp[z]; ov[..., 2] = ov[..., 0]
    axs[i, 4].imshow(ov); axs[i, 4].set_title("overlay: OCT green / MRI magenta (inside mask)", fontsize=8)
for a in axs.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/qc_mask_T.png", dpi=80); plt.close()

np.save(f"{OUT}/oct03.npy", op); np.save(f"{OUT}/oct03_affine.npy", A_op); np.save(f"{OUT}/spec_mask03.npy", sp)
np.save(f"{OUT}/mri_in_oct03_at_T.npy", mri_r0); np.save(f"{OUT}/mri_tissue_in_oct03_at_T.npy", mt_r)
P(f"done in {time.time()-t0:.0f}s; wrote {OUT}/landscape.png landscape.json qc_mask_T.png")
