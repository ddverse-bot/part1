#!/usr/bin/env python
"""Probe P2b: follow-up on the similarity landscape.
 (1) regional polarity of the OCT/MRI intensity relation at T (sub-block NCC),
 (2) finer 1-D profiles for sign-handled candidates (-NCC raw / flat, LCC^2 with larger windows, MI, in-plane |grad| NCC),
 (3) 2-D translation landscapes (depth x y, depth x x) for the top candidates,
 (4) forward+backward wall time of differentiable torch versions of the candidates.
Reuses the pooled 0.3 mm arrays written by p2_sim_landscape.py (work/probe_xr/sim/*.npy) and recomputes the rest.
"""
import os, sys, time, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage
BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration"); sys.path.insert(0, f"{BASE}/octreg")
from octreg.common import to_t, apply_affine, resample_to_grid, avg_pool_iso, grid_points, sample_at_world  # noqa: E402
torch.set_num_threads(4)
W = f"{BASE}/work/xiangrui_I58bs"; RUN = f"{BASE}/work/runs/xiangrui_I58bs_novasc"; OUT = f"{BASE}/work/probe_xr/sim"
log = open(f"{OUT}/followup_log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

t0 = time.time()
A_oct = np.load(f"{W}/oct150_affine.npy"); A_mri = np.load(f"{W}/mri_affine.npy"); T = np.load(f"{RUN}/T_oct2mri.npy")
oct_shape_full = (194, 267, 212)
oct_p = torch.from_numpy(np.load(f"{OUT}/oct03.npy")); A_op = np.load(f"{OUT}/oct03_affine.npy"); spec = torch.from_numpy(np.load(f"{OUT}/spec_mask03.npy"))
mri_p, A_mp = avg_pool_iso(torch.from_numpy(np.asarray(np.load(f"{W}/mri.npy", mmap_mode="r")).astype(np.float32))[None], A_mri, 4)
mrit_p, _ = avg_pool_iso(torch.from_numpy(np.load(f"{W}/mri_tissue.npy").astype(np.float32))[None], A_mri, 4)
mri_p, mrit_p = mri_p[0], mrit_p[0]; vox_o = float(np.linalg.norm(A_op[:3, 0])); vox_m = float(np.linalg.norm(A_mp[:3, 0])); shp = tuple(oct_p.shape)

def gauss1d(sigma):
    r = int(math.ceil(3 * sigma)); x = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-0.5 * (x / sigma) ** 2); return k / k.sum()
def gauss3d(x, sigma):
    k = gauss1d(sigma); r = (k.numel() - 1) // 2; y = x[None, None]
    for ax in range(3):
        s = [1, 1, 1, 1, 1]; s[2 + ax] = k.numel(); p = [0, 0, 0]; p[ax] = r; y = F.conv3d(y, k.view(s), padding=tuple(p))
    return y[0, 0]
def flatten(x, m, sigma_vox):
    lm = gauss3d(x * m, sigma_vox) / (gauss3d(m, sigma_vox) + 1e-6); return x / (lm + 1e-3 * float(x[m > 0.5].mean()))
def box(x, w):
    """separable box mean with zero padding (== avg_pool3d w^3, count_include_pad) ; x [D,H,W]"""
    y = x[None, None]
    for ax in range(3):
        k = [1, 1, 1]; k[ax] = w; p = [0, 0, 0]; p[ax] = w // 2; y = F.avg_pool3d(y, tuple(k), stride=1, padding=tuple(p), count_include_pad=True)
    return y[0, 0]
def grad3(x):
    xp = F.pad(x[None, None], (1,) * 6, mode="replicate")[0, 0]
    return torch.stack([(xp[2:, 1:-1, 1:-1] - xp[:-2, 1:-1, 1:-1]) / 2, (xp[1:-1, 2:, 1:-1] - xp[1:-1, :-2, 1:-1]) / 2, (xp[1:-1, 1:-1, 2:] - xp[1:-1, 1:-1, :-2]) / 2])
def ncc_t(a, b, m):
    a = a[m]; b = b[m]; a = a - a.mean(); b = b - b.mean(); return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12)
def lcc_t(a, b, m, w, eps=1e-2, sq=True):
    mf = m.float(); n = box(mf, w) + 1e-6; am = a * mf; bm = b * mf
    mu_a = box(am, w) / n; mu_b = box(bm, w) / n
    va = (box(am * a, w) / n - mu_a ** 2).clamp(min=0); vb = (box(bm * b, w) / n - mu_b ** 2).clamp(min=0); cov = box(am * b, w) / n - mu_a * mu_b
    l = cov ** 2 / (va * vb + eps) if sq else cov / torch.sqrt(va * vb + eps)
    valid = m & (n > 0.5); return l[valid].mean()
def mi_np(a, b, m, ra, rb, bins=32):
    a = a[m]; b = b[m]
    ia = ((a - ra[0]) / (ra[1] - ra[0]) * bins).long().clamp(0, bins - 1); ib = ((b - rb[0]) / (rb[1] - rb[0]) * bins).long().clamp(0, bins - 1)
    h = torch.bincount(ia * bins + ib, minlength=bins * bins).float().view(bins, bins); p = h / h.sum(); pa = p.sum(1); pb = p.sum(0); nz = p > 0
    return float((p[nz] * torch.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
def zs(x, m): v = x[m]; return (x - v.mean()) / (v.std() + 1e-6)

# ---- MRI-side prep (pose independent): flattened variants at sigma 3 and 6 mm
mtf = (mrit_p > 0.5).float()
SIGS = (3.0, 6.0)
mri_flat = {s: flatten(mri_p, mtf, s / vox_m) for s in SIGS}
mov = torch.stack([mri_p, mrit_p] + [mri_flat[s] for s in SIGS])
A_mp_t, A_op_t = to_t(A_mp, device="cpu"), to_t(A_op, device="cpu")
def resample(Tp): return resample_to_grid(mov, A_mp_t, A_op_t, shp, T_grid_to_vol=to_t(Tp, device="cpu"))
specf = spec.float()
oct_flat = {s: flatten(oct_p, specf, s / vox_o) for s in SIGS}
oct_flat13 = flatten(oct_p, specf, 1.3 / vox_o)
oct_g = grad3(gauss3d(oct_flat13, 1.0)); oct_gip = oct_g[1:].norm(dim=0)
oct_z = zs(oct_p, spec)
res0 = resample(T); m0 = spec & (res0[1] > 0.5)
ra = tuple(np.percentile(oct_p[m0].numpy(), [0.5, 99.5]).tolist()); rb = tuple(np.percentile(res0[0][m0].numpy(), [0.5, 99.5]).tolist())
P(f"prep {time.time()-t0:.1f}s; mask {m0.sum().item()*vox_o**3/1000:.2f} cm3")

# ---- (1) regional polarity at T: NCC in sub-blocks of the mask bounding box
idx = np.argwhere(m0.numpy()); lo = idx.min(0); hi = idx.max(0) + 1
P(f"mask bbox voxels {lo.tolist()}..{hi.tolist()} = {(hi-lo)*vox_o} mm")
for name, a, b in (("raw", oct_p, res0[0]), ("flat3", oct_flat[3.0], res0[2]), ("flat6", oct_flat[6.0], res0[3])):
    for nb in (2, 3, 4):
        edges = [np.linspace(lo[k], hi[k], nb + 1).astype(int) for k in range(3)]; vals = []
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    sub = torch.zeros_like(m0); sub[edges[0][i]:edges[0][i + 1], edges[1][j]:edges[1][j + 1], edges[2][k]:edges[2][k + 1]] = True
                    mm = m0 & sub
                    if mm.sum() > 2000: vals.append((float(ncc_t(a, b, mm)), int(mm.sum())))
        v = np.array([x[0] for x in vals]); n = np.array([x[1] for x in vals])
        P(f"regional NCC ({name}, {nb}^3 blocks, {len(v)} blocks with >2000 vox): neg {np.sum(v<0)}/{len(v)}  vol-weighted mean {np.sum(v*n)/n.sum():+.3f}  "
          f"min {v.min():+.3f} max {v.max():+.3f}  (block edge ~{((hi-lo)/nb*vox_o).round(1).tolist()} mm)")

# ---- (2) finer 1-D profiles
c_o = (A_oct @ np.r_[(np.array(oct_shape_full) - 1) / 2.0, 1.0])[:3]; axes_u = [A_oct[:3, k] / np.linalg.norm(A_oct[:3, k]) for k in range(3)]
AX = ["depth(z)", "y", "x"]
def rodrigues(u, deg):
    th = math.radians(deg); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]); return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
def pert(tvec_mm=(0, 0, 0), rot=None):
    Pm = np.eye(4); Pm[:3, 3] = sum(d * u for d, u in zip(tvec_mm, axes_u))
    if rot is not None:
        R = rodrigues(axes_u[rot[0]], rot[1]); Pm[:3, :3] = R; Pm[:3, 3] += c_o - R @ c_o
    return T @ Pm
TIMES = {}
def timed(name, fn, *a):
    t = time.perf_counter(); v = fn(*a); TIMES.setdefault(name, []).append(time.perf_counter() - t); return float(v)
def evaluate(Tp):
    t = time.perf_counter(); res = resample(Tp); TIMES.setdefault("resample", []).append(time.perf_counter() - t)
    mri_r, mrit_r = res[0], res[1]; m = spec & (mrit_r > 0.5); mri_z = zs(mri_r, m); o = {}
    o["negNCC_raw"] = -timed("ncc", ncc_t, oct_p, mri_r, m)
    o["negNCC_flat3"] = -timed("ncc", ncc_t, oct_flat[3.0], res[2], m)
    o["negNCC_flat6"] = -timed("ncc", ncc_t, oct_flat[6.0], res[3], m)
    for w in (9, 15, 21):
        o[f"LCC2_w{w}"] = timed(f"lcc_w{w}", lcc_t, oct_z, mri_z, m, w, 1e-2, True)
        o[f"negLCC_w{w}"] = -timed(f"lcc_w{w}", lcc_t, oct_z, mri_z, m, w, 1e-2, False)
    o["MI32"] = timed("mi", mi_np, oct_p, mri_r, m, ra, rb)
    t = time.perf_counter(); mg = grad3(gauss3d(flatten(mri_r, (mrit_r > 0.5).float(), 1.3 / vox_o), 1.0)); TIMES.setdefault("grad_prep", []).append(time.perf_counter() - t)
    o["gmag_inplane"] = timed("ncc", ncc_t, oct_gip, mg[1:].norm(dim=0), m)
    o["mask_cm3"] = float(m.sum()) * vox_o ** 3 / 1000
    return o
TR = [-4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4]; RO = [-10, -7, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 7, 10]
v0 = evaluate(T); SIMS = [k for k in v0 if k != "mask_cm3"]; prof = {}
t1 = time.time()
for ax in range(3):
    prof[f"trans_{AX[ax]}"] = {a: (v0 if a == 0 else evaluate(pert(tvec_mm=tuple(a if k == ax else 0 for k in range(3))))) for a in TR}
    P(f"trans {AX[ax]} done {time.time()-t1:.0f}s")
for ax in range(3):
    prof[f"rot_{AX[ax]}"] = {a: (v0 if a == 0 else evaluate(pert(rot=(ax, a)))) for a in RO}
    P(f"rot {AX[ax]} done {time.time()-t1:.0f}s")
P("time per pose (s): " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in TIMES.items()))

def parab(xs, ys):
    i = int(np.argmax(ys))
    if 0 < i < len(xs) - 1:
        c = np.polyfit(np.array(xs[i - 1:i + 2], float), np.array(ys[i - 1:i + 2]), 2)
        if c[0] < 0: return float(np.clip(-c[1] / (2 * c[0]), xs[i - 1], xs[i + 1]))
    return None
summary = {}
P("\n=== fine 1-D profiles (overlap mask; all 'higher is better') ===")
P(f"{'sim':14s} {'at T':>8s} {'best':>8s} {'where':>20s} {'contrast':>9s} {'locmax(0.5mm/1deg)':>18s} " + " ".join(f"{k[:10]:>10s}" for k in prof))
for s in SIMS:
    v_T = v0[s]; best = (v_T, "T", 0); sh = []; loc = 0; pk = {}; hw = {}
    for key, d in prof.items():
        xs = sorted(d); ys = [d[x][s] for x in xs]
        for x, y in zip(xs, ys):
            if y > best[0]: best = (y, key, x)
            if abs(x) == (4 if key.startswith("trans") else 10): sh.append(y)
        c = 0.5 if key.startswith("trans") else 1.0; loc += int(v_T >= d[c][s] and v_T >= d[-c][s]); pk[key] = parab(xs, ys)
        # half-width: offset where the profile drops halfway from peak to shoulder mean (linear interp), on each side
        thr = 0.5 * (max(ys) + np.mean([ys[0], ys[-1]])); side = []
        for sgn in (-1, 1):
            xx = [x for x in xs if sgn * x >= 0]; xx = xx if sgn > 0 else xx[::-1]; yy = [d[x][s] for x in xx]; hwv = None
            for q in range(1, len(xx)):
                if yy[q] < thr <= yy[q - 1]: hwv = abs(xx[q - 1] + (thr - yy[q - 1]) / (yy[q] - yy[q - 1]) * (xx[q] - xx[q - 1])); break
            side.append(hwv)
        hw[key] = side
    contrast = (best[0] - float(np.mean(sh))) / (abs(best[0]) + 1e-12)
    summary[s] = {"value_at_T": v_T, "best": best, "contrast": contrast, "n_axes_local_max_fine": loc, "parabolic_peak": pk, "half_width": hw,
                  "profiles": {k: {str(x): d[x][s] for x in sorted(d)} for k, d in prof.items()}}
    P(f"{s:14s} {v_T:8.4f} {best[0]:8.4f} {best[1]+' '+str(best[2]):>20s} {contrast:9.3f} {loc:18d} " + " ".join(f"{(pk[k] if pk[k] is not None else float('nan')):10.2f}" for k in prof))
P("half-widths (mm or deg, to half way between peak and +-4mm/+-10deg shoulders; -side/+side):")
for s in SIMS: P(f"{s:14s} " + " ".join(f"{k[:10]}:{'/'.join('%.1f' % v if v is not None else 'na' for v in summary[s]['half_width'][k])}" for k in prof))

# ---- (3) 2-D translation landscapes for the top candidates
TOP = ["negNCC_raw", "negNCC_flat3", "LCC2_w15", "MI32", "gmag_inplane"]
G = np.arange(-4, 4.01, 1.0); maps = {}
for pair, name in (((0, 1), "depth_x_y"), ((0, 2), "depth_x_x")):
    M = {s: np.zeros((len(G), len(G))) for s in TOP}
    for i, a in enumerate(G):
        for j, b in enumerate(G):
            tv = [0, 0, 0]; tv[pair[0]] = a; tv[pair[1]] = b; o = evaluate(pert(tvec_mm=tuple(tv)))
            for s in TOP: M[s][i, j] = o[s]
    maps[name] = {s: M[s].tolist() for s in TOP}; P(f"2D map {name} done {time.time()-t1:.0f}s")
    for s in TOP:
        i, j = np.unravel_index(np.argmax(M[s]), M[s].shape); P(f"  {s:14s} 2D argmax at ({AX[pair[0]]} {G[i]:+.0f} mm, {AX[pair[1]]} {G[j]:+.0f} mm) value {M[s][i,j]:.4f} (at T {M[s][4,4]:.4f})")

# ---- (4) differentiable torch cost: forward+backward wall time at 0.3 mm
from octreg.refine import params_from_matrix, compose
c_t = to_t(c_o, device="cpu"); r0, t0_, ls0, sh0, mirror = params_from_matrix(T, c_o)
r = to_t(r0, device="cpu").requires_grad_(True); tt = to_t(t0_, device="cpu").requires_grad_(True); ls = to_t(ls0, device="cpu"); shv = to_t(sh0, device="cpu")
pts = grid_points(A_op_t, shp).reshape(-1, 3); mov_flat = mri_flat[3.0][None]; mov_raw = mri_p[None]; mov_t = mrit_p[None]
def loss_ncc(Tt):
    pm = apply_affine(Tt, pts); fm = sample_at_world(mov_flat, A_mp_t, pm)[0].reshape(shp); mt_ = sample_at_world(mov_t, A_mp_t, pm)[0].reshape(shp)
    m = spec & (mt_.detach() > 0.5); return ncc_t(oct_flat[3.0], fm, m)           # maximise -NCC -> minimise NCC
def loss_lcc2(Tt, w=15):
    pm = apply_affine(Tt, pts); fm = sample_at_world(mov_raw, A_mp_t, pm)[0].reshape(shp); mt_ = sample_at_world(mov_t, A_mp_t, pm)[0].reshape(shp)
    m = spec & (mt_.detach() > 0.5); return -lcc_t(oct_z, zs(fm, m), m, w, 1e-2, True)
def loss_gip(Tt):
    pm = apply_affine(Tt, pts); fm = sample_at_world(mov_flat, A_mp_t, pm)[0].reshape(shp); mt_ = sample_at_world(mov_t, A_mp_t, pm)[0].reshape(shp)
    m = spec & (mt_.detach() > 0.5); g = grad3(gauss3d(fm, 1.0)); return -ncc_t(oct_gip, g[1:].norm(dim=0), m)
for name, fn in (("NCC_flat3 (points->NCC)", loss_ncc), ("LCC2_w15 (grid)", loss_lcc2), ("gmag_inplane (grid)", loss_gip)):
    ts = []
    for rep in range(3):
        t = time.perf_counter(); Tt = compose(r, tt, ls, shv, c_t, mirror); L = fn(Tt); L.backward(); ts.append(time.perf_counter() - t); r.grad = None; tt.grad = None
    P(f"autograd fwd+bwd {name}: {np.mean(ts[1:]):.2f} s (loss {L.item():.4f}, grad wrt rotvec {r.grad if r.grad is not None else ''})")

json.dump({"summary": summary, "maps2d": maps, "grid2d_mm": G.tolist(), "times_s": {k: float(np.mean(v)) for k, v in TIMES.items()}}, open(f"{OUT}/followup.json", "w"), indent=1)

# ---- plots
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(SIMS), 6, figsize=(20, 2.1 * len(SIMS)))
for i, s in enumerate(SIMS):
    for j, key in enumerate(prof):
        xs = sorted(prof[key]); a = axs[i, j]; a.plot(xs, [prof[key][x][s] for x in xs], "-o", ms=2.5, lw=1.2); a.axvline(0, color="k", lw=0.5, alpha=0.5); a.grid(alpha=0.3)
        if i == 0: a.set_title(key, fontsize=9)
        if j == 0: a.set_ylabel(s, fontsize=8)
fig.suptitle("P2b fine 1-D profiles around T (0.3 mm; overlap mask; higher = better; negNCC = polarity-flipped NCC)", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(f"{OUT}/landscape_fine.png", dpi=90); plt.close()
fig, axs = plt.subplots(2, len(TOP), figsize=(4 * len(TOP), 7.5))
for i, (name, pair) in enumerate((("depth_x_y", (0, 1)), ("depth_x_x", (0, 2)))):
    for j, s in enumerate(TOP):
        M = np.array(maps[name][s]); a = axs[i, j]; im = a.imshow(M, origin="lower", extent=(G[0] - 0.5, G[-1] + 0.5, G[0] - 0.5, G[-1] + 0.5), cmap="viridis")
        a.plot(0, 0, "r+", ms=12, mew=2); ii, jj = np.unravel_index(np.argmax(M), M.shape); a.plot(G[jj], G[ii], "wx", ms=8, mew=2)
        a.set_xlabel(f"{AX[pair[1]]} [mm]"); a.set_ylabel(f"{AX[pair[0]]} [mm]"); a.set_title(f"{s}  (T=+, max=x)", fontsize=9); plt.colorbar(im, ax=a, fraction=0.046)
fig.suptitle("P2b 2-D translation landscapes (1 mm grid) around T", fontsize=11); plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(f"{OUT}/landscape_2d.png", dpi=90); plt.close()
P(f"done in {time.time()-t0:.0f}s")
