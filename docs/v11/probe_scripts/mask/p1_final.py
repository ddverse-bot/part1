"""Final candidate: texture markers (0.04-mm band-passed directional std, min over axes / local mean, pooled to 0.16 mm)
+ watershed on the 0.04-mm rim strength (max-pooled to 0.16 mm) -> mask at 0.16 -> resampled to the 0.15 and 0.04 grids."""
import sys, json; sys.path.insert(0, "/tmp")
from p1_common import *
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
POOL = 4; VOX16 = VOX04 * POOL
cfg_final = (int(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (4, 1.0)
do_fine = "--fine" in sys.argv
o = np.load(f"{W}/oct150.npy"); old = np.load(f"{W}/oct150_mask.npy"); zero = o <= 0
mri_oct = np.load(f"{OUT}/mri_tissue_in_oct150_T.npy")
A150 = np.load(f"{W}/oct150_affine.npy"); A16 = np.load(f"{OUT}/fields16_affine.npy"); Am = np.load(f"{W}/mri_affine.npy"); T = np.load("work/runs/xiangrui_I58bs_novasc/T_oct2mri.npy")
ii, jj, kk = np.meshgrid(*[np.arange(s) for s in o.shape], indexing="ij")
P = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], 0).astype(np.float64); q150 = (np.linalg.inv(A16) @ A150 @ P)[:3]
qm = (np.linalg.inv(Am) @ T @ A150 @ P)[:3]; mri_shape = np.load(f"{W}/mri.npy", mmap_mode="r").shape
covered = ((qm >= 0) & (qm < np.array(mri_shape)[:, None] - 1)).all(0).reshape(o.shape); del qm
def gmm2_1d(v, iters=100):
    v = np.asarray(v, np.float64); v = v[::max(1, v.size // 500_000)]
    mu = np.percentile(v, [25, 75]); sd = np.array([v.std() / 2] * 2); w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = np.stack([w[k] / sd[k] * np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) for k in range(2)], 1) + 1e-300
        r = ll / ll.sum(1, keepdims=True); n = r.sum(0); w = n / n.sum(); mu = (r * v[:, None]).sum(0) / n; sd = np.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / n) + 1e-6
    lo, hi = np.argsort(mu); grid = np.linspace(mu[lo], mu[hi], 400)
    post = [w[k] / sd[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) for k in range(2)]
    return float(grid[np.argmin(np.abs(post[lo] - post[hi]))]), {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}
# ---- fields at 0.16
fields = np.load(f"{OUT}/fields16_bp2_w9.npy"); den = np.maximum(fields[4], 1e-3); valid = fields[4] > 0.5; mv = valid.astype(np.float32)
def msmooth(f, sig): return gaussian_filter(f * mv, sig) / np.maximum(gaussian_filter(mv, sig), 1e-3)
Ss = [msmooth(fields[i] / den, 3.0) for i in range(3)]; MU = msmooth(fields[3] / den, 3.0)
F = np.minimum(np.minimum(Ss[0], Ss[1]), Ss[2]) / np.maximum(MU, 1.0); F[~valid] = 0
lthr, ginfo = gmm2_1d(np.log(F[valid & (F > 0)])); thr = float(np.exp(lthr)); say("mincv16 log-GMM thr", thr, ginfo)
rimf = np.load(f"{OUT}/rim16_from04_w5.npy"); r16, tmax = rimf[0], rimf[1]; elev = tmax.copy(); elev[~valid] = 0
rimbin = (r16 > 0.25) & valid
# ---- does the 0.04-derived rim seal the interior at 0.16?
passable = ~rimbin & valid; edt = distance_transform_edt(passable)
for r in (0, 1, 2, 3):
    core = edt > r; lab, n = label(core); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    faces = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]); fc = np.bincount(faces, minlength=sizes.size)
    top = np.argsort(sizes)[::-1][:4]; say(f"rim16 seal test r={r}: comps {n}; top (cm3/face-contact): " + "; ".join(f"{sizes[k]*VOX16**3/1000:.2f}/{fc[k]}" for k in top))
# ---- watershed configs
def run(r_open, scale, close_r=0):
    inner = ndimage.binary_opening((F > thr * scale) & valid, structure=ball(r_open)); outer = ndimage.binary_opening((F < thr) & valid, structure=ball(r_open)) | ~valid
    mk = np.zeros(F.shape, np.int32); mk[outer] = 1; mk[inner] = 2
    lab = watershed(elev, mk); m16 = (lab == 2) & valid
    # exterior pockets enclosed by the specimen (no path to a box face or a zero block through the exterior label) belong to the specimen
    ext = (lab == 1) & valid; le, ne = label(ext); touch = np.zeros(ne + 1, bool)
    for f in (le[0], le[-1], le[:, 0], le[:, -1], le[:, :, 0], le[:, :, -1]): touch[np.unique(f)] = True
    touch[np.unique(le[ndimage.binary_dilation(~valid, structure=ball(1))])] = True; touch[0] = False
    m16 = valid & ~np.isin(le, np.where(touch)[0])
    m16_all = binary_fill_holes(m16); labc, n = label(m16_all); sizes = np.bincount(labc.ravel()); sizes[0] = 0
    keep = np.isin(labc, np.where(sizes * VOX16**3 / 1000 >= 0.05)[0]); m16_lcc = cleanup(m16, close_r=close_r)
    return m16_lcc, keep, mk, float(inner.sum()) * VOX16**3 / 1000, float((outer & valid).sum()) * VOX16**3 / 1000, int((sizes * VOX16**3 / 1000 >= 0.05).sum())
def to150(m16):
    f = map_coordinates(m16.astype(np.float32), q150, order=1, mode="nearest").reshape(o.shape); return (f > 0.5) & ~zero
summary = []
for r_open in (() if '--nosweep' in sys.argv else (3, 4, 6)):
    for scale in (1.0, 1.5, 2.0):
        m16, m16k, mk, vin, vout, ncomp = run(r_open, scale)
        m150 = cleanup(to150(m16)); nm = f"g_ws16_open{r_open}_scale{scale}"
        rec = float((m150 & mri_oct).sum() / mri_oct.sum()); excess = float((m150 & ~mri_oct & covered).sum() / max(1, (m150 & covered).sum()))
        d = report(m150, nm, o, old, mri_oct, {"thr": thr, "r_open_16": r_open, "scale": scale, "cm3_16_lcc": float(m16.sum()) * VOX16**3 / 1000, "cm3_16_keep005": float(m16k.sum()) * VOX16**3 / 1000, "n_comp_ge_0.05cm3": ncomp,
                                                 "inner_marker_cm3": vin, "outer_marker_cm3": vout, "mri_recall": rec, "excess_outside_mri_in_covered": excess})
        np.save(f"{OUT}/{nm}_mask150.npy", m150); np.save(f"{OUT}/{nm}_mask16.npy", m16)
        summary.append((nm, d["cm3"], round(d["cm3_16_keep005"], 2), ncomp, round(vin, 1), round(vout, 1), d["keep_of_old"], round(rec, 3), round(excess, 3), d["dice_vs_mri_in_oct"]))
say("SUMMARY (name, cm3@0.15 lcc, cm3@0.16 keep>=0.05, ncomp, inner, outer, keep_old, mri_recall, excess, dice)"); [say(*s) for s in summary]
# old mask reference metrics
rec_old = float((old & mri_oct).sum() / mri_oct.sum()); exc_old = float((old & ~mri_oct & covered).sum() / (old & covered).sum()); say("OLD mask: recall", round(rec_old, 3), "excess", round(exc_old, 3))
if do_fine:
    r_open, scale = cfg_final; m16, m16k, mk, *_ = run(r_open, scale, close_r=3); nm = f"g_ws16_open{r_open}_scale{scale}_close3"
    m150 = cleanup(to150(m16)); np.save(f"{OUT}/final_mask16.npy", m16); np.save(f"{OUT}/final_mask150.npy", m150)
    rec = float((m150 & mri_oct).sum() / mri_oct.sum()); excess = float((m150 & ~mri_oct & covered).sum() / max(1, (m150 & covered).sum()))
    dfin = report(m150, "final_mask150", o, old, mri_oct, {"config": nm, "thr": thr, "cm3_16": float(m16.sum()) * VOX16**3 / 1000, "mri_recall": rec, "excess_outside_mri_in_covered": excess}); say("FINAL 0.15:", json.dumps(dfin))
    v = np.load(f"{W}/octv.npy", mmap_mode="r"); Av = np.load(f"{W}/octv_affine.npy"); Z, Y, X = v.shape
    out = np.lib.format.open_memmap(f"{OUT}/final_mask04.npy", mode="w+", dtype=bool, shape=v.shape)
    mf = m16.astype(np.float32); M = np.linalg.inv(A16) @ Av; tot = 0
    jj, kk = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
    for z0 in range(0, Z, 64):
        z1 = min(Z, z0 + 64); zz = np.repeat(np.arange(z0, z1)[:, None, None], Y, 1); zz = np.repeat(zz, X, 2)
        pts = np.stack([zz.ravel(), np.tile(jj.ravel(), z1 - z0), np.tile(kk.ravel(), z1 - z0), np.ones(zz.size)], 0)
        q = (M @ pts)[:3]; f = map_coordinates(mf, q, order=1, mode="nearest").reshape(z1 - z0, Y, X)
        blk = np.asarray(v[z0:z1]); m = (f > 0.5) & (blk > 0); out[z0:z1] = m; tot += int(m.sum())
    out.flush(); say("final mask04 voxels", tot, "cm3", tot * VOX04**3 / 1000)
    fig, ax = plt.subplots(3, 3, figsize=(18, 18))
    for r in range(3):
        for c, frac in enumerate((0.3, 0.5, 0.7)):
            i = int(v.shape[r] * frac); im = np.asarray(np.take(v, i, axis=r)).astype(np.float32); mm = np.take(out, i, axis=r)
            ax[r, c].imshow(im, cmap="gray", vmax=np.percentile(im[im > 0], 99.5)); ax[r, c].contour(mm.astype(float), levels=[0.5], colors="r", linewidths=0.6); ax[r, c].set_title(f"final mask @0.04, axis{r} idx{i}", fontsize=9); ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/final_mask04.png", dpi=60); plt.close(fig)
    json.dump({"config": nm, "thr_mincv16": thr, "gmm": ginfo, "mask04_voxels": tot, "mask04_cm3": tot * VOX04**3 / 1000, "mask04_fraction": tot / float(np.prod(v.shape)), "mask150": dfin}, open(f"{OUT}/final_info.json", "w"), indent=1)
say("done final")
