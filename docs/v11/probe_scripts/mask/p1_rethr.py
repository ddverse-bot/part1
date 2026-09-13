import sys; sys.path.insert(0, "/tmp")
from p1_common import *
from skimage.filters import threshold_multiotsu
from skimage.segmentation import watershed
o = np.load(f"{W}/oct150.npy"); old = np.load(f"{W}/oct150_mask.npy"); zero = o <= 0
mri_oct = np.load(f"{OUT}/mri_tissue_in_oct150_T.npy"); ag = np.load(f"{OUT}/ref_agarose150.npy"); ti = np.load(f"{OUT}/ref_tissue150.npy")
A150 = np.load(f"{W}/oct150_affine.npy"); A16 = np.load(f"{OUT}/fields16_affine.npy")
ii, jj, kk = np.meshgrid(*[np.arange(s) for s in o.shape], indexing="ij")
P = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], 0).astype(np.float64); q150 = (np.linalg.inv(A16) @ A150 @ P)[:3]
x = o.astype(np.float32); m = (~zero).astype(np.float32)
lstd3 = gaussian_filter(local_std(x, 3)[0] * m, 0.7) / np.maximum(gaussian_filter(m, 0.7), 1e-3)
def gmm2_1d(v, iters=100):
    """2-component 1-D Gaussian mixture by EM on v (log domain); returns the decision threshold between the means."""
    v = np.asarray(v, np.float64); v = v[::max(1, v.size // 500_000)]
    mu = np.percentile(v, [25, 75]); sd = np.array([v.std() / 2] * 2); w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = np.stack([w[k] / sd[k] * np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) for k in range(2)], 1) + 1e-300
        r = ll / ll.sum(1, keepdims=True); n = r.sum(0); w = n / n.sum(); mu = (r * v[:, None]).sum(0) / n; sd = np.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / n) + 1e-6
    lo, hi = np.argsort(mu); grid = np.linspace(mu[lo], mu[hi], 400)
    post = [w[k] / sd[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) for k in range(2)]
    t = grid[np.argmin(np.abs(post[lo] - post[hi]))]
    return float(t), {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}
def thresholds(F, valid):
    v = F[valid]; v = v[v > 0]; lv = np.log(v[::max(1, v.size // 2_000_000)])
    t_mo3 = float(np.exp(threshold_multiotsu(lv, classes=3)[0])); t_gm, info = gmm2_1d(lv); t_gm = float(np.exp(t_gm))
    return {"multiotsu3_lo": t_mo3, "gmm2_log": t_gm, "gmm_info": info}
def refine_ws(field, thr, erode_r=3):
    inner = ndimage.binary_erosion((field > thr) & ~zero, structure=ball(erode_r)); outer = ndimage.binary_erosion((field <= thr) & ~zero, structure=ball(erode_r)) | zero
    mk = np.zeros(o.shape, np.int32); mk[outer] = 1; mk[inner] = 2
    return (watershed(lstd3, mk) == 2) & ~zero
summary = []
for src in ("e_minstd_w7_sig3", "e_mincv_w7_sig3", "a04_minstd_bp2_w9_sig3", "a04_mincv_bp2_w9_sig3", "a04_minstd_bp2_w9_sig2", "a04_mincv_bp2_w9_sig2"):
    if src.startswith("e_"):
        F = np.load(f"{OUT}/{src}_field150.npy"); valid = ~zero; F150 = F
    else:
        F16 = np.load(f"{OUT}/{src}_field16.npy"); valid = F16 > 0
        F150 = map_coordinates(F16, q150, order=1, mode="nearest").reshape(o.shape); F150[zero] = 0; F = F16
    th = thresholds(F, valid)
    for tname in ("multiotsu3_lo", "gmm2_log"):
        thr = th[tname]; raw = (F150 > thr) & ~zero; mk = cleanup(raw, close_r=2)
        nm = f"{src}_{tname}"; sep = (float((F150[ag] < thr).mean()), float((F150[ti] > thr).mean()))
        d = report(mk, nm, o, old, mri_oct, {"thr": thr, "raw_cm3": float(raw.sum()) * VOX150**3 / 1000, "agarose_ref_below_thr": sep[0], "tissue_ref_above_thr": sep[1], "gmm": th["gmm_info"]})
        np.save(f"{OUT}/{nm}_mask150.npy", mk)
        summary.append((nm, d["cm3"], round(d["raw_cm3"], 2), round(thr, 4), d["keep_of_old"], d["dice_vs_mri_in_oct"], np.round(sep, 3).tolist(), d["n_cc"]))
        if tname == "gmm2_log":
            mw = cleanup(refine_ws(F150, thr), close_r=1); d2 = report(mw, nm + "_ws", o, old, mri_oct, {"thr": thr}); np.save(f"{OUT}/{nm}_ws_mask150.npy", mw)
            summary.append((nm + "_ws", d2["cm3"], "-", round(thr, 4), d2["keep_of_old"], d2["dice_vs_mri_in_oct"], "-", d2["n_cc"]))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4)); v = F150[~zero]; v = v[v > 0]
    ax.hist(np.log(v[::5]), bins=200, alpha=.4, label="all non-zero"); ax.hist(np.log(np.maximum(F150[ag], 1e-6)), bins=200, alpha=.6, label="agarose ref"); ax.hist(np.log(np.maximum(F150[ti], 1e-6)), bins=200, alpha=.6, label="tissue ref")
    for tname, col in (("multiotsu3_lo", "k"), ("gmm2_log", "r")): ax.axvline(np.log(th[tname]), color=col, label=tname)
    ax.set_yscale("log"); ax.legend(); ax.set_title(f"log {src}"); plt.tight_layout(); plt.savefig(f"{OUT}/{src}_loghist.png", dpi=80); plt.close(fig)
say("SUMMARY (name, cm3, raw, thr, keep_old, dice, (ag<thr, ti>thr), n_cc)"); [say(*s) for s in summary]
say("done rethr")
