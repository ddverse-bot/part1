import sys; sys.path.insert(0, "/tmp")
from p1_common import *
from scipy.ndimage import gaussian_gradient_magnitude, distance_transform_edt
o = np.load(f"{W}/oct150.npy"); zero = o <= 0; x = o.astype(np.float32)
nz = ~ndimage.binary_dilation(zero, structure=ball(1), iterations=3)          # away from zero blocks
ag = np.zeros(o.shape, bool); ag[:11] = True; ag[189:] = True; ag &= nz
zc, yc, xc = [s // 2 for s in o.shape]; zz, yy, xx = np.ogrid[:o.shape[0], :o.shape[1], :o.shape[2]]
ti = ((zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2) <= 22 ** 2
say("ref voxels: agarose", ag.sum(), "tissue", ti.sum())
def var1d(a, w, axis):
    sz = [1, 1, 1]; sz[axis] = w; m = uniform_filter(a, sz); v = uniform_filter(a * a, sz) - m * m; return np.sqrt(np.maximum(v, 0))
feats = {}
feats["lstd3"] = local_std(x, 3)[0]; feats["lstd5"] = local_std(x, 5)[0]
for axn, nm in enumerate("zyx"):
    feats[f"std_{nm}7"] = var1d(x, 7, axn)
feats["grad1"] = gaussian_gradient_magnitude(x, 1.0)
# band-pass texture: remove speckle with a 0.15-mm-level gaussian 1.0 then local std w=7 (structure at ~0.3-1 mm)
xs = gaussian_filter(x, 1.0); feats["bp_lstd7"] = local_std(xs, 7)[0]
feats["bp_std_y7"] = var1d(xs, 7, 1); feats["bp_std_z7"] = var1d(xs, 7, 0)
# intensity relative to slice median
feats["int"] = x.copy()
rows = []
for k, f in feats.items():
    for sig in (0, 3, 6):
        fs = gaussian_filter(f, sig) if sig else f
        a = fs[ag]; t = fs[ti]
        # separation: fraction of tissue above the agarose 95th pct, fraction of agarose below the tissue 5th pct
        thr_hi = np.percentile(a, 95); thr_lo = np.percentile(t, 5)
        rows.append((k, sig, np.percentile(a, [10, 50, 90]).round(0).tolist(), np.percentile(t, [10, 50, 90]).round(0).tolist(), round(float((t > thr_hi).mean()), 3), round(float((a < thr_lo).mean()), 3)))
        say(f"{k:10s} sig{sig}: agarose p10/50/90 {rows[-1][2]}  tissue {rows[-1][3]}  tissue>ag_p95 {rows[-1][4]}  ag<tissue_p5 {rows[-1][5]}")
# figure of a few smoothed features (sig=3) on mid-slices
keys = ["lstd3", "std_y7", "std_z7", "std_x7", "bp_lstd7", "bp_std_y7", "int"]
fig, ax = plt.subplots(3, len(keys), figsize=(4 * len(keys), 13))
for c, k in enumerate(keys):
    fs = gaussian_filter(feats[k], 3.0); fs[zero] = 0
    for r in range(3):
        i = o.shape[r] // 2; im = np.take(fs, i, axis=r); ax[r, c].imshow(im, cmap="magma", vmax=np.percentile(fs[~zero], 98)); ax[r, c].set_title(f"{k} sig3 axis{r}", fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/feat150_sig3.png", dpi=55); plt.close(fig)
np.save(f"{OUT}/ref_agarose150.npy", ag); np.save(f"{OUT}/ref_tissue150.npy", ti)
say("done feat")
