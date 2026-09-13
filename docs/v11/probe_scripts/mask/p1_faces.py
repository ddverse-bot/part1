import sys; sys.path.insert(0, "/tmp")
from p1_common import *
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import distance_transform_edt
o = np.load(f"{W}/oct150.npy"); zero = o <= 0
t, mu = local_std(o, 3); ts = gaussian_filter(t, 0.7); hi = otsu(ts[~zero])
rim = apply_hysteresis_threshold(ts, 0.5 * hi, hi); rimd = ndimage.binary_dilation(rim, structure=ball(1))
vmax = np.percentile(o[o > 0], 99.5)
# slices near faces + a few z levels
sel = [(0, 0), (0, 5), (0, 12), (0, 19), (0, 40), (0, 150), (0, 176), (0, 193), (1, 0), (1, 3), (1, 263), (1, 266), (2, 0), (2, 3), (2, 208), (2, 211)]
fig, ax = plt.subplots(4, 4, figsize=(22, 22))
for a, (axn, i) in zip(ax.ravel(), sel):
    im = np.take(o, i, axis=axn); rm = np.take(rimd, i, axis=axn); zz = np.take(zero, i, axis=axn)
    a.imshow(im, cmap="gray", vmax=vmax); a.contour(rm.astype(float), levels=[0.5], colors="r", linewidths=0.5); a.contour(zz.astype(float), levels=[0.5], colors="y", linewidths=0.5)
    a.set_title(f"axis{axn} idx{i} (red=rim w3 hyst0.5 dil1, yellow=zero)", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/faces.png", dpi=55); plt.close(fig)
# cores of the passable space
passable = ~rimd & ~zero
edt = distance_transform_edt(passable)
near_zero = ndimage.binary_dilation(zero, structure=ball(1), iterations=3)
for r in (2, 3, 4, 6):
    core = edt > r; lab, n = label(core); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    faces = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
    fc = np.bincount(faces, minlength=sizes.size); zc = np.bincount(lab[near_zero].ravel(), minlength=sizes.size)
    top = np.argsort(sizes)[::-1][:8]
    say(f"r={r}: n cores {n}; top (cm3 / face-contact / near-zero): " + "; ".join(f"{sizes[k]*VOX150**3/1000:.2f}/{fc[k]}/{zc[k]}" for k in top))
    if r == 3:
        fig, ax = plt.subplots(3, 3, figsize=(16, 16))
        show = np.zeros(o.shape, np.int8)
        for rank, k in enumerate(top[:6]): show[lab == k] = rank + 1
        for rr in range(3):
            for c, frac in enumerate((0.3, 0.5, 0.7)):
                i = int(o.shape[rr] * frac); ax[rr, c].imshow(np.take(o, i, axis=rr), cmap="gray", vmax=vmax); ax[rr, c].imshow(np.ma.masked_equal(np.take(show, i, axis=rr), 0), cmap="tab10", alpha=0.5, vmin=0, vmax=9); ax[rr, c].set_title(f"cores r=3, axis{rr} idx{i}", fontsize=9); ax[rr, c].axis("off")
        plt.tight_layout(); plt.savefig(f"{OUT}/cores_r3.png", dpi=55); plt.close(fig)
say("done faces")
