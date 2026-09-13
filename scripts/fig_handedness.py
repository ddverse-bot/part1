"""Side-by-side of the two I58 handedness candidates in MRI space: MRI | OCT via P_R5 (proper in the prep frame) | OCT via the mirrored
solution, three orthogonal MRI planes through the OCT block centre, with the MRI tissue contour on the OCT panels."""
import sys, numpy as np, nibabel as nib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
mri_p, a_p, b_p, out = sys.argv[1:5]
M = np.asarray(nib.load(mri_p).dataobj, np.float32); A = np.asarray(nib.load(a_p).dataobj, np.float32); B = np.asarray(nib.load(b_p).dataobj, np.float32)
tis = ndimage.binary_opening(M > 1.0211 * 0.55, iterations=1)                   # display contour only
sup = (A > 0) | (B > 0); c = np.array(ndimage.center_of_mass(sup)).round().astype(int)
def norm(x, m):
    v = x[m] if m.any() else x[x > 0]; lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1); return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
planes = [("axis0 = %d" % c[0], lambda V: V[c[0], :, :]), ("axis1 = %d" % c[1], lambda V: V[:, c[1], :]), ("axis2 = %d" % c[2], lambda V: V[:, :, c[2]])]
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
for r, (name, sl) in enumerate(planes):
    m2 = sl(M); t2 = sl(tis)
    for col, (title, V) in enumerate((("MRI", M), ("OCT via P_R5 (shipped, proper in prep frame)", A), ("OCT via mirrored solution (proper in OCT header frame)", B))):
        img = sl(V); a = ax[r, col]; a.imshow(norm(img, img > 0).T if col else norm(m2, m2 > 0).T, cmap="gray", origin="lower")
        if col: a.contour(t2.T.astype(float), levels=[0.5], colors="c", linewidths=0.6)
        a.set_title(f"{title}\n{name}", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=80); print("saved", out)
