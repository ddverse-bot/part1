import sys; sys.path.insert(0, "/tmp")
from p1_common import *
v = np.load(f"{W}/octv.npy", mmap_mode="r"); say("octv", v.shape)
# reference slabs: agarose z in [8,40) (0.32-1.6 mm; first imaged sections are agarose), tissue: slab around centre
def var1d(a, w, axis):
    sz = [1, 1, 1]; sz[axis] = w; m = uniform_filter(a, sz); vv = uniform_filter(a * a, sz) - m * m; return np.sqrt(np.maximum(vv, 0))
def feats_of(blk):
    blk = blk.astype(np.float32); f = {}
    f["lstd5"] = local_std(blk, 5)[0]; f["lstd9"] = local_std(blk, 9)[0]
    for axn, nm in enumerate("zyx"): f[f"std_{nm}9"] = var1d(blk, 9, axn)
    bs = gaussian_filter(blk, 2.0); f["bp2_lstd9"] = local_std(bs, 9)[0]; f["bp2_std_y9"] = var1d(bs, 9, 1); f["bp2_std_z9"] = var1d(bs, 9, 0); f["bp2_std_x9"] = var1d(bs, 9, 2)
    bs4 = gaussian_filter(blk, 4.0); f["bp4_lstd15"] = local_std(bs4, 15)[0]
    f["int"] = blk
    return f
zc, yc, xc = [s // 2 for s in v.shape]
A = np.asarray(v[8:40]); Az = A <= 0
T = np.asarray(v[zc - 16:zc + 16]); 
fa = feats_of(A); ft = feats_of(T)
# masks: agarose ref = non-zero voxels away from zero in A (all of it is agarose); tissue ref = ball radius 60 vox (2.4 mm) around slab centre in y,x
nzA = ~ndimage.binary_dilation(Az, structure=ball(1), iterations=6)
yy, xx = np.ogrid[:T.shape[1], :T.shape[2]]; ti = np.broadcast_to(((yy - yc) ** 2 + (xx - xc) ** 2) <= 60 ** 2, T.shape)
say("ref voxels", nzA.sum(), ti.sum())
for k in fa:
    for sig in (0, 4, 8):
        a = gaussian_filter(fa[k], sig)[nzA] if sig else fa[k][nzA]; t = gaussian_filter(ft[k], sig)[ti] if sig else ft[k][ti]
        a = a[::7]; t = t[::7]
        thr_hi = np.percentile(a, 95); thr_lo = np.percentile(t, 5)
        say(f"{k:11s} sig{sig}: agarose p10/50/90 {np.percentile(a,[10,50,90]).round(0).tolist()}  tissue {np.percentile(t,[10,50,90]).round(0).tolist()}  tissue>ag_p95 {float((t>thr_hi).mean()):.3f}  ag<tissue_p5 {float((a<thr_lo).mean()):.3f}")
say("done feat04")
