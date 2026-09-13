import sys; sys.path.insert(0, "/tmp")
from p1_common import *
from multiprocessing import Pool
V = f"{W}/octv.npy"; POOL = 4; SIG_BP = 2.0; WIN = 9; CH = 128
def masked_std_axes(bs, m, w):
    out = []
    for axn in range(3):
        sz = [1, 1, 1]; sz[axn] = w
        den = uniform_filter(m, sz); mu = uniform_filter(bs * m, sz) / np.maximum(den, 1e-3); v = uniform_filter(bs * bs * m, sz) / np.maximum(den, 1e-3) - mu * mu
        s = np.sqrt(np.maximum(v, 0)); s[den < 0.5] = 0; out.append(s)
    return out
def pool4(a):  # mean-pool by POOL along all axes (trailing remainder dropped)
    z, y, x = (s // POOL * POOL for s in a.shape)
    return a[:z, :y, :x].reshape(z // POOL, POOL, y // POOL, POOL, x // POOL, POOL).mean(axis=(1, 3, 5))
def work(args):
    z0, z1 = args; v = np.load(V, mmap_mode="r"); Z = v.shape[0]; pad = 16
    a, b = max(0, z0 - pad), min(Z, z1 + pad)
    blk = np.asarray(v[a:b]).astype(np.float32); m = (blk > 0).astype(np.float32)
    bs = gaussian_filter(blk * m, SIG_BP) / np.maximum(gaussian_filter(m, SIG_BP), 1e-3); bs[m == 0] = 0
    S = masked_std_axes(bs, m, WIN)
    mu = uniform_filter(bs * m, WIN) / np.maximum(uniform_filter(m, WIN), 1e-3)
    sl = slice(z0 - a, z1 - a)
    outs = [pool4(s[sl] * m[sl]) for s in S] + [pool4(mu[sl] * m[sl]), pool4(m[sl])]
    return z0, np.stack(outs, 0).astype(np.float32)
if __name__ == "__main__":
    v = np.load(V, mmap_mode="r"); Z, Y, X = v.shape; say("octv", v.shape)
    Zp = Z // POOL; jobs = [(z0, min(Z, z0 + CH)) for z0 in range(0, Zp * POOL, CH)]
    fields = np.zeros((5, Zp, Y // POOL, X // POOL), np.float32)
    with Pool(4) as p:
        for z0, arr in p.imap_unordered(work, jobs):
            fields[:, z0 // POOL:z0 // POOL + arr.shape[1]] = arr; say("chunk", z0, "done")
    den = np.maximum(fields[4], 1e-3); Sz, Sy, Sx, MU = (fields[i] / den for i in range(4)); valid = fields[4] > 0.5
    np.save(f"{OUT}/fields16_bp2_w9.npy", fields)
    A16 = pooled_affine(np.load(f"{W}/octv_affine.npy"), POOL) if False else None
    Av = np.load(f"{W}/octv_affine.npy"); A16 = Av.copy(); A16[:3, 3] = Av[:3, 3] + Av[:3, :3] @ (np.ones(3) * (POOL - 1) / 2.0); A16[:3, :3] = Av[:3, :3] * POOL
    np.save(f"{OUT}/fields16_affine.npy", A16)
    mv = valid.astype(np.float32)
    def msmooth(f, sig): return gaussian_filter(f * mv, sig) / np.maximum(gaussian_filter(mv, sig), 1e-3)
    o = np.load(f"{W}/oct150.npy"); old = np.load(f"{W}/oct150_mask.npy"); zero150 = o <= 0; A150 = np.load(f"{W}/oct150_affine.npy")
    mri_oct = np.load(f"{OUT}/mri_tissue_in_oct150_T.npy")
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in o.shape], indexing="ij")
    P = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], 0).astype(np.float64); q150 = (np.linalg.inv(A16) @ A150 @ P)[:3]
    for sig in (2, 3):
        Ss = [msmooth(s, sig) for s in (Sz, Sy, Sx)]; mus = msmooth(MU, sig)
        mn = np.minimum(np.minimum(Ss[0], Ss[1]), Ss[2])
        for fname, F in (("minstd", mn), ("mincv", mn / np.maximum(mus, 1.0))):
            F = F.copy(); F[~valid] = 0; thr = otsu(F[valid])
            m16 = cleanup((F > thr) & valid, close_r=2); v16 = float(m16.sum()) * (VOX04 * POOL) ** 3 / 1000
            # to the 0.15 grid: trilinear field -> threshold -> cleanup
            F150 = map_coordinates(F, q150, order=1, mode="nearest").reshape(o.shape); F150[zero150] = 0
            m150 = cleanup((F150 > thr) & ~zero150, close_r=2)
            nm = f"a04_{fname}_bp2_w9_sig{sig}"
            d = report(m150, nm, o, old, mri_oct, {"thr": thr, "cm3_at_016": round(v16, 3), "bp_sigma_mm": SIG_BP * VOX04, "window_mm": WIN * VOX04, "smooth_sigma_mm": sig * VOX04 * POOL})
            np.save(f"{OUT}/{nm}_mask150.npy", m150); np.save(f"{OUT}/{nm}_field16.npy", F.astype(np.float32)); np.save(f"{OUT}/{nm}_mask16.npy", m16)
            with open(f"{OUT}/{nm}_thr.txt", "w") as f: f.write(str(thr))
    say("done aniso04")
