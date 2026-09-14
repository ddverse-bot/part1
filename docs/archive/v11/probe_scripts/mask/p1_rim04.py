import sys; sys.path.insert(0, "/tmp")
from p1_common import *
from multiprocessing import Pool
V = f"{W}/octv.npy"; POOL = 4; CH = 128; WIN = 5
def work(args):
    z0, z1, thr = args; v = np.load(V, mmap_mode="r"); Z = v.shape[0]; pad = 8
    a, b = max(0, z0 - pad), min(Z, z1 + pad)
    blk = np.asarray(v[a:b]).astype(np.float32); t = local_std(blk, WIN)[0]; t = gaussian_filter(t, 1.0)
    sl = slice(z0 - a, z1 - a); ts = t[sl]
    if thr is None:   # pass 1: subsample of the texture map for the global threshold
        return z0, ts[::4, ::4, ::4].copy()
    rim = ts > thr; z, y, xx = (s // POOL * POOL for s in rim.shape)
    r16 = rim[:z, :y, :xx].reshape(z // POOL, POOL, y // POOL, POOL, xx // POOL, POOL).mean(axis=(1, 3, 5))     # fraction of rim voxels per 0.16 mm cell
    tm = ts[:z, :y, :xx].reshape(z // POOL, POOL, y // POOL, POOL, xx // POOL, POOL).max(axis=(1, 3, 5))
    return z0, np.stack([r16, tm], 0).astype(np.float32)
if __name__ == "__main__":
    v = np.load(V, mmap_mode="r"); Z, Y, X = v.shape; Zp = Z // POOL; jobs = [(z0, min(Zp * POOL, z0 + CH)) for z0 in range(0, Zp * POOL, CH)]
    with Pool(4) as p: subs = [s for _, s in p.imap_unordered(work, [(a, b, None) for a, b in jobs])]
    sub = np.concatenate([s.ravel() for s in subs]); sub = sub[sub > 0]; thr = otsu(sub); say("rim04 otsu thr", thr, "frac above", float((sub > thr).mean()))
    out = np.zeros((2, Zp, Y // POOL, X // POOL), np.float32)
    with Pool(4) as p:
        for z0, arr in p.imap_unordered(work, [(a, b, thr) for a, b in jobs]): out[:, z0 // POOL:z0 // POOL + arr.shape[1]] = arr
    np.save(f"{OUT}/rim16_from04_w5.npy", out)
    say("saved rim16 fields; rim frac at 0.16 (any rim voxel in cell):", float((out[0] > 0).mean()))
    say("done rim04")
