#!/usr/bin/env python
"""P3 probe part 3: apply the x-stripe flat-field to the full octv (0.04 mm), derive oct150_destriped, measure before/after,
characterise the black missing-tile blocks.  CPU only, 4 worker processes (each single-threaded).
usage: python p3_full.py [sz sy sx mode]   (defaults 12 12 6 add)"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy import ndimage
from multiprocessing import Pool

SZ, SY, SX = (float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (12.0, 12.0, 6.0)
MODE = sys.argv[4] if len(sys.argv) > 4 else "add"
BASE = os.path.expanduser("~/autodl-tmp/oct-mri-registration"); W = os.path.join(BASE, "work/xiangrui_I58bs"); OUT = os.path.join(BASE, "work/probe_xr/stripes")
OCTV = f"{W}/octv.npy"; DST = f"{OUT}/octv_destriped.npy"; SMAP = f"{OUT}/octv_stripeS.npy"
t0 = time.time()
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)
TR = 3.0
PAD = int(np.ceil(TR * SZ)) + 2

def masked_lp(img, m, sig):
    num = ndimage.gaussian_filter(np.where(m, img, 0.0).astype(np.float32), sig, truncate=TR); den = ndimage.gaussian_filter(m.astype(np.float32), sig, truncate=TR)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0), den > 0.05

def work(args):
    z0, z1 = args
    octv = np.load(OCTV, mmap_mode="r"); D = octv.shape[0]
    a, b = max(0, z0 - PAD), min(D, z1 + PAD)
    blk = np.asarray(octv[a:b]).astype(np.float32); valid = blk > 0
    A_, ok = masked_lp(blk, valid, (SZ, SY, 0)); B_, _ = masked_lp(A_, ok, (0, 0, SX)); S = np.where(ok, A_ - B_, 0.0).astype(np.float32)
    if MODE == "add":
        out = np.where(valid, blk - S, 0.0)
    else:
        g = np.where(ok & (A_ > 0.2 * B_), B_ / np.maximum(A_, 1e-3), 1.0); g = np.clip(g, 0.5, 2.0); out = np.where(valid, blk * g, 0.0)
    out = np.clip(out, 1.0, 65000.0); out[~valid] = 0.0            # keep zeros as the missing-data marker
    dst = np.load(DST, mmap_mode="r+"); sm = np.load(SMAP, mmap_mode="r+")
    dst[z0:z1] = out[z0 - a:z1 - a].astype(np.float16); sm[z0:z1] = S[z0 - a:z1 - a].astype(np.float16)
    dst.flush(); sm.flush()
    return (z0, z1, float(S[z0 - a:z1 - a][valid[z0 - a:z1 - a]].std()))

if __name__ == "__main__":
    octv = np.load(OCTV, mmap_mode="r"); D, H, Wd = octv.shape
    say("octv", octv.shape, "params", SZ, SY, SX, MODE, "pad", PAD)
    if not os.path.exists(DST):
        np.lib.format.open_memmap(DST, mode="w+", dtype=np.float16, shape=octv.shape).flush()
    if not os.path.exists(SMAP):
        np.lib.format.open_memmap(SMAP, mode="w+", dtype=np.float16, shape=octv.shape).flush()
    CH = 64; jobs = [(z0, min(D, z0 + CH)) for z0 in range(0, D, CH)]
    with Pool(4) as p:
        for z0, z1, s in p.imap_unordered(work, jobs):
            say("chunk", z0, z1, "S std (valid)", round(s, 1))
    say("full destripe written", DST, SMAP)

    # ------------------------------------------------------------------ oct150_destriped = oct150 - S resampled (pool 4 -> trilinear onto the oct150 grid)
    import torch; torch.set_num_threads(4)
    sys.path.insert(0, os.path.join(BASE, "octreg"))
    from octreg.common import pool_mean_np, pooled_affine, resample_to_grid, to_t
    Av = np.load(f"{W}/octv_affine.npy"); A150 = np.load(f"{W}/oct150_affine.npy"); o150 = np.load(f"{W}/oct150.npy"); m150 = np.load(f"{W}/oct150_mask.npy")
    S = np.load(SMAP, mmap_mode="r")
    Sp = pool_mean_np(S, 4); Ap = pooled_affine(Av, 4)
    S150 = resample_to_grid(to_t(Sp, device="cpu")[None], to_t(Ap, device="cpu"), to_t(A150, device="cpu"), o150.shape).numpy()[0]
    o150d = np.where(o150 > 0, np.clip(o150 - S150, 1.0, None), 0.0).astype(np.float32)
    np.save(f"{OUT}/oct150_destriped.npy", o150d); np.save(f"{OUT}/oct150_stripeS.npy", S150.astype(np.float32))
    say("oct150_destriped written; S150 std in mask", float(S150[m150].std()))

    # ------------------------------------------------------------------ before/after measures
    R = {"params": {"sz": SZ, "sy": SY, "sx": SX, "mode": MODE, "truncate": TR}}
    m150e = np.load(f"{OUT}/mask150_eroded2mm.npy")
    M_ = np.linalg.inv(A150) @ Av
    idx = [np.clip(np.rint(M_[r, r] * np.arange(octv.shape[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
    mve = m150e[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
    dst = np.load(DST, mmap_mode="r")
    def axis_profile(vol, mask, axis, chunk=32):
        n = vol.shape[axis]; s = np.zeros(n); c = np.zeros(n)
        for z0 in range(0, vol.shape[0], chunk):
            blk = np.asarray(vol[z0:z0 + chunk]).astype(np.float32); mk = np.asarray(mask[z0:z0 + chunk]); v = np.where(mk, blk, 0.0)
            axes = tuple(i for i in range(3) if i != axis)
            if axis == 0: s[z0:z0 + blk.shape[0]] += v.sum(axes); c[z0:z0 + blk.shape[0]] += mk.sum(axes)
            else: s += v.sum(axes); c += mk.sum(axes)
        return np.where(c > 2000, s / np.maximum(c, 1), np.nan), c
    def resid(p, win):
        ok = np.isfinite(p); ii = np.where(ok)[0]; pp = p[ii.min():ii.max() + 1].copy(); okk = ok[ii.min():ii.max() + 1]
        pp[~okk] = np.interp(np.where(~okk)[0], np.where(okk)[0], pp[okk]); d = pp - ndimage.uniform_filter1d(pp, win, mode="nearest")
        n = d.size; f = np.fft.rfftfreq(n); P = np.abs(np.fft.rfft(d * np.hanning(n))) ** 2; P[f < 1.0 / win] = 0; k = int(np.argmax(P))
        return {"resid_std": float(d.std()), "p2p95": float(np.percentile(d, 97.5) - np.percentile(d, 2.5)), "fft_period": float(1 / f[k]) if f[k] > 0 else None, "fft_peak_frac": float(P[k] / P.sum())}
    R["profiles"] = {}
    for nm, vol, mk, win in (("octv_before", octv, mve, 60), ("octv_after", dst, mve, 60), ("oct150_before", o150, m150e, 16), ("oct150_after", o150d, m150e, 16)):
        R["profiles"][nm] = {}
        for ax_i, an in enumerate(["z", "y", "x"]):
            p, c = axis_profile(vol, mk, ax_i); R["profiles"][nm][an] = resid(p, win)
        say(nm, json.dumps(R["profiles"][nm]))
    # adjacent-plane NCC along x (z-y planes), high-passed, before/after (subsample of x)
    def ncc(a, b, m):
        a = a[m]; b = b[m]; a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    def adj_ncc(vol, mask, axis, lo, hi, step):
        out = []
        for i in range(lo, hi, step):
            p0 = np.take(vol, i, axis=axis).astype(np.float32); p1 = np.take(vol, i + 1, axis=axis).astype(np.float32); m = np.take(mask, i, axis=axis) & np.take(mask, i + 1, axis=axis)
            if m.sum() < 5000: continue
            out.append(ncc(p0 - ndimage.gaussian_filter(p0, 3), p1 - ndimage.gaussian_filter(p1, 3), m))
        return float(np.median(out)), float(np.percentile(out, 5))
    octv_ram = np.asarray(octv); dst_ram = np.asarray(dst)
    R["adjacent_plane_ncc"] = {}
    for nm, vol in (("before", octv_ram), ("after", dst_ram)):
        R["adjacent_plane_ncc"][nm] = {"x": adj_ncc(vol, mve, 2, 60, 735, 3), "y": adj_ncc(vol, mve, 1, 60, 940, 4), "z": adj_ncc(vol, mve, 0, 60, 668, 3)}
        say("adjacent plane NCC (median, p5)", nm, R["adjacent_plane_ncc"][nm])
    # figures: z-x and x-y planes before/after at both resolutions
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cz, cy, cx = [int(v) for v in ndimage.center_of_mass(mve)]; cz1, cy1, cx1 = [int(v) for v in ndimage.center_of_mass(m150e)]
    v0, v1 = 3000.0, 34000.0
    fig, ax = plt.subplots(3, 3, figsize=(27, 24))
    ax[0, 0].imshow(octv_ram[:, cy, :].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[0, 0].set_title(f"octv z-x (y={cy}) BEFORE")
    ax[0, 1].imshow(dst_ram[:, cy, :].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[0, 1].set_title(f"octv z-x AFTER x-flat-field (sz{SZ:.0f} sy{SY:.0f} sx{SX:.0f} {MODE})")
    Sram = np.asarray(S[:, cy, :]).astype(np.float32); ax[0, 2].imshow(Sram, cmap="gray", vmin=-4000, vmax=4000); ax[0, 2].set_title("removed stripe S (z-x)")
    ax[1, 0].imshow(octv_ram[cz].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[1, 0].set_title(f"octv x-y (z={cz}) BEFORE")
    ax[1, 1].imshow(dst_ram[cz].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[1, 1].set_title("octv x-y AFTER")
    ax[1, 2].imshow(np.asarray(S[cz]).astype(np.float32), cmap="gray", vmin=-4000, vmax=4000); ax[1, 2].set_title("removed stripe S (x-y)")
    ax[2, 0].imshow(o150[:, cy1, :], cmap="gray", vmin=v0, vmax=v1); ax[2, 0].set_title(f"oct150 z-x (y={cy1}) BEFORE")
    ax[2, 1].imshow(o150d[:, cy1, :], cmap="gray", vmin=v0, vmax=v1); ax[2, 1].set_title("oct150 z-x AFTER (oct150 - S150)")
    ax[2, 2].imshow(S150[:, cy1, :], cmap="gray", vmin=-4000, vmax=4000); ax[2, 2].set_title("S150")
    plt.tight_layout(); plt.savefig(f"{OUT}/full_before_after.png", dpi=60); plt.close(fig)
    # zoom crops 300x300 full res
    fig, ax = plt.subplots(2, 3, figsize=(24, 16)); c = 150
    for j, (nm, vol) in enumerate((("BEFORE", octv_ram), ("AFTER", dst_ram))):
        ax[j, 0].imshow(vol[cz - c:cz + c, cy, cx - c:cx + c].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[j, 0].set_title(f"{nm} z-x crop")
        ax[j, 1].imshow(vol[cz, cy - c:cy + c, cx - c:cx + c].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[j, 1].set_title(f"{nm} x-y crop")
        ax[j, 2].imshow(vol[cz - c:cz + c, cy - c:cy + c, cx].astype(np.float32), cmap="gray", vmin=v0, vmax=v1); ax[j, 2].set_title(f"{nm} z-y crop (should be unchanged)")
    plt.tight_layout(); plt.savefig(f"{OUT}/full_before_after_zoom.png", dpi=70); plt.close(fig)

    # ------------------------------------------------------------------ black blocks (oct150): rectangles inside the scanned FOV bounding box
    nzb = o150 > 0; fov = [np.where(nzb.any(axis=tuple(j for j in range(3) if j != i)))[0] for i in range(3)]
    fov_box = [[int(f.min()), int(f.max()) + 1] for f in fov]
    inside = np.zeros_like(nzb); inside[fov_box[0][0]:fov_box[0][1], fov_box[1][0]:fov_box[1][1], fov_box[2][0]:fov_box[2][1]] = True
    zin = (~nzb) & inside
    lab, n = ndimage.label(zin); objs = ndimage.find_objects(lab); sizes = ndimage.sum(zin, lab, np.arange(1, n + 1))
    dt_old = ndimage.distance_transform_edt(~m150) * 0.15; dt_er = ndimage.distance_transform_edt(~m150e) * 0.15
    blocks = []
    for k in np.argsort(sizes)[::-1][:10]:
        sl = objs[k]; comp = lab == (k + 1)
        blocks.append({"bbox_zyx_vox": [[int(s.start), int(s.stop)] for s in sl], "bbox_mm": [[round(s.start * 0.15, 2), round(s.stop * 0.15, 2)] for s in sl], "n_vox": int(sizes[k]),
                       "fill_of_bbox": round(float(sizes[k] / np.prod([s.stop - s.start for s in sl])), 3), "n_in_oldmask": int((comp & m150).sum()), "n_in_eroded": int((comp & m150e).sum()),
                       "min_dist_to_oldmask_mm": round(float(dt_old[comp].min()), 2), "min_dist_to_eroded2mm_mask_mm": round(float(dt_er[comp].min()), 2)})
    R["black_blocks_oct150"] = {"fov_bbox_zyx": fov_box, "n_zero_components_inside_fov": int(n), "blocks": blocks, "exact_zero": True,
                                "note": "zeros are exactly 0.0 (min nonzero 2.19 in octv); all-zero region = outside FOV + corner rectangles"}
    # mask boundary vs zeros: fraction of old-mask boundary voxels adjacent to a zero
    bnd = m150 & ~ndimage.binary_erosion(m150); adj0 = ndimage.binary_dilation(~nzb) & bnd
    R["black_blocks_oct150"]["oldmask_boundary_frac_touching_zeros"] = float(adj0.sum() / max(bnd.sum(), 1))
    say("black blocks", json.dumps(R["black_blocks_oct150"])[:2000])
    json.dump(R, open(f"{OUT}/full_results.json", "w"), indent=1, default=float)
    say("done")
