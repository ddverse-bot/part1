#!/usr/bin/env python3
"""CPU test of octreg.destripe on a subject's cached vessel-level volume (never writes into --work):

    python scripts/test_destripe.py --work work/xiangrui_I58bs --out work/v11/tests/destripe [--ref work/probe_xr/stripes/octv_destriped.npy]

Loads W/octv.npy into RAM (a copy), runs detect_stripe_axis + destripe_inplace, prints before/after per-axis profile residual
stds, FFT peak, zero fraction, min/max, derives the 0.15 mm level from the (un)destriped octv (block means = prep's pooling,
then resampling onto the oct150 grid) and reports the same numbers there, compares against a reference destriped volume
(NCC inside the 2-mm-eroded mask on a subsample), writes DIR/octv_destriped.npy, DIR/oct150_destriped.npy,
DIR/test_destripe.png (mid z-x and x-y planes before/after/S + stripe-axis profiles) and DIR/test_destripe.json.
PASS criteria (I58, acceptance M2): axis 2 with ratio <= 0.8; stripe-axis residual <= 55 (octv 138 -> ~51; oct150 85 -> ~49);
other axes change < 3 %; zero fraction unchanged; min nonzero >= 1; max <= 60000; no inf/nan."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.destripe import detect_stripe_axis, destripe_inplace, plane_profiles, profile_residual, profile_stats, CAP_F16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--axis", type=int, default=None, help="force the stripe axis (default: detect_stripe_axis)")
    ap.add_argument("--ref", type=Path, default=None, help="reference destriped octv (.npy, same grid) for an NCC check")
    ap.add_argument("--n-workers", type=int, default=4); ap.add_argument("--save-stripe-field", action="store_true")
    ap.add_argument("--sigma-smooth-mm", type=float, default=0.48); ap.add_argument("--sigma-hp-mm", type=float, default=0.24)
    ap.add_argument("--target-mm", type=float, default=0.15); ap.add_argument("--resid-max", type=float, default=55.0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)
    def fmt(v, nd=1): return "n/a" if v is None else f"{v:.{nd}f}"
    R = {"work": str(a.work), "out": str(a.out)}
    prep = json.load(open(a.work / "prep.json")) if (a.work / "prep.json").exists() else {}
    A_v = np.load(a.work / "octv_affine.npy"); A150 = np.load(a.work / "oct150_affine.npy")
    vox_v = float(np.linalg.norm(A_v[:3, :3], axis=0).mean()); vox_150 = float(np.linalg.norm(A150[:3, :3], axis=0).mean())
    vol = np.load(a.work / "octv.npy")                                                    # in RAM: the COPY we destripe
    o150_ref = np.load(a.work / "oct150.npy"); m150 = np.load(a.work / "oct150_mask.npy")
    say("octv", vol.shape, vol.dtype, f"{vox_v*1000:.1f} um; oct150", o150_ref.shape, f"{vox_150*1000:.0f} um; mask frac150 {m150.mean():.3f}")
    # statistics mask: the 0.15 mm mask eroded by 2 mm, nearest-resampled onto the octv grid (prep_subject lines 142-144 pattern)
    er_mm = 2.0                                                                            # P3 statistics mask; smaller blocks fall back to 1 / 0.5 mm
    while True:
        er = int(round(er_mm / vox_150)); m150e = ndimage.binary_erosion(m150, iterations=er)
        if m150e.sum() >= 0.2 * m150.sum() or er_mm <= 0.5: break
        er_mm /= 2
    R["stats_mask_erosion_mm"] = er_mm
    M_ = np.linalg.inv(A150) @ A_v
    idx = [np.clip(np.rint(M_[r, r] * np.arange(vol.shape[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
    mve = m150e[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
    say(f"statistics mask: eroded {er} vox = {er_mm:g} mm at {vox_150*1000:.0f} um -> octv frac {mve.mean():.3f} ({int(mve.sum())} voxels)")
    if mve.sum() < 1e5:
        say("WARNING: eroded mask nearly empty; statistics fall back to octv > 0"); mve = None

    # ---------------------------------------------------------------- axis detection
    axis, dinfo = detect_stripe_axis(vol, mask=mve, n_planes=40, hp_sigma_vox=3.0)
    R["detect"] = dinfo | {"axis": axis}
    say("adjacent-plane NCC medians (z,y,x):", np.round(dinfo["ncc_median_per_axis"], 3).tolist(), "p5:", np.round(dinfo["ncc_p5_per_axis"], 3).tolist(),
        f"ratio {dinfo['ratio']:.3f} -> axis {axis} ({'decisive' if axis is not None else 'NOT decisive'}); period {dinfo['period_vox']} vox (peak/median {dinfo['fft_peak_over_median']})")
    say("per-axis profile residual std:", [fmt(v) for v in dinfo["profile_resid_std_per_axis"]], "fft periods:", [fmt(v, 2) for v in dinfo["fft_period_per_axis"]])
    if a.axis is not None: say(f"forcing axis {a.axis} (detected {axis})"); axis = a.axis
    if axis is None:
        say("axis not decisive -> destripe skipped (this is the F2 fallback, not an error)"); R["destripe"] = {"applied": False, "reason": "axis_not_decisive"}
        json.dump(R, open(a.out / "test_destripe.json", "w"), indent=1, default=float); return 0

    # ---------------------------------------------------------------- 0.15 mm level BEFORE (block means from octv == prep's pooling)
    import torch; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from octreg.common import pool_mean_np, pooled_affine, resample_to_grid, to_t
    sp = np.asarray(prep.get("oct", {}).get("spacing_um_zyx", [vox_v * 1000 / 2] * 3), float)
    k150 = np.maximum(1, np.round(a.target_mm * 1000 / sp)).astype(int); kv = np.maximum(1, np.ceil(24.0 / sp - 1e-6)).astype(int)
    if not (k150 % kv == 0).all(): k150 = np.maximum(1, np.round(a.target_mm * 1000 / (sp * kv))).astype(int) * kv
    kk = k150 // kv; A_p = pooled_affine(A_v, kk)
    def level150(v):
        p = pool_mean_np(v, kk)
        return resample_to_grid(to_t(p, device="cpu")[None], to_t(A_p, device="cpu"), to_t(A150, device="cpu"), o150_ref.shape).numpy()[0].astype(np.float32)
    o150_b = level150(vol)
    def ncc(x, y, m):
        x = x[m].astype(np.float64); y = y[m].astype(np.float64); x -= x.mean(); y -= y.mean(); return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))
    R["oct150_pooling_check_ncc_vs_stored"] = ncc(o150_b, o150_ref, m150e); say(f"oct150 re-derived from octv (pool {kk.tolist()}): NCC vs stored oct150 inside eroded mask {R['oct150_pooling_check_ncc_vs_stored']:.5f}")
    win150 = max(8, int(round(2.4 / vox_150))); stats150_b = [profile_residual(p, win150)[0] for p in plane_profiles(o150_b, m150e)[0]]
    stats150_stored = [profile_residual(p, win150)[0] for p in plane_profiles(o150_ref, m150e)[0]]
    # mid-plane snapshots and profiles before
    cz, cy, cx = [int(v) for v in ndimage.center_of_mass(mve if mve is not None else (vol > 0))]
    zx_b = vol[:, cy, :].astype(np.float32).copy(); xy_b = vol[cz].astype(np.float32).copy()
    means_b, counts_b, _ = plane_profiles(vol, mve); win_v = max(8, int(round(2.4 / vox_v)))

    # ---------------------------------------------------------------- destripe (in place on the RAM copy)
    dinfo2 = destripe_inplace(vol, axis=axis, vox_mm=vox_v, sigma_smooth_mm=a.sigma_smooth_mm, sigma_hp_mm=a.sigma_hp_mm, cap=CAP_F16, zchunk=64, pad=40,
                              n_workers=a.n_workers, mask=mve, save_S=(a.out / "octv_stripeS.npy" if a.save_stripe_field else None))
    R["destripe"] = dinfo2
    say(f"destripe axis {axis} done in {dinfo2['seconds']:.0f}s ({dinfo2['n_chunks']} chunks x {dinfo2['n_workers']} workers); S std {fmt(dinfo2['S_std_mean'], 0)}")
    say("profile residual std before -> after (z,y,x):", [f"{fmt(b)} -> {fmt(c)}" for b, c in zip(dinfo2["profile_resid_std_before"], dinfo2["profile_resid_std_after"])])
    say("stripe-axis FFT peak before/after:", dinfo2["fft_peak_before"], dinfo2["fft_peak_after"])
    say(f"zero fraction {dinfo2['zero_fraction_before']:.6f} -> {dinfo2['zero_fraction_after']:.6f}; min nonzero {dinfo2['min_nonzero_after']}; max {dinfo2['max_after']}; nonfinite {dinfo2['n_nonfinite_after']}")
    means_a, counts_a, _ = plane_profiles(vol, mve)
    zx_a = vol[:, cy, :].astype(np.float32); xy_a = vol[cz].astype(np.float32)
    np.save(a.out / "octv_destriped.npy", vol); say("wrote", a.out / "octv_destriped.npy")
    o150_a = level150(vol); np.save(a.out / "oct150_destriped.npy", o150_a)
    stats150_a = [profile_residual(p, win150)[0] for p in plane_profiles(o150_a, m150e)[0]]
    R["oct150"] = {"before": stats150_b, "after": stats150_a, "stored_oct150": stats150_stored, "detrend_planes": win150,
                   "zero_fraction_before": float((o150_b == 0).mean()), "zero_fraction_after": float((o150_a == 0).mean()), "min_nonzero_after": float(o150_a[o150_a > 0].min()), "max_after": float(o150_a.max())}
    say("oct150 profile residual std before -> after (z,y,x):", [f"{fmt(b['resid_std'])} -> {fmt(c['resid_std'])}" for b, c in zip(stats150_b, stats150_a)], "| stored oct150:", [fmt(s["resid_std"]) for s in stats150_stored])

    # ---------------------------------------------------------------- reference comparison
    if a.ref is not None and a.ref.exists():
        ref = np.load(a.ref, mmap_mode="r")
        if ref.shape == vol.shape:
            st = 3; sub = (slice(None, None, st),) * 3
            rv = np.asarray(ref[sub]).astype(np.float32); vv = vol[sub].astype(np.float32); mm = (mve[sub] if mve is not None else (vv > 0)) & (rv > 0) & (vv > 0)
            d = np.abs(rv[mm] - vv[mm])
            R["ref"] = {"file": str(a.ref), "ncc_in_mask_subsample": ncc(rv, vv, mm), "abs_diff_median": float(np.median(d)), "abs_diff_p99": float(np.percentile(d, 99)), "frac_diff_gt_64": float((d > 64).mean()), "subsample_step": st}
            say(f"vs reference {a.ref.name}: NCC {R['ref']['ncc_in_mask_subsample']:.5f}, |diff| median {R['ref']['abs_diff_median']:.1f} p99 {R['ref']['abs_diff_p99']:.1f}, frac > 64: {R['ref']['frac_diff_gt_64']:.4f}")
        else: say("reference shape mismatch", ref.shape, "vs", vol.shape)

    # ---------------------------------------------------------------- pass/fail
    ax_names = ["z", "y", "x"]; others = [i for i in range(3) if i != axis]
    sb, sa = dinfo2["profile_resid_std_before"], dinfo2["profile_resid_std_after"]
    checks = {"axis_decisive_ratio<=0.8": (dinfo["ratio"] <= 0.8) if np.isfinite(dinfo["ratio"]) else False,
              f"octv_{ax_names[axis]}_resid<= {a.resid_max:g}": None if sa[axis] is None else sa[axis] <= a.resid_max,
              "octv_other_axes_change<3%": None if any(sb[i] is None or sa[i] is None for i in others) else all(abs(sa[i] - sb[i]) / max(sb[i], 1e-9) < 0.03 for i in others),
              f"oct150_{ax_names[axis]}_resid<= {a.resid_max:g}": None if stats150_a[axis]["resid_std"] is None else stats150_a[axis]["resid_std"] <= a.resid_max,
              "zero_fraction_unchanged": dinfo2["zero_fraction_before"] == dinfo2["zero_fraction_after"],
              "min_nonzero>=1": dinfo2["min_nonzero_after"] is not None and dinfo2["min_nonzero_after"] >= 1.0,
              "max<=60000": dinfo2["max_after"] is not None and dinfo2["max_after"] <= 60000.0, "finite": dinfo2["n_nonfinite_after"] == 0,
              "oct150_finite": bool(np.isfinite(o150_a).all()),
              "runtime<300s": (time.time() - t0 < 300) if a.n_workers >= 4 else None}         # M2 timing is defined for 4 workers (I58 serial: 276 s destripe alone)
    if "ref" in R: checks["ref_ncc>=0.999"] = R["ref"]["ncc_in_mask_subsample"] >= 0.999
    R["checks"] = checks; R["all_pass"] = all(v is not False for v in checks.values()); R["seconds"] = time.time() - t0   # None = n/a (too few voxels)
    for k, v in checks.items(): say(f"  {'n/a ' if v is None else ('PASS' if v else 'FAIL')}  {k}")
    json.dump(R, open(a.out / "test_destripe.json", "w"), indent=1, default=float)

    # ---------------------------------------------------------------- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    v0, v1 = np.percentile(zx_b[zx_b > 0], [1, 99]) if (zx_b > 0).any() else (0, 1)
    sw = max(4.0 * (dinfo2["S_std_mean"] or 0.0), 1e-3)                                     # S display window follows the data level (I58 ~16k, I46 ~100)
    fig, ax = plt.subplots(3, 3, figsize=(21, 18))
    for r, (nm, b, af) in enumerate((("z-x plane (y=%d)" % cy, zx_b, zx_a), ("x-y plane (z=%d)" % cz, xy_b, xy_a))):
        S = np.where(b > 0, b - af, 0.0)
        ax[r, 0].imshow(b, cmap="gray", vmin=v0, vmax=v1); ax[r, 0].set_title(f"{a.work.name} octv {nm} BEFORE")
        ax[r, 1].imshow(af, cmap="gray", vmin=v0, vmax=v1); ax[r, 1].set_title(f"AFTER destripe (axis {axis} = {ax_names[axis]}, sigma {a.sigma_smooth_mm}/{a.sigma_hp_mm} mm)")
        ax[r, 2].imshow(S, cmap="gray", vmin=-sw, vmax=sw); ax[r, 2].set_title(f"removed stripe field S = before - after (window +-{sw:.0f})")
        for c in range(3): ax[r, c].axis("off")
    for c, axi in enumerate([axis] + others):
        db, i0 = profile_residual(means_b[axi], win_v)[1:]; da, _ = profile_residual(means_a[axi], win_v)[1:]
        x = np.arange(i0, i0 + db.size)
        ax[2, c].plot(x, db, lw=0.7, label=f"before std {fmt(sb[axi])}"); ax[2, c].plot(x[:da.size], da, lw=0.7, label=f"after std {fmt(sa[axi])}", alpha=0.85)
        ax[2, c].axhline(0, color="k", lw=0.5); ax[2, c].legend(fontsize=9); ax[2, c].set_xlabel(f"{ax_names[axi]} plane index (octv)")
        ax[2, c].set_title(("STRIPE AXIS " if axi == axis else "") + f"{ax_names[axi]}: per-plane mean inside eroded mask, detrended ({win_v} planes)")
    fig.suptitle(f"test_destripe {a.work.name}: NCC medians z/y/x {np.round(dinfo['ncc_median_per_axis'], 3).tolist()} ratio {dinfo['ratio']:.2f} -> axis {axis}; period {dinfo['period_vox'] and round(dinfo['period_vox'], 2)} vox; "
                 f"oct150 {ax_names[axis]} resid {fmt(stats150_b[axis]['resid_std'], 0)} -> {fmt(stats150_a[axis]['resid_std'], 0)}; {'ALL PASS' if R['all_pass'] else 'FAILURES: ' + ', '.join(k for k, v in checks.items() if v is False)}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(a.out / "test_destripe.png", dpi=70); plt.close(fig)
    say("wrote", a.out / "test_destripe.png", a.out / "test_destripe.json"); return 0 if R["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
