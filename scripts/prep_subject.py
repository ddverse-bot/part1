#!/usr/bin/env python3
"""Prepare one OCT-block / MRI pair for octreg (any subject, any resolution).  Writes cached arrays into --work:

  mri.npy, mri_affine.npy          MRI on its native grid (float32) + NIfTI affine (voxel -> world mm)
  mri_tissue.npy                   tissue mask (outlier-robust Otsu between air/fluid and tissue)
  labels4.npy                      [optional, evaluation only] 0 bg, 1 WM, 2 infra GM, 3 supra GM on the MRI grid
  ba_labels.npy                    [optional, evaluation only] the block-region manual labels (same coding)
  mri_vessels.npy                  [optional, evaluation only] manual MRI vessel label (bool)
  oct150.npy/_affine/_mask         OCT block at --target-mm (default 0.15 mm), slab-normalised, isotropic grid in the OCT frame
  octv.npy/_affine/_mask           OCT 'vessel level' (native if >= 24 um, else pooled to ~24 um), float16
  octv_vessels.npy                 our own OCT vessel segmentation at the vessel level (dark tubes 24-106 um, top-1% inside tissue)
  oct_ves_dist48_own.npy/_affine   EDT (um) to the nearest own OCT vessel on a ~48 um grid (evaluation)
  oct_ves_dist48_vesseg.npy        [optional] same from a provided OCT vessel segmentation (e.g. the dataset's ves_seg)
  oct_native_affine.npy, prep.json

OCT spacing comes from --oct-spacing-um (z,y,x) or the BIDS sidecar next to the OCT (PixelSize = [x,y,z])."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import (layout_affine, oct_slab_normalize, oct_tissue_mask, pool_mean_np, pooled_affine, resample_to_grid, to_t, write_json)
from octreg.specimen import oct_specimen_mask, choose_oct_mask, want_texture, mask_qc_png, erode_mm

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--mri", type=Path, required=True); ap.add_argument("--oct", type=Path, required=True)
ap.add_argument("--oct-spacing-um", type=str, default=None, help="z,y,x in um (default: sidecar PixelSize)")
ap.add_argument("--oct-layout", default="SPR", help="arbitrary RAS letters for the OCT array axes (the search covers all orientations)")
ap.add_argument("--labels-wholehemi", type=Path, default=None); ap.add_argument("--labels-ba", type=Path, default=None)
ap.add_argument("--mri-vessels", type=Path, default=None); ap.add_argument("--oct-vesseg", type=Path, default=None)
ap.add_argument("--target-mm", type=float, default=0.15); ap.add_argument("--vessel-level-um", type=float, default=24.0)
ap.add_argument("--oct-tissue-thresh", type=float, default=None, help="raw-intensity threshold for the slab normalisation (default: 0.5 x Otsu of a subsample)")
ap.add_argument("--mri-tissue-thresh", type=float, default=None, help="override the automatic MRI tissue threshold")
ap.add_argument("--skip-mri", action="store_true"); ap.add_argument("--skip-oct", action="store_true")
ap.add_argument("--oct-mask", choices=["intensity", "texture", "auto"], default="auto", help="0.15 mm OCT mask: intensity (v1 histogram threshold), texture (specimen.oct_specimen_mask), auto = texture iff V_intensity > 1.5 x V_MRI_tissue and it passes --mask-band")
ap.add_argument("--destripe", choices=["off", "on"], default="off", help="on = flat-field the section stripes of octv (octreg.destripe) and derive oct150 from the destriped octv")
ap.add_argument("--stripe-axis", default="auto", help="array axis of the section sheets for the destripe: auto (data-driven) | 0 | 1 | 2")
ap.add_argument("--skip-vessels", action="store_true", help="no Frangi / octv_vessels.npy / EDT (register.py then runs without the vascular stage)")
ap.add_argument("--mri-tissue-cm3", type=float, default=None, help="MRI tissue volume (cm3) for the auto mask rule when the MRI was not prepped into this --work")
ap.add_argument("--mask-band", type=str, default="0.5,1.75", help="accepted V_texture / V_MRI band 'lo,hi'")
ap.add_argument("--save-stripe-field", action="store_true", help="write octv_stripeS.npy (float16, diagnostics only)")
a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True); t0 = time.time()
info = json.load(open(a.work / "prep.json")) if (a.work / "prep.json").exists() else {}
def say(*x): print(*x, f"[{time.time()-t0:.0f}s]", flush=True)

def load_nii(p):
    import nibabel as nib
    img = nib.load(str(p)); return np.asanyarray(img.dataobj), np.asarray(img.affine)

def map_labels_to_grid(lab, A_lab, shape, A_ref):
    """Nearest-voxel mapping of a label volume onto the MRI grid (handles axis permutation / flips / offsets / resolution)."""
    M = np.linalg.inv(A_lab) @ A_ref                       # MRI voxel -> label voxel
    if np.allclose(np.abs(M[:3, :3]).sum(1), 1, atol=1e-3) and np.allclose(np.abs(M[:3, :3]).sum(0), 1, atol=1e-3) and np.allclose(np.round(M[:3, 3]), M[:3, 3], atol=1e-2):
        perm = [int(np.argmax(np.abs(M[r, :3]))) for r in range(3)]; sign = [int(np.sign(M[r, perm[r]])) for r in range(3)]; off = np.round(M[:3, 3]).astype(int)
        idx = [sign[r] * np.arange(shape[perm[r]]) + off[r] for r in range(3)]          # label index along label axis r as a function of the MRI index along axis perm[r]
        ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij", sparse=True); mri_idx = [ii, jj, kk]
        li = [idx[r][mri_idx[perm[r]]] for r in range(3)]
        valid = np.ones(shape, dtype=bool)
        for r in range(3): valid &= (li[r] >= 0) & (li[r] < lab.shape[r])
        vals = lab[np.clip(li[0], 0, lab.shape[0] - 1), np.clip(li[1], 0, lab.shape[1] - 1), np.clip(li[2], 0, lab.shape[2] - 1)]
        return np.where(valid, vals, 0).astype(np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    step = 64
    for i0 in range(0, shape[0], step):
        ii, jj, kk = np.meshgrid(np.arange(i0, min(shape[0], i0 + step)), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        v = np.stack([ii, jj, kk, np.ones_like(ii)], -1).reshape(-1, 4).astype(np.float64) @ M.T
        ijk = np.rint(v[:, :3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(lab.shape)), 1)
        vals = np.zeros(len(ijk), dtype=np.uint8); vals[ok] = lab[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        out[i0:i0 + ii.shape[0]] = vals.reshape(ii.shape)
    return out

# ------------------------------------------------------------------ MRI (+ optional annotations, evaluation only)
if not a.skip_mri:
    mri, A_mri = load_nii(a.mri); mri = mri.astype(np.float32)
    if mri.ndim == 4: mri = mri[..., 0]
    nbad = int((~np.isfinite(mri)).sum())
    if nbad: np.nan_to_num(mri, copy=False, nan=0.0, posinf=0.0, neginf=0.0); say(f"sanitized {nbad} non-finite MRI voxels -> 0")
    np.save(a.work / "mri.npy", mri); np.save(a.work / "mri_affine.npy", A_mri)
    sub = mri[::4, ::4, ::4]; sub = sub[sub > 0]
    from octreg.common import histogram_tissue_threshold
    thr, info_thr = histogram_tissue_threshold(sub)
    if a.mri_tissue_thresh is not None: thr = a.mri_tissue_thresh
    tissue = mri > thr
    np.save(a.work / "mri_tissue.npy", tissue)
    vox = np.linalg.norm(A_mri[:3, :3], axis=0)
    info["mri"] = {"file": str(a.mri), "shape": list(mri.shape), "voxel_mm": vox.round(4).tolist(), "tissue_thresh": thr, "tissue_fraction": float(tissue.mean()), **info_thr,
                   "intensity_p50_p99": [float(np.percentile(sub, 50)), float(np.percentile(sub, 99))]}
    say("MRI", mri.shape, "voxel mm", vox.round(3), "tissue thr", round(thr, 2), "tissue frac", round(float(tissue.mean()), 3))
    labels4 = None
    if a.labels_wholehemi and a.labels_wholehemi.exists():
        wh, A_wh = load_nii(a.labels_wholehemi); labels4 = map_labels_to_grid(np.asarray(wh).astype(np.uint8), A_wh, mri.shape, A_mri); del wh
        info["labels_wholehemi"] = str(a.labels_wholehemi)
    if a.labels_ba and a.labels_ba.exists():
        ba, A_ba = load_nii(a.labels_ba); ba = map_labels_to_grid(np.asarray(ba).astype(np.uint8), A_ba, mri.shape, A_mri)
        np.save(a.work / "ba_labels.npy", ba); info["labels_ba"] = str(a.labels_ba)
        if labels4 is None: labels4 = ba
        else:
            m = ba > 0; info["labels_agreement_wholehemi_vs_ba"] = float((labels4[m] == ba[m]).mean()) if m.any() else None
    if labels4 is not None:
        np.save(a.work / "labels4.npy", labels4.astype(np.uint8)); u, c = np.unique(labels4, return_counts=True)
        info["labels4_counts"] = dict(zip(u.tolist(), c.tolist())); say("labels4 counts", info["labels4_counts"])
    if a.mri_vessels and a.mri_vessels.exists():
        ves, A_ves = load_nii(a.mri_vessels); ves = map_labels_to_grid((np.asarray(ves) > 0).astype(np.uint8), A_ves, mri.shape, A_mri) > 0
        np.save(a.work / "mri_vessels.npy", ves); info["mri_vessels"] = {"file": str(a.mri_vessels), "n_voxels": int(ves.sum())}
        say("MRI vessel label voxels", int(ves.sum()))
    del mri

# ------------------------------------------------------------------ OCT
if not a.skip_oct:
    nii_oct = str(a.oct).endswith((".nii", ".nii.gz"))
    if str(a.oct).endswith(".npy"): raw = np.load(a.oct, mmap_mode="r")
    elif nii_oct:
        import nibabel as nib
        _im = nib.load(str(a.oct)); raw = np.asanyarray(_im.dataobj)
        if a.oct_spacing_um is None:
            sp = (np.asarray(_im.header.get_zooms()[:3]) * 1000.0)    # per array axis (i,j,k) -> treated as (z,y,x)
    else:
        import tifffile
        try: raw = tifffile.memmap(str(a.oct))                        # zero-copy for uncompressed (OME-)TIFF
        except Exception: raw = tifffile.imread(str(a.oct))
    if raw.ndim == 4: raw = raw[:, 0] if raw.shape[1] < raw.shape[-1] else raw[..., 0]
    if a.oct_spacing_um: sp = [float(x) for x in a.oct_spacing_um.split(",")]
    elif not nii_oct:
        side = json.load(open(a.oct.with_suffix("").with_suffix(".json") if a.oct.name.endswith((".ome.tiff", ".ome.tif")) else a.oct.with_suffix(".json")))
        px = side["PixelSize"]; sp = [float(px[2]), float(px[1]), float(px[0])] if len(px) == 3 else [float(px[1]), float(px[1]), float(px[0])]
    sp = np.array(sp); raw_shape = list(map(int, raw.shape)); say("OCT", raw.shape, raw.dtype, "spacing um (z,y,x)", sp.tolist())
    A_nat = layout_affine(raw.shape, sp / 1000.0, a.oct_layout); np.save(a.work / "oct_native_affine.npy", A_nat)
    thr_raw = a.oct_tissue_thresh
    if thr_raw is None:
        from skimage.filters import threshold_otsu
        s4 = raw[::4, ::8, ::8].astype(np.float32); s4 = s4[np.isfinite(s4) & (s4 > 0)]; thr_raw = 0.5 * float(threshold_otsu(s4))   # well below the tissue mode, above the background
    window = max(10, int(round(1200.0 / sp[0])))                 # ~1.2 mm of depth = one section period at 12 um
    octn, slab = oct_slab_normalize(raw, tissue_thresh=thr_raw, window=window, out_dtype=np.float16); del raw
    say("slab-normalised (window", window, "slices, raw tissue thr", round(thr_raw, 1), ")")
    # 0.15 mm level: anisotropic integer pooling, then resample onto an isotropic grid in the same frame
    k150 = np.maximum(1, np.round(a.target_mm * 1000 / sp)).astype(int)
    kv = np.maximum(1, np.ceil(a.vessel_level_um / sp - 1e-6)).astype(int)     # never finer than the vessel level (20 um data -> 40 um)
    if a.stripe_axis not in ("auto", "0", "1", "2"): raise SystemExit(f"--stripe-axis must be auto|0|1|2, got {a.stripe_axis}")
    band = tuple(float(x) for x in a.mask_band.split(","))
    V_mri = a.mri_tissue_cm3                                                   # MRI tissue volume (cm3) for the mask rule; from prep.json when the MRI was prepped into this --work
    if V_mri is None and all(k in info.get("mri", {}) for k in ("tissue_fraction", "shape", "voxel_mm")):
        V_mri = float(info["mri"]["tissue_fraction"] * np.prod(info["mri"]["shape"]) * np.prod(info["mri"]["voxel_mm"]) / 1000.0)
    vol150 = lambda m: float(m.sum()) * a.target_mm ** 3 / 1000.0             # cm3 at the 0.15 mm level
    mask_tex = None; V_tex = None; tex_info = None; ds_info = {"applied": False, "reason": "off"}; ds_diag = None; axis = None; provenance = "v1"
    def vgrid(m150):
        """nearest resample of a 0.15 mm mask onto the vessel grid (same construction as maskv below)."""
        M_ = np.linalg.inv(A150) @ A_v
        idx = [np.clip(np.rint(M_[r, r] * np.arange(octv.shape[r]) + M_[r, 3]).astype(int), 0, m150.shape[r] - 1) for r in range(3)]
        return m150[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
    def override_axis(diag, axis_det, axis_new):
        """--stripe-axis override: keep the detection record, but take period_vox / FFT peak from the OVERRIDDEN axis
        (detect_stripe_axis fills them for the detected axis only; same peak >= 3 x spectrum-median rule)."""
        diag = dict(diag); diag["axis_detected"] = axis_det; diag["axis_overridden"] = True
        snr = (diag.get("fft_peak_over_median_per_axis") or [None] * 3)[axis_new]; per = (diag.get("fft_period_per_axis") or [None] * 3)[axis_new]
        diag["fft_peak_over_median"] = snr; diag["period_vox"] = per if (snr is not None and np.isfinite(snr) and snr >= 3.0) else None
        return diag
    if a.destripe == "off":
        p150 = pool_mean_np(octn, k150); A_p = pooled_affine(A_nat, k150)
        shape150 = tuple(int(np.floor(p150.shape[i] * (sp[i] * k150[i]) / (a.target_mm * 1000))) for i in range(3))
        A150 = A_p.copy(); A150[:3, :3] = A_nat[:3, :3] / (sp[None, :] / 1000.0) * a.target_mm
        o150 = resample_to_grid(to_t(p150)[None], to_t(A_p), to_t(A150), shape150).cpu().numpy()[0]; del p150
        mask_int, thr150 = oct_tissue_mask(o150, thresh=None, closing_iter=2)
        np.save(a.work / "oct150.npy", o150.astype(np.float32)); np.save(a.work / "oct150_affine.npy", A150)
        say("oct150", shape150, "tissue frac", round(float(mask_int.mean()), 3))
        # vessel level (~24 um or native if coarser)
        octv = octn if kv.max() == 1 else pool_mean_np(octn, kv).astype(np.float16); A_v = pooled_affine(A_nat, kv); sp_v = sp * kv
        del octn
        V_int = vol150(mask_int)
        want_tex, want_reason = want_texture(a.oct_mask, V_int, V_mri)
        if want_tex:    # data-driven: never fires on the DANDI cortex blocks (V_int << 1.5 V_mri)
            say(f"texture specimen mask ({want_reason}; V_int {V_int:.2f} cm3, V_mri {V_mri})")
            mask_tex, _, tex_info = oct_specimen_mask(octv, A_v, o150, A150, a.work, n_workers=4); V_tex = tex_info["V_cm3_150"]
            say(f"texture mask {V_tex:.2f} cm3 (W {tex_info['V_cm3_W']:.2f}, octv {tex_info['V_cm3_v']:.2f}), thr {tex_info['thr']:.4g}, n_cc {tex_info['n_cc_before_largest']}, {tex_info['seconds_total']:.0f} s")
    else:
        # a. vessel level first (the destripe works on octv; octn is freed before anything else is allocated)
        octv = octn if kv.max() == 1 else pool_mean_np(octn, kv).astype(np.float16); A_v = pooled_affine(A_nat, kv); sp_v = sp * kv
        del octn
        # b. oct150 must be a block mean of octv blocks (kv | k150) so that its grid equals v1's
        if not (k150 % kv == 0).all(): raise SystemExit(f"--destripe on: k150 {k150.tolist()} not divisible by kv {kv.tolist()} for spacing {sp.tolist()} um")
        # c. 0.15 mm level from octv: pooled_affine(A_nat, k150) == v1's A_p (block means and centre offsets coincide)
        A_p = pooled_affine(A_nat, k150)
        def level150(vol_v):
            p150 = pool_mean_np(vol_v, k150 // kv)
            shape150 = tuple(int(np.floor(p150.shape[i] * (sp[i] * k150[i]) / (a.target_mm * 1000))) for i in range(3))
            A150 = A_p.copy(); A150[:3, :3] = A_nat[:3, :3] / (sp[None, :] / 1000.0) * a.target_mm
            return shape150, A150, resample_to_grid(to_t(p150)[None], to_t(A_p), to_t(A150), shape150).cpu().numpy()[0]
        shape150, A150, o150_nd = level150(octv)
        np.save(a.work / "oct150_nodestripe.npy", o150_nd.astype(np.float32))
        mask_int_nd, _ = oct_tissue_mask(o150_nd, thresh=None, closing_iter=2); V_int_nd = vol150(mask_int_nd)
        say("oct150 (no destripe)", shape150, "tissue frac", round(float(mask_int_nd.mean()), 3))
        # d. texture mask on the ORIGINAL octv (P1 validated it on un-destriped data)
        want_tex, want_reason = want_texture(a.oct_mask, V_int_nd, V_mri)
        if want_tex:
            say(f"texture specimen mask ({want_reason}; V_int {V_int_nd:.2f} cm3, V_mri {V_mri})")
            mask_tex, _, tex_info = oct_specimen_mask(octv, A_v, o150_nd, A150, a.work, n_workers=4); V_tex = tex_info["V_cm3_150"]
            say(f"texture mask {V_tex:.2f} cm3 (W {tex_info['V_cm3_W']:.2f}, octv {tex_info['V_cm3_v']:.2f}), thr {tex_info['thr']:.4g}, n_cc {tex_info['n_cc_before_largest']}, {tex_info['seconds_total']:.0f} s")
        # e. stripe axis on the chosen mask eroded 2 mm, nearest-resampled to the octv grid
        m_diag = choose_oct_mask(mask_int_nd, V_int_nd, mask_tex, V_tex, V_mri, a.oct_mask, band)[0]
        maskv_diag = vgrid(erode_mm(m_diag, 2.0, a.target_mm)); del m_diag
        from octreg import destripe as _ds
        axis, ds_diag = _ds.detect_stripe_axis(octv, mask=maskv_diag, n_planes=40, hp_sigma_vox=3.0)
        if a.stripe_axis != "auto": ds_diag = override_axis(ds_diag, axis, int(a.stripe_axis)); axis = int(a.stripe_axis)
        say("stripe axis", axis, {k: v for k, v in ds_diag.items()})
        if axis is None:
            ds_info = {"applied": False, "reason": "axis_not_decisive", "detect": ds_diag}; provenance = "from_octv_nodestripe"
            say("WARNING: stripe axis not decisive (min adjacent-plane NCC >= 0.8 x second) -> destripe SKIPPED; oct150 = pooled un-destriped octv")
            o150 = o150_nd; mask_int = mask_int_nd
        else:
            # f. flat-field in place (zeros stay zero, values clipped to [1, cap] before the float16 write-back)
            ds_info = _ds.destripe_inplace(octv, axis=axis, vox_mm=float(sp_v.mean()) / 1000.0, sigma_smooth_mm=0.48, sigma_hp_mm=0.24, cap=60000.0, zchunk=64, pad=40, n_workers=4,
                                           mask=maskv_diag, save_S=(a.work / "octv_stripeS.npy" if a.save_stripe_field else None))
            ds_info["detect"] = ds_diag; provenance = "from_destriped_octv"
            say("destriped octv:", {k: v for k, v in ds_info.items() if k not in ("detect",)})
            # g. oct150 from the destriped octv (same grid); intensity mask on it
            _, _, o150 = level150(octv)
            mask_int, thr150 = oct_tissue_mask(o150, thresh=None, closing_iter=2)
        del maskv_diag, o150_nd, mask_int_nd
        np.save(a.work / "oct150.npy", o150.astype(np.float32)); np.save(a.work / "oct150_affine.npy", A150)
        V_int = vol150(mask_int)
        say("oct150", shape150, "tissue frac", round(float(mask_int.mean()), 3), f"({provenance})")
    # ---- mask choice (v1 path: no texture mask computed -> resolves to mask_int) and saves; oct150_mask.npy = the CHOSEN mask
    mask150, mask_mode, mask_reason = choose_oct_mask(mask_int, V_int, mask_tex, V_tex, V_mri, a.oct_mask, band)
    if a.oct_mask == "auto" and V_mri is None: say("WARNING: auto mask rule needs the MRI volume (prep the MRI into this --work or pass --mri-tissue-cm3); using intensity")
    np.save(a.work / "oct150_mask.npy", mask150); np.save(a.work / "oct150_mask_intensity.npy", mask_int)
    if mask_tex is not None: np.save(a.work / "oct150_mask_texture.npy", mask_tex)
    try:    # diagnostic figure only: a matplotlib / font-cache failure must never abort the prep
        mask_qc_png(o150, mask_int, mask150, a.work / "oct150_mask_qc.png", title=f"{a.work.name}: mask {mask_mode} ({mask_reason}); V_int {V_int:.2f} cm3, V_tex {V_tex if V_tex is None else round(V_tex, 2)}, V_mri {V_mri if V_mri is None else round(V_mri, 2)} cm3")
    except Exception as e: say(f"WARNING: oct150_mask_qc.png not written ({type(e).__name__}: {e})")
    say(f"oct150 mask: {mask_mode} ({mask_reason}); V_int {V_int:.2f} cm3, V_tex {V_tex}, V_mri {V_mri}, chosen frac {float(mask150.mean()):.3f}")
    # tissue mask at the vessel level: the (cleaned) 0.15 mm mask resampled nearest onto the vessel grid (no full-res morphology)
    M_ = np.linalg.inv(A150) @ A_v                                   # vessel-grid voxel -> 0.15 mm voxel (same axes, diagonal + offset)
    idx = [np.clip(np.rint(M_[r, r] * np.arange(octv.shape[r]) + M_[r, 3]).astype(int), 0, mask150.shape[r] - 1) for r in range(3)]
    maskv = mask150[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
    np.save(a.work / "octv.npy", octv); np.save(a.work / "octv_affine.npy", A_v); np.save(a.work / "octv_mask.npy", maskv)
    say("octv", octv.shape, "spacing um", sp_v.round(1).tolist(), "tissue frac", round(float(maskv.mean()), 3))
    if a.destripe == "off" and mask_tex is not None and ds_diag is None:       # diagnostic stripe axis (no destripe) for the vessel section-phase check whenever the texture mask was computed (forced OR auto-selected; never on the v1 default path)
        try:
            from octreg import destripe as _ds
            axis, ds_diag = _ds.detect_stripe_axis(octv, mask=vgrid(erode_mm(mask150, 2.0, a.target_mm)), n_planes=40, hp_sigma_vox=3.0)
            if a.stripe_axis != "auto": ds_diag = override_axis(ds_diag, axis, int(a.stripe_axis)); axis = int(a.stripe_axis)
            ds_info["detect"] = ds_diag; say("stripe axis (diagnostic only)", axis, ds_diag)
        except ImportError: say("octreg.destripe not available: no stripe-axis diagnostic")
    # evaluation EDT on a ~48 um grid (max-pool of the mask)
    from scipy import ndimage
    def edt48(mask, A_mask, sp_mask):
        k = np.maximum(1, np.round(48.0 / sp_mask)).astype(int)
        z, y, x = (mask.shape[i] // k[i] * k[i] for i in range(3))
        pooled = mask[:z, :y, :x].reshape(z // k[0], k[0], y // k[1], k[1], x // k[2], k[2]).max(axis=(1, 3, 5))
        dist = ndimage.distance_transform_edt(~pooled, sampling=tuple(sp_mask * k)).astype(np.float32)   # um
        return dist, pooled_affine(A_mask, k), float(pooled.mean())
    vmask = None; fr = None; sig_vox = None; gam = None
    if not a.skip_vessels:
        # own vessel segmentation at the vessel level: dark tubes of 24-106 um (sigmas in um -> voxels of the in-plane spacing)
        from octreg.vascular import frangi_chunked
        from octreg.features import frangi_gamma
        sig_um = (24.0, 36.0, 53.0, 72.0, 106.0); sig_vox = tuple(float(s / np.mean(sp_v)) for s in sig_um)
        D = octv.shape[0]; ves = np.zeros(octv.shape, np.float16); ov = int(np.ceil(3 * max(sig_vox))) + 1
        zc = int(np.clip(30e6 // (octv.shape[1] * octv.shape[2]) - 2 * ov, 8, 64))   # slab sized to a ~30M-voxel GPU budget
        with torch.no_grad():   # one structureness constant per scale for the whole volume (from a strided subsample of slabs)
            nsl = int(np.clip(20e6 // (octv.shape[1] * octv.shape[2]), 4, 8))
            _z0s = np.linspace(0, max(0, D - nsl), 6).astype(int); _sub = np.concatenate([octv[z:z + nsl] for z in _z0s], 0).astype(np.float32)   # 6 full-res slabs
            gam = frangi_gamma(to_t(_sub)[None], sig_vox); del _sub
            for z0 in range(0, D, zc):
                lo_, hi_ = max(0, z0 - ov), min(D, z0 + zc + ov)
                vg = frangi_chunked(to_t(octv[lo_:hi_].astype(np.float32)), sig_vox, zchunk=hi_ - lo_ + 1, overlap=0, gamma=gam)
                ves[z0:min(D, z0 + zc)] = vg[z0 - lo_:z0 - lo_ + min(zc, D - z0)].cpu().numpy().astype(np.float16); del vg
        torch.cuda.empty_cache()
        st = max(1, int(round((maskv.size / 20_000_000) ** (1 / 3)))); vt = ves[::st, ::st, ::st][maskv[::st, ::st, ::st]].astype(np.float32); thr_v = float(np.percentile(vt, 99.0)); del vt
        vmask = (ves > thr_v); vmask &= maskv; del ves
        np.save(a.work / "octv_vessels.npy", vmask)
        say("own OCT vessels (top-1% dark-tube vesselness inside tissue):", int(vmask.sum()), "voxels; sigmas vox", np.round(sig_vox, 2).tolist())
        d, A48, fr = edt48(vmask, A_v, sp_v); np.save(a.work / "oct_ves_dist48_own.npy", d); np.save(a.work / "oct_ves_dist48_own_affine.npy", A48)
    else:
        say("vessel block skipped (--skip-vessels): no octv_vessels.npy / EDT")
    info["oct"] = {"file": str(a.oct), "shape_native": raw_shape, "spacing_um_zyx": sp.tolist(), "layout": a.oct_layout,
                   "raw_tissue_thresh": thr_raw, "slab_window": window, "shape150": list(shape150), "tissue_frac150": float(mask150.mean()),
                   "vessel_level_spacing_um": sp_v.tolist(), "shape_v": list(octv.shape), "own_vessel_voxels": None if vmask is None else int(vmask.sum()), "own_vessel_frac48": fr,
                   "frangi_sigmas_vox": sig_vox, "frangi_gamma": None if gam is None else list(gam),
                   "per_slice_median_head": [round(x, 1) if x == x else None for x in slab["per_slice_median"][:5]],
                   "mask_mode": mask_mode, "mask_reason": mask_reason, "tissue_frac150_intensity": float(mask_int.mean()),
                   "tissue_frac150_texture": None if mask_tex is None else float(mask_tex.mean()), "V_oct_cm3_int": V_int, "V_oct_cm3_tex": V_tex, "V_mri_cm3": V_mri,
                   "mask_band": list(band), "texture": tex_info, "destripe": ds_info, "oct150_provenance": provenance, "vessels_skipped": bool(a.skip_vessels)}
    if a.destripe == "on": info["oct"]["V_oct_cm3_int_nodestripe"] = V_int_nd
    if vmask is not None and ds_diag is not None and axis is not None and ds_diag.get("period_vox") is not None:   # vessel density vs section phase (I58: ratio 2.54)
        mod = _ds.section_phase_modulation(vmask, maskv, axis=axis, period_vox=ds_diag["period_vox"])
        info["oct"]["vessel_section_modulation"] = {"ratio": mod["ratio"], "var_frac": mod["var_frac"], "period_vox": ds_diag["period_vox"], "axis": axis}
        say("vessel section-phase modulation:", info["oct"]["vessel_section_modulation"])
    elif vmask is not None and ds_diag is not None:     # key stays absent (register.py then behaves as v1), but never silently
        say("vessel section-phase diagnostic skipped:", "stripe axis not decisive" if axis is None else f"no FFT period on axis {axis} (peak/median {ds_diag.get('fft_peak_over_median')})")
    if a.oct_vesseg and a.oct_vesseg.exists():
        import tifffile
        seg = tifffile.imread(str(a.oct_vesseg)) > 0
        # the segmentation may have fewer slices than the stack; assume alignment at z=0 (as verified on sub-I46)
        full = np.zeros((raw_shape[0], seg.shape[1], seg.shape[2]), dtype=bool)
        nz = min(full.shape[0], seg.shape[0]); full[:nz] = seg[:nz]; del seg
        A_seg = A_nat; sp_seg = sp
        d2, A2, fr2 = edt48(full, A_seg, sp_seg); np.save(a.work / "oct_ves_dist48_vesseg.npy", d2); np.save(a.work / "oct_ves_dist48_vesseg_affine.npy", A2)
        info["oct"]["vesseg"] = {"file": str(a.oct_vesseg), "frac48": fr2, "z_slices": int(nz)}; say("provided OCT vessel segmentation EDT saved, frac", round(fr2, 5))
    del octv
write_json(info, a.work / "prep.json")
say("done")
