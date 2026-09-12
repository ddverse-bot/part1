#!/usr/bin/env python3
"""Cross-modal self-distillation (real data only): train an OCT structural parser on the REAL OCT block using
targets transferred from the MRI parser through the current registration T (OCT->MRI).  No synthetic images,
no OCT annotations.  Then re-register with parser-vs-parser features and report held-out metrics.

    python distill_oct_parser.py --work work/i46 --parser work/parser_real --T work/runs/crop_mixed_real/T_oct2mri.npy --out work/oct_parser_v1
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, grid_points, world_bbox_to_voxel, avg_pool_iso, pool_mean_np, oct_slab_normalize, resample_to_grid
from octreg.parser import UNet3D, soft_dice_loss, load_parser, parse_volume
from octreg.synth import standardize, random_flip_perm, gaussian_blur3d, random_bias_field
from octreg.features import parser_features
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap, transform_diff

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--dandi", type=Path, default=None, help="if given, also build a 0.072 mm OCT level from the OME-TIFF for texture-rich training")
ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--patch", type=int, default=64); ap.add_argument("--conf", type=float, default=0.65)
ap.add_argument("--rounds", type=int, default=2, help="distillation -> re-registration rounds")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
def say(*x): print(*x, flush=True)

A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); labels4 = np.load(a.work / "labels4.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
T = np.load(a.T)
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]
centre = (T @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 25, centre + 25)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
# MRI parser 4-class probabilities are not cached (only wm/gm/tissue); rebuild infra/supra split by re-parsing the region
net_mri = load_parser(a.parser / "parser.pt")
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
prob_mri = parse_volume(net_mri, mri_reg, patch=96, stride=64).float().to(dev)          # [4,D,H,W]
mri_tis = to_t(load_region("tissue") > 0.5)[None].float()

# --------------------------------------------------------- OCT training levels: 0.15 mm (given) and optionally 0.072 mm
levels = {0.15: (oct150.astype(np.float32), oct_mask, A_oct)}
if a.dandi:
    import tifffile
    oct12 = tifffile.imread(str(a.dandi / "sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff"))
    octn, _ = oct_slab_normalize(oct12, tissue_thresh=30.0, window=100); del oct12
    p6 = pool_mean_np(octn, 6); del octn                                                    # 0.072 mm
    A12 = np.load(a.work / "oct12_affine.npy"); A72 = A12.copy(); A72[:3, :3] *= 6; A72[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * 2.5)
    from octreg.common import oct_tissue_mask
    m72, _ = oct_tissue_mask(p6, thresh=None, closing_iter=2)
    levels[0.072] = (p6.astype(np.float32), m72, A72)
    say("built 0.072 mm OCT level", p6.shape)

def targets_for(vol_shape, A_lvl, T_cur):
    """Soft targets [4,D,H,W] on an OCT grid = MRI parser probabilities at the transformed positions."""
    pts = grid_points(to_t(A_lvl), vol_shape).reshape(-1, 3)
    pm = apply_affine(to_t(T_cur), pts)
    tg = torch.empty((4, pts.shape[0]), device=dev)
    for s in range(0, pts.shape[0], 4_000_000):
        tg[:, s:s + 4_000_000] = sample_at_world(prob_mri, to_t(A_reg), pm[s:s + 4_000_000])
    return tg.reshape(4, *vol_shape)

def train_round(T_cur, round_id):
    net = UNet3D().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.iters, pct_start=0.05, final_div_factor=20)
    scaler = torch.cuda.amp.GradScaler()
    data = []
    for sp, (vol, msk, A_l) in levels.items():
        tg = targets_for(vol.shape, A_l, T_cur)
        conf = tg.max(0).values
        mt = to_t(msk.astype(np.float32))
        # confident targets inside tissue; outside tissue -> background target; ambiguous voxels ignored
        valid = ((conf > a.conf) & (mt > 0.5)) | (mt <= 0.5)
        tgt = torch.where(mt > 0.5, tg.argmax(0), torch.zeros_like(tg.argmax(0)))
        tgt = torch.where(valid, tgt, torch.full_like(tgt, -1))
        x = to_t(vol)
        # pad every level to at least one patch per axis (the 0.15 mm block is only ~50 voxels deep): outside = background
        pad = [max(0, a.patch - s_) for s_ in x.shape]
        if any(pad):
            pd = (0, pad[2], 0, pad[1], 0, pad[0])
            x = F.pad(x, pd, value=float(x[mt > 0.5].min())); tgt = F.pad(tgt, pd, value=0); mt = F.pad(mt, pd, value=0.0)
        say(f"  level {sp} mm: targets valid {valid.float().mean().item():.2f}, class fractions (valid tissue) {[round(float(((tgt == c) & (mt > 0.5)).sum() / max((mt > 0.5).sum().item(), 1)), 3) for c in range(4)]}")
        data.append((x, tgt, mt))
    P = a.patch; t0 = time.time()
    for it in range(1, a.iters + 1):
        x, tgt, mt = data[it % len(data)]
        D, H, W = x.shape
        # random tissue-centred patch
        while True:
            c = torch.tensor([np.random.randint(P // 2, max(D - P // 2, P // 2 + 1)), np.random.randint(P // 2, max(H - P // 2, P // 2 + 1)), np.random.randint(P // 2, max(W - P // 2, P // 2 + 1))])
            l = (c - P // 2).clamp(min=0); h = l + P
            if h[0] <= D and h[1] <= H and h[2] <= W and mt[l[0]:h[0], l[1]:h[1], l[2]:h[2]].mean() > 0.2:
                break
        xb = x[l[0]:h[0], l[1]:h[1], l[2]:h[2]][None, None].clone(); tb = tgt[l[0]:h[0], l[1]:h[1], l[2]:h[2]][None].clone()
        tb, xb = random_flip_perm(tb, xb)
        # real-data augmentation: gain, gamma, bias, blur, noise
        xb = xb * random_bias_field(xb.shape, dev, strength=(0.0, 0.3))
        lo_, hi_ = torch.quantile(xb.flatten()[::7], 0.005), torch.quantile(xb.flatten()[::7], 0.995)
        xb = ((xb - lo_) / (hi_ - lo_ + 1e-6)).clamp(0, 1) ** float(np.exp(0.3 * np.random.randn()))
        xb = gaussian_blur3d(xb, float(np.random.rand() * 1.0))
        xb = standardize(xb) + 0.05 * np.random.rand() * torch.randn_like(xb)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(xb)
        logits = logits.float()
        loss = F.cross_entropy(logits, tb, ignore_index=-1) + soft_dice_loss(logits, tb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if it % 250 == 0: say(f"  round {round_id} it {it} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
    torch.save({"state_dict": net.state_dict(), "chans": (16, 32, 64, 128)}, a.out / f"oct_parser_round{round_id}.pt")
    net.eval()
    prob = parse_volume(net, oct150, patch=64, stride=32).float().numpy()
    np.save(a.out / f"oct150_prob_round{round_id}.npy", prob.astype(np.float16))
    return net, prob

# --------------------------------------------------------- registration with parser-vs-parser features (reuse the demo machinery)
from octreg.search import FFTSearcher
from octreg.refine import Refiner
FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves))
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
def register_with(prob_oct, name):
    FO = parser_features(to_t(prob_oct), "wm_gm"); MO = to_t(oct_mask)[None].float()
    lv = {}
    for f in (4, 2, 1):
        if f == 1: lv[f] = (FM, A_reg, FO, A_oct, MO, mri_tis)
        else:
            a1, A1 = avg_pool_iso(FM, A_reg, f); a2, A2 = avg_pool_iso(FO, A_oct, f); a3, _ = avg_pool_iso(MO, A_oct, f); a4, _ = avg_pool_iso(mri_tis, A_reg, f); lv[f] = (a1, A1, a2, A2, a3, a4)
    srch = FFTSearcher(lv[4][0], lv[4][1], lv[4][5], lv[4][2], lv[4][3], lv[4][4], spacing=0.6)
    cands, info = srch.run(n_rot=4000, topk=16, seed=0, log_every=0, mirror=True)
    refs = {f: Refiner(lv[f][0], lv[f][1], lv[f][2], lv[f][3], lv[f][4]) for f in (4, 2, 1)}
    best = None
    for c in cands[:8]:
        Tc = c["T"]
        for f, dofs, it in ((4, ("rigid", "similarity"), 120), (2, ("rigid", "affine"), 200), (1, ("affine",), 200)):
            for dof in dofs: Tc, l = refs[f].refine(Tc, dof=dof, iters=it)
        if best is None or l < best[0]: best = (l, Tc)
    Tn = best[1]
    ev = vessel_distance_stats(Tn, ves_ijk, A_mri, d0, A0, n_ctrl=0)["registered"]
    oct_class = np.where(oct_mask, np.where(prob_oct.argmax(0) == 1, 1, np.where(prob_oct.argmax(0) >= 2, 2, 0)), 0)
    gw = gm_wm_overlap(Tn, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
    d = transform_diff(Tn, T, c_o)
    say(f"[{name}] search top1 {info['top1']:.3f} top2 {info['top2']:.3f} | final ncc {1-best[0]:.3f} | Dice(parser classes) WM {gw['dice_WM']:.3f} GM {gw['dice_GM']:.3f} | vessels med {ev.get('median_um',-1):.0f} f150 {ev.get('frac_within_150um',-1):.2f} | vs T0 centre {d['centre_mm']:.2f} mm rot {d['rotation_deg']:.2f} deg mirror {np.linalg.det(Tn[:3,:3])<0}")
    return Tn, {"search_top1": info["top1"], "search_top2": info["top2"], "final_ncc": 1 - best[0], "dice_WM": gw["dice_WM"], "dice_GM": gw["dice_GM"], "vessel_median_um": ev.get("median_um"), "vessel_f150": ev.get("frac_within_150um"), **{f"vs_T0_{k}": v for k, v in d.items()}, "T": Tn.tolist()}

log = {"rounds": []}
# baseline: the ORIGINAL (synthetic-free) MRI parser applied to OCT directly (parser-vs-parser without distillation)
prob0 = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
_, r0 = register_with(prob0, "round0: MRI-trained parser on OCT")
log["rounds"].append({"round": 0, **r0})
T_cur = T
for rd in range(1, a.rounds + 1):
    say(f"=== distillation round {rd} (targets from T with centre {np.round((T_cur @ np.r_[c_o,1])[:3],1).tolist()}) ===")
    net, prob = train_round(T_cur, rd)
    fr = [round(float((prob.argmax(0)[oct_mask] == c).mean()), 3) for c in range(4)]
    say(f"  OCT parser round {rd}: class fractions in tissue bg/WM/infra/supra {fr}")
    T_new, r = register_with(prob, f"round{rd}: distilled OCT parser")
    log["rounds"].append({"round": rd, "oct_class_fractions": fr, **r})
    T_cur = T_new
json.dump(log, open(a.out / "distill_log.json", "w"), indent=1, default=float)
np.save(a.out / "T_final.npy", T_cur)
say("saved", a.out)
