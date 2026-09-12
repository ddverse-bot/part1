"""3D U-Net structural parser: any-contrast volume -> {background, WM, infragranular GM, supragranular GM}.

Trained on synthetic images generated from the whole-hemisphere GM/WM labels (see synth.py) plus a
fraction of real MRI patches, so that at inference it parses *both* the MRI and the (downsampled)
OCT block into the same anatomical classes.  The class probabilities are the registration features.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synth import N_CLASSES, random_block_cut, random_flip_perm, standardize, synthesize


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, cin=1, cout=N_CLASSES, chans=(16, 32, 64, 128)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = cin
        for ch in chans:
            self.enc.append(ConvBlock(c, ch)); c = ch
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(len(chans) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(chans[i], chans[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(chans[i - 1] * 2, chans[i - 1]))
        self.head = nn.Conv3d(chans[0], cout, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool3d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            s = skips.pop()
            x = dec(torch.cat([x, s], 1))
        return self.head(x)


def soft_dice_loss(logits, target, n_classes=N_CLASSES, eps=1.0):
    p = torch.softmax(logits, 1)
    onehot = F.one_hot(target.clamp(min=0), n_classes).permute(0, 4, 1, 2, 3).float()
    valid = (target >= 0).float()[:, None]
    p = p * valid; onehot = onehot * valid
    inter = (p * onehot).sum((0, 2, 3, 4)); den = p.sum((0, 2, 3, 4)) + onehot.sum((0, 2, 3, 4))
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice[1:].mean()


class PatchSampler:
    """Random label/MRI patches from the whole-hemisphere volumes (numpy, CPU) -> GPU tensors."""

    def __init__(self, labels: np.ndarray, mri: np.ndarray | None, patch: int = 96, tissue_mask: np.ndarray | None = None,
                 centre_labels=(2, 3), n_centres: int = 200_000, seed: int = 0, exclude_ijk=None):
        """exclude_ijk: optional (lo, hi) voxel box; patch centres inside it are never sampled (held-out region)."""
        self.labels = labels; self.mri = mri; self.patch = patch; self.tissue = tissue_mask
        rng = np.random.default_rng(seed)
        cand = np.argwhere(np.isin(labels[::4, ::4, ::4], centre_labels)) * 4     # cortex-centred patches
        if exclude_ijk is not None:
            lo, hi = np.asarray(exclude_ijk[0]), np.asarray(exclude_ijk[1])
            keep = ~np.all((cand >= lo - patch // 2) & (cand <= hi + patch // 2), axis=1)
            cand = cand[keep]
        self.centres = cand[rng.choice(len(cand), size=min(n_centres, len(cand)), replace=False)]
        self.rng = rng
        self.shape = np.array(labels.shape)

    def sample(self, n: int, with_mri: bool, device):
        P = self.patch
        L = np.zeros((n, P, P, P), dtype=np.int64)
        I = np.zeros((n, 1, P, P, P), dtype=np.float32) if with_mri else None
        for b in range(n):
            while True:
                c = self.centres[self.rng.integers(len(self.centres))]
                lo = c - P // 2 + self.rng.integers(-P // 4, P // 4 + 1, size=3)
                lo = np.clip(lo, 0, self.shape - P)
                hi = lo + P
                lab = self.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                if (lab > 0).mean() > 0.15:
                    break
            L[b] = lab
            if with_mri:
                m = self.mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
                if self.tissue is not None:
                    # label 0 that is actually tissue (unlabelled deep WM etc.) -> ignore index -1
                    t = self.tissue[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                    L[b][(lab == 0) & t] = -1
                I[b, 0] = m
        Lt = torch.from_numpy(L).to(device)
        It = torch.from_numpy(I).to(device) if with_mri else None
        return Lt, It


def train_parser(labels: np.ndarray, mri: np.ndarray | None, out_dir: Path, iters: int = 4000, batch: int = 2,
                 patch: int = 96, p_real: float = 0.3, lr: float = 1e-3, device="cuda", tissue_mask=None,
                 log_every: int = 50, seed: int = 0, exclude_ijk=None, p_invert: float = 0.0) -> Path:
    """p_invert: probability of inverting the intensity polarity of a real-MRI training patch (contrast-agnostic parser:
    the target modality may have WM brighter or darker than GM).  This is an augmentation of REAL images, not synthesis."""
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    net = UNet3D().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05, final_div_factor=20)
    scaler = torch.cuda.amp.GradScaler()
    sampler = PatchSampler(labels, mri, patch=patch, tissue_mask=tissue_mask, seed=seed, exclude_ijk=exclude_ijk)
    log = open(out_dir / "train_log.txt", "a")
    t0 = time.time(); run_loss = 0.0; run_dice = 0.0
    for it in range(1, iters + 1):
        use_real = (mri is not None) and (torch.rand(()) < p_real)
        L, I = sampler.sample(batch, with_mri=use_real, device=device)
        if use_real:
            L, I = random_flip_perm(L, I)
            # augmentation of the REAL MRI only (no synthetic images): bias field, gain, gamma, blur, noise
            from .synth import random_bias_field, gaussian_blur3d
            I = I * random_bias_field(I.shape, device, strength=(0.0, 0.3))
            lo_, hi_ = torch.quantile(I.flatten(1)[:, ::13], 0.005, dim=1), torch.quantile(I.flatten(1)[:, ::13], 0.995, dim=1)
            I = ((I - lo_.reshape(-1, 1, 1, 1, 1)) / (hi_ - lo_ + 1e-6).reshape(-1, 1, 1, 1, 1)).clamp(0, 1)
            if p_invert > 0:
                # invert the polarity INSIDE the tissue only (background stays dark): a WM-bright / GM-dark contrast
                inv = (torch.rand(batch, 1, 1, 1, 1, device=device) < p_invert).float()
                fg = (I > 0.04).float()
                I = inv * (fg * (1.0 - I)) + (1.0 - inv) * I
            gam = torch.exp(0.3 * torch.randn(batch, 1, 1, 1, 1, device=device))
            I = I ** gam
            for b in range(batch):
                sig = float(torch.rand(()) * 1.2)
                I[b:b + 1] = gaussian_blur3d(I[b:b + 1], sig)
            x = standardize(I)
            x = x + 0.05 * torch.rand(()) * torch.randn_like(x)
        else:
            L, _ = random_flip_perm(L)
            L = random_block_cut(L, p=0.6)
            x = synthesize(L, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(x)
        logits = logits.float()
        ce = F.cross_entropy(logits, L, ignore_index=-1)
        dl = soft_dice_loss(logits, L)
        loss = ce + dl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run_loss += loss.item(); run_dice += 1 - dl.item()
        if it % log_every == 0:
            msg = f"it {it:5d}  loss {run_loss / log_every:.4f}  fgDice {run_dice / log_every:.3f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            run_loss = run_dice = 0.0
        if it % 1000 == 0 or it == iters:
            torch.save({"state_dict": net.state_dict(), "iters": it, "chans": (16, 32, 64, 128)}, out_dir / "parser.pt")
    log.close()
    return out_dir / "parser.pt"


def load_parser(path: Path, device="cuda") -> UNet3D:
    ck = torch.load(str(path), map_location=device)
    net = UNet3D(chans=tuple(ck.get("chans", (16, 32, 64, 128)))).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net


@torch.no_grad()
def parse_volume(net: UNet3D, vol: np.ndarray | torch.Tensor, device="cuda", patch: int = 128, stride: int = 96,
                 batch: int = 4, out_dtype=torch.float16, norm: str = "window", min_fg: float = 0.05) -> torch.Tensor:
    """Sliding-window inference on a whole volume [D,H,W] (any size) -> probabilities [C,D,H,W] (CPU tensor).

    norm="window": every window is z-scored on its own (exactly like the training patches, which are z-scored per
    patch); windows whose foreground fraction (voxels above an Otsu threshold of the volume) is below `min_fg` are
    air and get background probability 1 without running the network.  norm="global": z-score the whole volume once
    (old behaviour; mislabels air-dominated windows as tissue because their statistics never occur in training)."""
    x = torch.as_tensor(np.asarray(vol), dtype=torch.float32)
    fg_thr = None
    if norm == "global":
        x = standardize(x[None, None])[0, 0]                 # [D,H,W]
    else:
        # foreground level: a quarter of the (outlier-robust) bright end; air / noise floor is far below in ex-vivo MRI
        sub = x[::4, ::4, ::4].flatten()
        fg_thr = 0.25 * float(torch.quantile(sub[torch.randperm(sub.numel())[:2_000_000]], 0.995))
    D, H, W = x.shape
    P = patch
    pad = [max(0, P - D), max(0, P - H), max(0, P - W)]
    xp = F.pad(x[None, None], (0, pad[2], 0, pad[1], 0, pad[0]), mode="constant", value=float(x.min()))[0, 0]
    Dp, Hp, Wp = xp.shape
    prob = torch.zeros((N_CLASSES, Dp, Hp, Wp), dtype=torch.float32)
    wsum = torch.zeros((1, Dp, Hp, Wp), dtype=torch.float32)
    # smooth blending window
    t = torch.linspace(-1, 1, P); w1 = (1 - t.abs()) * 0.9 + 0.1
    win = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :])
    starts = lambda n: sorted(set(list(range(0, max(n - P, 0) + 1, stride)) + [max(n - P, 0)]))
    coords = [(a, b, c) for a in starts(Dp) for b in starts(Hp) for c in starts(Wp)]
    bg_only = torch.zeros((N_CLASSES, P, P, P)); bg_only[0] = 1.0
    for s in range(0, len(coords), batch):
        chunk = coords[s:s + batch]
        xb = torch.stack([xp[a:a + P, b:b + P, c:c + P] for a, b, c in chunk])[:, None].to(device)
        if norm == "global":
            with torch.autocast("cuda", dtype=torch.float16):
                pb = torch.softmax(net(xb).float(), 1).cpu()
        else:
            fg = (xb > fg_thr).float().mean((1, 2, 3, 4))                       # foreground fraction per window
            pb = bg_only[None].repeat(len(chunk), 1, 1, 1, 1)
            run = fg >= min_fg
            if run.any():
                with torch.autocast("cuda", dtype=torch.float16):
                    pb[run.cpu()] = torch.softmax(net(standardize(xb[run])).float(), 1).cpu()
        for (a, b, c), p in zip(chunk, pb):
            prob[:, a:a + P, b:b + P, c:c + P] += p * win
            wsum[:, a:a + P, b:b + P, c:c + P] += win
    prob = (prob / wsum.clamp(min=1e-6))[:, :D, :H, :W]
    return prob.to(out_dtype)
