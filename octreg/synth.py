"""Synthetic image generator (SynthSeg-style) with OCT-specific degradations.

Given a label patch (0 = background/other, 1 = WM, 2 = infragranular GM, 3 = supragranular GM) we
draw a random image whose only stable content is the *geometry* of the tissue classes:

  * random per-class intensity (Gaussian mixture, so any contrast/polarity: MRI-like or OCT-like)
  * random block cut (labels outside a random box -> background)      -> tissue-block field of view
  * multiplicative speckle (Gamma)                                     -> OCT
  * sawtooth depth attenuation along a random axis with random period  -> serial-sectioning OCT slabs
  * smooth multiplicative bias field                                   -> MRI B1 / OCT illumination
  * random Gaussian blur, anisotropic resolution loss, gamma, additive noise

Everything runs on the GPU in torch so it can be sampled on the fly during training.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_CLASSES = 4  # bg, WM, infra GM, supra GM


def _rand(lo, hi, size=(), device="cuda"):
    return torch.rand(size, device=device) * (hi - lo) + lo


def random_block_cut(labels: torch.Tensor, p: float = 0.6) -> torch.Tensor:
    """labels: [B,D,H,W] long. With prob p per sample, zero everything outside a random box."""
    B, D, H, W = labels.shape
    out = labels.clone()
    for b in range(B):
        if torch.rand(()) < p:
            keep = torch.ones((D, H, W), dtype=torch.bool, device=labels.device)
            for ax, n in enumerate((D, H, W)):
                frac = float(_rand(0.35, 1.0))
                size = max(8, int(n * frac))
                start = int(torch.randint(0, n - size + 1, ()))
                sl = [slice(None)] * 3
                sl[ax] = slice(start, start + size)
                m = torch.zeros((D, H, W), dtype=torch.bool, device=labels.device)
                m[tuple(sl)] = True
                keep &= m
            out[b][~keep] = 0
    return out


def gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: [B,1,D,H,W]; separable Gaussian, sigma in voxels (per-batch scalar)."""
    if sigma < 0.05:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2); k = k / k.sum()
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + ax] = -1
        kk = k.reshape(shape)
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax)] = r; pad[2 * (2 - ax) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), kk)
    return x


def random_bias_field(shape, device, strength=(0.0, 0.5), grid=(3, 7)) -> torch.Tensor:
    B = shape[0]
    g = int(torch.randint(grid[0], grid[1] + 1, ()))
    field = torch.randn((B, 1, g, g, g), device=device)
    field = F.interpolate(field, size=shape[2:], mode="trilinear", align_corners=True)
    s = _rand(*strength, size=(B, 1, 1, 1, 1), device=device)
    return torch.exp(s * field / (field.flatten(1).std(1).reshape(B, 1, 1, 1, 1) + 1e-6))


def sawtooth_attenuation(shape, device, p=0.5) -> torch.Tensor:
    """Per-sample sawtooth exp(-beta * frac(z/period)) along a random axis (serial-sectioning OCT)."""
    B, _, D, H, W = shape
    out = torch.ones(shape, device=device)
    for b in range(B):
        if torch.rand(()) < p:
            ax = int(torch.randint(0, 3, ()))
            n = (D, H, W)[ax]
            period = float(_rand(4.0, 14.0))          # voxels (0.6-2.1 mm at 0.15 mm)
            phase = float(_rand(0.0, period))
            beta = float(_rand(0.2, 1.4))
            z = torch.arange(n, device=device, dtype=torch.float32)
            frac = ((z + phase) % period) / period
            prof = torch.exp(-beta * frac)
            shape_b = [1, 1, 1]; shape_b[ax] = n
            out[b, 0] = prof.reshape(shape_b)
    return out


@torch.no_grad()
def synthesize(labels: torch.Tensor, device="cuda") -> torch.Tensor:
    """labels: [B,D,H,W] long in {0..3} -> image [B,1,D,H,W] float32, per-sample standardized."""
    B, D, H, W = labels.shape
    x = torch.zeros((B, 1, D, H, W), device=device)
    for b in range(B):
        # per-class means; sometimes make the two GM layers nearly identical, sometimes distinct
        means = _rand(0.0, 1.0, size=(N_CLASSES,), device=device)
        if torch.rand(()) < 0.5:
            means[3] = means[2] + _rand(-0.08, 0.08)
        if torch.rand(()) < 0.7:          # background darkest most of the time (OCT/MRI both)
            means[0] = _rand(0.0, 0.15)
        stds = _rand(0.02, 0.15, size=(N_CLASSES,), device=device)
        lb = labels[b]
        img = means[lb] + stds[lb] * torch.randn_like(lb, dtype=torch.float32)
        # speckle (multiplicative gamma noise), OCT-like
        if torch.rand(()) < 0.6:
            k = float(_rand(2.0, 25.0))
            g = torch.distributions.Gamma(torch.tensor(k, device=device), torch.tensor(k, device=device)).sample(lb.shape)
            img = img * g
        x[b, 0] = img
    # slab attenuation, bias field
    x = x * sawtooth_attenuation(x.shape, device, p=0.5)
    x = x * random_bias_field(x.shape, device, strength=(0.0, 0.4))
    # per-sample blur / resolution loss
    for b in range(B):
        sig = float(_rand(0.0, 1.5))
        x[b:b + 1] = gaussian_blur3d(x[b:b + 1], sig)
        if torch.rand(()) < 0.4:  # anisotropic resolution loss then upsample
            f = [float(_rand(1.0, 3.0)) for _ in range(3)]
            small = [max(4, int(round(s / fi))) for s, fi in zip((D, H, W), f)]
            x[b:b + 1] = F.interpolate(F.interpolate(x[b:b + 1], size=small, mode="trilinear", align_corners=False),
                                       size=(D, H, W), mode="trilinear", align_corners=False)
    # gamma + noise, then standardize per sample
    for b in range(B):
        v = x[b:b + 1]
        lo, hi = torch.quantile(v.flatten()[::7], 0.005), torch.quantile(v.flatten()[::7], 0.995)
        v = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        gam = float(_rand(0.6, 1.6))
        v = v ** gam
        v = v + float(_rand(0.0, 0.08)) * torch.randn_like(v)
        x[b:b + 1] = v
    return standardize(x)


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score over the whole patch/volume (same at train and inference)."""
    m = x.flatten(1).mean(1).reshape(-1, 1, 1, 1, 1)
    s = x.flatten(1).std(1).reshape(-1, 1, 1, 1, 1) + 1e-6
    return (x - m) / s


def random_flip_perm(labels: torch.Tensor, image: torch.Tensor | None = None):
    """Random axis permutation + flips applied identically to labels [B,D,H,W] and image [B,1,D,H,W]."""
    perm = torch.randperm(3).tolist()
    labels = labels.permute(0, *[1 + p for p in perm])
    if image is not None:
        image = image.permute(0, 1, *[2 + p for p in perm])
    for ax in range(3):
        if torch.rand(()) < 0.5:
            labels = labels.flip(1 + ax)
            if image is not None:
                image = image.flip(2 + ax)
    return labels.contiguous(), (image.contiguous() if image is not None else None)
