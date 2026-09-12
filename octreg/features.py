"""Registration representations for a volume [1,D,H,W] (+ tissue mask):

  parser : class probabilities from the trained structural parser (WM, GM[, infra, supra])
  mind   : MIND-SSC (Heinrich et al. 2013), 12 channels, training-free
  otsu   : two-class intensity split inside tissue (WM/GM by brightness; polarity given per modality)
  intensity : standardized intensity (single channel; only meaningful with matching polarity)

All return float32 tensors [C,D,H,W] on the same grid as the input.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- MIND-SSC
def _pdist_sq(x):  # x: [1,6,3]
    xx = (x * x).sum(-1, keepdim=True)
    return xx - 2 * x @ x.transpose(1, 2) + xx.transpose(1, 2)


def mind_ssc(img: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
    """img: [1,1,D,H,W] float -> [1,12,D,H,W]. Faithful re-implementation of MINDSSC (ConvexAdam repo)."""
    device = img.device
    kernel_size = radius * 2 + 1
    six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]], dtype=torch.long)
    dist = _pdist_sq(six.float().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    mask = ((x > y).reshape(-1) & (dist == 2).reshape(-1))
    idx1 = six.unsqueeze(1).repeat(1, 6, 1).reshape(-1, 3)[mask]
    idx2 = six.unsqueeze(0).repeat(6, 1, 1).reshape(-1, 3)[mask]
    m1 = torch.zeros(12, 1, 3, 3, 3); m2 = torch.zeros(12, 1, 3, 3, 3)
    m1.reshape(-1)[torch.arange(12) * 27 + idx1[:, 0] * 9 + idx1[:, 1] * 3 + idx1[:, 2]] = 1
    m2.reshape(-1)[torch.arange(12) * 27 + idx2[:, 0] * 9 + idx2[:, 1] * 3 + idx2[:, 2]] = 1
    m1, m2 = m1.to(device), m2.to(device)
    rpad1 = nn.ReplicationPad3d(dilation); rpad2 = nn.ReplicationPad3d(radius)
    xi = rpad1(img)
    ssd = F.avg_pool3d(rpad2((F.conv3d(xi, m1, dilation=dilation) - F.conv3d(xi, m2, dilation=dilation)) ** 2), kernel_size, stride=1)
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind = torch.exp(-mind / mind_var)
    return mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]), :, :, :]


# ----------------------------------------------------------------------------- Otsu tissue classes
def otsu_two_class(vol: torch.Tensor, tissue: torch.Tensor, wm_bright: bool, smooth_sigma_vox: float = 1.0):
    """vol [1,D,H,W], tissue [1,D,H,W] bool -> [2,D,H,W] soft (WM, GM) maps via a sigmoid around Otsu."""
    from skimage.filters import threshold_otsu
    x = vol.clone()
    if smooth_sigma_vox > 0:
        from .synth import gaussian_blur3d
        x = gaussian_blur3d(x[None], smooth_sigma_vox)[0]
    vals = x[tissue].detach().cpu().numpy()
    thr = float(threshold_otsu(vals)) if vals.size > 100 else float(np.median(vals)) if vals.size else 0.0
    scale = float(np.std(vals) + 1e-6) * 0.25
    s = torch.sigmoid((x - thr) / scale)
    wm = s if wm_bright else 1 - s
    wm = wm * tissue
    gm = (1 - wm) * tissue
    return torch.cat([wm, gm], 0), thr


def otsu_two_class_lowmem(vol_np: np.ndarray, tissue_np: np.ndarray, wm_bright: bool, smooth_sigma_vox: float = 1.0, zchunk: int = 64,
                          flatten_sigma_mm: float = 10.0, vox_mm: float = 0.15, device=None):
    """Two-class (WM/GM) soft split of a large MRI, label-free: the intensity is first flattened by its local tissue mean
    (Gaussian, `flatten_sigma_mm`; removes bias field / regional contrast drift), then an Otsu threshold on the flattened
    tissue intensities gives [P(WM), P(GM)] = sigmoid split inside the tissue mask.  Built chunk by chunk on the GPU
    (whole-hemisphere volumes).  Returns ([2,D,H,W] float32 on GPU, threshold)."""
    from skimage.filters import threshold_otsu
    from scipy.ndimage import gaussian_filter
    from .common import DEVICE, to_t, pool_mean_np
    device = device or DEVICE
    D, H, W = vol_np.shape
    # local tissue mean on a pooled grid (fast), upsampled to the full grid on the CPU
    pool = 4
    num = pool_mean_np(np.where(tissue_np, vol_np, 0).astype(np.float32), pool); den = pool_mean_np(tissue_np.astype(np.float32), pool)
    sig = flatten_sigma_mm / (vox_mm * pool)
    field = gaussian_filter(num, sig) / (gaussian_filter(den, sig) + 1e-3)
    floor = float(np.percentile(field[den > 0.5], 5)) * 0.5 if (den > 0.5).any() else 1e-3
    field = np.maximum(field, max(floor, 1e-3)).astype(np.float32)
    field_full = F.interpolate(torch.from_numpy(field)[None, None], size=(D, H, W), mode="trilinear", align_corners=False)[0, 0].numpy()
    sub = (vol_np[::3, ::3, ::3] / field_full[::3, ::3, ::3])[tissue_np[::3, ::3, ::3]].astype(np.float32)
    sub = np.clip(sub, 0, np.percentile(sub, 99.5)) if sub.size else sub
    thr = float(threshold_otsu(sub)) if sub.size > 100 else float(np.median(sub)) if sub.size else 1.0
    scale = float(np.std(sub) + 1e-6) * 0.25
    out = np.empty((2, D, H, W), dtype=np.float16)                  # CPU-resident; callers pool / crop what they need onto the GPU
    pad = int(3 * smooth_sigma_vox) + 1 if smooth_sigma_vox > 0 else 0
    for z0 in range(0, D, zchunk):
        a, b = max(0, z0 - pad), min(D, z0 + zchunk + pad)
        x = to_t(np.asarray(vol_np[a:b]).astype(np.float32) / field_full[a:b], device=device)[None]; t = to_t(np.asarray(tissue_np[a:b]), dtype=torch.bool, device=device)[None]
        if smooth_sigma_vox > 0:
            from .synth import gaussian_blur3d
            x = gaussian_blur3d(x[None], smooth_sigma_vox)[0]
        sg = torch.sigmoid((x - thr) / scale); wm = (sg if wm_bright else 1 - sg) * t
        out[0, z0:min(D, z0 + zchunk)] = wm[0, z0 - a:z0 - a + min(zchunk, D - z0)].half().cpu().numpy()
        out[1, z0:min(D, z0 + zchunk)] = ((1 - wm) * t)[0, z0 - a:z0 - a + min(zchunk, D - z0)].half().cpu().numpy()
        del x, t, sg, wm
    return out, thr


# ----------------------------------------------------------------------------- feature builders
def parser_features(prob: torch.Tensor, channels: str = "wm_gm") -> torch.Tensor:
    """prob [4,D,H,W] (bg, WM, infra, supra) -> feature channels."""
    p = prob.float()
    if channels == "wm_gm":
        return torch.stack([p[1], p[2] + p[3]], 0)
    if channels == "wm_infra_supra":
        return p[1:4]
    if channels == "tissue_wm_gm":
        return torch.stack([1 - p[0], p[1], p[2] + p[3]], 0)
    raise ValueError(channels)


def build_features(kind: str, vol: torch.Tensor, tissue: torch.Tensor, prob: torch.Tensor | None = None,
                   wm_bright: bool = True, parser_channels: str = "wm_gm", vessel_sigmas=(1.0, 1.5, 2.2)) -> torch.Tensor:
    """Dispatch. vol [1,D,H,W] float, tissue [1,D,H,W] bool/float, prob [4,D,H,W] (parser) or None."""
    if kind == "parser":
        assert prob is not None
        return parser_features(prob, parser_channels)
    if kind == "mind":
        x = vol.float()
        m = (x - x[tissue.bool()].mean()) / (x[tissue.bool()].std() + 1e-6)
        return mind_ssc(m[None])[0]
    if kind == "otsu":
        f, _ = otsu_two_class(vol.float(), tissue.bool(), wm_bright)
        return f
    if kind == "intensity":
        x = vol.float()
        m = (x - x[tissue.bool()].mean()) / (x[tissue.bool()].std() + 1e-6)
        return m if wm_bright else -m
    if kind == "vessel":
        v = frangi_dark_vesselness(vol.float(), sigmas_vox=vessel_sigmas)
        v = v * tissue.float()
        hi = torch.quantile(v[v > 0].flatten()[::max(1, int((v > 0).sum().item()) // 200000)], 0.995) if (v > 0).any() else 1.0
        return (v / (hi + 1e-6)).clamp(0, 1)
    if kind == "parser_vessel":
        assert prob is not None
        return torch.cat([parser_features(prob, parser_channels), build_features("vessel", vol, tissue, vessel_sigmas=vessel_sigmas)], 0)
    if kind == "mind_vessel":
        return torch.cat([build_features("mind", vol, tissue), build_features("vessel", vol, tissue, vessel_sigmas=vessel_sigmas)], 0)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- vesselness (Frangi, dark tubes)
def _gauss_deriv_kernels(sigma: float, device):
    r = int(max(2, round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-0.5 * (t / sigma) ** 2); g = g / g.sum()
    d1 = -(t / sigma ** 2) * g
    d2 = ((t ** 2 - sigma ** 2) / sigma ** 4) * g
    return g, d1, d2, r


def _sep_conv3d(x: torch.Tensor, kz, ky, kx, r) -> torch.Tensor:
    """x [1,1,D,H,W]; separable filtering with 1D kernels along z,y,x (replicate padding)."""
    x = F.pad(x, (r, r, r, r, r, r), mode="replicate")
    x = F.conv3d(x, kz.reshape(1, 1, -1, 1, 1))
    x = F.conv3d(x, ky.reshape(1, 1, 1, -1, 1))
    x = F.conv3d(x, kx.reshape(1, 1, 1, 1, -1))
    return x


@torch.no_grad()
def frangi_gamma(vol: torch.Tensor, sigmas_vox=(1.0, 1.5, 2.2), q: float = 99.9, chunk: int = 4_000_000):
    """Per-scale structureness constant c for frangi_dark_vesselness, from the volume itself (a strided subsample is
    fine): 0.5 x the q-th percentile of the scale-normalised Hessian norm.  Using one c per scale for the whole volume
    (instead of per processing chunk) makes the response comparable everywhere."""
    x = vol.float()[None]; out = []
    for s in sigmas_vox:
        g, d1, d2, r = _gauss_deriv_kernels(s, x.device)
        Hzz = _sep_conv3d(x, d2, g, g, r); Hyy = _sep_conv3d(x, g, d2, g, r); Hxx = _sep_conv3d(x, g, g, d2, r)
        Hzy = _sep_conv3d(x, d1, d1, g, r); Hzx = _sep_conv3d(x, d1, g, d1, r); Hyx = _sep_conv3d(x, g, d1, d1, r)
        H = torch.stack([Hzz, Hzy, Hzx, Hzy, Hyy, Hyx, Hzx, Hyx, Hxx], -1).reshape(-1, 3, 3) * (s * s)
        Sm = []
        for i in range(0, H.shape[0], chunk):
            ev = sym3x3_eigvals(H[i:i + chunk]); Sm.append(torch.sqrt((ev ** 2).sum(1)))
        S = torch.cat(Sm); n = S.numel()
        out.append(0.5 * float(S.kthvalue(max(1, int(q / 100.0 * n))).values.clamp(min=1e-6)))
        del H, Hzz, Hyy, Hxx, Hzy, Hzx, Hyx, S, Sm
    return tuple(out)


def frangi_dark_vesselness(vol: torch.Tensor, sigmas_vox=(1.0, 1.5, 2.2), alpha=0.5, beta=0.5, gamma=None,
                           chunk: int = 4_000_000) -> torch.Tensor:
    """vol [1,D,H,W] float (bright tissue, dark vessels) -> vesselness [1,D,H,W] in [0,1] (max over scales).
    Frangi 1998 with the sign convention for dark tubular structures (lambda2, lambda3 > 0).
    gamma: None (0.5*max S per processing chunk), a float, or one value per scale (see frangi_gamma)."""
    x = vol.float()[None]
    best = torch.zeros_like(x[0])
    gam = list(gamma) if isinstance(gamma, (tuple, list)) else [gamma] * len(sigmas_vox)
    for s, gs in zip(sigmas_vox, gam):
        g, d1, d2, r = _gauss_deriv_kernels(s, x.device)
        Hzz = _sep_conv3d(x, d2, g, g, r); Hyy = _sep_conv3d(x, g, d2, g, r); Hxx = _sep_conv3d(x, g, g, d2, r)
        Hzy = _sep_conv3d(x, d1, d1, g, r); Hzx = _sep_conv3d(x, d1, g, d1, r); Hyx = _sep_conv3d(x, g, d1, d1, r)
        H = torch.stack([Hzz, Hzy, Hzx, Hzy, Hyy, Hyx, Hzx, Hyx, Hxx], -1).reshape(-1, 3, 3) * (s * s)   # scale-normalized
        out = torch.zeros(H.shape[0], device=x.device)
        for i in range(0, H.shape[0], chunk):
            ev = sym3x3_eigvals(H[i:i + chunk])                              # closed form, ascending
            # sort by absolute value ascending: l1 (smallest |.|), l2, l3
            order = ev.abs().argsort(dim=1)
            l = torch.gather(ev, 1, order)
            l1, l2, l3 = l[:, 0], l[:, 1], l[:, 2]
            Ra = (l2.abs() / (l3.abs() + 1e-6))
            Rb = (l1.abs() / torch.sqrt((l2 * l3).abs() + 1e-6))
            S = torch.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
            c = gs if gs is not None else 0.5 * float(S.max().clamp(min=1e-6))
            v = (1 - torch.exp(-(Ra ** 2) / (2 * alpha ** 2))) * torch.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1 - torch.exp(-(S ** 2) / (2 * c ** 2)))
            v = torch.where((l2 > 0) & (l3 > 0), v, torch.zeros_like(v))       # dark tubes only
            out[i:i + chunk] = v
        best = torch.maximum(best, out.reshape(x.shape[1:]))
        del H, Hzz, Hyy, Hxx, Hzy, Hzx, Hyx
    return best.reshape(1, *vol.shape[1:])


def sym3x3_eigvals(A: torch.Tensor) -> torch.Tensor:
    """Closed-form eigenvalues of symmetric 3x3 matrices [N,3,3] -> [N,3] ascending (no cuSOLVER)."""
    a11, a22, a33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    a12, a13, a23 = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    p1 = a12 ** 2 + a13 ** 2 + a23 ** 2
    q = (a11 + a22 + a33) / 3
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2 * p1
    p = torch.sqrt(p2 / 6 + 1e-30)
    B = (A - q[:, None, None] * torch.eye(3, device=A.device, dtype=A.dtype)) / p[:, None, None]
    r = (torch.det(B) / 2).clamp(-1, 1)
    phi = torch.acos(r) / 3
    e1 = q + 2 * p * torch.cos(phi)
    e3 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
    e2 = 3 * q - e1 - e3
    ev = torch.stack([e3, e2, e1], 1)                     # ascending
    diag = p1 < 1e-20
    if diag.any():
        ev[diag] = torch.sort(torch.stack([a11, a22, a33], 1)[diag], 1).values
    return ev
