"""preprocess: histogram-valley status, isotropic-texture specimen mask against the intensity rule (and chunk invariance),
per-plane hole filling, gain invariance and the channel pairs."""
import dataclasses

import numpy as np
import pytest
from scipy import ndimage

from octreg import preprocess as pp
from octreg.params import Params

P = Params()


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _ball(n, c, r):
    z, y, x = np.mgrid[:n, :n, :n]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def test_foreground_status():
    rng = np.random.default_rng(0)
    ball = _ball(64, (32, 32, 32), 26)
    bimodal = np.where(ball, rng.normal(500, 30, ball.shape), rng.normal(100, 20, ball.shape)).astype(np.float32)
    m, info = pp.foreground(bimodal, 0.15, P)
    assert info["status"] == "ok" and 150 < info["threshold"] < 400 and _dice(m, ball) > 0.98
    assert info["n_components"] == 1 and info["volume_cm3"] == pytest.approx(m.sum() * 0.15 ** 3 / 1e3)
    m, info = pp.foreground(rng.normal(500, 40, (48,) * 3).astype(np.float32), 0.15, P)
    assert info["status"] == "no_valley" and m.all()


def _embedded_block(seed, n=96):
    """Tissue ball (isotropic texture) cut by one face, in a medium of equal mean intensity whose structure is anisotropic
    (random section offsets along axis 2, tile seams along axis 0, quiet along axis 1) plus a black tile column."""
    rng = np.random.default_rng(seed)
    ball = _ball(n, (48, 46, 66), 36)
    tex = ndimage.gaussian_filter(rng.standard_normal((n,) * 3), 1.5)
    stripes = rng.normal(0, 0.05, n)[None, None, :] + 0.1 * (np.arange(n) % 24 < 2)[:, None, None]
    vol = 1000 * (1 + stripes) * np.where(ball, 1 + 0.25 * tex / tex.std(), 1.0) * (1 + 0.1 * rng.standard_normal((n,) * 3))
    vol = np.maximum(vol, 1).astype(np.float32)
    vol[:, :20, :20] = 0
    return vol, ball & (vol > 0)


@pytest.mark.parametrize("voxel_mm, min_dice", [(0.08, 0.9), (P.fine_mm, 0.85)])   # 0.04 mm: the default fine grid
def test_specimen_mask_texture_not_intensity(monkeypatch, voxel_mm, min_dice):
    vol, truth = _embedded_block(0)
    Q = dataclasses.replace(P, texture_smooth_mm=0.32)                  # a block of a few mm (the default 1.2 mm suits real blocks)
    m, info = pp.specimen_mask(vol, voxel_mm, Q)
    assert _dice(m, truth) > min_dice and not m[vol == 0].any()
    assert info["volume_cm3"] == pytest.approx(m.sum() * voxel_mm ** 3 / 1e3) and info["threshold"] > 0
    assert _dice(pp.foreground(vol, voxel_mm, P)[0], truth) < 0.6        # equal intensities: the histogram rule fails
    monkeypatch.setattr(pp, "_CHUNK", 16)
    assert np.array_equal(pp.specimen_mask(vol, voxel_mm, Q)[0], m)      # chunked == unchunked


def _classes(seed, n=96):
    rng = np.random.default_rng(seed)
    truth = ndimage.gaussian_filter(rng.standard_normal((n,) * 3), 4.0) > 0
    img = 500 * np.where(truth, 1.6, 1.0) * (1 + 0.05 * rng.standard_normal((n,) * 3))
    return img.astype(np.float32), truth, _ball(n, (n / 2 - 0.5,) * 3, 0.48 * n)


def test_two_class_classes_and_gain_invariance():
    img, truth, mask = _classes(1, 64)
    p = pp.two_class(img, mask, P)
    assert p.dtype == np.float32 and 0 <= p.min() and p.max() <= 1 and ((p > 0.5) == truth)[mask].mean() > 0.9
    np.testing.assert_allclose(pp.two_class(img * 100, mask, P), p, atol=1e-5)


def test_channels():
    img, _, mask = _classes(2, 32)
    p = pp.two_class(img, mask, P)
    u, w = pp.oct_channels(p, mask)
    v = pp.mri_channels(p, mask)
    assert u.shape == v.shape == (2, 32, 32, 32) and u.dtype == v.dtype == w.dtype == np.float32
    np.testing.assert_allclose(u.sum(0), 1, atol=1e-6)
    assert np.array_equal(w, mask.astype(np.float32)) and not v[:, ~mask].any()
    np.testing.assert_allclose(v.sum(0), mask, atol=1e-6)


def test_fill_planes_fills_holes_open_to_a_face():
    """A cavity that reaches a cut face is not enclosed in 3-D but is enclosed in the planes across it; outside stays outside."""
    m = np.zeros((12, 12, 12), bool)
    m[2:10, 2:10, :] = True
    m[5:7, 5:7, :] = False                                   # a tunnel through the block along axis 2, open at both faces
    m[4:8, 4:8, 4:8] &= ~_ball(4, (1.5, 1.5, 1.5), 1.2)      # and an enclosed cavity
    assert not ndimage.binary_fill_holes(m)[5, 5, 0]
    f = pp._fill_planes(m)
    assert f[2:10, 2:10, :].all() and not f[:2].any() and not f[10:].any()
