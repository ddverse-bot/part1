"""preprocess: histogram-valley status, isotropic-texture specimen mask against the intensity rule, stripe detection and
flat field (each axis, no-op, zeros, chunk invariance), flattening, gain invariance and the channel pairs."""
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


def _striped(seed, axis, n=80):
    """Isotropic texture; with an axis: texture independent between planes along it plus a sawtooth of plane offsets with
    random dropouts (period 7.5 planes); a black tile column."""
    rng = np.random.default_rng(seed)
    tex = ndimage.gaussian_filter(rng.standard_normal((n,) * 3), 1.5)
    vol = 1000 + 100 * tex / tex.std()
    if axis is not None:
        sig = [1.5, 1.5, 1.5]
        sig[axis] = 0
        planar = ndimage.gaussian_filter(rng.standard_normal((n,) * 3), sig)
        prof = 60 * ((np.arange(n) + rng.uniform(0, 7.5)) % 7.5 / 7.5 - 0.5) + rng.normal(0, 20, n)
        vol = vol + 100 * planar / planar.std() + prof.reshape([-1 if d == axis else 1 for d in range(3)])
    vol = vol.astype(np.float32)
    vol[:16, :16] = 0
    return vol


def _profile_residual(vol, axis):
    v = np.moveaxis(vol, axis, 0)
    prof = v.sum((1, 2)) / (v > 0).sum((1, 2))
    return float((prof - ndimage.uniform_filter1d(prof, 15, mode="nearest")).std())


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_destripe_axis(axis, monkeypatch):
    vol = _striped(10 + axis, axis)
    orig = vol.copy()
    out, info = pp.destripe(vol, orig > 0, 0.04, P)
    assert out is not vol and np.array_equal(vol, orig)                  # a new array; the input is not modified
    assert info["applied"] and info["axis"] == axis and info["ratio"] < P.destripe_ncc_ratio
    assert _profile_residual(out, axis) < 0.2 * _profile_residual(orig, axis)
    assert (out[orig == 0] == 0).all() and (out[orig > 0] > 0).all()
    monkeypatch.setattr(pp, "_CHUNK", 12)                                # padding (48 planes) spans several chunks
    assert np.array_equal(pp.destripe(orig.copy(), None, 0.04, P)[0], out)


def test_destripe_noop():
    vol = _striped(2, None)
    orig = vol.copy()
    out, info = pp.destripe(vol, None, 0.04, P)
    assert info["reason"] == "not_decisive" and not info["applied"] and np.array_equal(out, orig)
    out, info = pp.destripe(_striped(3, 1), None, 0.04, dataclasses.replace(P, destripe=False))
    assert info["reason"] == "off" and info["ncc_by_axis"] is None


def _classes(seed, n=96):
    rng = np.random.default_rng(seed)
    truth = ndimage.gaussian_filter(rng.standard_normal((n,) * 3), 4.0) > 0
    img = 500 * np.where(truth, 1.6, 1.0) * (1 + 0.05 * rng.standard_normal((n,) * 3))
    return img.astype(np.float32), truth, _ball(n, (n / 2 - 0.5,) * 3, 0.48 * n)


def test_flattening_removes_linear_bias():
    img, truth, mask = _classes(0)
    biased = img * (1 + 0.6 * (np.arange(96) - 47.5) / 47.5)[None, None, :].astype(np.float32)
    acc = lambda p: ((p > 0.5) == truth)[mask].mean()
    flat = acc(pp.two_class(biased, mask, 0.6, P))
    assert flat > 0.9 and flat > acc(pp.two_class(biased, mask, 0.6, P, flatten=False)) + 0.1


@pytest.mark.parametrize("flatten", [True, False])
def test_gain_invariance(flatten):
    img, _, mask = _classes(1, 64)
    p = pp.two_class(img, mask, 0.6, P, flatten)
    assert p.dtype == np.float32 and 0 <= p.min() and p.max() <= 1
    np.testing.assert_allclose(pp.two_class(img * 100, mask, 0.6, P, flatten), p, atol=1e-5)


def test_channels():
    img, _, mask = _classes(2, 32)
    p = pp.two_class(img, mask, 0.6, P)
    u, w = pp.oct_channels(p, mask)
    v = pp.mri_channels(p, mask)
    assert u.shape == v.shape == (2, 32, 32, 32) and u.dtype == v.dtype == w.dtype == np.float32
    np.testing.assert_allclose(u.sum(0), 1, atol=1e-6)
    assert np.array_equal(w, mask.astype(np.float32)) and not v[:, ~mask].any()
    np.testing.assert_allclose(v.sum(0), mask, atol=1e-6)
    z = pp.two_class(img, mask, 0.6, dataclasses.replace(P, features="intensity"))
    assert abs(z[mask].mean()) < 0.05 and abs(z[mask].std() - 1) < 0.05
    u, _ = pp.oct_channels(z, mask, "intensity")
    v = pp.mri_channels(z, mask, "intensity")
    assert np.array_equal(u[1], -u[0]) and np.array_equal(v[1], -v[0]) and np.array_equal(v[0], z * mask)
