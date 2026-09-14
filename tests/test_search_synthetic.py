"""Search + affine refinement (register.align) on a 64^3 two-class random field: an OCT block cut through a known rotation of the
search set (optionally mirrored) with inverted contrast and a 5 % spacing error is recovered to < 0.5 mm with polarity -1; the
overlap gate of the refinement and the overlap rule of the search."""
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from octreg import geometry as G
from octreg.params import Params
from octreg.refine import refine
from octreg.register import align
from octreg.search import MIRROR, Searcher

H = 0.15                                                 # base grid (mm); Params().search_mm 0.6 = box average over 4 voxels
FAST = dataclasses.replace(Params(), n_rot=24, topk=8)


def pool(x, A, k):
    """Box average of numpy [C, D, H, W] over k voxels, affine moved to the pooled voxel centres."""
    A = np.asarray(A, float).copy()
    A[:3, 3] += A[:3, :3] @ np.full(3, (k - 1) / 2.0)
    A[:3, :3] *= k
    return F.avg_pool3d(torch.as_tensor(x)[None], k)[0].numpy(), A


def phantom(rot_index, mirror):
    """MRI: p = sigmoid of a smoothed Gaussian field on 64^3 voxels, foreground ball M (radius 4.4 mm); channels (p M, (1 - p) M).
    OCT: 40 x 36 x 32 voxels on a flipped-axis grid; voxel x shows 1 - p at T_true x, weight M(T_true x), channels unmasked.
    T_true = R diag(1.05, 1, 1) (x - c) + (0.4, -0.3, 0.2) mm, R = rotation rot_index of the search set (x MIRROR if mirror).
    -> (mri_base (v, M, affine), oct_base (u, w, affine), T_true, OCT foreground points in mm), base grids of H mm."""
    g = gaussian_filter(np.random.default_rng(1).normal(size=(64, 64, 64)), 5.0)
    p = (1 / (1 + np.exp(-g / (0.25 * g.std())))).astype(np.float32)
    A_M = np.diag([H, H, H, 1.0])
    A_M[:3, 3] = -H * 63 / 2
    M = (np.linalg.norm(G.apply_affine(A_M, np.indices(p.shape).transpose(1, 2, 3, 0).astype(float)), axis=-1) < 4.4).astype(np.float32)
    shape = (40, 36, 32)
    A_O = np.diag([H, -H, H, 1.0])
    A_O[:3, 3] = (12.0, -3.0, 7.0)
    c = G.apply_affine(A_O, (np.array(shape) - 1) / 2.0)
    T = np.eye(4)
    T[:3, :3] = G.rotations(FAST.n_rot, FAST.seed)[rot_index] @ (MIRROR if mirror else np.eye(3)) @ np.diag([1.05, 1.0, 1.0])
    T[:3, 3] = np.array([0.4, -0.3, 0.2]) - T[:3, :3] @ c
    x = G.apply_affine(A_O, np.indices(shape).reshape(3, -1).T.astype(float))
    pm = G.sample_world(torch.as_tensor(np.stack([p, M])), A_M, torch.as_tensor(G.apply_affine(T, x), dtype=torch.float32))
    q, w = 1 - pm[0].numpy().reshape(shape), pm[1].numpy().reshape(shape)
    return (np.stack([p * M, (1 - p) * M]), M, A_M), (np.stack([q, 1 - q]), w, A_O), T, x[w.reshape(-1) > 0.5]


@pytest.mark.parametrize("rot_index, mirror", [(5, False), (17, True)])
def test_recovery(rot_index, mirror):
    mri, oct, T_true, pts = phantom(rot_index, mirror)
    poses, info = align(oct, mri, FAST, "cpu")
    s, r = info["search"], info["refine"]
    assert s["tau"] == pytest.approx(FAST.overlap_rho) and not s["overlap_floor"] and s["n_orientations"] == 2 * FAST.n_rot
    assert r["n_poses"] == FAST.topk and len(poses) == FAST.topk - r["n_below_tau"] and all(p["overlap"] >= s["tau"] for p in poses)
    assert [p["L"] for p in poses] == sorted(p["L"] for p in poses)
    best = poses[0]
    assert best["polarity"] == -1 and best["mirror"] == mirror and best["S"] > 0.8
    assert np.linalg.norm(G.apply_affine(best["T"], pts) - G.apply_affine(T_true, pts), axis=1).mean() < 0.5
    assert 0.01 < best["log_scales"][0] < np.log(1.05) + 0.01 and best["overlap"] > 0.95     # the scale prior shrinks ls toward 0
    if not mirror:                                                                          # the gate and the polarity check
        with pytest.raises(ValueError, match="overlap"):
            refine([{"T": T_true, "polarity": -1}], mri, oct, dataclasses.replace(FAST, iters=2), "cpu", tau=1.01)
        with pytest.raises(ValueError, match="polarity"):
            refine([{"T": T_true, "polarity": 0}], mri, oct, FAST, "cpu")


def test_overlap_rule():
    """tau = overlap_rho min(1, V_MRI / V_OCT), never below 0.15 (flagged); the MRI foreground is cut to smaller balls."""
    (v, m, A), (u, w, B), _, _ = phantom(0, False)
    (v, A), m, (u, B), w = pool(v, A, 4), pool(m[None], A, 4)[0][0], pool(u, B, 4), pool(w[None], B, 4)[0][0]
    r = np.linalg.norm(G.apply_affine(A, np.indices(m.shape).transpose(1, 2, 3, 0).astype(float)), axis=-1)
    for radius_mm, ratio_range, floor in ((10.0, (1.0, 9.0), False), (2.6, (0.25, 0.9), False), (1.6, (0.02, 0.15), True)):
        cut = m * (r < radius_mm)
        s = Searcher(v, cut, A, u, w, B, FAST, "cpu")
        ratio = cut.sum() / w.sum()                                    # equal voxel volumes
        assert ratio_range[0] < ratio < ratio_range[1] and s.overlap_floor == floor
        assert s.tau == pytest.approx(max(0.15, FAST.overlap_rho * min(1.0, ratio)), rel=1e-5)
