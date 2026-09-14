"""Search + affine ladder on a 64^3 two-class random field: an OCT block cut through a known rotation of the search set (optionally
mirrored) with inverted contrast and a 5 % spacing error is recovered to < 0.5 mm with polarity -1, by the ladder and by the direct
affine stage (ladder off); the overlap rule."""
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from octreg import geometry as G
from octreg.params import Params
from octreg.refine import refine
from octreg.search import MIRROR, Searcher, search

H = 0.15                                                 # finest grid (mm); Params().levels 0.6 / 0.3 / 0.15 = box average 4 / 2 / 1
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
    -> (mri_pyr, oct_pyr, T_true, OCT foreground points in mm)."""
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
    mri, mask, oct, weight = np.stack([p * M, (1 - p) * M]), M[None], np.stack([q, 1 - q]), w[None]
    mri_pyr, oct_pyr = {}, {}
    for level, k in zip(FAST.levels, (4, 2, 1)):
        (v, A), m = pool(mri, A_M, k), pool(mask, A_M, k)[0][0]
        (u, B), ww = pool(oct, A_O, k), pool(weight, A_O, k)[0][0]
        mri_pyr[level], oct_pyr[level] = (v, m, A), (u, ww, B)
    return mri_pyr, oct_pyr, T, x[w.reshape(-1) > 0.5]


@pytest.mark.parametrize("rot_index, mirror", [(5, False), (17, True)])
def test_recovery(rot_index, mirror):
    mri_pyr, oct_pyr, T_true, pts = phantom(rot_index, mirror)
    level = FAST.levels[0]
    candidates, info = search(*mri_pyr[level], *oct_pyr[level], FAST, "cpu")
    assert info["tau"] == pytest.approx(FAST.overlap_rho) and not info["overlap_floor"] and info["n_orientations"] == 2 * FAST.n_rot
    assert candidates[0]["polarity"] == -1 and candidates[0]["mirror"] == mirror
    best, finalists = refine(candidates, mri_pyr, oct_pyr, FAST, "cpu", info["tau"])
    assert len(finalists) == FAST.keep[-1] and best is finalists[0] and all(best["L"] <= f["L"] for f in finalists)
    assert best["polarity"] == -1 and best["mirror"] == mirror and best["S"] > 0.8
    assert G.pose_distance(best["T"], T_true, pts) < 0.5
    assert 0.01 < best["log_scales"][0] < np.log(1.05) + 0.01 and best["overlap"] > 0.95     # the scale prior shrinks ls toward 0
    if not mirror:                                                                          # ladder off: one affine stage at 0.15 mm
        best, finalists = refine(candidates[:2], mri_pyr, oct_pyr, dataclasses.replace(FAST, ladder=False, keep=(8, 3, 2)), "cpu")
        assert len(finalists) == 2 and best["polarity"] == -1 and G.pose_distance(best["T"], T_true, pts) < 0.5
        with pytest.raises(ValueError, match="overlap"):                                    # the ladder keeps the overlap rule
            refine(candidates[:1], mri_pyr, oct_pyr, dataclasses.replace(FAST, iters=(2, 2, 2)), "cpu", tau=1.01)


def test_overlap_rule():
    """tau = overlap_rho min(1, V_MRI / V_OCT), never below 0.15 (flagged); the MRI foreground is cut to smaller balls."""
    mri_pyr, oct_pyr, _, _ = phantom(0, False)
    (v, m, A), (u, w, B) = mri_pyr[0.6], oct_pyr[0.6]
    r = np.linalg.norm(G.apply_affine(A, np.indices(m.shape).transpose(1, 2, 3, 0).astype(float)), axis=-1)
    for radius_mm, ratio_range, floor in ((10.0, (1.0, 9.0), False), (2.6, (0.25, 0.9), False), (1.6, (0.02, 0.15), True)):
        cut = m * (r < radius_mm)
        s = Searcher(v, cut, A, u, w, B, FAST, "cpu")
        ratio = cut.sum() / w.sum()                                    # equal voxel volumes
        assert ratio_range[0] < ratio < ratio_range[1] and s.overlap_floor == floor
        assert s.tau == pytest.approx(max(0.15, FAST.overlap_rho * min(1.0, ratio)), rel=1e-5)
