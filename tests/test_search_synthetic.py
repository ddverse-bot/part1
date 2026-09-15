"""Search + affine refinement (register.align) on a 64^3 two-class random field: an OCT block cut through a known rotation of the
search set with inverted contrast and a 5 % spacing error is recovered to < 0.5 mm with polarity -1."""
import dataclasses

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from octreg import geometry as G
from octreg.params import Params
from octreg.refine import refine
from octreg.register import align

H = 0.15                                                 # base grid (mm); Params().search_mm 0.6 = box average over 4 voxels
FAST = dataclasses.replace(Params(), n_rot=24, topk=8)


def phantom(rot_index, cut=None):
    """MRI: p = sigmoid of a smoothed Gaussian field on 64^3 voxels, foreground ball M (radius 4.4 mm); channels (p M, (1 - p) M).
    OCT: 40 x 36 x 32 voxels on a flipped-axis grid, all measured; voxel x shows 1 - p at T_true x, specimen mask M(T_true x),
    channels unmasked.
    T_true = R diag(1.05, 1, 1) (x - c) + (0.4, -0.3, 0.2) mm, R = rotation rot_index of the search set.
    cut: OCT axis-0 index from which the block was cut off the specimen (mask 0 there, over MRI tissue and background alike).
    -> (mri_base (v, M, affine), oct_base (u, w, measured, affine), T_true, OCT foreground points in mm), base grids of H mm."""
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
    T[:3, :3] = G.rotations(FAST.n_rot, FAST.seed)[rot_index] @ np.diag([1.05, 1.0, 1.0])
    T[:3, 3] = np.array([0.4, -0.3, 0.2]) - T[:3, :3] @ c
    x = G.apply_affine(A_O, np.indices(shape).reshape(3, -1).T.astype(float))
    pm = G.sample_world(torch.as_tensor(np.stack([p, M])), A_M, torch.as_tensor(G.apply_affine(T, x), dtype=torch.float32))
    q, w = 1 - pm[0].numpy().reshape(shape), pm[1].numpy().reshape(shape)
    if cut is not None:
        w = w * (np.arange(shape[0]) < cut)[:, None, None]
    return (np.stack([p * M, (1 - p) * M]), M, A_M), (np.stack([q, 1 - q]), w, np.ones(shape, np.float32), A_O), T, x[w.reshape(-1) > 0.5]


@pytest.mark.parametrize("rot_index", [5, 17])
def test_recovery(rot_index):
    mri, oct, T_true, pts = phantom(rot_index)
    poses, info = align(oct, mri, FAST, "cpu")
    s, r = info["search"], info["refine"]
    assert s["n_orientations"] == FAST.n_rot and r["n_poses"] == len(poses) == FAST.topk
    assert [p["L"] for p in poses] == sorted(p["L"] for p in poses)
    best = poses[0]
    assert best["polarity"] == -1 and best["S_class"] < -0.8 and best["S_outline"] > 0.9 and np.linalg.det(best["T"][:3, :3]) > 0
    assert np.linalg.norm(G.apply_affine(best["T"], pts) - G.apply_affine(T_true, pts), axis=1).mean() < 0.5
    assert 0.01 < best["log_scales"][0] < np.log(1.05) + 0.01                                 # the scale prior shrinks ls toward 0
    with pytest.raises(ValueError, match="polarity"):
        refine([{"T": T_true, "polarity": 0}], mri, oct, FAST, "cpu")


def test_recovery_block_cut_from_larger_specimen():
    """The MRI holds tissue beyond a cut face of the block, which lies under OCT embedding: recovered to < 0.1 mm (counting that
    embedding as a mismatch gives 0.16 mm and an outline score of 0.26)."""
    mri, oct, T_true, pts = phantom(5, cut=24)
    assert ((oct[1] == 0) & (G.sample_world(torch.as_tensor(mri[1][None]), mri[2], torch.as_tensor(G.apply_affine(
        T_true, G.apply_affine(oct[3], np.indices(oct[1].shape).reshape(3, -1).T.astype(float))), dtype=torch.float32))[0]
        .numpy().reshape(oct[1].shape) > 0.5)).sum() > 1000
    best = align(oct, mri, FAST, "cpu")[0][0]
    assert best["polarity"] == -1 and np.linalg.det(best["T"][:3, :3]) > 0 and best["S_outline"] > 0.9
    assert np.linalg.norm(G.apply_affine(best["T"], pts) - G.apply_affine(T_true, pts), axis=1).mean() < 0.1
