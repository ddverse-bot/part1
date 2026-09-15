"""Swapping the OCT classes negates the two-class score exactly and leaves the outline score, in the FFT search and in the refiner,
so the poses are the same and the contrast polarity is the sign of one score (docs/METHOD.md §2-4); a forced polarity (search
argument, an ablation) keeps its sign."""
import dataclasses

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from octreg import geometry as G
from octreg.params import Params
from octreg.refine import BaseGrid, compose, masked_ncc
from octreg.search import Searcher, search

MIRROR = np.diag([1.0, 1.0, -1.0])          # the score map negates for any orientation, mirrored ones included

TOL = 1e-5
FAST = dataclasses.replace(Params(), n_rot=16, topk=4)


@pytest.fixture(scope="module")
def pair():
    """MRI channels (p M, (1 - p) M) on a 0.6 mm grid with an ellipsoid foreground M; the OCT block is a sub-block of the same
    field with unmasked channels (p, 1 - p), a specimen weight that is > 0 on five of its six array faces and every voxel
    measured."""
    g = gaussian_filter(np.random.default_rng(0).normal(size=(36, 32, 30)), 2.0)
    p = (1 / (1 + np.exp(-g / (0.25 * g.std())))).astype(np.float32)
    x = (np.indices(p.shape).transpose(1, 2, 3, 0) - np.array(p.shape) / 2) / (0.48 * np.array(p.shape))
    M = ((x ** 2).sum(-1) < 1).astype(np.float32)
    A_M = np.diag([0.6, 0.6, 0.6, 1.0])
    A_M[:3, 3] = (-10.0, 4.0, 2.0)
    lo, d = np.array([8, 7, 7]), (21, 19, 17)
    A_O = A_M.copy()
    A_O[:3, 3] = G.apply_affine(A_M, lo.astype(float))
    q = p[lo[0]:lo[0] + d[0], lo[1]:lo[1] + d[1], lo[2]:lo[2] + d[2]]
    w = np.broadcast_to(np.clip(np.arange(d[0]) / 6.0, 0, 1)[:, None, None], d).astype(np.float32)
    return {"mri": (np.stack([p * M, (1 - p) * M]), M, A_M), "oct": (np.stack([q, 1 - q]), w, np.ones(d, np.float32), A_O)}


def swap(oct):
    u, w, q, A = oct
    return u[::-1].copy(), w, q, A


@pytest.mark.parametrize("k, mirror", [(0, False), (5, False), (9, True)])
def test_search_map_negates(pair, k, mirror):
    R = G.rotations(16, 0)[k] @ (MIRROR if mirror else np.eye(3))
    a, b = Searcher(*pair["mri"], *pair["oct"], device="cpu"), Searcher(*pair["mri"], *swap(pair["oct"]), device="cpu")
    (S, S_o), (S_b, S_ob) = a.score(R), b.score(R)
    assert int(a.valid.sum()) > 10 and float(S[a.valid].abs().max()) > 0.1
    assert float((S + S_b)[a.valid].abs().max()) < TOL and torch.equal(S_o, S_ob)


def test_search_candidates_flip_polarity(pair):
    ca, ia = search(*pair["mri"], *pair["oct"], FAST, "cpu")
    cb, ib = search(*pair["mri"], *swap(pair["oct"]), FAST, "cpu")
    assert len(ca) == len(cb) == FAST.topk and abs(ia["top1"] - ib["top1"]) < TOL and ia["n_orientations"] == FAST.n_rot
    for x, y in zip(ca, cb):
        assert np.allclose(x["T"], y["T"]) and abs(x["S"] - y["S"]) < TOL and abs(x["S_class"] + y["S_class"]) < TOL
        assert abs(x["S_outline"] - y["S_outline"]) < TOL and x["polarity"] == -y["polarity"] and x["S_class"] * x["polarity"] > 0
    assert np.allclose(ca[0]["T"], np.eye(4), atol=1e-6) and ca[0]["polarity"] == 1 and ca[0]["S_class"] > 0.9


def test_forced_polarity(pair):
    top = {}
    for pol in (0, 1, -1):
        c, info = search(*pair["mri"], *swap(pair["oct"]), FAST, "cpu", polarity=pol)
        top[pol] = info["top1"]
        if pol:
            assert all(x["polarity"] == pol for x in c) and abs(c[0]["S"] - info["top1"]) < 1e-12
    assert top[0] == max(top[1], top[-1]) and top[-1] > 0.5 > top[1]
    with pytest.raises(ValueError, match="polarity"):
        search(*pair["mri"], *pair["oct"], FAST, "cpu", polarity=2)


def test_refiner_score_negates(pair):
    a, b = BaseGrid(pair["mri"], pair["oct"], "cpu"), BaseGrid(pair["mri"], swap(pair["oct"]), "cpu")
    c = torch.as_tensor(a.c, dtype=torch.float32)
    f = lambda *v: torch.tensor(v, dtype=torch.float32)
    T = compose(f(0.03, -0.02, 0.04), c + f(0.2, -0.1, 0.3), f(0.02, 0.0, -0.03), f(0.01, 0.0, 0.02), c)
    (S, Sc, So), (S_b, Sc_b, So_b), (S_neg, _, _) = ([x.item() for x in g.score(T, pol)] for g, pol in ((a, 1), (b, 1), (a, -1)))
    assert abs(Sc + Sc_b) < TOL and So == So_b and abs(S_neg - S_b) < TOL
    assert Sc > 0.5


def test_outline_ignores_embedding_over_mri_tissue(pair):
    """A block cut out of a larger specimen: the OCT specimen mask is the MRI foreground inside the block up to a cut, and the MRI
    tissue beyond the cut lies under OCT embedding. At the true pose the outline NCC is 1 (that embedding is left out), whereas
    counting it would lower the score; the FFT map equals the refiner's direct weighted NCC at shifted poses as well, also with
    partial-volume masks (M^3 != M^2 != M, q (1 - w) w != 0) and with the block partly outside the MRI crop."""
    v, M, A_M = pair["mri"]
    u, _, q, A_O = pair["oct"]
    x = (np.indices(M.shape).transpose(1, 2, 3, 0) - np.array(M.shape) / 2) / (0.33 * np.array(M.shape))
    M = ((x ** 2).sum(-1) < 1).astype(np.float32)                                  # a specimen smaller than the block
    v = v * (M > 0)
    lo = np.rint(G.apply_affine(np.linalg.inv(A_M), A_O[:3, 3])).astype(int)
    spec = M[lo[0]:lo[0] + q.shape[0], lo[1]:lo[1] + q.shape[1], lo[2]:lo[2] + q.shape[2]]
    w = (spec * (np.arange(q.shape[1])[None, :, None] < 12)).astype(np.float32)
    assert ((w == 0) & (spec > 0)).sum() > 100 and ((w == 0) & (spec == 0)).sum() > 10
    s = Searcher(v, M, A_M, u, w, q, A_O, device="cpu")
    _, S_o = s.score(np.eye(3))
    g = BaseGrid((v, M, A_M), (u, w, q, A_O), "cpu")
    for shift in ((0, 0, 0), (1, 0, 0), (0, 2, -1)):
        T = torch.eye(4)
        T[:3, 3] = torch.tensor(shift, dtype=torch.float32) * 0.6
        c = G.apply_affine(np.linalg.inv(s.A_M), G.apply_affine(T.numpy(), s.c_o)) - (s.n - 1) / 2.0
        idx = np.ravel_multi_index(tuple(np.rint(c).astype(int)), s.shape)
        assert np.allclose(s.pose(np.eye(3), idx), T.numpy(), atol=1e-6)
        direct = g.score(T, 1)[2].item()
        assert abs(float(S_o.reshape(-1)[idx]) - direct) < 1e-3
        if shift == (0, 0, 0):
            m = G.sample_world(g.v, g.A_M, G.apply_affine(T, g.pts))[2:]
            assert direct > 0.99 and masked_ncc(m, g.w[None], g.q).item() < 0.9
    Ms, ws = gaussian_filter(M, 1.0), gaussian_filter(w, 1.0)
    s2, g2 = Searcher(v * Ms, Ms, A_M, u, ws, q, A_O, device="cpu"), BaseGrid((v * Ms, Ms, A_M), (u, ws, q, A_O), "cpu")
    _, S_o2 = s2.score(np.eye(3))
    for shift in ((0, 0, 0), (1, 0, 0), (0, 2, -1), (8, 0, 0)):                     # (8, 0, 0): part of the block leaves the crop
        T = torch.eye(4)
        T[:3, 3] = torch.tensor(shift, dtype=torch.float32) * 0.6
        c = G.apply_affine(np.linalg.inv(s2.A_M), G.apply_affine(T.numpy(), s2.c_o)) - (s2.n - 1) / 2.0
        idx = np.ravel_multi_index(tuple(np.rint(c).astype(int)), s2.shape)
        assert abs(float(S_o2.reshape(-1)[idx]) - g2.score(T, 1)[2].item()) < 1e-4
