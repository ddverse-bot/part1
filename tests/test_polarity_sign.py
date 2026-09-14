"""Swapping the OCT classes negates S exactly, in the FFT search and in the refiner, so the contrast polarity is the sign of one
score (method steps 4-5); a forced polarity (search argument, an ablation) keeps its sign."""
import dataclasses

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from octreg import geometry as G
from octreg.params import Params
from octreg.refine import BaseGrid, compose
from octreg.search import MIRROR, Searcher, search

TOL = 1e-5
FAST = dataclasses.replace(Params(), n_rot=16, topk=4)


@pytest.fixture(scope="module")
def pair():
    """MRI channels (p M, (1 - p) M) on a 0.6 mm grid with an ellipsoid foreground M; the OCT block is a sub-block of the same
    field with unmasked channels (p, 1 - p) and a weight that is > 0 on five of its six array faces."""
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
    return {"mri": (np.stack([p * M, (1 - p) * M]), M, A_M), "oct": (np.stack([q, 1 - q]), w, A_O)}


def swap(oct):
    u, w, A = oct
    return u[::-1].copy(), w, A


@pytest.mark.parametrize("k, mirror", [(0, False), (5, False), (9, True)])
def test_search_map_negates(pair, k, mirror):
    R = G.rotations(16, 0)[k] @ (MIRROR if mirror else np.eye(3))
    a, b = Searcher(*pair["mri"], *pair["oct"], device="cpu"), Searcher(*pair["mri"], *swap(pair["oct"]), device="cpu")
    (S, _, adm), (S_b, _, adm_b) = a.score(R), b.score(R)
    assert int(adm.sum()) > 10 and torch.equal(adm, adm_b)
    assert float(S[adm].abs().max()) > 0.1
    assert float((S + S_b)[a.valid].abs().max()) < TOL


def test_search_candidates_flip_polarity(pair):
    ca, ia = search(*pair["mri"], *pair["oct"], FAST, "cpu")
    cb, ib = search(*pair["mri"], *swap(pair["oct"]), FAST, "cpu")
    assert len(ca) == len(cb) == FAST.topk and abs(ia["top1"] - ib["top1"]) < TOL and ia["n_admissible"] == ib["n_admissible"]
    for x, y in zip(ca, cb):
        assert np.allclose(x["T"], y["T"]) and abs(x["S"] + y["S"]) < TOL and x["polarity"] == -y["polarity"]
        assert x["S"] * x["polarity"] > 0 and x["mirror"] == y["mirror"]
    assert np.allclose(ca[0]["T"], np.eye(4), atol=1e-6) and ca[0]["polarity"] == 1 and ca[0]["S"] > 0.5


def test_forced_polarity(pair):
    top = {}
    for pol in (0, 1, -1):
        c, info = search(*pair["mri"], *swap(pair["oct"]), FAST, "cpu", polarity=pol)
        top[pol] = info["top1"]
        if pol:
            assert all(x["polarity"] == pol for x in c) and abs(c[0]["S"] * pol - info["top1"]) < 1e-12
    assert top[0] == max(top[1], top[-1]) and top[-1] > 0.5 > top[1]
    with pytest.raises(ValueError, match="polarity"):
        search(*pair["mri"], *pair["oct"], FAST, "cpu", polarity=2)


@pytest.mark.parametrize("mirror", [False, True])
def test_refiner_score_negates(pair, mirror):
    a, b = BaseGrid(pair["mri"], pair["oct"], "cpu"), BaseGrid(pair["mri"], swap(pair["oct"]), "cpu")
    c = torch.as_tensor(a.c, dtype=torch.float32)
    f = lambda *v: torch.tensor(v, dtype=torch.float32)
    T = compose(f(0.03, -0.02, 0.04), c + f(0.2, -0.1, 0.3), f(0.02, 0.0, -0.03), f(0.01, 0.0, 0.02), c, mirror)
    S, S_b, S_neg = a.score(T, 1).item(), b.score(T, 1).item(), a.score(T, -1).item()
    assert abs(S + S_b) < TOL and S_neg == S_b
    assert mirror or S > 0.5
