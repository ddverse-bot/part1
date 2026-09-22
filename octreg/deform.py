"""§6 Smooth deformation: a small displacement field on top of the affine of §1-5, fitted to two kinds of label-free local
evidence, with its smoothness chosen by held-out error (docs/METHOD.md §6).

Convention. The field u is in mm along the MRI world axes and is defined at MRI points x: the OCT content that belongs at x sits
at x + u(x) in the affinely registered OCT, so warped_OCT(x) = OCT_affine(x + u(x)) = OCT(T^-1 (x + u(x))), T the affine (OCT
world -> MRI world). It is a pull-back field on the MRI base grid; T is not changed and the field is not inverted anywhere.

Everything runs on the MRI base grid, onto which the OCT and its specimen mask are resampled through T (mask kept above INSIDE).
Evidence, measured on that volume:
  interior  block matches of the contrast-free structure feature of the two volumes, over the blocks with at least 70 % of
            their volume inside both masks eroded by Params.df_erode_mm (octreg.blockmatch): centres W [N, 3] and
            displacements D [N, 3] (mm, world); matches longer than the search range df_range_mm are dropped, the range
            being per axis.
  boundary  points of the MRI foreground surface (surface) with outward unit normals, and the offset between the surface edges
            of the two volumes along each normal (edge_offsets: the steepest fall of the intensity going outwards, found in the
            same way in the MRI and in the OCT, so that what the rule does to an edge cancels): points P [M, 3], normals n
            [M, 3], offsets delta [M] (mm; > 0: the OCT surface lies outside the MRI surface). The edge, not the bright rim of
            the OCT: the rim is a layer below the surface, its ridge lies inside the surface by half its thickness. Only
            profiles with exactly one prominent fall are used, so faces crossed by section stripes drop out without any
            knowledge of the sectioning axis. And only points with an interior match within Params.df_support_mm: where the
            inside of the two volumes does not correspond (torn or missing tissue, a bubble), an edge nearby is not evidence of
            a deformation, and the field stays at what its neighbourhood supports.
Model (Lattice, fit): displacements c on a regular control lattice of spacing Params.df_grid_mm over the bounding box of the MRI
grid, trilinear in between. Regularised least squares: rows N(W) c = D, rows n . (N(P) c) = delta weighted by sqrt(3 N / M) so
that both kinds carry the same total weight, a membrane penalty lam (first differences of c along the lattice axes)^2, a ridge
row RIDGE c, and HUBER_ROUNDS solutions re-weighted by the Huber rule (Params.df_huber_mm) on both residual types.
Model choice (choose): lam by the held-out error over FOLDS spatial folds (cells of FOLD_MM), score = held-out median interior
error + held-out median boundary error. The field has to stay plausible for fixed tissue: its strain (largest first difference
of c / spacing) stays below Params.df_max_strain. The candidates are the weights of Params.df_lams whose full fit does, and the
smallest weight that does (smallest_lam, bisection), so the choice moves continuously with the evidence instead of jumping
between two weights of the list. Without a candidate below Params.df_gain x the score of no deformation, with no candidate at
all (even the largest weight of df_lams strains Params.df_max_strain or more), or with too little evidence, there is no
deformation (status 'not_supported', zero field).
The evidence is measured once and the field fitted once; both kinds are measured again on the warped OCT for the read-outs.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from . import geometry as G
from .blockmatch import SLAB, _erode, _on_grid, match, structure_feature
from .params import Params
from .search import to_torch

INSIDE = 0.99              # the resampled (and the warped) OCT specimen mask is kept above this
MAX_SURFACE = 50000        # boundary points used at most, drawn with SURFACE_SEED
SURFACE_SEED = 0
SDF_SIGMA = 2.5            # voxels: Gaussian smoothing of the signed distance whose gradient is the normal
PROFILE_SIGMA = 1.0        # samples: Gaussian of the derivative along a profile
MEASURED = 0.95            # an OCT profile is used when at least this fraction of it is measured (> 0)
FOLD_MM, FOLDS = 7.0, 4    # spatial folds: cells of FOLD_MM, fold = (cell index . FOLD_KEY) mod FOLDS
FOLD_KEY = (1, 3, 5)
RIDGE = 1e-2               # ridge row RIDGE * c: keeps lattice nodes without evidence determined
HUBER_ROUNDS = 4
LAM_TOL = 1.05             # the smallest weight that keeps the strain limit is found to this factor


def surface(mask, affine):
    """Surface points of a foreground mask (bool [D, H, W], voxel -> world affine of an isotropic grid) and their outward unit
    normals: the foreground voxels with a background 6-neighbour (grid faces are no surface), at most MAX_SURFACE of them drawn
    with SURFACE_SEED; normals from the gradient of the signed distance to the mask surface smoothed by SDF_SIGMA voxels.
    -> (points [M, 3] world mm, normals [M, 3] world)."""
    mask, A = np.asarray(mask, bool), np.asarray(affine, float)
    idx = np.argwhere(mask & ~ndimage.binary_erosion(mask, border_value=1))
    if not len(idx):
        return np.zeros((0, 3)), np.zeros((0, 3))
    idx = idx[np.random.default_rng(SURFACE_SEED).permutation(len(idx))[:MAX_SURFACE]]
    sdf = (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)).astype(np.float32)
    sdf = ndimage.gaussian_filter(sdf, SDF_SIGMA)
    g = np.stack([x[tuple(idx.T)] for x in np.gradient(sdf)], 1).astype(float)
    n = g @ np.linalg.inv(A[:3, :3])                           # a gradient is a covector: inv(A)^T g
    norm = np.linalg.norm(n, axis=1)
    idx, n = idx[norm > 1e-9], n[norm > 1e-9] / norm[norm > 1e-9, None]
    return G.apply_affine(A, idx.astype(float)), n


def edge_offsets(vol, affine, points, normals, params: Params = Params(), measured=MEASURED):
    """Position of the surface edge of vol (torch [D, H, W] on the grid `affine`; 0 = not measured) along each normal: the
    profile from -df_profile_mm to +df_profile_mm in grid steps (trilinear) and its derivative by a Gaussian of PROFILE_SIGMA
    samples; a prominent fall is a local maximum of the fall going outwards more than df_edge_mad MADs above the median of
    the fall along the profile, within +-df_reach_mm; a point is used when at least `measured` of its profile is > 0 and it has exactly one
    prominent fall, whose position gets a parabolic sub-sample correction. -> (used bool [M], positions [M] mm, 0 where not used)."""
    P, h = params, float(np.linalg.norm(np.asarray(affine, float)[:3, 0]))
    if not len(points):
        return np.zeros(0, bool), np.zeros(0)
    k = max(2, round(P.df_profile_mm / h))
    t = np.arange(-k, k + 1) * h
    pts = torch.as_tensor(points[:, None, :] + normals[:, None, :] * t[None, :, None], dtype=torch.float32, device=vol.device)
    prof = G.sample_world(vol[None], affine, pts)[0].double().cpu().numpy()
    ok = (prof > 0).mean(1) >= measured
    fall = -ndimage.gaussian_filter1d(prof, PROFILE_SIGMA, axis=1, order=1)
    base = np.median(fall, 1, keepdims=True)
    z = (fall - base) / (np.median(np.abs(fall - base), 1, keepdims=True) + 1e-12)
    peak = ((fall[:, 1:-1] > fall[:, :-2]) & (fall[:, 1:-1] >= fall[:, 2:]) & (z[:, 1:-1] > P.df_edge_mad)
            & (np.abs(t[1:-1]) <= P.df_reach_mm + 1e-9))
    ok &= peak.sum(1) == 1
    j, r = peak.argmax(1) + 1, np.arange(len(prof))
    y0, y1, y2 = fall[r, j - 1], fall[r, j], fall[r, j + 1]
    den = y0 - 2 * y1 + y2
    sub = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    return ok, np.where(ok, t[j] + sub * h, 0.0)


class Evidence:
    """Both kinds of evidence of an OCT volume given on the MRI base grid. mri = (arr, foreground, affine), numpy."""

    def __init__(self, mri, params: Params = Params(), device="cuda"):
        arr, mask, A = mri
        P = self.P = params
        self.A, self.device = np.asarray(A, float), device
        self.h = float(np.linalg.norm(self.A[:3, 0]))
        self.erode = max(0, round(P.df_erode_mm / self.h))
        self.feat_m = structure_feature(arr, mask, self.h, P.df_sigma_mm, device)
        self.core_m = _erode(to_torch(mask, device), self.erode)
        self.points, self.normals = surface(mask, A)
        self.edge_m = edge_offsets(to_torch(arr, device), self.A, self.points, self.normals, P, measured=0.0)

    def interior(self, vol, inside):
        """Block matches of the MRI in vol (torch [D, H, W], specimen mask `inside` 0 / 1 torch) -> (centres W [N, 3] world mm,
        displacements D [N, 3] mm world axes: the OCT content of the block at W sits at W + D), all confident matches."""
        P = self.P
        feat_o = structure_feature(vol, inside, self.h, P.df_sigma_mm, self.device)
        c, d, *_ = match(self.feat_m, feat_o, self.core_m * _erode(inside, self.erode), self.h, block_mm=P.df_block_mm,
                         step_mm=P.df_step_mm, range_mm=P.df_range_mm, z_min=P.df_z_min)
        return G.apply_affine(self.A, c), d @ self.A[:3, :3].T

    def boundary(self, vol):
        """-> (points [M, 3], normals [M, 3], offsets [M] mm: the edge of vol minus the edge of the MRI along the normal) of the
        surface points with exactly one edge in both volumes."""
        ok, edge = edge_offsets(vol, self.A, self.points, self.normals, self.P)
        ok &= self.edge_m[0]
        return self.points[ok], self.normals[ok], (edge - self.edge_m[1])[ok]

    def measure(self, vol, inside):
        """-> dict: W, D (matches up to df_range_mm), P, n, delta (the supported boundary points: those with one of these
        matches within df_support_mm), and the read-outs interior_mm (median |D| of all confident matches, taken before the
        df_range_mm filter, so over slightly more matches than the fit uses), boundary_mm (median |delta|) and within (fraction
        of |delta| < 0.3 mm) over all boundary points with one edge; None without evidence."""
        with torch.no_grad():
            W, D = self.interior(vol, inside)
            Pb, n, delta = self.boundary(vol)
        length, off = np.linalg.norm(D, axis=1), np.abs(delta)
        W, D = W[length <= self.P.df_range_mm], D[length <= self.P.df_range_mm]      # the range is per axis
        near = cKDTree(W).query(Pb)[0] <= self.P.df_support_mm if len(W) and len(Pb) else np.zeros(len(Pb), bool)
        return {"W": W, "D": D, "P": Pb[near], "n": n[near], "delta": delta[near],
                "interior_mm": float(np.median(length)) if len(length) else None,
                "boundary_mm": float(np.median(off)) if len(off) else None,
                "within": float((off < 0.3).mean()) if len(off) else None}


class Lattice:
    """Regular control lattice of spacing `spacing` (mm) along the world axes over the box lo..hi (mm): n = ceil((hi - lo) /
    spacing) + 1 nodes per axis, node (a, b, c) at lo + spacing (a, b, c), flat index (a n1 + b) n2 + c."""

    def __init__(self, lo, hi, spacing):
        self.lo, self.spacing = np.asarray(lo, float), float(spacing)
        self.n = np.maximum(np.ceil((np.asarray(hi, float) - self.lo) / self.spacing - 1e-9).astype(int), 1) + 1
        self.K = int(self.n.prod())
        self.affine = np.diag([self.spacing] * 3 + [1.0])
        self.affine[:3, 3] = self.lo
        idx, rows = np.arange(self.K).reshape(self.n), []
        for ax in range(3):                                     # first differences along the three lattice axes
            a, b = (np.take(idx, range(s, self.n[ax] - 1 + s), ax).ravel() for s in (0, 1))
            r = np.arange(len(a))
            rows.append(sparse.csr_matrix((np.r_[-np.ones(len(a)), np.ones(len(a))], (np.r_[r, r], np.r_[a, b])), (len(a), self.K)))
        Dm = sparse.vstack(rows)
        self.penalty = sparse.block_diag([Dm.T @ Dm] * 3, format="csr")

    def basis(self, X):
        """Trilinear weights of the nodes at world points X [N, 3] -> sparse [N, K]."""
        q = (np.asarray(X, float).reshape(-1, 3) - self.lo) / self.spacing
        i0 = np.clip(np.floor(q).astype(int), 0, self.n - 2)
        f, r = q - i0, np.arange(len(q))
        rows, cols, vals = [], [], []
        for d in np.ndindex(2, 2, 2):
            rows.append(r)
            cols.append(((i0[:, 0] + d[0]) * self.n[1] + i0[:, 1] + d[1]) * self.n[2] + i0[:, 2] + d[2])
            vals.append(np.prod(np.where(np.array(d, bool), f, 1 - f), axis=1))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (len(q), self.K))

    def strain(self, c):
        """Largest first difference of the node displacements c [3, K] divided by the spacing."""
        g = np.asarray(c).reshape(3, *self.n)
        return max([float(np.abs(np.diff(g, axis=ax + 1)).max()) for ax in range(3) if self.n[ax] > 1] + [0.0]) / self.spacing

    def on_grid(self, c, shape, affine, device="cuda"):
        """The field of node displacements c [3, K] on the grid (shape, affine), trilinear -> torch float32 [3, *shape]."""
        return _on_grid(to_torch(np.asarray(c).reshape(3, *self.n), device), self.affine, tuple(shape), affine, np.eye(4))


def fit(lattice: Lattice, ev, lam, huber_mm, rows=None):
    """Node displacements of the lattice fitted to the evidence ev (Evidence.measure; rows = (interior bool [N], boundary bool
    [M]) restricts it, e.g. to the training folds) by the regularised, Huber re-weighted least squares of the module docstring;
    ev may carry the bases 'NW', 'NP' of this lattice. -> c [3, K]."""
    K = lattice.K
    kb, ke = rows if rows is not None else (slice(None), slice(None))
    Nb, Ne = (ev[k][s] if k in ev else lattice.basis(ev[x][s]) for k, x, s in (("NW", "W", kb), ("NP", "P", ke)))
    D, n, delta = ev["D"][kb], ev["n"][ke], ev["delta"][ke]
    Ae = sparse.hstack([Ne.multiply(n[:, a:a + 1]) for a in range(3)], format="csr")                 # rows n . (N(P) c)
    w_e = 3.0 * len(D) / max(len(delta), 1) if len(D) else 1.0  # squared row weight: equal total weight of both kinds
    fixed = lam * lattice.penalty + RIDGE ** 2 * sparse.identity(3 * K, format="csr")
    wb, we = np.ones(len(D)), np.ones(len(delta))               # squared Huber row weights
    for _ in range(HUBER_ROUNDS):
        Hb = Nb.T @ Nb.multiply(wb[:, None])
        H = sparse.block_diag([Hb] * 3) + w_e * (Ae.T @ Ae.multiply(we[:, None])) + fixed
        rhs = np.concatenate([Nb.T @ (wb * D[:, a]) for a in range(3)]) + w_e * (Ae.T @ (we * delta))
        c = splu(sparse.csc_matrix(H)).solve(rhs)
        rb = np.linalg.norm(Nb @ c.reshape(3, K).T - D, axis=1)
        re = np.abs(Ae @ c - delta)
        wb, we = (np.minimum(1.0, huber_mm / np.maximum(r, 1e-6)) for r in (rb, re))
    return c.reshape(3, K)


def folds(X):
    """Spatial fold of world points X [N, 3]: cells of FOLD_MM, fold = (cell index . FOLD_KEY) mod FOLDS."""
    return (np.floor(np.asarray(X, float).reshape(-1, 3) / FOLD_MM).astype(int) @ np.array(FOLD_KEY)) % FOLDS


def held_out(lattice: Lattice, ev, lam, huber_mm):
    """Held-out median errors (interior mm, boundary mm) of fit over the spatial folds of both kinds of evidence."""
    fb, fe, eb, ee = folds(ev["W"]), folds(ev["P"]), [], []
    for k in range(FOLDS):
        c = fit(lattice, ev, lam, huber_mm, (fb != k, fe != k)).T
        eb.append(np.linalg.norm(ev["NW"][fb == k] @ c - ev["D"][fb == k], axis=1))
        ee.append(np.abs(((ev["NP"][fe == k] @ c) * ev["n"][fe == k]).sum(1) - ev["delta"][fe == k]))
    return float(np.median(np.concatenate(eb))), float(np.median(np.concatenate(ee)))


def smallest_lam(lattice: Lattice, ev, params: Params = Params()):
    """The smallest membrane weight between the smallest and the largest of df_lams whose full fit strains less than
    df_max_strain (the strain falls as the weight grows: bisection in log lam down to a factor LAM_TOL); None when even the
    largest weight strains more."""
    P = params
    ok = lambda lam: lattice.strain(fit(lattice, ev, lam, P.df_huber_mm)) < P.df_max_strain
    lo, hi = min(P.df_lams), max(P.df_lams)
    if ok(lo):
        return lo
    if not ok(hi):
        return None
    while hi / lo > LAM_TOL:
        mid = float(np.sqrt(lo * hi))
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def choose(lattice: Lattice, ev, none, params: Params = Params()):
    """Model choice by held-out error over the weights of df_lams that keep the strain limit and the smallest weight that does
    (module docstring). `none` is the score of no deformation, the sum of the two medians the caller reports as cv['none'].
    -> (lam, [b, e] held out) of the best candidate, or None when none beats df_gain x `none`; and the table of all
    candidates."""
    P, best, table = params, None, []
    e = {**ev, "NW": lattice.basis(ev["W"]), "NP": lattice.basis(ev["P"])}
    lam0 = smallest_lam(lattice, e, P)
    for lam in sorted({lam0, *(x for x in P.df_lams if x > lam0)}, reverse=True) if lam0 is not None else ():
        b, d = held_out(lattice, e, lam, P.df_huber_mm)
        table.append({"lam": lam, "interior_mm": b, "boundary_mm": d, "max_strain": lattice.strain(fit(lattice, e, lam, P.df_huber_mm))})
        if b + d < P.df_gain * none and (best is None or b + d < sum(best[1])):
            best = (lam, [b, d])
    return best, table


def warp(src, affine, field):
    """src (torch [C, D, H, W] on the grid `affine`) pulled back by field (torch [3, D, H, W], mm, world axes, same grid):
    out(x) = src(x + u(x)), trilinear, 0 outside, SLAB voxels at a time. -> torch [C, D, H, W]."""
    out, shape, dev = torch.empty_like(src), src.shape[1:], src.device
    j, k = (torch.arange(s, dtype=torch.float32, device=dev) for s in shape[1:])
    step = max(1, SLAB // (shape[1] * shape[2]))
    for i0 in range(0, shape[0], step):
        i = torch.arange(i0, min(shape[0], i0 + step), dtype=torch.float32, device=dev)
        x = G.apply_affine(affine, torch.stack(torch.meshgrid(i, j, k, indexing="ij"), -1)) + field[:, i0:i0 + len(i)].permute(1, 2, 3, 0)
        out[:, i0:i0 + len(i)] = G.sample_world(src, affine, x)
    return out


def smooth_deformation(mri, oct, T, params: Params = Params(), device="cuda"):
    """§6 on base-grid arrays (module docstring, which states the convention of the field). mri = (arr, foreground, affine),
    oct = (arr, specimen mask, affine): the raw intensities and masks on the isotropic base grids (numpy); T: 4x4 OCT world ->
    MRI world, the final affine.
    -> (field float32 numpy [3, D, H, W] on the MRI base grid, mm along the MRI world axes, zero when not supported; info).
    info: status 'applied' | 'not_supported' (too little evidence, every weight of df_lams straining df_max_strain or more, or
    no held-out gain over df_gain); grid_mm, lam (None when not supported); max_strain of the field; n_interior,
    n_boundary (the evidence of the fit: matches up to df_range_mm, supported boundary points); cv {'none': [b, e], 'chosen':
    [b, e]}: held-out median interior and boundary error (mm) without a field and of the chosen weight; candidates (the table
    of the model choice); residual {'interior_mm', 'boundary_mm', 'boundary_within_0.3mm': [before, after]}: median length of
    all confident matches, median |offset| and fraction of offsets below 0.3 mm over all boundary points, measured on the affine
    OCT and again on the warped one (after = before when not supported; None without evidence); field {'median_mm', 'p95_mm',
    'max_mm'} over the MRI foreground; seconds."""
    P, t0, T = params, time.time(), np.asarray(T, float)
    (m_arr, m_mask, A_m), (o_arr, o_mask, A_o) = mri, oct
    A_m, shape = np.asarray(A_m, float), tuple(np.shape(m_arr))
    with torch.no_grad():
        evidence = Evidence(mri, P, device)
        src = _on_grid(torch.stack([to_torch(o_arr, device), to_torch(o_mask, device)]), A_o, shape, A_m, np.linalg.inv(T))
        ev = last = evidence.measure(src[0], (src[1] > INSIDE).to(torch.float32))
    corners = G.apply_affine(A_m, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                            for k in (0, shape[2] - 1)], float))
    lattice = Lattice(corners.min(0), corners.max(0), P.df_grid_mm)
    info = {"status": "not_supported", "grid_mm": None, "lam": None, "max_strain": 0.0, "n_interior": len(ev["W"]),
            "n_boundary": len(ev["P"]), "cv": {"none": None, "chosen": None}, "candidates": []}
    best, field = None, torch.zeros((3, *shape), dtype=torch.float32, device=device)
    if len(ev["W"]) >= P.df_min_interior and len(ev["P"]) >= P.df_min_boundary:
        info["cv"]["none"] = [float(np.median(np.linalg.norm(ev["D"], axis=1))), float(np.median(np.abs(ev["delta"])))]
        best, info["candidates"] = choose(lattice, ev, sum(info["cv"]["none"]), P)
    if best is not None:
        c = fit(lattice, ev, best[0], P.df_huber_mm)
        field = lattice.on_grid(c, shape, A_m, device)
        info.update(status="applied", grid_mm=P.df_grid_mm, lam=best[0], max_strain=lattice.strain(c))
        info["cv"]["chosen"] = best[1]
        with torch.no_grad():
            moved = warp(src, A_m, field)
            last = evidence.measure(moved[0], (moved[1] > INSIDE).to(torch.float32))
    info["residual"] = {name: [ev[k], last[k]] for name, k in (("interior_mm", "interior_mm"), ("boundary_mm", "boundary_mm"),
                                                              ("boundary_within_0.3mm", "within"))}
    u = field.cpu().numpy()
    mag = np.linalg.norm(u, axis=0)[np.asarray(m_mask, bool)]
    info["field"] = {k: float(f(mag)) if len(mag) else 0.0
                     for k, f in (("median_mm", np.median), ("p95_mm", lambda x: np.percentile(x, 95)), ("max_mm", np.max))}
    info["seconds"] = time.time() - t0
    return u, info
