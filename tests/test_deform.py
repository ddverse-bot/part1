"""§6 smooth deformation (octreg/deform.py) and the interior evidence it is built on (octreg/blockmatch.py), CPU. The structure
feature does not change when the intensities are inverted and hardly under a monotone remapping, and match finds a known shift.
The stage itself runs on a synthetic specimen: smooth random structure inside an ellipsoid, bright on
a zero background (the MRI); the OCT is an inverted, gamma-remapped copy on its own flipped grid with a bright rim as a layer
below its surface, embedded in darker, textured agarose with bright parallel stripes across one OCT axis. The boundary
evidence (the offset between the surface edges of the two volumes) measures a pure 0.5 mm shift and mostly drops the faces
crossed by the stripes; boundary points without an interior match within df_support_mm are not used; smallest_lam finds the
smallest membrane weight that keeps the strain limit; a known smooth displacement (a Gaussian bump of 1.0 mm and 8 mm width)
is recovered; an aligned pair is left alone (status not_supported, zero field); a field written and read back reproduces the
warped volume through geometry.resample_to."""
import dataclasses

import numpy as np
import pytest
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from octreg import blockmatch as bm, deform, geometry as G, io
from octreg.params import Params

H = 0.2                                                           # isotropic grid of the two evidence tests
PARAMS = Params.from_dict({"df_range_mm": 1.5, "df_sigma_mm": 0.45, "df_block_mm": 3.6, "df_step_mm": 1.8, "df_erode_mm": 2.4})
#     a 19 mm specimen on a 0.3 mm grid: a search range of 5 voxels, smaller blocks than the default and the feature sigma in
#     proportion; the rim layer of pair() reaches 1.8 mm below the surface and a block may lie 30 % outside the core, so the
#     erosion is 2.4 mm (with 1.2 mm the rim pulls the outer matches 0.17 mm inwards)


def structure(shape, rng):
    """Band-limited noise with oriented ridges, unit variance."""
    s = ndimage.gaussian_filter(rng.standard_normal(shape), 2.0) * 8 + ndimage.gaussian_filter(rng.standard_normal(shape), (1.2, 4.0, 1.2)) * 5
    return (s / s.std()).astype(np.float32)


def grid(shape, h, flipped):
    A = np.eye(4)
    A[:3, :3] = np.diag([-h, -h, h]) if flipped else np.array([[h, 0, 0], [0, 0, h], [0, -h, 0]])
    A[:3, 3] = -A[:3, :3] @ ((np.array(shape) - 1) / 2)
    return A


def world(shape, A):
    return G.apply_affine(A, np.indices(shape).reshape(3, -1).T.astype(float))


def bump(y, amplitude):
    """The displacement v at affinely registered positions y: the content of x sits at y = x + u(x), u(x) = v(y)."""
    return amplitude * np.array([0.6, -0.64, 0.48]) * np.exp(-((y - np.array([2.0, 1.0, -1.0])) ** 2).sum(-1, keepdims=True) / (2 * 8.0 ** 2))


def pair(v, h=0.3, semi=(9.6, 8.7, 8.1), seed=0, period=6, stripe=1.5, rim=1.5):
    """-> (mri, oct, T, stripe axis in the MRI world, semi-axes) on grids of spacing h for the displacement v(y) ([N, 3] ->
    [N, 3]): an ellipsoid (semi-axes semi, MRI world mm) of smooth structure, zero outside (the MRI); the OCT with tissue at
    0.55-1.45, a bright rim as a layer below the surface (amplitude rim at the surface, falling inwards with a Gaussian of 2
    voxels), agarose at 0.3 with a little texture and, in the agarose, one bright plane every `period` voxels along the OCT
    array axis 2 (amplitude stripe)."""
    rng, semi = np.random.default_rng(seed), np.asarray(semi, float)
    shape_m = tuple(int(np.ceil((2 * a + 6.0) / h)) for a in semi[[0, 2, 1]])              # array axes (R, I, A) of the MRI grid
    shape_o = tuple(int(np.ceil((2 * a + 7.0) / h)) for a in semi)
    A_M, A_O = grid(shape_m, h, False), grid(shape_o, h, True)
    s = structure(shape_m, rng)
    mask_m = (((world(shape_m, A_M) / semi) ** 2).sum(-1) <= 1).reshape(shape_m)
    mri = np.where(mask_m, 1.0 + 0.3 * s, 0.0).astype(np.float32)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.05, -0.08, 0.06]).as_matrix()
    T[:3, 3] = [0.3, -0.2, 0.25]
    y = G.apply_affine(T, world(shape_o, A_O))
    x = y - v(y)
    f = np.sqrt(((x / semi) ** 2).sum(-1))
    dist = (f - 1) * f / np.maximum(np.linalg.norm(x / semi ** 2, axis=1), 1e-9)          # signed distance to the surface, first order
    so = G.sample_world(torch.as_tensor(s[None]), A_M, torch.as_tensor(x, dtype=torch.float32))[0].numpy()
    tissue = 0.55 + 0.9 * (1 - np.clip((so + 3.5) / 7, 0, 1)) ** 0.6                        # inverted, gamma-remapped
    agarose = 0.3 + 0.3 * ndimage.gaussian_filter(rng.standard_normal(shape_o), 1.0).ravel()  # darker than tissue, a little texture
    stripes = stripe * (np.indices(shape_o)[2].ravel() % period == 0)
    layer = rim * np.exp(-0.5 * (dist / (2 * h)) ** 2)                                     # the bright rim: a layer below the surface
    oct_ = np.where(f <= 1, tissue + layer, agarose + stripes)
    oct_ = (oct_ + 0.01 * rng.standard_normal(len(oct_))).reshape(shape_o).astype(np.float32)
    return (mri, mask_m, A_M), (oct_, (f <= 1).reshape(shape_o), A_O), T, T[:3, :3] @ A_O[:3, 2] / h, semi


def on_mri_grid(mri, oct_, T):
    src = deform._on_grid(torch.stack([torch.as_tensor(oct_[0]), torch.as_tensor(oct_[1], dtype=torch.float32)]), oct_[2], mri[0].shape,
                          mri[2], np.linalg.inv(T))
    return src[0], (src[1] > deform.INSIDE).to(torch.float32)


def lattice_of(mri):
    corners = G.apply_affine(mri[2], np.array([[i, j, k] for i in (0, mri[0].shape[0] - 1) for j in (0, mri[0].shape[1] - 1)
                                               for k in (0, mri[0].shape[2] - 1)], float))
    return deform.Lattice(corners.min(0), corners.max(0), PARAMS.df_grid_mm)


# ------------------------------------------------------------------------------- interior evidence (octreg/blockmatch.py)
def test_structure_feature_contrast_free():
    rng = np.random.default_rng(0)
    shape = (40, 44, 36)
    arr = 1.0 + 0.3 * structure(shape, rng)
    mask = np.zeros(shape, bool)
    mask[3:-3, 4:-4, 3:-3] = True
    f = bm.structure_feature(arr, mask, H, 0.3, "cpu")
    assert f.shape == (6, *shape) and f.dtype == torch.float32 and float(f[:, ~torch.as_tensor(mask)].abs().max()) == 0
    assert float((f[0] + f[1] + f[2]).abs().max()) < 1e-6                              # trace-free
    inverted = bm.structure_feature(7.0 - 3.0 * arr, mask, H, 0.3, "cpu")
    assert float((f - inverted).abs().max()) < 1e-3
    remapped = bm.structure_feature(np.exp(2.0 * arr), mask, H, 0.3, "cpu")             # monotone, strongly non-linear
    ncc = float((f * remapped).sum() / torch.sqrt((f * f).sum() * (remapped * remapped).sum()))
    assert ncc > 0.9, ncc
    ramp = bm.structure_feature(5.0 + 0.1 * np.indices(shape)[0], mask, H, 0.3, "cpu")[:, torch.as_tensor(mask)]
    assert float(ramp[0].min()) > 0.1 and float(ramp[3:].abs().max()) < 1e-3          # the gradient of a ramp along axis 0 ...
    assert float((ramp[1] + ramp[0] / 2).abs().max()) < 1e-3 and float((ramp[2] + ramp[0] / 2).abs().max()) < 1e-3   # ... up to the mask edge


def test_match_finds_a_shift():
    rng = np.random.default_rng(1)
    shape = (48, 48, 48)
    mask = np.ones(shape, bool)
    f = bm.structure_feature(structure(shape, rng), mask, H, 0.3, "cpu")
    shift = (2, -3, 1)
    moved = torch.roll(f, shift, (1, 2, 3))
    core = torch.zeros(shape)
    core[8:-8, 8:-8, 8:-8] = 1
    c, d, z, ncc0, ncc, tried = bm.match(f, moved, core, H, block_mm=3.2, step_mm=1.6, range_mm=1.0, z_min=4.0)   # 16, 8, +-5 vox
    assert tried == 27 and len(c) >= 20 and np.abs(d - shift).max() < 0.05 and ncc.min() > 0.99 and np.median(ncc0) < 0.5 and z.min() >= 4
    assert np.allclose(c.min(0), 15.5) and np.allclose(c.max(0), 31.5)                  # centres of the 16-voxel blocks inside the core
    assert bm.match(f, moved, core, H, block_mm=12.8, step_mm=1.6, range_mm=1.0, z_min=4.0)[-1] == 0     # no block fits


# ---------------------------------------------------------------------------------- smooth deformation (octreg/deform.py)
def test_evidence_measures_a_shift_and_drops_striped_faces():
    shift = np.array([0.3, -0.32, 0.24])                                                   # 0.5 mm
    small = {"v": lambda y: np.broadcast_to(shift, y.shape), "h": 0.15, "semi": (5.0, 4.5, 4.2), "rim": 2.5}
    mri, oct_, T, axis, semi = pair(stripe=0.0, **small)                                   # no stripes: the accuracy of the rule
    points, normals = deform.surface(mri[1], mri[2])
    assert len(points) > 5000 and np.allclose(np.linalg.norm(normals, axis=1), 1)
    radial = points / semi ** 2
    assert ((normals * radial).sum(1) / np.linalg.norm(radial, axis=1)).min() > 0.95       # outward, along the ellipsoid normal
    f = np.sqrt(((points / semi) ** 2).sum(1))
    depth = (1 - f) * f / np.linalg.norm(radial, axis=1)                                   # the outermost foreground voxels (0.15 mm)
    assert depth.min() > 0 and depth.max() < 0.15 and 0.04 < np.median(depth) < 0.09
    ev = deform.Evidence(mri, Params(), "cpu")
    used_m, edge_m = ev.edge_m                                                             # the MRI edge: the ellipsoid surface
    assert used_m.mean() > 0.65 and np.abs(np.median((edge_m - depth)[used_m])) < 0.03, (used_m.mean(), np.median((edge_m - depth)[used_m]))
    vol = on_mri_grid(mri, oct_, T)[0]
    P, n, delta = ev.boundary(vol)
    error = delta - n @ shift
    assert len(P) > 6000 and np.median(np.abs(error)) < 0.04 and abs(np.median(error)) < 0.01, (len(P), np.median(np.abs(error)), np.median(error))
    used_o, edge_o = deform.edge_offsets(vol, mri[2], points, normals)                     # the offset is the difference of the two edges
    assert np.array_equal(P, points[used_o & used_m]) and np.array_equal(delta, (edge_o - edge_m)[used_o & used_m])
    assert not edge_o[~used_o].any() and not edge_m[~used_m].any()
    assert not deform.edge_offsets(vol * 0, mri[2], points, normals)[0].any() and not len(ev.boundary(vol * 0)[0])
    assert not deform.edge_offsets(vol, mri[2], points[:0], normals[:0])[0].size

    mri, oct_, T, axis, semi = pair(stripe=3.0, period=4, **small)                         # stripes three times the tissue / agarose step
    used, edge = deform.edge_offsets(on_mri_grid(mri, oct_, T)[0], mri[2], points, normals)
    across = np.abs(normals @ axis)                                                        # 1 on the two faces the stripes cross
    assert used[across > 0.9].mean() < 0.35 and used[across < 0.2].mean() > 0.7, (used[across > 0.9].mean(), used[across < 0.2].mean())
    error = (edge - edge_m - normals @ shift)[used & used_m]                               # what is left: a stripe at the surface moves
    assert np.median(np.abs(error)) < 0.06 and abs(np.median(error)) < 0.04, (np.median(np.abs(error)), np.median(error))  # the edge out

    mri, oct_, T, *_ = pair(lambda y: np.broadcast_to(shift, y.shape))                     # interior matches: 0.3 mm voxels
    W, D = deform.Evidence(mri, PARAMS, "cpu").interior(*on_mri_grid(mri, oct_, T))
    assert len(W) > 100 and np.median(np.linalg.norm(D - shift, axis=1)) < 0.05, (len(W), np.median(np.linalg.norm(D - shift, axis=1)))


def test_boundary_points_need_an_interior_match_nearby():
    mri, oct_, T, *_ = pair(lambda y: bump(y, 1.0))
    ev = deform.Evidence(mri, PARAMS, "cpu")
    vol, inside = on_mri_grid(mri, oct_, T)
    P, n, delta = ev.boundary(vol)
    whole = ev.measure(vol, inside)
    assert len(whole["W"]) > 100 and len(whole["P"]) > 0.9 * len(P) > 2000                 # matches all over the specimen
    x = torch.as_tensor(world(mri[0].shape, mri[2])[:, 0].reshape(mri[0].shape))
    half = ev.measure(vol, inside * (x < 0))                                               # the inside corresponds only where x < 0
    near = cKDTree(half["W"]).query(P)[0] <= PARAMS.df_support_mm
    assert len(half["W"]) >= 10 and half["W"][:, 0].max() < 0 and 0.1 < near.mean() < 0.6, (len(half["W"]), near.mean())
    assert all(np.array_equal(half[k], a[near]) for k, a in (("P", P), ("n", n), ("delta", delta)))
    assert half["P"][:, 0].max() < PARAMS.df_support_mm and P[:, 0].max() > 9
    tight = deform.Evidence(mri, dataclasses.replace(PARAMS, df_support_mm=3.5), "cpu").measure(vol, inside)
    assert 0 < len(tight["P"]) < 0.8 * len(whole["P"]) and cKDTree(tight["W"]).query(tight["P"])[0].max() <= 3.5
    empty = ev.measure(vol, 0 * inside)                                                    # no interior match: no boundary point
    assert not len(empty["W"]) and not len(empty["P"]) and not len(empty["delta"]) and empty["interior_mm"] is None
    for m in (whole, half, empty):                                                         # the read-outs: all boundary points
        assert m["boundary_mm"] == np.median(np.abs(delta)) and m["within"] == (np.abs(delta) < 0.3).mean()
    u, info = deform.smooth_deformation(mri, (oct_[0], oct_[1] & False, oct_[2]), T, PARAMS, "cpu")
    assert info["status"] == "not_supported" and info["n_interior"] == info["n_boundary"] == 0 and not u.any()


def test_smallest_lam_keeps_the_strain_limit():
    mri, oct_, T, *_ = pair(lambda y: bump(y, 1.0))
    ev, lat = deform.Evidence(mri, PARAMS, "cpu").measure(*on_mri_grid(mri, oct_, T)), lattice_of(mri)
    strain = lambda lam: lat.strain(deform.fit(lat, ev, lam, PARAMS.df_huber_mm))
    assert strain(min(PARAMS.df_lams)) < PARAMS.df_max_strain                              # the bump strains by 0.08
    assert deform.smallest_lam(lat, ev, PARAMS) == min(PARAMS.df_lams)
    for limit in (0.05, 0.02):                                                             # a limit inside the range of the list
        P = dataclasses.replace(PARAMS, df_max_strain=limit)
        lam = deform.smallest_lam(lat, ev, P)
        assert min(P.df_lams) < lam < max(P.df_lams) and lam not in P.df_lams
        assert strain(lam) < limit <= strain(lam / 1.2), (lam, strain(lam), strain(lam / 1.2))
        none = float(np.median(np.linalg.norm(ev["D"], axis=1)) + np.median(np.abs(ev["delta"])))
        best, table = deform.choose(lat, ev, none, P)
        assert [r["lam"] for r in table] == sorted([lam, *(x for x in P.df_lams if x > lam)], reverse=True)
        assert all(r["max_strain"] < limit and set(r) == {"lam", "interior_mm", "boundary_mm", "max_strain"} for r in table)
        assert best[0] == min(table, key=lambda r: r["interior_mm"] + r["boundary_mm"])["lam"]
    P = dataclasses.replace(PARAMS, df_max_strain=1e-3)                                    # even the largest weight strains more
    assert strain(max(P.df_lams)) > 1e-3 and deform.smallest_lam(lat, ev, P) is None
    assert deform.choose(lat, ev, 1.0, P) == (None, [])
    u, info = deform.smooth_deformation(mri, oct_, T, P, "cpu")
    assert info["status"] == "not_supported" and info["candidates"] == [] and info["cv"]["none"] and not u.any()


def test_recovers_a_smooth_displacement():
    v = lambda y: bump(y, 1.0)
    mri, oct_, T, *_ = pair(v)
    u, info = deform.smooth_deformation(mri, oct_, T, PARAMS, "cpu")
    assert info["status"] == "applied" and u.shape == (3, *mri[0].shape) and u.dtype == np.float32
    assert set(info) == {"status", "grid_mm", "lam", "max_strain", "n_interior", "n_boundary", "cv", "candidates", "residual", "field", "seconds"}
    assert info["grid_mm"] == PARAMS.df_grid_mm and 1 <= len(info["candidates"]) <= len(PARAMS.df_lams) + 1
    assert info["lam"] in [r["lam"] for r in info["candidates"]] and min(PARAMS.df_lams) <= info["lam"] <= max(PARAMS.df_lams)
    assert info["n_interior"] >= 100 and info["n_boundary"] >= 300 and sum(info["cv"]["chosen"]) < 0.2 * sum(info["cv"]["none"])
    res = info["residual"]
    assert res["interior_mm"][0] > 0.5 and res["interior_mm"][1] < 0.1 * res["interior_mm"][0], res
    assert res["boundary_mm"][0] > 0.2 and res["boundary_mm"][1] < 0.4 * res["boundary_mm"][0], res
    assert res["boundary_within_0.3mm"][1] > 0.97 and 0 < info["max_strain"] < PARAMS.df_max_strain, (res, info["max_strain"])
    x = world(mri[0].shape, mri[2])[mri[1].ravel()]
    truth = np.zeros_like(x)
    for _ in range(20):                                                                    # u(x) = v(x + u(x))
        truth = v(x + truth)
    err = np.linalg.norm(u[:, mri[1]].T - truth, axis=1)
    assert np.median(err) < 0.08 and np.percentile(err, 95) < 0.18 and err.max() < 0.32, (np.median(err), np.percentile(err, 95), err.max())
    assert abs(info["field"]["max_mm"] - np.linalg.norm(truth, axis=1).max()) < 0.1


def test_aligned_pair_is_left_alone():
    mri, oct_, T, *_ = pair(lambda y: np.zeros_like(y))
    u, info = deform.smooth_deformation(mri, oct_, T, PARAMS, "cpu")
    assert info["status"] == "not_supported" and not u.any() and info["grid_mm"] is None and info["lam"] is None, info
    assert info["n_interior"] >= 100 and info["n_boundary"] >= 300 and 1 <= len(info["candidates"]) <= len(PARAMS.df_lams) + 1
    assert all(r["interior_mm"] + r["boundary_mm"] >= PARAMS.df_gain * sum(info["cv"]["none"]) for r in info["candidates"])
    assert info["cv"]["chosen"] is None and info["max_strain"] == 0 and info["field"]["max_mm"] == 0
    assert all(a == b for a, b in info["residual"].values()) and info["residual"]["interior_mm"][0] < 0.05
    assert info["residual"]["boundary_mm"][0] < 0.1 and info["residual"]["boundary_within_0.3mm"][0] > 0.97
    few, info = deform.smooth_deformation(mri, oct_, T, dataclasses.replace(PARAMS, df_min_boundary=10 ** 6), "cpu")
    assert info["status"] == "not_supported" and not few.any() and info["candidates"] == [] and info["cv"]["none"] is None


def test_field_file_and_resample(tmp_path):
    """io.save_field / load_field and geometry.resample_to(field=...): out(x) = src(T^-1 (x + u(x)))."""
    rng = np.random.default_rng(0)
    shape_f, shape_s = (24, 22, 20), (40, 36, 44)
    A_F = np.array([[0.5, 0, 0, -6.0], [0, 0, 0.5, -5.0], [0, -0.5, 0, 5.5], [0, 0, 0, 1.0]])
    A_S = np.diag([-0.3, -0.3, 0.3, 1.0])
    A_S[:3, 3] = -A_S[:3, :3] @ ((np.array(shape_s) - 1) / 2)
    T = np.eye(4)                                                                           # source world -> field world
    T[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.15]).as_matrix() @ np.diag([1.03, 0.98, 1.0])
    T[:3, 3] = [0.4, -0.3, 0.2]
    field = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape_f), 3.0) for _ in range(3)]).astype(np.float32)
    field *= 0.8 / np.abs(field).max()
    src = ndimage.gaussian_filter(rng.standard_normal(shape_s), 1.5).astype(np.float32)
    io.save_field(field, A_F, tmp_path / "warp.nii.gz")
    import nibabel as nib
    img = nib.load(str(tmp_path / "warp.nii.gz"))
    assert img.shape == (*shape_f, 1, 3) and img.get_data_dtype() == np.float32 and img.header.get_intent()[0] == "vector"
    back, A_back = io.load_field(tmp_path / "warp.nii.gz")
    assert np.array_equal(back, field) and np.allclose(A_back, A_F)
    io.save_nifti(src, A_S, tmp_path / "src.nii.gz")
    with pytest.raises(ValueError, match="displacement field"):
        io.load_field(tmp_path / "src.nii.gz")
    # a finer destination grid inside the field grid, in the same world
    shape_d = (30, 28, 26)
    A_D = A_F.copy()
    A_D[:3, :3] *= 0.5
    A_D[:3, 3] = G.apply_affine(A_F, np.array([4.0, 4.0, 3.0]))
    x = torch.as_tensor(world(shape_d, A_D), dtype=torch.float64)
    ux = G.sample_world(torch.as_tensor(field, dtype=torch.float64), A_F, x).T
    want = G.sample_world(torch.as_tensor(src[None], dtype=torch.float64), A_S, G.apply_affine(np.linalg.inv(T), x + ux))[0].numpy()
    got = G.resample_to(src, A_S, shape_d, A_D, np.linalg.inv(T), field=back, affine_field=A_back)
    assert got.dtype == np.float32 and np.abs(got.ravel() - want).max() < 1e-5 and np.abs(want).max() > 0.05
    plain = G.resample_to(src, A_S, shape_d, A_D, np.linalg.inv(T))
    assert np.abs(got - plain).max() > 0.01                                                 # the field matters
    assert np.array_equal(G.resample_to(src, A_S, shape_d, A_D, np.linalg.inv(T), field=0 * back, affine_field=A_back), plain)
    # the stage's own warp of a volume already on the field grid
    on_grid = G.resample_to(src, A_S, shape_f, A_F, np.linalg.inv(T))
    warped = deform.warp(torch.as_tensor(on_grid[None]), A_F, torch.as_tensor(field))[0].numpy()
    assert np.abs(warped - G.resample_to(on_grid, A_F, shape_f, A_F, np.eye(4), field=back, affine_field=A_back)).max() < 1e-5


def test_lattice_and_fit():
    lat = deform.Lattice([-3.0, 0.0, 1.0], [7.5, 9.9, 11.0], 5.0)
    assert lat.n.tolist() == [4, 3, 3] and lat.K == 36
    rng = np.random.default_rng(0)
    X = rng.uniform([-3, 0, 1], [7.5, 9.9, 11], (50, 3))
    N = lat.basis(X)
    assert np.allclose(N.sum(1), 1) and N.nnz <= 8 * 50
    nodes = np.stack(np.meshgrid(*[lat.lo[a] + 5.0 * np.arange(lat.n[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
    c = np.stack([nodes @ g + b for g, b in (([0.01, 0.02, -0.01], 0.3), ([0.0, -0.03, 0.02], -0.1), ([0.04, 0.0, 0.0], 0.0))])
    linear = X @ np.array([[0.01, 0.02, -0.01], [0.0, -0.03, 0.02], [0.04, 0.0, 0.0]]).T + [0.3, -0.1, 0.0]
    assert np.allclose(N @ c.T, linear)                                                    # trilinear: exact on linear fields
    assert abs(lat.strain(c) - 0.04) < 1e-12
    A = np.diag([0.5, 0.5, 0.5, 1.0])
    A[:3, 3] = [-3, 0, 1]
    dense = lat.on_grid(c, (21, 19, 20), A, "cpu").numpy()
    assert np.allclose(dense.reshape(3, -1).T, lat.basis(world((21, 19, 20), A)) @ c.T, atol=1e-5)
    assert (deform.folds(np.array([[0.0, 0, 0], [6.9, 0, 0], [7.0, 0, 0], [0, 7.0, 0], [0, 0, -0.1]])) == [0, 0, 1, 3, 3]).all()
    n = rng.standard_normal((400, 3))                                                      # boundary rows alone (weight 1): a sphere
    n /= np.linalg.norm(n, axis=1, keepdims=True)                                          # moved by u0 shows n . u0 along its normals
    P, u0 = np.array([2.0, 5.0, 6.0]) + 3.0 * n, np.array([0.3, -0.2, 0.1])
    c = deform.fit(lat, {"W": np.zeros((0, 3)), "D": np.zeros((0, 3)), "P": P, "n": n, "delta": n @ u0}, 0.3, 0.3)
    assert np.abs(lat.basis(P) @ c.T - u0).max() < 1e-3 and lat.strain(c) < 1e-3
