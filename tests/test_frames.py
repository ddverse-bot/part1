"""params, io and geometry: header frames, plane streaming, isotropic resampling against direct pooling, sampling, the rotation
set, transform files (the ITK file checked with SimpleITK). Everything is synthetic (<= 96^3) in pytest temp dirs.
"""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import tifffile
import torch
from scipy import ndimage

from octreg import geometry as g
from octreg import io
from octreg.params import Params

ROOT = Path(__file__).resolve().parents[1]


def oblique(spacing, origin, seed=7, mirror=True):
    """voxel -> world affine with rotated axes; det < 0 when mirror."""
    A = np.eye(4)
    A[:3, :3] = g.rotations(20, seed)[11] @ np.diag(spacing) @ np.diag([1.0, 1.0, -1.0 if mirror else 1.0])
    A[:3, 3] = origin
    return A


def corners(shape):
    return np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)


def direct(data, A, k, shape_out, A_out):
    """Reference pooling on the whole array: k^3 box means (float64), then trilinear at the output voxel centres, blending with
    0 beyond the outer pooled centres."""
    n = np.array(data.shape) // k
    pooled = data[:n[0] * k[0], :n[1] * k[1], :n[2] * k[2]].reshape(n[0], k[0], n[1], k[1], n[2], k[2]).mean((1, 3, 5), dtype=np.float64)
    A_p = np.array(A, float)
    A_p[:3, 3] += A_p[:3, :3] @ ((k - 1) / 2.0)
    A_p[:3, :3] *= k
    c = g.apply_affine(np.linalg.inv(A_p) @ A_out, np.indices(shape_out).reshape(3, -1).T.astype(float))
    return ndimage.map_coordinates(pooled, c.T, order=1, mode="grid-constant", cval=0.0).reshape(shape_out), pooled.shape


def stack(vol):
    planes = list(io.iter_planes(vol))
    assert [k for k, _ in planes] == list(range(vol.shape[2])) and all(p.dtype == np.float32 for _, p in planes)
    return np.stack([p for _, p in planes], -1)


# ---------------------------------------------------------------------------------------------------------------- params
def test_params_defaults_dict_and_hash():
    p = Params()
    assert (p.search_mm, p.base_mm, p.fine_mm, p.n_rot, p.topk, p.iters, p.lam, p.clamp) == (0.6, 0.15, 0.04, 8000, 24, 200, 2.0, 0.15)
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    assert all(num(v) or (isinstance(v, tuple) and v and all(isinstance(x, float) for x in v)) for v in p.to_dict().values())  # constants
    assert Params.from_dict({"ngf_sigmas_mm": [0.5, 1]}).ngf_sigmas_mm == (0.5, 1.0)
    q = Params.from_dict(json.loads(json.dumps(dataclasses.replace(p, topk=12, lam=0.0).to_dict())))
    assert q == dataclasses.replace(p, topk=12, lam=0.0) and q.hash() != p.hash() and len(p.hash()) == 16
    assert Params.from_dict({"lam": 1, "clamp": 1}) == dataclasses.replace(p, lam=1.0, clamp=1.0)
    for bad in ({"rho": 0.8}, {"topk": 12.5}, {"topk": True}, {"polarity": "+1"}, {"destripe": False}, {"levels": [0.6, 0.15]},
                {"fine_mm": 0.2}, {"search_mm": 0.1}, {"search_mm": 0.55}, {"ngf_sigmas_mm": []}, {"ngf_sigmas_mm": [0.3, -0.1]},
                {"ngf_sigmas_mm": 0.3}, {"ngf_iters": 0}, {"ngf_erode_mm": 0.0}):
        with pytest.raises(ValueError):
            Params.from_dict(bad)
    assert p.hash() == "892a1f3b4fd6f7ed"                                             # octreg 1.1; 1.0: 7d01b8a167a83631
    assert {k for k in p.to_dict() if k.startswith("df_")} == {                       # §6: the two kinds of evidence, one
        "df_sigma_mm", "df_block_mm", "df_step_mm", "df_range_mm", "df_z_min", "df_erode_mm",                # lattice, one fit
        "df_reach_mm", "df_profile_mm", "df_edge_mad", "df_support_mm", "df_huber_mm", "df_grid_mm", "df_lams",
        "df_max_strain", "df_gain", "df_min_interior", "df_min_boundary"}
    assert (p.df_sigma_mm, p.df_block_mm, p.df_step_mm, p.df_range_mm, p.df_z_min, p.df_erode_mm) == (0.24, 4.5, 1.5, 1.35, 4.0, 0.6)
    assert (p.df_grid_mm, p.df_support_mm, p.df_edge_mad, p.df_lams) == (5.0, 5.0, 5.0, (30.0, 10.0, 3.0, 1.0, 0.3))
    assert Params.from_dict({"df_grid_mm": 7, "df_lams": [3, 1]}) == dataclasses.replace(p, df_grid_mm=7.0, df_lams=(3.0, 1.0))
    for bad, message in (({"df_grids_mm": [10.0, 7.0, 5.0]}, "unknown key"), ({"df_rounds": 2}, "unknown key"),
                         ({"df_ridge_mad": 5.0}, "unknown key"), ({"df_grid_mm": [5.0]}, "not a float"), ({"df_lams": []}, "df_lams"),
                         ({"df_lams": [1.0, 0.0]}, "df_lams"), ({"df_reach_mm": 3.0}, "df_reach_mm <= df_profile_mm"),
                         ({"df_min_interior": 0}, "df_min_interior"),
                         *(({k: v}, "df_ lengths") for k, v in (("df_sigma_mm", 0.0), ("df_block_mm", -1.0), ("df_step_mm", -0.5),
                                                                ("df_z_min", 0.0), ("df_erode_mm", 0.0), ("df_grid_mm", 0.0),
                                                                ("df_support_mm", -1.0), ("df_edge_mad", 0.0)))):
        with pytest.raises(ValueError, match=message):
            Params.from_dict(bad)
    env = dict(os.environ, PYTHONHASHSEED="4242", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", "from octreg.params import Params; print(Params().hash())"], env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == p.hash()


# ------------------------------------------------------------------------------------------------------------------- io
@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
@pytest.mark.parametrize("form", ["sform", "qform"])
def test_oblique_det_negative_nifti_roundtrip(tmp_path, suffix, form):
    shape = (10, 12, 14)
    A = oblique((20.0, 25.0, 30.0), (-7300.0, 4100.0, 6900.0))                       # um
    assert np.linalg.det(A[:3, :3]) < 0
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    img = nib.Nifti1Image(data, None)
    img.header.set_xyzt_units("micron")
    img.header.set_qform(A, code=1)
    img.header.set_sform(A if form == "sform" else None, code=1 if form == "sform" else 0)
    nib.save(img, str(tmp_path / f"o{suffix}"))
    v = io.load_volume(tmp_path / f"o{suffix}")
    assert v.frame == "header" and v.shape == shape and np.allclose(v.spacing_mm, (0.02, 0.025, 0.03), atol=1e-7)
    assert np.abs(corners(shape) @ (v.affine[:3, :3] - A[:3, :3] / 1000).T + v.affine[:3, 3] - A[:3, 3] / 1000).max() < 1e-6
    io.save_nifti(data, v.affine, tmp_path / f"s{suffix}")                            # writer -> reader round trip
    w = io.load_volume(tmp_path / f"s{suffix}")
    assert np.abs(corners(shape) @ (w.affine - v.affine)[:3, :3].T + (w.affine - v.affine)[:3, 3]).max() < 1e-6
    assert np.array_equal(stack(w), data)
    s = io.load_volume(tmp_path / f"o{suffix}", spacing_um=(45, 30, 25))              # Z,Y,X = along (k, j, i)
    assert np.allclose(s.spacing_mm, (0.025, 0.03, 0.045)) and np.allclose(s.affine[:3, 3], A[:3, 3] / 1000, atol=1e-6)
    assert np.allclose(s.affine[:3, :3] / s.spacing_mm, A[:3, :3] / (20.0, 25.0, 30.0), atol=1e-6)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_streamed_planes_equal_nibabel(tmp_path, suffix):
    rng = np.random.default_rng(1)
    raw = rng.integers(-3000, 3000, (9, 11, 13, 1)).astype(np.int16)                  # trailing singleton, scaled int16
    img = nib.Nifti1Image(raw, np.diag([0.08, 0.08, 0.08, 1.0]))
    img.header.set_slope_inter(0.5, 3.0)
    nib.save(img, str(tmp_path / f"i{suffix}"))
    f = rng.random((9, 11, 13)).astype(np.float32)                                     # float with NaN / inf
    f[1, 2, 3], f[4, 5, 6] = np.nan, np.inf
    nib.save(nib.Nifti1Image(f, None), str(tmp_path / f"f{suffix}"))                     # no sform / qform: array frame
    for name in ("i", "f"):
        v = io.load_volume(tmp_path / f"{name}{suffix}")
        ref = np.nan_to_num(np.asarray(nib.load(str(tmp_path / f"{name}{suffix}")).get_fdata(), np.float32), posinf=0.0)
        assert v.shape == (9, 11, 13) and np.array_equal(stack(v), ref.reshape(v.shape))
    assert io.load_volume(tmp_path / f"f{suffix}").frame == "array"
    with pytest.raises(ValueError, match="3-D"):
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 2), np.float32), np.eye(4)), str(tmp_path / f"t{suffix}"))
        io.load_volume(tmp_path / f"t{suffix}")


def test_tiff_npy_axes_spacing_and_refusals(tmp_path):
    a = (np.random.default_rng(2).random((6, 8, 10)) * 4000).astype(np.uint16)          # numpy (z, y, x)
    tifffile.imwrite(tmp_path / "o.ome.tif", a, ome=True, metadata={"axes": "ZYX", "PhysicalSizeX": 0.012, "PhysicalSizeXUnit": "mm",
                     "PhysicalSizeY": 13.0, "PhysicalSizeZ": 14.0})
    v = io.load_volume(tmp_path / "o.ome.tif")
    assert v.frame == "array" and v.shape == (10, 8, 6) and np.allclose(v.affine, np.diag([0.012, 0.013, 0.014, 1.0]))
    assert np.array_equal(stack(v), a.transpose(2, 1, 0))
    tifffile.imwrite(tmp_path / "c.tif", a, compression="zlib")                         # read page by page
    with pytest.raises(ValueError, match="no voxel spacing"):
        io.load_volume(tmp_path / "c.tif")
    v = io.load_volume(tmp_path / "c.tif", spacing_um=(35, 31, 30))
    assert np.allclose(v.spacing_mm, (0.030, 0.031, 0.035)) and np.array_equal(stack(v), a.transpose(2, 1, 0))
    np.save(tmp_path / "n.npy", a)
    v = io.load_volume(tmp_path / "n.npy", spacing_um=(14, 12, 10))                    # Z,Y,X
    assert np.allclose(v.spacing_mm, (0.010, 0.012, 0.014)) and np.array_equal(stack(v), a.transpose(2, 1, 0))
    with pytest.raises(ValueError, match="no voxel spacing"):
        io.load_volume(tmp_path / "n.npy")
    np.save(tmp_path / "flat.npy", np.zeros((5, 6), np.float32))
    with pytest.raises(ValueError, match="3-D"):
        io.load_volume(tmp_path / "flat.npy", spacing_um=(1, 1, 1))
    with pytest.raises(ValueError, match="spacing_um"):
        io.load_volume(tmp_path / "n.npy", spacing_um=(12, 0, 12))
    (tmp_path / "x.mgz").write_bytes(b"0")
    for bad, msg in (("x.mgz", "unsupported"), ("missing.nii.gz", "no such file")):
        with pytest.raises(ValueError, match=msg):
            io.load_volume(tmp_path / bad)


# ------------------------------------------------------------------------------------------------------------- geometry
def test_resample_iso_equals_direct_pooling(tmp_path):
    rng = np.random.default_rng(3)
    shape, sp = (64, 72, 80), (0.020, 0.025, 0.030)
    A = oblique(sp, (1.0, -2.0, 3.0))
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).astype(float)
    ramp = (ijk @ A[:3, :3].T + A[:3, 3]) @ np.array([0.7, -1.1, 0.4])                # linear in world mm
    noise = rng.random(shape).astype(np.float32)
    for name, data in (("ramp", ramp.astype(np.float32)), ("noise", noise)):
        io.save_nifti(data, A, tmp_path / f"{name}.nii.gz")
        vol = io.load_volume(tmp_path / f"{name}.nii.gz")
        data = stack(vol)
        for level in (0.15, 0.1, 0.06):
            out, A_out = g.resample_iso(vol, level)
            assert np.allclose(np.linalg.norm(A_out[:3, :3], axis=0), level)
            p2, A2 = g.pool_iso(data, vol.affine, level)
            assert np.array_equal(out, p2) and np.array_equal(A_out, A2)
            k = np.maximum(1, np.rint(np.round(level / np.array(sp), 9))).astype(int)
            ref, pooled_shape = direct(data, vol.affine, k, out.shape, A_out)
            assert out.shape == tuple(int(np.floor(np.round(pooled_shape[i] * sp[i] * k[i] / level, 9))) for i in range(3))
            np.testing.assert_allclose(out, ref, atol=2e-4 * max(1.0, np.abs(data).max()))
            if name == "ramp":                                                        # trilinear of a linear field is exact
                ijk_out = np.stack(np.meshgrid(*[np.arange(s) for s in out.shape], indexing="ij"), -1).astype(float)
                inside = np.all(ijk_out * level <= (np.array(pooled_shape) - 1) * vol.spacing_mm * k + 1e-9, -1)
                truth = (ijk_out @ A_out[:3, :3].T + A_out[:3, 3]) @ np.array([0.7, -1.1, 0.4])
                assert inside.mean() > 0.5 and np.abs(out - truth)[inside].max() < 1e-3
    io.save_nifti(noise, np.diag([0.02, -0.02, 0.02, 1.0]), tmp_path / "iso.nii.gz")    # float32 pixdim 0.0199999996
    out, A_out = g.resample_iso(io.load_volume(tmp_path / "iso.nii.gz"), 0.04)
    assert out.shape == (32, 36, 40) and np.allclose(out, noise.reshape(32, 2, 36, 2, 40, 2).mean((1, 3, 5)), atol=1e-6)
    mask = noise > 0.7
    io.save_nifti((mask * 255).astype(np.uint8), np.diag([0.02, -0.02, 0.02, 1.0]), tmp_path / "mask.nii.gz")
    frac, _ = g.resample_iso(io.load_volume(tmp_path / "mask.nii.gz"), 0.04, binary=True)       # indicator, not the values
    assert np.allclose(frac, mask.reshape(32, 2, 36, 2, 40, 2).mean((1, 3, 5)), atol=1e-6)
    A = oblique((0.02, 0.02, 0.02), (0.0, 0.0, 0.0))                  # k = 8, step 0.9375: 15 pooled -> 16 planes, the last
    ones, _ = g.pool_iso(np.ones((120, 24, 120), np.float32), A, 0.15)                 # one past the last pooled centre
    assert ones.shape == (16, 3, 16) and np.allclose(ones[:15, :, :15], 1) and np.allclose(ones[15, :, :15], 0.9375)
    assert np.allclose(ones[15, :, 15], 0.9375 ** 2)
    edge = rng.random((120, 24, 120)).astype(np.float32)
    out, A_out = g.pool_iso(edge, A, 0.15)
    assert np.allclose(out, direct(edge, A, np.array([8, 8, 8]), out.shape, A_out)[0], atol=1e-5)
    with pytest.raises(ValueError, match="fewer than 2"):
        g.pool_iso(noise[:4, :4, :4], np.diag([0.02, 0.02, 0.02, 1]), 0.6)


def test_sampling_primitives():
    rng = np.random.default_rng(4)
    shape = (12, 14, 16)
    A = oblique((0.2, 0.3, 0.25), (-1.0, 2.0, 0.5))
    x = rng.random(shape).astype(np.float32)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = g.rotations(10, 3)[4] * 1.05, (0.3, -0.2, 0.1)
    A_dst = oblique((0.22, 0.22, 0.22), (0.0, 0.0, 0.0), seed=2, mirror=False)
    A_dst[:3, 3] = g.apply_affine(np.linalg.inv(T), g.apply_affine(A, (np.array(shape) - 1) / 2)) - A_dst[:3, :3] @ [7.0, 6.0, 8.0]
    out = g.resample_to(x, A, (15, 13, 17), A_dst, T)                                 # scipy map_coordinates
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in (15, 13, 17)], indexing="ij"), -1).reshape(-1, 3).astype(float)
    pts = torch.as_tensor(g.apply_affine(T, g.apply_affine(A_dst, ijk)))
    ref = g.sample_world(torch.as_tensor(x)[None], A, pts)[0].numpy().reshape(out.shape)  # torch grid_sample
    assert (out != 0).mean() > 0.3 and (out == 0).mean() > 0.05 and np.allclose(out, ref, atol=1e-5)
    m = x > 0.5
    assert np.array_equal(g.resample_to(m, A, shape, A, np.eye(4), order=0), m)


def test_rotation_set():
    """The fixed rotation set (seed 0) that every search uses: stored entries and column sums."""
    R = g.rotations(8000, 0)
    assert np.array_equal(R[0], np.eye(3)) and np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[1], [[-0.722930144505603, 0.649299596414965, -0.236182218340989],
                              [-0.023748891083315, 0.318282527797234, 0.947698381697107],
                              [0.690512850233376, 0.690728793807212, -0.214675888418364]], atol=1e-12)
    assert np.allclose(R[7999], [[0.838506604404748, -0.337798770672833, 0.427549605193997],
                                 [-0.533017324098544, -0.345580223090142, 0.772312657943527],
                                 [-0.113133578483348, -0.875480610830926, -0.469823896240561]], atol=1e-12)
    assert np.allclose(R.sum(0), [[26.68730805, -12.852368481, 44.494519067], [23.638512738, -0.248947076, 14.27878513],
                                  [-53.911386299, -74.17969957, -6.435933718]], atol=1e-8)


# ------------------------------------------------------------------------------------------------------- transform files
def test_transform_txt_lta_itk_roundtrip(tmp_path):
    shape_o, shape_m = (40, 30, 20), (34, 48, 50)
    A_o, A_m = oblique((0.02, 0.02, 0.02), (5.0, -3.0, 2.0)), oblique((0.08, 0.08, 0.08), (-10.0, 12.0, -7.0), seed=3, mirror=False)
    io.save_nifti(np.zeros(shape_o, np.float32), A_o, tmp_path / "oct.nii.gz")
    io.save_nifti(np.zeros(shape_m, np.float32), A_m, tmp_path / "mri.nii.gz")
    src, dst = io.load_volume(tmp_path / "oct.nii.gz"), io.load_volume(tmp_path / "mri.nii.gz")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = g.rotations(10, 5)[6] @ np.diag([1.04, 0.97, -1.01]), (3.5, -8.25, 11.0)
    io.write_transform_txt(T, tmp_path / "T.txt")
    assert np.array_equal(np.loadtxt(tmp_path / "T.txt"), T)
    io.write_lta(T, src, dst, tmp_path / "x.lta")
    assert b"\r" not in (tmp_path / "x.lta").read_bytes() + (tmp_path / "T.txt").read_bytes()
    text = (tmp_path / "x.lta").read_text()
    rows = text.split("1 4 4\n")[1].splitlines()[:4]
    assert "LINEAR_RAS_TO_RAS" in text and np.allclose(np.array([r.split() for r in rows], float), T, rtol=1e-14, atol=1e-14)
    for block, vol in ((text.split("src volume info")[1].split("dst volume info")[0], src), (text.split("dst volume info")[1], dst)):
        info = {k.strip(): val.split("#")[0].split() for k, val in (ln.split("=", 1) for ln in block.splitlines() if "=" in ln)}
        num = lambda key: np.array(info[key], float)
        hdr = nib.MGHImage(np.zeros(vol.shape, np.float32), vol.affine).header               # FreeSurfer's own geometry
        assert [int(s) for s in info["volume"]] == list(vol.shape) and info["filename"] == [str(Path(vol.path).resolve())]
        assert np.allclose(num("cras"), hdr["Pxyz_c"], atol=1e-5) and np.allclose(num("voxelsize"), hdr["delta"], atol=1e-6)
        assert np.allclose(np.stack([num("xras"), num("yras"), num("zras")]), hdr["Mdc"], atol=1e-6)
    io.write_json({"T": T, "p": Params(), "nan": float("nan"), "f": np.float32(1.5), "path": tmp_path}, tmp_path / "r.json")
    back = json.loads((tmp_path / "r.json").read_text())
    assert np.array_equal(back["T"], T) and back["p"]["search_mm"] == 0.6 and back["nan"] is None and back["f"] == 1.5
    sitk = pytest.importorskip("SimpleITK")                   # ITK file: dst voxel -> dst physical -> transform -> src voxel
    np.save(tmp_path / "a.npy", np.zeros((14, 12, 10), np.float32))                  # numpy (z, y, x): volume (10, 12, 14)
    arr = io.load_volume(tmp_path / "a.npy", spacing_um=(40, 30, 20))
    a_img = sitk.GetImageFromArray(np.zeros((14, 12, 10), np.float32))
    a_img.SetSpacing(tuple(arr.spacing_mm))                                             # ITK index order (x, y, z)
    images = {id(src): sitk.ReadImage(str(tmp_path / "oct.nii.gz")), id(dst): sitk.ReadImage(str(tmp_path / "mri.nii.gz")), id(arr): a_img}
    q = np.array([3.0, 4.0, 5.0])
    for s, d in ((src, dst), (arr, dst), (src, arr)):
        io.write_itk(T, s, d, tmp_path / "x_itk.txt")
        assert b"\r" not in (tmp_path / "x_itk.txt").read_bytes()
        p = sitk.ReadTransform(str(tmp_path / "x_itk.txt")).TransformPoint(images[id(d)].TransformContinuousIndexToPhysicalPoint(tuple(q)))
        want = g.apply_affine(np.linalg.inv(s.affine) @ np.linalg.inv(T) @ d.affine, q)
        assert np.allclose(images[id(s)].TransformPhysicalPointToContinuousIndex(p), want, atol=1e-4)
