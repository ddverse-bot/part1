"""End to end through the command line on a small synthetic pair, CPU: an agarose-embedded OCT block (equal mean intensity,
section offsets, tile seams, a black tile, inverted contrast, LPI-like header) and an MRI crop (RIA-like header) related by
a known affine. Reduced search and refinement through --params; the pose is recovered to < 0.4 mm (0.12-0.28 mm over seeds 0-2),
the texture mask finds the specimen where the histogram rule has no valley, every output is written, `octreg apply`
reproduces oct_in_mri.nii.gz and `octreg qc` reproduces the visual QC figures; the 6 mm specimen holds no 4.5 mm block, so the
smooth deformation (§6) reports not_supported and writes no field, and `octreg apply` uses a field file put into the run
directory; user masks replace both foregrounds; the same OCT as an NPY stack, whose array frame is the mirror image of the
NIfTI world, is registered in its correct handedness and replayed by `octreg apply --oct-spacing-um` from another path."""
import json

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from octreg import geometry as G, io, preprocess as pp
from octreg.cli import main
from octreg.params import Params
from octreg.register import OUTPUTS, OUTPUTS_DEFORM, QC_CELL_IN, QC_COLOURS, QC_DPI, QC_HEAD_IN, WARP, deform_label


def distance(T1, T2, pts):
    return np.linalg.norm(G.apply_affine(T1, pts) - G.apply_affine(T2, pts), axis=1).mean()


FAST = {"fine_mm": 0.08, "n_rot": 24, "topk": 6, "iters": 120,
        "texture_smooth_mm": 0.32}                  # the synthetic block is a few mm across (the default 1.2 mm suits real blocks)


def field(world, grid):
    """Trilinear samples of a volume (numpy [D, H, W], affine) at world points [N, 3]."""
    vol, A = grid
    return G.sample_world(torch.as_tensor(vol[None]), A, torch.as_tensor(world, dtype=torch.float32))[0].numpy()


def pair(tmp, seed=0):
    """-> (OCT path, MRI path, T_true OCT world -> MRI world, OCT specimen points in OCT world mm)."""
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.standard_normal((96,) * 3), 5.0)                    # class structure, world grid 0.15 mm
    A_g = np.diag([0.15, 0.15, 0.15, 1.0])
    A_g[:3, 3] = -0.15 * 47.5
    classes = ((1 / (1 + np.exp(-g / (0.25 * g.std())))).astype(np.float32), A_g)
    centre, semi = np.array([0.3, -0.2, 0.1]), np.array([3.0, 2.6, 2.3])                # specimen ellipsoid in the MRI world

    def tissue(x):
        return (((x - centre) / semi) ** 2).sum(-1) <= 1

    A_M = np.array([[0.15, 0, 0, 0], [0, 0, 0.15, 0], [0, -0.15, 0, 0], [0, 0, 0, 1.0]])   # i -> R, j -> I, k -> A
    shape_m = (48, 52, 44)
    A_M[:3, 3] = -A_M[:3, :3] @ ((np.array(shape_m) - 1) / 2)
    x = G.apply_affine(A_M, np.indices(shape_m).reshape(3, -1).T.astype(float))
    m = tissue(x)
    mri = np.where(m, 600 + 200 * field(x, classes), 80) + rng.normal(0, 70, m.shape)     # overlapping classes, one valley
    io.save_nifti(np.maximum(mri, 0).reshape(shape_m).astype(np.float32), A_M, tmp / "mri.nii.gz")

    shape_o = (96, 92, 84)
    A_O = np.diag([-0.08, -0.08, -0.08, 1.0])                                            # i -> L, j -> P, k -> I
    A_O[:3, 3] = np.array([12.0, -3.0, 7.0]) - A_O[:3, :3] @ ((np.array(shape_o) - 1) / 2)
    c_o = G.apply_affine(A_O, (np.array(shape_o) - 1) / 2)
    T = np.eye(4)
    T[:3, :3] = G.rotations(FAST["n_rot"], 0)[7] @ _rot([0.6, -0.5, 0.62], 4.0) @ np.diag([1.03, 1.0, 1.0])
    T[:3, 3] = centre + np.array([0.25, 0.1, -0.15]) - T[:3, :3] @ c_o
    xo = G.apply_affine(A_O, np.indices(shape_o).reshape(3, -1).T.astype(float))
    xm = G.apply_affine(T, xo)
    inside = tissue(xm).reshape(shape_o)
    tex = ndimage.gaussian_filter(rng.standard_normal(shape_o), 1.5)
    stripes = rng.normal(0, 0.05, shape_o[2])[None, None, :] + 0.1 * (np.arange(shape_o[0]) % 24 < 2)[:, None, None]
    bright = (0.8 + 0.4 * (1 - field(xm, classes))).reshape(shape_o)                     # inverted contrast
    oct_ = 1000 * (1 + stripes) * np.where(inside, bright * (1 + 0.25 * tex / tex.std()), 1.0)
    oct_ = np.maximum(oct_ * (1 + 0.1 * rng.standard_normal(shape_o)), 1).astype(np.float32)
    oct_[:, :10, :10] = 0
    io.save_nifti(oct_, A_O, tmp / "oct.nii.gz")
    return tmp / "oct.nii.gz", tmp / "mri.nii.gz", T, xo[inside.ravel() & (oct_.ravel() > 0)]


def png(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return torch.linalg.matrix_exp(torch.as_tensor(K)).numpy()


def test_register_and_apply(tmp_path, capsys):
    oct_path, mri_path, T_true, pts = pair(tmp_path)
    (tmp_path / "params.json").write_text(json.dumps(FAST))
    out = tmp_path / "run"
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "params.json")]) == 0
    assert all((out / f).is_file() for f in OUTPUTS)
    res = json.loads((out / "result.json").read_text())
    T = np.loadtxt(out / "T_oct2mri.txt")
    assert distance(T, T_true, pts) < 0.4
    assert res["pose"]["polarity"] == -1 and np.linalg.det(np.loadtxt(out / "T_oct2mri.txt")[:3, :3]) > 0 and "nondefault_params" in res["flags"]
    assert np.allclose(np.loadtxt(out / "T_mri2oct.txt") @ T, np.eye(4), atol=1e-9) and np.allclose(res["T_oct2mri"], T)
    assert res["foreground"]["oct"]["source"] == "texture" and res["foreground"]["mri"]["status"] == "ok"
    oct_img = nib.load(str(oct_path))                          # equal intensities: no histogram valley; texture: the specimen
    raw_h = G.pool_iso(np.asarray(oct_img.dataobj, np.float32), oct_img.affine, 0.15)[0]
    assert pp.foreground(raw_h, 0.15, Params())[1]["status"] == "no_valley"
    assert 0.9 < res["foreground"]["oct"]["volume_cm3"] / (len(pts) * 0.08 ** 3 / 1e3) < 1.3          # 1.10-1.12 over seeds
    scale = res["pose"]["scale_per_oct_axis"]                   # true 1.03 / 1 / 1; the mask margin biases it low (0.91-0.97)
    assert 0.85 < min(scale) and max(scale) < 1.1
    assert res["params"]["n_rot"] == FAST["n_rot"] and set(res["seconds"]) >= {"oct_mask", "search_refine_ngf", "total"}
    assert res["refine"]["n_poses"] == FAST["topk"] and res["search"]["n_orientations"] == FAST["n_rot"]
    d = res["deform"]                                          # §6: no 4.5 mm block fits the specimen, so no deformation
    assert d["status"] == "not_supported" and "deformation_not_supported" in res["flags"] and d["n_interior"] == 0 and "deform" in res["seconds"]
    assert set(d) == {"status", "grid_mm", "lam", "max_strain", "n_interior", "n_boundary", "cv", "candidates", "residual", "field", "seconds"}
    assert d["n_boundary"] == 0 and d["candidates"] == [] and d["lam"] is None      # no match nearby: no boundary point is used,
    assert d["residual"]["interior_mm"] == [None, None] and d["residual"]["boundary_mm"][0] > 0   # but the edges are read out
    assert d["residual"]["boundary_mm"][0] == d["residual"]["boundary_mm"][1]
    assert "§6: not_supported (0 interior matches, 0 boundary points)" in capsys.readouterr().out
    assert d["field"]["max_mm"] == 0 and not any((out / f).exists() for f in OUTPUTS_DEFORM)
    assert np.array_equal(nib.load(str(out / "oct_in_mri_affine.nii.gz")).get_fdata(), nib.load(str(out / "oct_in_mri.nii.gz")).get_fdata())

    ref = nib.load(str(out / "oct_in_mri.nii.gz"))
    assert ref.shape == nib.load(str(mri_path)).shape and np.allclose(ref.affine, nib.load(str(mri_path)).affine, atol=1e-6)
    assert main(["apply", "--run", str(out), "--moving", str(oct_path), "--reference", str(mri_path), "-o",
                 str(tmp_path / "again.nii.gz")]) == 0
    np.testing.assert_allclose(nib.load(str(tmp_path / "again.nii.gz")).get_fdata(), ref.get_fdata(), rtol=1e-5, atol=1e-3)
    assert main(["apply", "--run", str(out), "--moving", str(mri_path), "--reference", str(oct_path), "-o",
                 str(tmp_path / "back.nii.gz"), "--inverse"]) == 0
    back = nib.load(str(tmp_path / "back.nii.gz"))
    assert back.shape == nib.load(str(oct_path)).shape and np.asarray(back.dataobj).max() > 500

    qc, montage = png(out / "qc.png"), png(out / "qc_montage.png")        # 3 and 12 rows of planes, both outlines drawn
    assert [round((x.shape[0] / QC_DPI - QC_HEAD_IN) / QC_CELL_IN, 2) for x in (qc, montage)] == [3, 12]
    assert all((x == list(bytes.fromhex(c[1:]))).all(-1).sum() > 100 for x in (qc, montage) for c in QC_COLOURS)
    run_qc = ["qc", "--run", str(out), "--oct", str(oct_path), "--mri", str(mri_path)]
    assert main(run_qc + ["--T", str(out / "T_oct2mri.txt"), "-o", str(tmp_path / "qc" / "again")]) == 0
    np.testing.assert_array_equal(png(tmp_path / "qc" / "again.png"), qc)
    np.testing.assert_array_equal(png(tmp_path / "qc" / "again_montage.png"), montage)
    np.savetxt(tmp_path / "T_true.txt", T_true)
    assert main(run_qc + ["--T", str(tmp_path / "T_true.txt")]) == 0 and (out / "qc_T_true_montage.png").is_file()
    assert main(run_qc + ["--T", str(tmp_path / "missing.txt")]) == 2
    assert main(["register", str(tmp_path / "missing.nii.gz"), str(mri_path), "-o", str(out), "--device", "cpu"]) == 2
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--oct-mask", "missing.nii.gz", "--device", "cpu"]) == 2
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "missing.json")]) == 2
    assert main(["apply", "--run", str(tmp_path / "missing_run"), "--moving", str(oct_path), "--reference", str(mri_path),
                 "-o", str(tmp_path / "never.nii.gz")]) == 2

    # `octreg apply` with a field in the run directory: u = one MRI voxel along MRI array axis 0, so the OCT content moves by one voxel
    A_f = ref.affine.copy()
    A_f[:3, :3] *= 16
    io.save_field(np.broadcast_to((ref.affine[:3, 0])[:, None, None, None], (3, 4, 4, 4)), A_f, out / WARP)
    run = ["apply", "--run", str(out), "--moving", str(oct_path), "--reference", str(mri_path), "-o"]
    assert main(run + [str(tmp_path / "warped.nii.gz")]) == 0 and main(run + [str(tmp_path / "plain.nii.gz"), "--affine-only"]) == 0
    warped, plain = (nib.load(str(tmp_path / f)).get_fdata() for f in ("warped.nii.gz", "plain.nii.gz"))
    np.testing.assert_allclose(plain, ref.get_fdata(), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(warped[:-1], plain[1:], rtol=1e-4, atol=0.05)
    capsys.readouterr()
    assert main(["apply", "--run", str(out), "--moving", str(mri_path), "--reference", str(oct_path), "-o", str(tmp_path / "back2.nii.gz"),
                 "--inverse"]) == 0 and "not inverted" in capsys.readouterr().out
    np.testing.assert_array_equal(nib.load(str(tmp_path / "back2.nii.gz")).get_fdata(), back.get_fdata())


def test_outputs_with_a_field(tmp_path, monkeypatch):
    """The outputs of register when §6 applies a field (here a given smooth one in place of deform.smooth_deformation, which the
    small specimen cannot support): the field file, both overlays, qc_deform.png, and `octreg apply` with and without the field."""
    oct_path, mri_path, T_true, pts = pair(tmp_path)

    def given(mri, oct_, T, params, device):
        x = G.apply_affine(mri[2], np.indices(mri[0].shape).reshape(3, -1).T.astype(float)).T.reshape(3, *mri[0].shape)
        u = (0.3 * np.stack([np.sin(x[1] / 2), np.cos(x[2] / 2), np.sin(x[0] / 3)])).astype(np.float32)
        return u, {"status": "applied", "grid_mm": 5.0, "lam": 0.324, "max_strain": 0.1, "n_interior": 120, "n_boundary": 400,
                   "cv": {"none": [0.3, 0.5], "chosen": [0.15, 0.25]},
                   "candidates": [{"lam": 0.324, "interior_mm": 0.15, "boundary_mm": 0.25, "max_strain": 0.1}],
                   "residual": {"interior_mm": [0.3, 0.1], "boundary_mm": [0.5, 0.2], "boundary_within_0.3mm": [0.3, 0.6]},
                   "field": {"median_mm": 0.3, "p95_mm": 0.5, "max_mm": float(np.linalg.norm(u, axis=0).max())}, "seconds": 0.0}

    monkeypatch.setattr("octreg.register.smooth_deformation", given)
    (tmp_path / "params.json").write_text(json.dumps(FAST))
    out = tmp_path / "run"
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--device", "cpu", "--params", str(tmp_path / "params.json")]) == 0
    res = json.loads((out / "result.json").read_text())
    assert all((out / f).is_file() for f in OUTPUTS + OUTPUTS_DEFORM) and "deformation_not_supported" not in res["flags"]
    assert res["deform"]["status"] == "applied" and distance(np.loadtxt(out / "T_oct2mri.txt"), T_true, pts) < 0.4      # T is the affine
    u, A_u = io.load_field(out / WARP)
    warp = nib.load(str(out / WARP))
    assert warp.shape[3:] == (1, 3) and warp.header.get_intent()[0] == "vector" and abs(np.abs(u).max() - 0.3) < 1e-3
    base, A_base = G.resample_iso(io.load_volume(mri_path), 0.15)                       # the field lives on the MRI base grid
    assert u.shape[1:] == base.shape and np.allclose(A_u, A_base, atol=1e-6)
    run = ["apply", "--run", str(out), "--moving", str(oct_path), "--reference", str(mri_path), "-o"]
    assert main(run + [str(tmp_path / "warped.nii.gz")]) == 0 and main(run + [str(tmp_path / "plain.nii.gz"), "--affine-only"]) == 0
    both = [nib.load(str(f)).get_fdata() for f in (out / "oct_in_mri.nii.gz", out / "oct_in_mri_affine.nii.gz")]
    np.testing.assert_allclose(nib.load(str(tmp_path / "warped.nii.gz")).get_fdata(), both[0], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(nib.load(str(tmp_path / "plain.nii.gz")).get_fdata(), both[1], rtol=1e-5, atol=1e-3)
    assert np.abs(both[0] - both[1]).max() > 50
    assert deform_label(res["deform"]).startswith("§6: lattice 5 mm, lam 0.32; interior 0.30 -> 0.10 mm, boundary 0.50 -> 0.20 mm "
                                                  "(within 0.3 mm: 30 -> 60 %); field median 0.30, max ")
    fig = png(out / "qc_deform.png")                           # 3 rows of planes, the MRI foreground outline drawn
    assert round((fig.shape[0] / QC_DPI - QC_HEAD_IN) / QC_CELL_IN, 2) == 3 and (fig == list(bytes.fromhex(QC_COLOURS[0][1:]))).all(-1).sum() > 100


def test_user_masks(tmp_path):
    """--oct-mask (the true specimen on its own 0.16 mm grid) and --mri-mask replace the texture mask and the histogram rule."""
    oct_path, mri_path, T_true, pts = pair(tmp_path, seed=1)
    A = io.load_volume(oct_path).affine.copy()
    A[:3, :3] *= 2
    mask = np.zeros((48, 46, 42), np.uint8)
    mask[tuple(np.rint(G.apply_affine(np.linalg.inv(A), pts)).astype(int).T)] = 1
    io.save_nifti(mask, A, tmp_path / "oct_mask.nii.gz")
    mri = nib.load(str(mri_path))
    io.save_nifti((np.asarray(mri.dataobj) > 350).astype(np.uint8), mri.affine, tmp_path / "mri_mask.nii.gz")
    (tmp_path / "params.json").write_text(json.dumps(FAST))
    out = tmp_path / "run"
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--oct-mask", str(tmp_path / "oct_mask.nii.gz"),
                 "--mri-mask", str(tmp_path / "mri_mask.nii.gz"), "--device", "cpu", "--params", str(tmp_path / "params.json")]) == 0
    res = json.loads((out / "result.json").read_text())
    assert res["foreground"]["oct"]["source"].endswith("oct_mask.nii.gz") and res["foreground"]["mri"]["source"].endswith("mri_mask.nii.gz")
    assert distance(np.loadtxt(out / "T_oct2mri.txt"), T_true, pts) < 0.4 and res["pose"]["polarity"] == -1


def test_array_frame_handedness(tmp_path):
    """An NPY stack has no orientation: both handednesses are tried and fine structure picks the mirrored one here."""
    oct_path, mri_path, T_true, pts = pair(tmp_path, seed=0)
    img = nib.load(str(oct_path))
    np.save(tmp_path / "oct.npy", np.asarray(img.dataobj, np.float32).transpose(2, 1, 0))       # numpy (z, y, x)
    A_arr = np.diag([0.08, 0.08, 0.08, 1.0])
    T_arr = T_true @ img.affine @ np.linalg.inv(A_arr)                                          # array world -> MRI world, det < 0
    pts_arr = G.apply_affine(A_arr @ np.linalg.inv(img.affine), pts)
    (tmp_path / "params.json").write_text(json.dumps(FAST))
    out = tmp_path / "run"
    assert main(["register", str(tmp_path / "oct.npy"), str(mri_path), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "params.json"), "--oct-spacing-um", "80,80,80"]) == 0
    res, T = json.loads((out / "result.json").read_text()), np.loadtxt(out / "T_oct2mri.txt")
    assert np.linalg.det(T_arr[:3, :3]) < 0 and np.linalg.det(T[:3, :3]) < 0
    assert res["pose"]["handedness"] == -1 and "mirrored_oct_frame" in res["flags"] and distance(T, T_arr, pts_arr) < 0.4
    assert res["pose"]["NGF"] > res["pose"]["NGF_other_handedness"]
    assert main(["apply", "--run", str(out), "--moving", str(tmp_path / "oct.npy"), "--reference", str(mri_path), "-o",
                 str(tmp_path / "again.nii.gz")]) == 0
    np.testing.assert_allclose(nib.load(str(tmp_path / "again.nii.gz")).get_fdata(),
                               nib.load(str(out / "oct_in_mri.nii.gz")).get_fdata(), rtol=1e-5, atol=1e-3)


def test_apply_moved_npy_with_spacing(tmp_path):
    """`octreg apply` replays a run whose OCT is an NPY stack from another path: the spacing recorded in result.json no longer
    applies to the file, so the run is refused until --oct-spacing-um gives it, and then the result is the same."""
    oct_path, mri_path = pair(tmp_path)[:2]
    stack = np.asarray(nib.load(str(oct_path)).dataobj, np.float32).transpose(2, 1, 0)       # numpy (z, y, x)
    np.save(tmp_path / "oct.npy", stack)
    (tmp_path / "moved").mkdir()
    np.save(tmp_path / "moved" / "oct.npy", stack)                                           # the same stack elsewhere
    (tmp_path / "params.json").write_text(json.dumps({**FAST, "n_rot": 8, "topk": 1, "iters": 20, "ngf_iters": 10}))
    out = tmp_path / "run"
    assert main(["register", str(tmp_path / "oct.npy"), str(mri_path), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "params.json"), "--oct-spacing-um", "80,80,80"]) == 0
    assert main(["apply", "--run", str(out), "--moving", str(tmp_path / "oct.npy"), "--reference", str(mri_path), "-o",
                 str(tmp_path / "here.nii.gz")]) == 0
    run = ["apply", "--run", str(out), "--moving", str(tmp_path / "moved" / "oct.npy"), "--reference", str(mri_path), "-o",
           str(tmp_path / "moved.nii.gz")]
    assert main(run) == 2                                      # an NPY without spacing: refused
    assert main(run + ["--oct-spacing-um", "80,80,80"]) == 0
    np.testing.assert_allclose(nib.load(str(tmp_path / "moved.nii.gz")).get_fdata(),
                               nib.load(str(tmp_path / "here.nii.gz")).get_fdata(), rtol=1e-6, atol=1e-6)
