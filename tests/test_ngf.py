"""§5 fine-structure refinement: world gradients of a ramp on a permuted, flipped grid; F is unchanged by inverting or rescaling
the OCT contrast; a pose 1 mm and 3 degrees off is recovered on a synthetic pair whose OCT is a non-linear, inverted function of
the MRI structure."""
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from octreg import geometry as G, ngf
from octreg.params import Params

H = 0.15


def grid(shape, perm=True):
    A = np.zeros((4, 4))
    A[3, 3] = 1
    if perm:
        A[0, 0], A[1, 2], A[2, 1] = H, H, -H
    else:
        A[:3, :3] = np.diag([-H, -H, H])
    A[:3, 3] = -A[:3, :3] @ ((np.array(shape) - 1) / 2)
    return A


def world(shape, A):
    return G.apply_affine(A, np.indices(shape).reshape(3, -1).T.astype(float)).reshape(*shape, 3)


def test_gradient_field_ramp():
    shape = (40, 36, 32)
    A = grid(shape)
    x = world(shape, A)
    arr = (2 * x[..., 0] - 3 * x[..., 2]).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[4:-4, 4:-4, 4:-4] = True
    g = ngf.gradient_field(arr, mask, A, 0.3, "cpu").numpy()
    inner = ndimage.binary_erosion(mask, iterations=9)                                # beyond the kernel radius (8 voxels)
    assert np.abs(g[:, inner] - np.array([2, 0, -3])[:, None]).max() < 2e-3
    flat = ngf.gradient_field(np.where(mask, 5.0, 0.0).astype(np.float32), mask, A, 0.3, "cpu").numpy()
    assert np.abs(flat).max() < 1e-3                                                  # the mask edge adds no gradient


def pair(seed=0):
    rng = np.random.default_rng(seed)
    shape_m = (64, 64, 64)
    A_M = grid(shape_m)
    xm = world(shape_m, A_M)
    s = ndimage.gaussian_filter(rng.standard_normal(shape_m), 2.0)
    s /= s.std()
    mask_m = ((xm / np.array([4.2, 4.0, 3.8])) ** 2).sum(-1) <= 1
    mri = np.where(mask_m, 1.0 + 0.3 * s, 0.0).astype(np.float32)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.2, -0.4, 0.3]).as_matrix()
    T[:3, 3] = [0.3, -0.2, 0.25]
    shape_o = (60, 62, 58)
    A_O = grid(shape_o, perm=False)
    xo = world(shape_o, A_O).reshape(-1, 3)
    xm_of_o = G.apply_affine(T, xo)
    samp = lambda vol: G.sample_world(torch.as_tensor(vol[None], dtype=torch.float32), A_M,
                                      torch.as_tensor(xm_of_o, dtype=torch.float32))[0].numpy().reshape(shape_o)
    so, mo = samp(s.astype(np.float32)), samp(mask_m.astype(np.float32)) > 0.5
    oct_ = np.where(mo, 2.0 - np.tanh(1.5 * so), 0.0).astype(np.float32)             # inverted, non-linear contrast
    return (mri, mask_m, A_M), (oct_, mo, A_O), T


def test_ngf_contrast_free():
    mri, (o, mo, A_O), T = pair()
    P = Params()
    Tt = torch.tensor(T, dtype=torch.float32)
    F1 = float(ngf.NGFGrid(mri, (o, mo, A_O), 0.3, P, "cpu").F(Tt))
    F2 = float(ngf.NGFGrid(mri, (np.where(mo, 7.0 - 3.0 * o, 0.0).astype(np.float32), mo, A_O), 0.3, P, "cpu").F(Tt))
    off = torch.tensor(np.r_[np.c_[np.eye(3), [0.5, 0.0, 0.0]], [[0, 0, 0, 1]]], dtype=torch.float32)
    F3 = float(ngf.NGFGrid(mri, (o, mo, A_O), 0.3, P, "cpu").F(off @ Tt))
    assert abs(F1 - F2) < 1e-4 and F3 < 0.8 * F1


def test_ngf_recovers_offset_pose():
    mri, oct_, T = pair()
    P = Params.from_dict({"ngf_sigmas_mm": [0.45, 0.3], "ngf_erode_mm": 0.6})
    off = np.eye(4)
    off[:3, :3] = Rotation.from_rotvec(np.radians(3.0) * np.array([1.0, 1.0, 0.0]) / np.sqrt(2)).as_matrix()
    off[:3, 3] = [0.6, -0.5, 0.6]
    T0 = off @ T
    pts = world(oct_[0].shape, oct_[2])[oct_[1]]
    d0 = np.linalg.norm(G.apply_affine(T0, pts) - G.apply_affine(T, pts), axis=1).mean()
    T1, info = ngf.refine_ngf(mri, oct_, T0, P, "cpu")
    d1 = np.linalg.norm(G.apply_affine(T1, pts) - G.apply_affine(T, pts), axis=1).mean()
    assert d0 > 0.9 and d1 < 0.08, (d0, d1)
    assert info["F"] > info["F_start"]


def test_ngf_separates_mirror_images():
    """The handedness rule of §3 for array frames: refined from the true pose, and from the same pose in the mirrored OCT frame
    (the mirror image of the block in the same place), the true handedness aligns more fine structure."""
    from octreg.register import MIRROR
    mri, (o, mo, A_O), T = pair(seed=2)
    P = Params.from_dict({"ngf_sigmas_mm": [0.45, 0.3], "ngf_erode_mm": 0.6})
    _, true = ngf.refine_ngf(mri, (o, mo, A_O), T, P, "cpu")
    _, mirrored = ngf.refine_ngf(mri, (o, mo, MIRROR @ A_O), T, P, "cpu")
    assert true["F"] > 1.5 * mirrored["F"], (true["F"], mirrored["F"])
