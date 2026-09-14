# octreg 1.0: method

## Data and assumptions

The OCT is a scattering or intensity map of a tissue block embedded in scatterer-doped agarose, with section stripes along one
array axis, tile seams in the section plane and exact zeros where data are missing. Its contrast may be inverted relative to
the MRI. The MRI is an ex-vivo scan already cropped to a region containing the block; the crop is the only position prior. Both
voxel sizes are taken as correct, so the true scales are close to 1, and both file headers as having the correct handedness.

Transforms map millimetres in the OCT file world to the MRI file world: the NIfTI header frame, or for TIFF and NPY
diag(spacing) on the axes (x, y, z) = numpy axes (2, 1, 0). The OCT is streamed plane by plane into a 0.04 mm fine grid, where
the specimen mask is computed; box averages of it give a 0.15 mm base grid (masks kept above 0.5), on which everything else
runs, and oct_in_mri.nii.gz. The search runs on the 4³ box average of the base grid, a 0.6 mm grid.

## 1. Specimen mask from isotropic texture

Doped agarose scatters about as strongly as tissue, but its structure is directional: a section stripe varies only along the
sectioning axis and a tile seam only across the seam. Tissue varies along every direction. The fine OCT is smoothed (normalised
Gaussian, σ 0.08 mm), the running coefficient of variation c_a is computed along each array axis over 0.36 mm, and the texture
field F = min_a c_a is averaged over 0.16 mm blocks. log F is smoothed (σ 1.2 mm) and split by Otsu's threshold, followed by
closing (0.48 mm) and the largest component. Uniform tissue such as a white-matter bundle leaves holes, and a hole that reaches a
cut face is not enclosed in 3-D, so holes are filled in every array plane. On I58 the mask measures 19.3 cm³; an intensity
threshold on the same OCT takes in the agarose and gives 29.9 cm³.

The MRI foreground threshold is the first histogram valley, walking down from the brightest peak, below 0.5 of its smaller
neighbouring peak (peaks found on the square root of the smoothed counts); without one, nearly the whole field of view is kept
and the run is flagged. Either mask can be supplied as a file.

## 2. One score for structure and outline, polarity as a sign

OCT scattering and MRI intensity follow no fixed mapping, but in both the tissue falls into a brighter and a darker class. Each
volume I with foreground M is divided by its local foreground mean G(I M) / G(M) (Gaussian σ 10 mm) and blurred by one voxel to
give x; with t the Otsu threshold and s the standard deviation of the foreground values of x clipped at their 99.5th
percentile, p = sigmoid((x − t) / (0.25 s)).

The OCT enters as channels u = (p_O, 1 − p_O) with its specimen mask w as weight, the MRI as v = (p_M M, (1 − p_M) M). The
class structure alone is a weak signal on real blocks, so the score also compares the outlines, w against M over the measured
OCT voxels q, so that the agarose has to fall on MRI background:

    S_class = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   S_outline = NCC_q(w, M),   S = (2 |S_class| + S_outline) / 3.

Weighted covariance is linear and ignores constants, so NCC_w(1 − u, v) = −NCC_w(u, v). Swapping the OCT classes therefore
negates S_class exactly and leaves S_outline unchanged: one correlation scores both contrasts, and the sign of S_class is the
polarity (−1: inverted). The identity needs the OCT channels to sum to 1 wherever w > 0, which is why the mask is a weight and
never multiplies them.

## 3. Orientation search in the crop

The crop fixes the position, so only orientation is sampled: 8,000 rotations uniform on SO(3). For each, the OCT template is
rotated about its box centre and S is computed over all translations with FFTs in the masked form of Padfield (IEEE Trans.
Image Process. 21:2706, 2012); outside the crop the MRI counts as background. Each rotation keeps its best translation, and
non-maximum suppression (3 mm, 10°) leaves 24 poses.

Mirror images are not searched, so the handedness is that of the file headers. Two physical specimens are never mirror images,
and the score cannot tell handedness on a nearly symmetric specimen: on I58 the best mirrored pose has the lower loss but its
anatomy in the wrong place (docs/figures/fig_handedness_xiangrui.png). A mirrored stack, for example with a reversed section
order, has to be fixed in its header, or for TIFF and NPY by reversing one array axis.

## 4. Prior-bounded affine refinement

Each of the 24 poses is refined on the base grid with its polarity fixed:

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, shears h, log-scales ℓ and c the OCT box centre; Adam, 200 iterations, cosine schedule, each ℓ_i and
h_i clamped to ±0.15. The lowest L wins. The prior is needed because S alone rewards distorting the block.

## Evaluation

A registration is judged by looking at it, because label-free numbers can prefer a wrong pose. In every plane of qc_montage.png
(four planes per OCT axis, each shown as OCT, MRI through the transform, checkerboard and OCT with both outlines) the OCT
specimen must lie on the same anatomy in the MRI, internal structures must continue across the checkerboard, cut faces and
folded pieces must correspond and lie on the same side, and the contrast must be consistently inverted or not. Other candidate
transforms can be rendered with `octreg qc --T` and compared side by side.

### Xiangrui's I58 brainstem pair

On the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) the run took 9 min 36 s, 5.2 GiB of
RAM and 1.8 GiB of GPU memory. S is 0.2747 (S_class −0.1175, S_outline 0.5891), polarity −1, scales 1.007 / 0.971 / 0.970. Raw
20 µm OCT values mapped through the header and the transform correlate with the exported overlay at Spearman 0.991, against at
most 0.270 with any OCT axis flipped.

In the QC images (docs/figures/fig_registration_xiangrui.png; all planes in docs/figures/fig_qc_montage_xiangrui.png) the MRI
outline follows the OCT specimen in every plane apart from the torn and folded cerebellar pieces, which have moved; the OCT
mask takes in a margin of agarose in most planes. The cerebellar folia of the MRI land on the folded folia of the OCT, and a
round nucleus at the top of the axis-2 planes corresponds. The best mirrored pose fits the outline as well (S_outline 0.6438)
and has the lower loss (L 0.7133 against 0.7296), but its folia lie at the upper right of the axis-1 plane and are missing from
the upper right of the axis-2 plane.

Each ablation changes one element and reruns search and refinement; pose changes are block-corner means against the result.

| change | pose change (mm) |
|---|---|
| polarity forced to +1 | 47.7 |
| no scale prior (λ 0, clamp 1.0): S 0.3610, one axis scaled to 0.43 | 44.5 |
| intensity-threshold OCT mask instead of the texture mask | 42.1 |
| no outline term | 41.8 |
| the other handedness (OCT mirrored) | 29.8 |
| holes filled in 3-D only | 1.69 |
| standardised intensities instead of two-class maps | 0.74 |
| an independently made rim-watershed specimen mask (Dice 0.92) | 0.32 |
| MRI flattening off / OCT flattening off | 0.12 / 0.10 |

Flattening barely moves the pose but keeps it stable: without it, the rim-watershed mask and 3-D hole filling moved the pose by
42.3 and 40.9 mm.

## Parameters

The tunable constants are fields of `octreg.params.Params` (override with `--params`); I58 used the defaults.

| parameter | default | role |
|---|---|---|
| `search_mm`, `base_mm`, `fine_mm` | 0.6, 0.15, 0.04 mm | search grid, base grid, OCT grid of the specimen mask |
| `valley_ratio`, `min_component` | 0.5, 0.01 | MRI histogram valley / smaller peak; smallest MRI foreground component kept, as a fraction of the foreground |
| `texture_bandpass_mm`, `texture_window_mm` | 0.08, 0.36 mm | smoothing before, and window of, the directional coefficient of variation |
| `texture_grid_mm`, `texture_smooth_mm`, `texture_close_mm` | 0.16, 1.2, 0.48 mm | texture blocks, smoothing of log F (set on I58), closing radius |
| `flatten_sigma_mm`, `sigmoid_std` | 10 mm, 0.25 | flattening scale; sigmoid width in foreground standard deviations |
| `n_rot`, `seed`, `topk` | 8000, 0, 24 | rotations, rotation set seed, poses refined |
| `nms_mm`, `nms_deg` | 3 mm, 10° | poses closer in both count as one |
| `iters`, `lam`, `clamp` | 200, 2.0, 0.15 | Adam iterations, prior weight λ, bound on log-scales and shears |
| `lr_rot`, `lr_t`, `lr_ls`, `lr_sh` | 0.02 rad, 0.3 mm, 0.01, 0.01 | Adam learning rates |
