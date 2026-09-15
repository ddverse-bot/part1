# octreg 1.0: method

## Data and assumptions

The OCT is a scattering or intensity map of a tissue block embedded in scatterer-doped agarose, with section stripes along one
array axis, tile seams in the section plane and exact zeros where data are missing. Its contrast may be inverted relative to
the MRI. The MRI is an ex-vivo scan roughly cropped to a region containing the block. The crop is larger than the block and may
hold tissue that is not in it, for example beyond a face where the block was cut out of a larger specimen; it is the only
position prior. Both voxel sizes are taken as correct, so the true scales are close to 1, and both file headers as having the
correct handedness.

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
OCT voxels q. OCT tissue has to lie on MRI tissue and agarose may lie on MRI background, but agarose over MRI tissue is no
mismatch, since the MRI can hold tissue that is not in the block. The outline weight therefore leaves that case out:
q' = q (1 − (1 − w) E), with E = M inside the crop and 1 outside it.

    S_class = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   S_outline = NCC_q'(w, M),   S = (2 |S_class| + S_outline) / 3.

Weighted covariance is linear and ignores constants, so NCC_w(1 − u, v) = −NCC_w(u, v). Swapping the OCT classes therefore
negates S_class exactly and leaves S_outline unchanged: one correlation scores both contrasts, and the sign of S_class is the
polarity (−1: inverted). The identity needs the OCT channels to sum to 1 wherever w > 0, which is why the mask is a weight and
never multiplies them.

## 3. Orientation search in the crop

The crop bounds the position, so no global search is needed: 8,000 rotations uniform on SO(3) are sampled, and for each the OCT
template is rotated about its box centre and S is computed over all translations with FFTs in the masked form of Padfield (IEEE
Trans. Image Process. 21:2706, 2012); with b = q (1 − w), every sum of the outline NCC is a correlation of a template with E,
M, M² or M³. Outside the crop, OCT tissue counts as over background (the crop contains the block) and agarose is left out. Each
rotation keeps its best translation, and non-maximum suppression (3 mm, 10°) leaves 24 poses.

Mirror images are not searched, so the handedness is that of the file headers. Two physical specimens are never mirror images,
and the score cannot tell handedness on a nearly symmetric specimen: on I58 the best mirrored pose has the lower loss but its
anatomy in the wrong place. A mirrored stack, for example with a reversed section order, has to be fixed in its header, or for
TIFF and NPY by reversing one array axis.

## 4. Prior-bounded affine refinement

Each of the 24 poses is refined on the base grid with its polarity fixed:

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, shears h, log-scales ℓ and c the OCT box centre; Adam, 200 iterations, cosine schedule, each ℓ_i and
h_i clamped to ±0.15. The lowest L wins. The prior is needed because S alone rewards distorting the block.

## Evaluation

A registration is judged by looking at it, because label-free numbers can prefer a wrong pose. In every plane of qc_montage.png
(four planes per OCT axis, each shown as OCT, MRI through the transform, checkerboard and OCT with both outlines) the OCT
specimen must lie on the same anatomy in the MRI, internal structures must continue across the checkerboard, folded pieces, and
cut faces that the MRI also shows, must correspond and lie on the same side (where the block was cut out of a larger specimen,
the MRI tissue continues beyond the OCT cut face), and the contrast must be consistently inverted or not. Other candidate
transforms can be rendered with `octreg qc --T` and compared side by side.

### I58 brainstem pair

On the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) the run took 10 min 28 s, 5.2 GiB of
RAM and 1.3 GiB of GPU memory allocated by torch. S is 0.2855 (S_class −0.1144, S_outline 0.6277), polarity −1, scales 0.994 / 0.966 / 0.970. Raw
20 µm OCT values mapped through the header and the transform correlate with the exported overlay at Spearman 0.990, against at
most 0.266 with any OCT axis flipped.

docs/figures/registration_I58.png shows the OCT on three MRI planes through the registered specimen before and after
registration, read from the original files; in the axis-0 plane the MRI tissue continues beyond the block. In the
qc_montage.png of the run the MRI outline follows the OCT specimen in every plane apart from the torn and folded cerebellar
pieces, which have moved; the OCT mask takes in a margin of agarose in several planes. The cerebellar folia of the MRI land on the
folded folia of the OCT, and a round nucleus at the top of the axis-2 planes corresponds. The best mirrored pose fits the
outline better (S_outline 0.6867) and has the lower loss (L 0.7035 against 0.7192), but its folia lie at the upper right of the
axis-1 plane and are missing from the upper right of the axis-2 plane.

Each ablation changes one element and reruns search and refinement (the cut-face row compares the two outlines on the same
cut); pose changes are block-corner means against the result.

| change | pose change (mm) |
|---|---|
| intensity-threshold OCT mask instead of the texture mask | 42.0 |
| no outline term | 41.7 |
| polarity forced to +1 | 41.1 |
| no scale prior (λ 0, clamp 1.0): S 0.5018, one axis scaled to 0.46 | 38.7 |
| the other handedness (OCT mirrored) | 29.5 |
| cut face simulated by removing the last 30 % of the mask along OCT axis 1: two-sided outline / method | 4.69 / 0.40 |
| OCT flattening off | 2.41 |
| an independently made rim-watershed specimen mask (Dice 0.92) | 2.26 |
| holes filled in 3-D only | 2.18 |
| standardised intensities instead of two-class maps | 0.79 |
| two-sided outline (agarose over MRI tissue counted as a mismatch) | 0.23 |
| MRI flattening off | 0.18 |

The two-sided outline scores the full pair almost as the method does, but once the block has a cut face beyond which the MRI
tissue continues, it pulls the pose by 4.7 mm and the method by 0.4 mm. OCT flattening off, the rim-watershed mask and 3-D hole
filling all end in nearly the same pose, 5.2 to 5.7° from the result (1.0 to 1.1 mm mean over the specimen). Refined under the
method from the first and the last of these, the loss stops at 0.7197 against 0.7192 for the result, and the overlay of the first places
the torn cerebellar pieces worse. The loss separates poses this close only weakly.

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
