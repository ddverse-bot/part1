# octreg 1.0: method

## Data and assumptions

The OCT is a scattering or intensity map of a tissue block embedded in scatterer-doped agarose, with section stripes along one
array axis, tile seams in the section plane and exact zeros where data are missing. Its contrast may be inverted relative to the
MRI. The MRI is an ex-vivo scan roughly cropped to a region containing the block. The crop is larger than the block and may hold
tissue that is not in it, for example beyond a face where the block was cut out of a larger specimen; it is the only position
prior. Both voxel sizes are taken as correct, so the true scales are close to 1, and orientation headers, where the files have
them, as having the correct handedness.

Transforms map millimetres in the OCT file world to the MRI file world: the NIfTI header frame, or for TIFF and NPY
diag(spacing) on the axes (x, y, z) = numpy axes (2, 1, 0). The OCT is streamed plane by plane into a 0.04 mm fine grid, where
the specimen mask is computed; box averages of it give a 0.15 mm base grid (masks kept above 0.5), on which everything else
runs, and oct_in_mri.nii.gz. The search runs on the 4³ box average of the base grid, a 0.6 mm grid.

## 1. Specimen mask from isotropic texture

Doped agarose scatters about as strongly as tissue, but its structure is directional: a section stripe varies only along the
sectioning axis and a tile seam only across the seam. Tissue varies along every direction. The fine OCT is smoothed (normalised
Gaussian, σ 0.08 mm), the running coefficient of variation c_a is computed along each array axis over 0.36 mm, and the texture
field F = min_a c_a is averaged over 0.16 mm blocks. log F is smoothed (σ 1.2 mm) and split by Otsu's threshold, followed by
closing (0.48 mm) and the largest component. Uniform tissue such as a white-matter bundle leaves holes, and a hole that reaches
a cut face is not enclosed in 3-D, so holes are filled in every array plane. On I58 the mask measures 19.3 cm³; an intensity
threshold on the same OCT takes in the agarose and gives 29.9 cm³.

The MRI foreground threshold is the first histogram valley, walking down from the brightest peak, below 0.5 of its smaller
neighbouring peak (peaks found on the square root of the smoothed counts); without one, nearly the whole field of view is kept
and the run is flagged. Either mask can be supplied as a file.

## 2. One score for structure and outline, polarity as a sign

OCT scattering and MRI intensity follow no fixed mapping, but in both the tissue falls into a brighter and a darker class. Each
volume I with foreground M is divided by its local foreground mean G(I M) / G(M) (Gaussian σ 10 mm) and blurred by one voxel to
give x; with t the Otsu threshold and s the standard deviation of the foreground values of x clipped at their 99.5th percentile,
p = sigmoid((x − t) / (0.25 s)).

The OCT enters as channels u = (p_O, 1 − p_O) with its specimen mask w as weight, the MRI as v = (p_M M, (1 − p_M) M). The class
structure alone is a weak signal on real blocks, so the score also compares the outlines, w against M over the measured OCT
voxels q. OCT tissue has to lie on MRI tissue and agarose may lie on MRI background, but agarose over MRI tissue is no mismatch,
since the MRI can hold tissue that is not in the block. The outline weight therefore leaves that case out: q' = q (1 − (1 − w)
E), with E = M inside the crop and 1 outside it.

    S_class = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   S_outline = NCC_q'(w, M),   S = (2 |S_class| + S_outline) / 3.

Weighted covariance is linear and ignores constants, so NCC_w(1 − u, v) = −NCC_w(u, v). Swapping the OCT classes therefore
negates S_class exactly and leaves S_outline unchanged: one correlation scores both contrasts, and the sign of S_class is the
polarity (−1: inverted). The identity needs the OCT channels to sum to 1 wherever w > 0, which is why the mask is a weight and
never multiplies them.

## 3. Orientation search in the crop

The crop bounds the position, so no global search is needed: 8,000 rotations uniform on SO(3) are sampled, and for each the OCT
template is rotated about its box centre and S is computed over all translations with FFTs in the masked form of Padfield (IEEE
Trans. Image Process. 21:2706, 2012); with b = q (1 − w), every sum of the outline NCC is a correlation of a template with E, M,
M² or M³. Outside the crop, OCT tissue counts as over background (the crop contains the block) and agarose is left out. Each
rotation keeps its best translation, and non-maximum suppression (3 mm, 10°) leaves 24 poses.


## 4. Prior-bounded affine refinement

Each of the 24 poses is refined on the base grid with its polarity fixed:

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, shears h, log-scales ℓ and c the OCT box centre; Adam, 200 iterations, cosine schedule, each ℓ_i and h_i
clamped to ±0.15. The lowest L wins. The prior is needed because S alone rewards distorting the block.

## 5. Fine-structure refinement

The score of §2 compares large bright and dark regions and the outline. It finds the block, but it is dominated by the outline,
and an outline that carries a margin of agarose, or a cut face, leaves the last millimetres open. The best pose of §4 is
therefore refined on the internal structure of both volumes, fibre bundles, vessels and nuclei, compared by normalised gradient
fields (Haber and Modersitzki, 2006). On its base grid each volume I with mask m is smoothed and differentiated inside the mask
by normalised convolution, g = ∇(G_σ(I m) / G_σ(m)), so the mask edge adds no gradient, and n = g / √(|g|² + ε²), with ε the
median |g| over the MRI points used and over the OCT specimen mask. Both masks are eroded by 0.8 mm, which leaves the outline to
§2-4. Over every second interior MRI voxel x, with the eroded OCT mask w,

    F(T) = Σ_x w(T⁻¹x) (n_O(x) · n_M(x))² / Σ_x w(T⁻¹x),   g_O(x) = A⁻ᵀ g_OCT(T⁻¹x),

A the linear part of T. A squared cosine needs neither an intensity mapping nor the polarity. L = 1 − F + λ (Σ ℓ_i² + Σ h_i²) is
minimised as in §4 from the §4 pose, with learning rates a fifth of those of §4, one pass of 150 iterations at each of σ = 0.6,
0.4 and 0.3 mm, at which section stripes a few tenths of a millimetre apart are smoothed away. §5 refines and does not search: F
alone does not find the block (among the 24 poses of §4 on I58, one 41 mm away reaches a higher F after refinement than the
result), and it can lower S, whose outline and tissue classes are coarse.

Handedness. Two physical specimens are never mirror images, so an orientation header on both files fixes the handedness and §3
searches proper rotations only; the score of §2 cannot tell mirror images apart on a nearly symmetric block (on I58 the best
mirrored pose has the lower L but its anatomy in the wrong place). A TIFF or NPY stack, or a NIfTI file without sform and qform,
has only an array frame, whose handedness depends on how the stack was written. Then §3-5 run once for each handedness, which
doubles their time, the pose with the higher F wins, and a mirrored array frame is flagged: fine internal structure does not
match in a mirror image, while outline and tissue classes still can (on I58 F is 0.103 for the true handedness and 0.078 for the
mirrored one).

## Evaluation

A registration is judged by looking at it, because label-free numbers can prefer a wrong pose. In every plane of qc_montage.png
(four planes per OCT axis, each shown as OCT, MRI through the transform, checkerboard and OCT with both outlines) the OCT
specimen must lie on the same anatomy in the MRI, internal structures must continue across the checkerboard, folded pieces, and
cut faces that the MRI also shows, must correspond and lie on the same side (where the block was cut out of a larger specimen,
the MRI tissue continues beyond the OCT cut face), and the contrast must be consistently inverted or not. Other candidate
transforms can be rendered with `octreg qc --T` and compared side by side.

### I58 brainstem pair

On the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) the run took 10 min 40 s, 5.3 GiB of
RAM and 1.3 GiB of GPU memory allocated by torch. §1-4 give S 0.2855 (S_class −0.1144, S_outline 0.6277) and polarity −1; §5
raises F from 0.088 to 0.103 and moves the block by 2.4 mm at its corners (1.5 mm over the specimen, 5.3°), with scales 0.977 /
0.974 / 0.985. Raw 20 µm OCT values mapped through the header and the transform correlate with oct_in_mri.nii.gz at Spearman
0.991, against at most 0.263 with any OCT axis flipped.

docs/figures/registration_I58.png shows the result on three MRI planes, read from the original files; in the axis-0 plane the
MRI tissue continues beyond the block. On 21 MRI planes at the MRI voxel size (MRI, OCT and a colour fusion on one grid), with
landmark positions measured by language-model image readers and re-measured by a second reader, vessels, fissures and
fibre-bundle edges of the brainstem agree to a median of about 0.3 mm, with no shift, rotation or scale common to the planes.
Two parts do not follow an affine transform: the torn cerebellar pieces, which moved during sectioning, and the sections at one
end of the block (below about 12 mm along MRI axis 1), which are 10 to 14 % larger in plane than the MRI there and misfit by up
to 2 mm, while vessels a few millimetres further in match again. The §5 pose was compared with the §4 pose on 15 MRI planes,
shown as A and B in random order and judged separately on internal structure and on boundaries by such readers: the §5 pose was
preferred 20 times (13 clearly), the §4 pose 3 times (1 clearly), with 7 ties. At the §5 pose S is lower (0.251, S_class
−0.067): the coarse score does not resolve the last millimetres.

Each ablation changes one element of the method (the §1 grids and masks are computed once per mask source, and the no-§5 row
stops after §4). Pose changes are block-corner means, of the pose after §4 against the method's pose after §4 and of the final
pose against the result; the cut-face row compares the two outlines on the same cut.

| change | after §4 (mm) | final (mm) |
|---|---|---|
| intensity-threshold OCT mask instead of the texture mask | 42.0 | 42.9 |
| no outline term | 41.7 | 40.9 |
| polarity forced to +1 | 41.1 | 42.3 |
| no scale prior (λ 0, clamp 1.0): S 0.502 at the §4 pose, final scales 0.51-0.56 | 38.7 | 35.8 |
| the other handedness (OCT mirrored) | 29.5 | 30.5 |
| no fine-structure refinement (§5) | 0 | 2.36 |
| OCT flattening off | 2.41 | 0.00 |
| holes filled in 3-D only | 2.18 | 0.17 |
| standardised intensities instead of two-class maps | 0.79 | 0.00 |
| two-sided outline (agarose over MRI tissue counted as a mismatch) | 0.23 | 0.00 |
| MRI flattening off | 0.18 | 0.00 |
| cut face simulated by removing the mask beyond 70 % of its extent along OCT axis 1: method / two-sided outline | 0.40 / 4.69 | 0.11 / 3.87 |

The first five turn the block over or misplace it, and §5, which only refines, does not bring it back. §5 itself moves the
result by 2.4 mm. The next five changes of the §4 pose lie within the reach of §5, which settles them; those steps stay for the
§4 pose they give. MRI flattening, which moves it least (0.18 mm), stays as an exception, so that both volumes pass through the
same class map. With a cut face beyond which the MRI tissue continues, only the one-sided outline keeps the block in place: the
two-sided outline leaves it 3.9 mm off after §5.

### Beyond the target data: DANDI:000026

The Broca-area blocks of DANDI:000026 (Costantini et al. 2023) are slabs cut from the cortex of a hemisphere without embedding,
imaged at 12-35 µm as TIFF stacks, with a whole-hemisphere MRI. Cropped with a 12 mm margin around the Broca-area labels (I46,
I55) or around the block where an earlier registration of this dataset put it (I38, I56, I62), and run in both handednesses, one
of these five blocks registered: I55 (a block 21 mm across), where §5 raised the Dice of OCT white and grey matter against the
MRI cortical labels from 0.88 and 0.91 to 0.90 and 0.92, lowered the median distance of MRI vessel labels to OCT vessels from
266 to 229 µm, and F chose the handedness under which OCT white and grey matter match the labels (Dice 0.90 and 0.92, against
0.60 and 0.70). On the other four (blocks 15-52 mm across) the search put the OCT cortex on the wrong gyri or on deep
structures, and by inspection two blocks without a known position were not placed correctly either. Without embedding the mask
is nearly the whole block (95 % on I46), so the outline adds almost nothing, and the bright-dark patterns and edges of cortex
recur throughout the crop. The method relies on what its target data provide: an embedded block, whose outline constrains the
pose.

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
| `lr_rot`, `lr_t`, `lr_ls`, `lr_sh` | 0.02 rad, 0.3 mm, 0.01, 0.01 | Adam learning rates (§5: a fifth of these) |
| `ngf_sigmas_mm`, `ngf_erode_mm`, `ngf_iters` | 0.6, 0.4, 0.3 mm; 0.8 mm; 150 | gradient scales of the §5 passes, mask erosion, Adam iterations per pass |
