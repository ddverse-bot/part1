# octreg 1.0: method

octreg registers a serial-sectioning OCT block to an ex-vivo MRI crop with one affine transform and no labels. Its three
innovations are a specimen mask from isotropic texture, two-class structure maps whose contrast polarity is the sign of one
score, and an FFT orientation search inside the crop followed by a prior-bounded, overlap-gated affine refinement. Results are
judged by visual inspection. Numbers come from Xiangrui's I58 brainstem pair (docs/BENCHMARK.md).

## Data and assumptions

The OCT is a scattering or intensity map of a tissue block embedded in scatterer-doped agarose, with voxels of tens of
micrometres, section stripes along one array axis and tile seams in the section plane. Exact zeros are missing data. Its
contrast may be inverted relative to the MRI. The MRI is an ex-vivo scan already cropped to a region containing the block. The
crop may cut the tissue and is the only position prior. Both voxel sizes are taken as correct, so the true scales are close to 1.

## Grids and streaming

Transforms map world coordinates in mm from the OCT file to the MRI file, the world being the NIfTI header frame or diag(spacing)
for TIFF and NPY. The OCT is streamed plane by plane into an isotropic 0.04 mm fine grid (box average over whole voxels, then
trilinear), so memory holds the output and two planes. The fine grid is the largest OCT array held and serves only the
specimen mask. The OCT, its mask and the MRI then go to a 0.15 mm base grid for the two-class maps and the refinement, and a
further 4³ box average gives the 0.6 mm search grid.

## Specimen mask from isotropic texture

Doped agarose scatters about as strongly as tissue, so an intensity threshold cannot separate them. Texture can. Tissue holds
fibre bundles, cell bodies and vessels, and its intensity varies along every direction. At a scale of a few hundred
micrometres the agarose is uniform apart from acquisition artefacts, and each artefact is directional. A section stripe varies
only along the sectioning axis and a tile seam only across the seam. Along at least one array axis the agarose is therefore
flat while tissue is not, and the minimum over axes of a directional texture measure is high only in tissue.

The fine OCT is smoothed with a normalised Gaussian (σ 0.08 mm) to suppress speckle. Along each array axis a, the running
coefficient of variation c_a is computed over a 0.36 mm window, and the texture field is F = min_a c_a, averaged over 0.16 mm
blocks. Single block estimates are too noisy to classify, so log F is smoothed (σ 1.2 mm) before Otsu's threshold splits it
into two classes. Closing (radius 0.48 mm), the largest component and hole filling give the mask. On I58 it measures 16.9 cm³,
with a Dice of 0.910 against the watershed mask of our earlier pipeline (18.05 cm³). An intensity-valley threshold on the same
OCT gives 29.5 cm³ because it takes in the agarose.

The MRI foreground comes from that intensity-valley rule. Peaks are found on the square root of the smoothed histogram counts,
so the narrow background peak of a box-averaged grid cannot hide the tissue peak, and the threshold is the first valley below
0.5 of its smaller neighbouring peak. A missing valley is flagged. Either mask can be supplied as a file.

## Two-class maps and polarity as a sign

OCT scattering and MRI intensity follow no fixed mapping, but in both the tissue falls into a brighter and a darker class, such
as fibre tracts and grey matter. Each volume I with foreground M is divided by its local foreground mean G(I M) / G(M)
(Gaussian σ 10 mm) to remove slow multiplicative changes, and blurred by one voxel to give x. With t the Otsu threshold and s
the standard deviation of the foreground values, p = sigmoid((x − t) / (0.25 s)). OCT and MRI use the same rule.

The OCT enters as channels u = (p_O, 1 − p_O) with a separate weight w, its specimen mask, and the MRI as
v = (p_M M, (1 − p_M) M), so the MRI foreground carries the outline into the score. With the MRI channels sampled at the OCT
voxels through a pose,

    S = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   NCC_w(a, b) = cov_w(a, b) / √(var_w(a) var_w(b)).

Weighted covariance is linear and ignores added constants, so cov_w(1 − a, b) = −cov_w(a, b) and var_w(1 − a) = var_w(a), hence

    NCC_w(1 − u, v) = −NCC_w(u, v).

Swapping the OCT classes turns (p, 1 − p) into (1 − p, p) = 1 − u, so it negates S exactly and one correlation scores both
polarities. The search takes the largest |S| and records its sign as the polarity, −1 meaning inverted contrast. The identity
needs the OCT channels to sum to 1 wherever w > 0, which is why the mask is a weight and never multiplies the OCT channels. On
I58 the rule chooses −1. Forcing +1 lowers S from 0.1455 to 0.1087 and moves the pose by 12.0 mm (block-corner mean).

## Orientation search in the MRI crop

The crop removes the need for a global position search, so only orientation is sampled: 8,000 rotations uniform on SO(3)
(fixed seed), each also mirrored by diag(1, 1, −1), 16,000 orientations in all. For each one, the OCT channels and weight are
rotated about the OCT box centre into a template on the MRI axes, and the weighted NCC over all translations is computed with
FFTs in the masked form of Padfield (IEEE Trans. Image Process. 21:2706, 2012). The same FFTs give the overlap, the fraction of
OCT weight on MRI foreground. A translation is admissible when

    overlap ≥ τ = ρ min(1, V_MRI / V_OCT),   ρ = 0.6,

with V_MRI and V_OCT the foreground volumes. The factor min(1, V_MRI / V_OCT) is the largest overlap possible when the crop holds
less tissue than the specimen. τ never drops below 0.15, which is flagged. Each orientation keeps its best admissible |S| with
the sign, and non-maximum suppression (3 mm, 10°) leaves 24 poses. On I58, τ was 0.493 and the search took 46 s on the GPU.

## Prior-bounded, overlap-gated affine refinement

Each of the 24 poses is refined on the base grid with its polarity fixed, using

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) M_m (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, three shears h, three log-scales ℓ, the mirror M_m of the search pose and c the OCT box centre. Adam runs
200 iterations with a cosine schedule, and each ℓ_i and h_i is clamped to ±0.15 after every step.

The prior is needed because S alone rewards distorting the block. With λ = 0 and the clamp at 1.0 (ablation A6), S rises from
0.1455 to 0.2421 while the OCT is squeezed to 0.547 and 0.424 of its length along two axes, and the pose moves by 22.4 mm into
the other handedness. The voxel sizes are known, so the prior only states that the specimen keeps its size within a few percent.
The final I58 scales are 1.000, 0.967 and 0.985.

A refined pose can slide off the MRI foreground, so the admissibility rule is applied again. Poses whose overlap falls below τ are
dropped (4 of 24 on I58), and the remaining pose with the lowest L is the result. On I58 it came from the 14th of the 24 search
poses, so refining only the first few would have missed it.

## Outputs and the raw-data frame check

The transform relates the two file worlds and applies to the original files without reorientation. It is written as a matrix,
a FreeSurfer LTA and an ITK transform, with overlays in both frames, QC images and result.json. S is computed on internal grids
and would not reveal a header or axis-order mistake in the export, so the benchmark checks the export against the raw data. Raw
20 µm OCT values mapped through the header affine and T correlate with the exported overlay at Spearman 0.990, while flipping
any raw OCT axis gives at most 0.205.

## Evaluation

### Visual inspection

These pairs have no labels, and label-free numbers can prefer a wrong pose, so a registration is judged by looking at it. The
protocol uses the overlays in freeview and qc_montage.png, which shows four planes per OCT axis with the OCT, the MRI through
the transform, a checkerboard and the outlines of both masks. In every plane of all three axes, not only through the centre, the
OCT specimen must lie on the same anatomy in the MRI and the two outlines must largely agree. Internal structures such as fibre
tracts must continue across the checkerboard squares. Cut faces and detached or folded pieces must correspond. The contrast must
be consistently inverted, or consistently not, over the whole specimen. Plausible alternative poses are rendered with
`octreg qc --T` and compared side by side.

On I58 the final pose puts the MRI brainstem body on the OCT specimen in all three planes, with no cerebellar folia inside the
body, and the folded side piece matches (docs/figures/fig_qc_xiangrui.png). Along parts of the specimen edge the OCT mask still
lies on MRI background. R5, the pose of our earlier research pipeline, lies 10.3 mm (block-corner mean) and 25.05° away. Its QC
panels, like those of the other-handedness pose and a low-overlap competitor, fail at least one of these checks. The outline
distance does not decide between the final pose and R5. With the method's masks the rim medians (forward / reverse) are
1.88 / 1.24 mm against 2.80 / 1.37 mm for R5, but with the masks of the earlier pipeline they are 1.83 / 1.14 mm against
2.02 / 1.05 mm.

### Ablations

Each ablation changes one element and reruns search and refinement from the same preprocessed grids, one set per OCT mask.
Replacing the texture mask, the two-class maps, the polarity rule or the scale prior, or switching off OCT flattening, moves the
pose away from the visually accepted one by 12 to 44 mm (block-corner mean, docs/figures/fig_ablation.png).

| variant | change | pose change (mm) | note |
|---|---|---|---|
| A0 | intensity-valley OCT mask instead of the texture mask | 44.2 | mask 29.5 cm³, pose turned by 176° |
| A0b | watershed mask of the earlier pipeline | 30.3 | other handedness |
| A2 | OCT flattening off | 30.9 | other handedness |
| A4 | standardised intensities (z, −z) instead of two-class maps | 24.9 | |
| A5+1 | polarity forced to +1 | 12.0 | S 0.1087 against 0.1455 |
| A6 | no scale prior (λ 0, clamp 1.0) | 22.4 | scales 1.05 / 0.55 / 0.42, other handedness |
| A1 | MRI flattening off | 0.30 | kept |
| A3 | section-stripe flat field off (first run) | 0.15 | removed |
| A7 | one affine fit instead of the refinement ladder (first run) | 0.0015 | removed |

A6 uses the same maps and masks as the method and reaches a higher S with a non-physical pose, which is why S cannot be the
criterion. MRI flattening stays because the two-class map keeps one definition for both modalities, and without it the gap
between the two best search scores shrinks from 0.0061 to 0.0016. A3 and A7 are measured against the base of the first ablation
run, which still had both steps. Against the final pose they lie 0.24 and 0.39 mm away, the values in the figure.

### Label-free signals

result.json flags failures but does not certify a pose. Warning signs are a scale far from 1, an overlap close to τ, or any flag.
On I58, S is 0.1455 with polarity −1, overlap 0.586 against τ 0.493, and there are no flags. The run took 8 min 42 s, 5.19 GiB of
peak RAM and 1.6 GB of GPU memory.

## Parameters

All constants are fields of `octreg.params.Params`. The I58 run used the defaults (Params hash 8900ced9d7d1677a).

| parameter | default | unit | role |
|---|---|---|---|
| `search_mm` | 0.6 | mm | grid of the orientation search, an integer multiple of `base_mm` |
| `base_mm` | 0.15 | mm | grid of the MRI foreground, the two-class maps and the refinement |
| `fine_mm` | 0.04 | mm | OCT grid of the specimen mask |
| `valley_ratio` | 0.5 | ratio | MRI histogram valley must be below this fraction of the smaller neighbouring peak |
| `min_component` | 0.01 | fraction | smallest MRI foreground component kept, relative to the foreground volume |
| `texture_bandpass_mm` | 0.08 | mm | Gaussian σ applied to the OCT before the directional variation |
| `texture_window_mm` | 0.36 | mm | window of the directional coefficient of variation |
| `texture_grid_mm` | 0.16 | mm | block size of the texture field |
| `texture_smooth_mm` | 1.2 | mm | Gaussian σ of the log texture field before the Otsu threshold (set on I58) |
| `texture_close_mm` | 0.48 | mm | closing radius of the specimen mask |
| `flatten_sigma_mm` | 10.0 | mm | Gaussian σ of the local foreground mean used for flattening |
| `sigmoid_std` | 0.25 | foreground std | width of the sigmoid of the two-class map |
| `n_rot` | 8000 | count | rotations of the search, each also mirrored |
| `seed` | 0 | | seed of the rotation set |
| `topk` | 24 | count | search poses kept after non-maximum suppression, each refined |
| `nms_mm` | 3.0 | mm | poses with closer centres ... |
| `nms_deg` | 10.0 | deg | ... and closer rotations count as one |
| `overlap_rho` | 0.6 | fraction | ρ of the overlap gate τ (set on I58) |
| `overlap_floor` | 0.15 | fraction | lower bound of τ, flagged when reached |
| `iters` | 200 | count | Adam iterations per pose |
| `lam` | 2.0 | | λ, weight of the scale and shear prior |
| `clamp` | 0.15 | log-scale, shear | bound on each log-scale and shear |
| `lr_rot` | 0.02 | rad | Adam learning rate of the rotation vector |
| `lr_t` | 0.3 | mm | Adam learning rate of the translation |
| `lr_ls` | 0.01 | log-scale | Adam learning rate of the log-scales |
| `lr_sh` | 0.01 | shear | Adam learning rate of the shears |

## What we removed

A section-stripe flat field (destripe) took 365 of 906 s in the first run and moved the pose by 0.15 mm when switched off. A
rigid, similarity and affine ladder over three grids gave the same pose as one affine fit per search pose to within 0.0015 mm.
Both were removed. The research code at the git tag v1.1-archive also has a vascular channel, a fine registration stage, a
trained tissue parser and non-rigid refinement. None is in 1.0. The vascular channel helped on one pair only (I46), the fine
stage was rejected on the brainstem, the parser registered no better than the two-class maps, and the non-rigid gains were small.
