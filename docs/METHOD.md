# octreg 1.0: method

octreg registers a serial-sectioning OCT block to an ex-vivo MRI crop with one affine transform and no labels. Its three
innovations are a specimen mask from isotropic texture, a score that combines two-class structure maps with the specimen
outline and reads the contrast polarity off a sign, and an FFT orientation search inside the crop followed by a prior-bounded
affine refinement. Results are judged by visual inspection. Numbers come from Xiangrui's I58 brainstem pair
(docs/BENCHMARK.md).

## Data and assumptions

The OCT is a scattering or intensity map of a tissue block embedded in scatterer-doped agarose, with voxels of tens of
micrometres, section stripes along one array axis and tile seams in the section plane. Exact zeros are missing data. Its
contrast may be inverted relative to the MRI. The MRI is an ex-vivo scan already cropped to a region containing the block. The
crop may cut the tissue and is the only position prior. Both voxel sizes are taken as correct, so the true scales are close to 1.
Both file headers are taken to have the correct handedness.

## Grids and streaming

Transforms map world coordinates in mm from the OCT file to the MRI file, the world being the NIfTI header frame or diag(spacing)
for TIFF and NPY. The OCT is streamed plane by plane into an isotropic 0.04 mm fine grid (box average over whole voxels, then
trilinear), so memory holds the output and two planes. The fine grid is the largest OCT array held and serves only the
specimen mask. The OCT, its mask, its measured-voxel fraction and the MRI then go to a 0.15 mm base grid for the scores and the
refinement, and a further 4³ box average gives the 0.6 mm search grid.

## Specimen mask from isotropic texture

Doped agarose scatters about as strongly as tissue, so an intensity threshold cannot separate them. Texture can. Tissue holds
fibre bundles, cell bodies and vessels, and its intensity varies along every direction. At a scale of a few hundred
micrometres the agarose is uniform apart from acquisition artefacts, and each artefact is directional. A section stripe varies
only along the sectioning axis and a tile seam only across the seam. Along at least one array axis the agarose is therefore
flat while tissue is not, and the minimum over axes of a directional texture measure is high only in tissue.

The fine OCT is smoothed with a normalised Gaussian (σ 0.08 mm) to suppress speckle. Along each array axis a, the running
coefficient of variation c_a is computed over a 0.36 mm window, and the texture field is F = min_a c_a, averaged over 0.16 mm
blocks. Single block estimates are too noisy to classify, so log F is smoothed (σ 1.2 mm) before Otsu's threshold splits it
into two classes. Closing (radius 0.48 mm) and the largest component follow. Uniform tissue such as a white-matter bundle has
little texture and leaves holes, and a hole that reaches a cut face of the block is not enclosed in 3-D, so holes are filled
in every array plane (the union over the three axes). Voxels without OCT data are removed from the mask. On I58 it measures
19.2 cm³, with a Dice of 0.919 against the watershed mask of our earlier pipeline (18.05 cm³). An
intensity-valley threshold on the same OCT gives 29.5 cm³ because it takes in the agarose.

The MRI foreground comes from that intensity-valley rule. Peaks are found on the square root of the smoothed histogram counts,
so the narrow background peak of a box-averaged grid cannot hide the tissue peak, and the threshold is the first valley below
0.5 of its smaller neighbouring peak. A missing valley is flagged. Either mask can be supplied as a file.

## One score: two-class maps, the outline, and polarity as a sign

OCT scattering and MRI intensity follow no fixed mapping, but in both the tissue falls into a brighter and a darker class, such
as fibre tracts and grey matter. Each volume I with foreground M is divided by its local foreground mean G(I M) / G(M)
(Gaussian σ 10 mm) to remove slow multiplicative changes, and blurred by one voxel to give x. With t the Otsu threshold and s
the standard deviation of the foreground values (clipped at the 99.5th percentile), p = sigmoid((x − t) / (0.25 s)). OCT and
MRI use the same rule.

The OCT enters as channels u = (p_O, 1 − p_O) with its specimen mask w as a separate weight, and the MRI as
v = (p_M M, (1 − p_M) M). The class structure alone is a weak signal on real blocks, so the score also compares the specimen
outlines: the OCT specimen mask w against the MRI foreground M, weighted by the measured OCT voxels q, so that the agarose has
to fall on MRI background as much as the tissue has to fall on MRI tissue. With the MRI sampled at the OCT voxels through a pose,

    S_class = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   S_outline = NCC_q(w, M),   S = (2 |S_class| + S_outline) / 3,

where NCC_w(a, b) = cov_w(a, b) / √(var_w(a) var_w(b)). S is the mean correlation over three channel pairs: bright tissue, dark
tissue and specimen. Weighted covariance is linear and ignores added constants, so cov_w(1 − a, b) = −cov_w(a, b) and
var_w(1 − a) = var_w(a), hence

    NCC_w(1 − u, v) = −NCC_w(u, v).

Swapping the OCT classes turns (p, 1 − p) into (1 − p, p) = 1 − u, so it negates S_class exactly and leaves S_outline
unchanged. One correlation scores both polarities, and the sign of S_class is the polarity, −1 meaning inverted contrast. The
identity needs the OCT channels to sum to 1 wherever w > 0, which is why the mask is a weight and never multiplies the OCT
channels. On I58 the rule chooses −1, with S_class −0.1175 and S_outline 0.5891 at the result. Forcing +1 (ablation A5+1)
moves the pose by 47.7 mm (block-corner mean), and dropping the outline term (A9) moves it by 6.6 mm.

## Orientation search in the MRI crop

The crop removes the need for a global position search, so only orientation is sampled: 8,000 rotations uniform on SO(3)
(fixed seed). For each one, the OCT channels, specimen mask and measured fraction are rotated about the OCT box centre into a
template on the MRI axes, and S is computed over all translations with FFTs in the masked form of Padfield (IEEE Trans. Image
Process. 21:2706, 2012). Outside the crop the MRI counts as background. Each orientation keeps its best translation, and
non-maximum suppression (3 mm, 10°) leaves 24 poses. On I58 the search took 30 s on the GPU.

The search contains rotations only, no mirror images, so the handedness of the result is that of the two file headers. Two
physical specimens are never mirror images of each other, and the score cannot tell handedness on a nearly symmetric specimen.
On I58 the best pose of the other handedness (ablation A8, the OCT world mirrored before the search) reaches a lower loss than
the result (L 0.7133 against 0.7296), yet it puts the cerebellar folia and a round nucleus of the MRI on the side opposite to
the folded folia and the nucleus of the OCT (docs/figures/fig_handedness_xiangrui.png). A stack whose header is mirrored, for
example with the section order reversed, has to be fixed in its header before registration.

## Prior-bounded affine refinement

Each of the 24 poses is refined on the base grid with its polarity fixed, using

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, three shears h, three log-scales ℓ and c the OCT box centre. Adam runs 200 iterations with a cosine
schedule, and each ℓ_i and h_i is clamped to ±0.15 after every step. The pose with the lowest L is the result. On I58 it came
from the 16th of the 24 search poses, so refining only the first few would have missed it.

The prior is needed because S alone rewards distorting the block. With λ = 0 and the clamp at 1.0 (ablation A6), S rises from
0.2747 to 0.3610 while the OCT is scaled to 0.91, 0.83 and 0.43 of its length along its three axes, and the pose moves by
44.5 mm. Both voxel sizes are known, so the prior only asks the specimen to keep roughly its size and shape. The final I58
scales are 1.007, 0.971 and 0.970.

## Outputs and the raw-data frame check

The transform relates the two file worlds and applies to the original files without reorientation. It is written as a matrix,
a FreeSurfer LTA and an ITK transform, with overlays in both frames, QC images and result.json. S is computed on internal grids
and would not reveal a header or axis-order mistake in the export, so the benchmark checks the export against the raw data. Raw
20 µm OCT values mapped through the header affine and T correlate with the exported overlay at Spearman 0.991, while
flipping any raw OCT axis gives at most 0.270.

## Evaluation

### Visual inspection

These pairs have no labels, and label-free numbers can prefer a wrong pose, so a registration is judged by looking at it. The
protocol uses the overlays in freeview and qc_montage.png, which shows four planes per OCT axis with the OCT, the MRI through
the transform, a checkerboard and the outlines of both masks. In every plane of all three axes, not only through the centre, the
OCT specimen must lie on the same anatomy in the MRI and the two outlines must largely agree. Internal structures such as fibre
tracts must continue across the checkerboard squares. Cut faces and detached or folded pieces must correspond. The contrast must
be consistently inverted, or consistently not, over the whole specimen. Plausible alternative poses, including the best pose of
the other handedness, are rendered with `octreg qc --T` and compared side by side.

On I58 (docs/figures/fig_qc_montage_xiangrui.png) the MRI foreground outline follows the OCT specimen outline in the planes of
all three axes, apart from the torn and folded cerebellar pieces, which have moved. The cerebellar folia of the MRI land on the
folded folia of the OCT (axis 0 at 17.47 and 23.17 mm, axis 1 at 24.07 mm), and the folia strip and the round nucleus at the
top of the axis-2 planes correspond. docs/figures/fig_handedness_xiangrui.png shows the result next to the best pose of the other
handedness in the planes through the specimen centre: the mirrored pose also fits the outline, but its folia and nucleus lie on
the wrong side.

### Ablations

Each ablation changes one element and reruns search and refinement from the same preprocessed grids, one set per OCT mask.
Pose changes are block-corner means against the result (docs/figures/fig_ablation.png).

| variant | change | pose change (mm) | note |
|---|---|---|---|
| A5+1 | polarity forced to +1 | 47.7 | turned by 158° |
| A6 | no scale prior (λ 0, clamp 1.0) | 44.5 | S 0.3610, scales 0.91 / 0.83 / 0.43 |
| A0 | intensity-valley OCT mask instead of the texture mask | 42.1 | mask 29.5 cm³, turned by 165° |
| A8 | the other handedness (OCT world mirrored) | 29.8 | L 0.7133 against 0.7296, anatomy on the wrong side |
| A9 | no outline term | 6.6 | turned by 15° |
| A0c | holes filled in 3-D only | 1.69 | turned by 4° |
| A4 | standardised intensities (z, −z) instead of two-class maps | 0.74 | 0.51 mm over the specimen points |
| A0b | watershed mask of the earlier pipeline | 0.32 | |
| A1 | MRI flattening off | 0.12 | kept, see below |
| A2 | OCT flattening off | 0.10 | kept, see below |
| A3 | section-stripe flat field off (first run) | 0.15 | removed |
| A7 | one affine fit instead of the refinement ladder (first run) | 0.0015 | removed |

A3 and A7 are measured against the base of the first ablation run, which still had both steps and the earlier score.

Forcing the other polarity, dropping the scale prior, thresholding the OCT by intensity or mirroring it sends the pose to another
orientation, 30 to 48 mm away. Without the outline term the pose moves by 6.6 mm and turns by 15°. The two-class maps change the
I58 pose only a little once the outline is in the score (A4), just above the 0.5 mm deletion threshold over the specimen points,
and flattening less than that (A1, A2). Flattening is kept because it makes the pose stable. In an ablation run without
flattening (docs/results/xiangrui_I58/no_flattening/ablations.json) the base pose moved by only 0.24 mm, but the watershed mask,
3-D hole filling and dropping the outline term then moved the pose by 42.3, 40.9 and 42.5 mm, against 0.32, 1.69 and 6.6 mm
with flattening. A6 reaches a higher S with a non-physical pose and A8 a lower L with the anatomy on the wrong side, so neither
number can decide a registration.

### Label-free signals

result.json flags failures but does not certify a pose. Warning signs are a scale far from 1, a low S_outline, or any flag. On
I58, S is 0.2747 (S_class −0.1175, S_outline 0.5891) with polarity −1, and there are no flags. The run took 9 min 36 s,
5.19 GiB of peak RAM and 1.8 GiB of GPU memory.

## Parameters

All constants are fields of `octreg.params.Params`. The I58 run used the defaults (Params hash 8f731954d824ccac).

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
| `n_rot` | 8000 | count | rotations of the search (no mirror images) |
| `seed` | 0 | | seed of the rotation set |
| `topk` | 24 | count | search poses kept after non-maximum suppression, each refined |
| `nms_mm` | 3.0 | mm | poses with closer centres ... |
| `nms_deg` | 10.0 | deg | ... and closer rotations count as one |
| `iters` | 200 | count | Adam iterations per pose |
| `lam` | 2.0 | | λ, weight of the scale and shear prior |
| `clamp` | 0.15 | log-scale, shear | bound on each log-scale and shear |
| `lr_rot` | 0.02 | rad | Adam learning rate of the rotation vector |
| `lr_t` | 0.3 | mm | Adam learning rate of the translation |
| `lr_ls` | 0.01 | log-scale | Adam learning rate of the log-scales |
| `lr_sh` | 0.01 | shear | Adam learning rate of the shears |

## Removed components

A section-stripe flat field (destripe) took 365 of 906 s in the first run and moved the pose by 0.15 mm when switched off. A
rigid, similarity and affine ladder over three grids gave the same pose as one affine fit per search pose to within 0.0015 mm.
Both were removed. An earlier version gated poses by the fraction of the OCT specimen on MRI foreground, with a threshold set on
I58, and searched mirrored orientations. The outline term replaces the gate, and the file headers fix the handedness. The
research code at the git tag v1.1-archive also has a vascular channel, a fine registration stage, a trained tissue parser and
non-rigid refinement. None is in 1.0. The vascular channel helped on one pair only (I46), the fine stage was rejected on the
brainstem, the parser registered no better than the two-class maps, and the non-rigid gains were small.
