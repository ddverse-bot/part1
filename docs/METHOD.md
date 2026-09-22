# octreg 1.1: method

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
a cut face is not enclosed in 3-D, so holes are filled in every array plane. On I58 the mask measures 19.24 cm³ on the base
grid; an intensity threshold on the same OCT takes in the agarose and gives 29.52 cm³ (both after the mask is intersected with
the measured OCT, as bench/BENCHMARK.md reports them).

The rule needs an embedding to separate. Without one there are not two classes, and the threshold splits the tissue itself,
the more so when section stripes are strong, since a stripe makes tissue quiet along one axis and so look like embedding. A
block that is not embedded should be given its mask with `--oct-mask`, and the OCT file itself serves, since a mask file is
read as its positive voxels ("Beyond the target data").

The MRI foreground threshold is the first histogram valley, walking down from the brightest peak, below 0.5 of its smaller
neighbouring peak (peaks found on the square root of the smoothed counts); without one, nearly the whole field of view is kept
and the run is flagged. The mask above the threshold is then closed with a ball of 2 voxels, its holes are filled in 3-D and
components below min_component of the foreground volume are dropped, as for the OCT. Either mask can be supplied as a file
instead, and a supplied mask is used as given.

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
clamped to ±0.15. The lowest L wins. All of the prior is the penalty, which is needed because S alone rewards distorting the
block; the clamp is a bound that has never been reached, and a run that reaches it says so with the flag `clamp_saturated`
rather than returning a distorted block quietly. The two are ablated apart below.

Refining more than the best pose of §3 is not optional: on I58 the winner is the third of the 24 by search score
(`search_rank` 2 in result.json), so a run that refined one pose would have returned a different answer. It is also where the
time goes, 349 s of the run's 724 s against 35 s for the search.

## 5. Fine-structure refinement

The score of §2 compares large bright and dark regions and the outline. It finds the block, but it is dominated by the outline,
and an outline that carries a margin of agarose, or a cut face, leaves the last millimetres open. The best pose of §4 is
therefore refined on the internal structure of both volumes, fibre bundles, vessels and nuclei, compared by normalised gradient
fields (Haber and Modersitzki, 2006). On its base grid each volume I with mask m is smoothed and differentiated inside the mask
by normalised convolution, g = ∇(G_σ(I m) / G_σ(m)), so the mask edge adds no gradient, and n = g / √(|g|² + ε²), with ε the
median |g| over the MRI points used and over the OCT specimen mask. Both masks are eroded by 0.8 mm, which leaves the outline to
§2-4. Over every second interior MRI voxel along each axis, one in eight, with the eroded OCT mask w,

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

## 6. Smooth deformation

A specimen deforms slightly between the MRI scan and the OCT embedding, and serial sectioning adds its own small distortions.
After the affine, local misfits of a fraction of a millimetre remain that no affine removes. §6 fits a small smooth
displacement field on top of the affine. The affine stays the primary result: every transform file holds it alone, and the
field is written next to them.

The field u is defined on the MRI base grid, in mm along the MRI world axes, as a pull-back. The OCT content that belongs at
the MRI point x sits at x + u(x) in the affinely registered OCT, so the registered OCT at x is OCT(T⁻¹(x + u(x))). The OCT and
its specimen mask are resampled onto the MRI base grid through T (mask kept above 0.99), and two kinds of label-free local
evidence are measured on that volume.

Interior. Blocks of the MRI (4.5 mm, every 1.5 mm, at least 70 % of the block inside both masks eroded by 0.6 mm) are
searched in the OCT within ±1.35 mm
by zero-mean normalised cross-correlation of a contrast-free structure feature, the trace-free tensor n nᵀ of the normalised
gradient n of §5 (σ 0.24 mm, `octreg.blockmatch`). A match counts when its peak stands 4 standard deviations above its score map
and off the border of the range. Matches longer than 1.35 mm are dropped, the search range being per axis. Each match gives a
displacement vector at its block centre.

Boundary. The surface of the MRI foreground is sampled at up to 50,000 of its outermost voxels, with outward normals from the
gradient of the signed distance to the surface (smoothed, σ 2.5 voxels). Along each normal the intensity is read from −2.1 to
+2.1 mm in base-grid steps and differentiated with a Gaussian of σ 1 sample. The surface edge of a volume is the steepest fall
of its intensity going outwards, which assumes that in both volumes the tissue at the surface is brighter than what surrounds
it. A prominent fall is a local maximum of the fall more than 5 MADs above the median of the fall along the profile and within
±1.35 mm, and its position gets a parabolic sub-sample correction. The edge is found by this one rule in the MRI and in
the OCT. The offset δ is the OCT edge minus the MRI edge along the normal n, and it constrains n · u. What the rule does to an
edge is the same in both volumes and cancels in the difference, and so does the place of the sampled surface voxels relative to
the true surface. A point is used only when at least 95 % of its OCT profile is measured and each of the two profiles has
exactly one prominent fall. Faces crossed by section stripes show several falls and mostly drop out, so the sectioning axis need
not be known. On the synthetic pair of tests/test_deform.py a quarter of the points on a striped face survive, against three
quarters elsewhere, and what survives there is what the Huber weights of the fit then hold down.

The edge is used and not the bright rim of the OCT, because the rim is a layer below the surface. Its ridge lies inside the
surface by half the thickness of the layer, which differs from face to face, so a rule that takes the ridge for the surface
reads an offset where there is none.

Support. A boundary point is used only when an interior match lies within 5 mm of it. Where the inside of the two volumes does
not correspond, at torn or missing tissue or at a bubble, an edge nearby is not evidence of a deformation. The field there stays
at what its neighbourhood supports, and the affine is left as it is.

Model. Displacements c on a regular control lattice of 5 mm spacing over the bounding box of the MRI grid, trilinear in between,
are fitted by regularised least squares: three rows per interior match, one row n · u = δ per boundary point with the weight
√(3N / M) for N matches and M points, so that both kinds of evidence carry the same total weight, a membrane penalty λ on the
first differences of c along the lattice axes, and a small ridge on c. The system is solved four times, re-weighted by Huber's
rule at 0.3 mm on both kinds of residual.

Model choice. Only λ is chosen. Both kinds of evidence are split into four spatial folds (cells of 7 mm), the field is fitted
without one fold and tested on it, and the score of a λ is the held-out median interior error plus the held-out median boundary
error. The field has to stay plausible for fixed tissue, so the strain of the fit to all the evidence, the largest first
difference of c divided by the spacing, must stay below 0.15. The strain falls as λ grows. The smallest λ between 0.3 and 30
that keeps the limit is found by bisection in log λ, to a factor of 1.05. The candidates are this λ and the weights of the list
30, 10, 3, 1, 0.3 above it, and the candidate with the lowest score is chosen. A list alone would make the choice jump between
two of its weights. With the bisected weight it moves continuously with the evidence.

A field is applied only when the best candidate scores below 0.9 times the score of no deformation and at least 100 interior
matches and 300 supported boundary points exist. A run is also left alone when no weight up to 30 keeps the strain limit, so
that there is no candidate at all. Otherwise the run is flagged `deformation_not_supported`, no field is written and
oct_in_mri.nii.gz is the affine overlay. The evidence is measured once and the field is fitted once. Both kinds of evidence
are then measured again on the warped OCT, the interior over every confident match before the length drop and the boundary over
every point with one edge, and these residuals are the read-outs of the stage.

The field is small by construction. Both kinds of evidence are searched within 1.35 mm, the smoothness is the one that
predicts evidence it has not seen, and the strain limit bounds the model that can be chosen. result.json holds, under `deform`,
the lattice spacing and the chosen λ, the table of candidates, the held-out errors, the residuals of both kinds measured before
and after, the field magnitude over the MRI foreground and the largest strain. oct2mri_warp.nii.gz holds u as a NIfTI vector
image on the MRI base grid, with components along the NIfTI world axes (ITK and ANTs expect the first two negated).
oct_in_mri.nii.gz is resampled through T and u, oct_in_mri_affine.nii.gz through T alone, and qc_deform.png shows the MRI, both
overlays and the field magnitude on three planes, each with the outline of the MRI foreground. qc.png and qc_montage.png keep
showing the affine. `octreg apply` takes OCT-frame volumes through T and u. The field is not inverted, so the MRI to OCT
direction stays affine.

## Evaluation

A registration is judged by looking at it, because label-free numbers can prefer a wrong pose. In every plane of qc_montage.png
(four planes per OCT axis, each shown as OCT, MRI through the transform on the grey scale of the OCT, a checkerboard of the two
wherever either map reaches, and the OCT with both outlines) the MRI boundary must follow the edge of the OCT specimen, the OCT
specimen must lie on the same anatomy in the MRI, internal structures must continue across the checkerboard, folded pieces, and
cut faces that the MRI also shows, must correspond and lie on the same side (where the block was cut out of a larger specimen,
the MRI tissue continues beyond the OCT cut face), and the contrast must be consistently inverted or not. Other candidate
transforms can be rendered with `octreg qc --T` and compared side by side.

Which panel carries the decision was measured, not assumed. On the I58 pair, the run's pose and the same pose translated by
3 mm were rendered as pairs, in a random order per image, and given to language-model readers one image each, with leave to
answer that they could not tell. Two panels were compared at the same three planes, one per OCT array axis through the
specimen centroid. On the OCT with the outline of the MRI foreground through the pose, which is the fourth column of qc.png
without its second outline, 3 images read 4 times each came back 12 right, 0 wrong, 0 refusals, mostly at high confidence and
for the right reason, that the line cuts inside the tissue on one side and floats in the background on the other. On the
checkerboard, 9 images read twice each, three ways of choosing the region to draw it over by three planes, came back 7 right,
5 wrong and 6 refusals: chance. Zooming does not rescue it. The same checkerboard in a 14 mm window, seven squares across,
where one square holds a structure a reader can follow, was read 3 images 4 times each and came back 5 right, 6 wrong and 1
refusal. One reader gave the mechanism: in that window the two volumes are inversely contrasted, so a correct alignment makes
a bright square abut a dark one and the grid of steps looks stronger, not weaker; that reader judged by whether the fibre
bands kept their angle across a seam, and got it right. A third panel sits between the two: the interior of the registered
OCT with the MRI's own structure edges on it, away from the boundary, read 3 images 4 times each, came back 9 right, 1 wrong
and 2 refusals, at low confidence where the boundary had been read at high.

So the boundary decides, the interior carries real but weaker evidence, and a checkerboard shows the result most directly,
with nothing drawn on it and nothing derived from either mask, while settling nothing finer than a block dropped on the wrong
structure, which is how the DANDI failures below read. The figure of the README is the checkerboard, for what it shows
directly; the panel that decides a pose is the fourth column of every run's qc.png.

The region the checkerboard is drawn over is part of that honesty. It covers everything either the specimen mask or the MRI
foreground reaches, and leaves out only what both call background, where the two cannot disagree. Drawing it over the
intersection instead was tried and refused: inside that region the two maps agree by construction, so the visible disagreement
is 0.000 at 0, 2 and 5 mm of imposed error, and the worse the pose, the more the panel hides.

### I58 brainstem pair

On the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) the run takes 12 min 4 s, 5.2 GiB of
RAM and 1.6 GiB of GPU memory allocated by torch (RTX 5090, Windows), of which §6 is 23 s. §1-4 give S 0.2855 (S_class −0.1144,
S_outline 0.6277) and polarity −1; §5
raises F from 0.088 to 0.103 and moves the block by 2.4 mm at its corners (1.5 mm over the specimen, 5.3°), with scales 0.977 /
0.974 / 0.985. Raw 20 µm OCT values mapped through the header and the transform correlate with the affine overlay
(oct_in_mri_affine.nii.gz) at Spearman 0.990, against at most 0.256 with any OCT axis flipped.

docs/figures/registration_I58.png shows the result on three MRI planes, read from the original files: the MRI, the registered
OCT, and those two panels cut into 8 mm squares and interleaved. The checkerboard is nothing but the two of them, on their own
grey scales and neither inverted nor matched nor masked, which is why the figure is built on it; what a checkerboard can and
cannot settle is measured under "Evaluation". In the axis-0 plane the MRI tissue continues beyond the block. On 21 MRI planes at
the MRI voxel size (MRI, OCT and a colour fusion on one grid), with landmark positions measured by language-model image readers
and re-measured by a second reader, vessels, fissures and fibre-bundle edges of the brainstem agree to a median of about 0.3 mm,
with no shift, rotation or scale common to the planes. Two parts do not follow an affine transform: the torn cerebellar pieces,
which moved during sectioning, and the sections at one end of the block (below about 12 mm along MRI axis 1), which are 10 to 14
% larger in plane than the MRI there and misfit by up to 2 mm, while vessels a few millimetres further in match again. The §5
pose was compared with the §4 pose on 15 MRI planes, shown as A and B in random order and judged separately on internal
structure and on boundaries by such readers: the §5 pose was preferred 20 times (13 clearly), the §4 pose 3 times (1 clearly),
with 7 ties. At the §5 pose S is lower (0.251, S_class −0.067): the coarse score does not resolve the last millimetres.

After §5, misfits of 0.5 to 1 mm remain on I58 between the OCT surface and the MRI tissue boundary, mostly on the faces towards
one end of the sectioning axis, and the interior fine structure asks for an OCT that is smaller by 4, 2 and 0 % along three
axes. No affine removes both. The interior signal is not an artefact of the rim: matches at least 5 mm below the surface give
the same local affine as shallow ones (singular values 1.043, 1.018, 1.004 against 1.033, 1.015, 1.005), also with the masks
eroded by 1.5 mm. In the full run §6 found 502 interior matches and 8,402 supported boundary points and chose λ 0.51. The
held-out median errors fell from 0.247 to 0.155 mm (interior) and from 0.249 to 0.093 mm (boundary). The re-measured residuals
fell from 0.249 to 0.121 mm (interior) and from 0.285 to 0.104 mm (boundary, with 74 % of the offsets within 0.3 mm, from 51 %).
Over the MRI foreground the field has a median of 0.22 mm and a maximum of 1.16 mm, its largest strain is 0.148, and the stage
took 23 s. The affine of the run is that of release 1.0 to 0.003 mm at the block corners, the float noise between two runs,
since §1-5 did not change.

Detached cerebellar parts that moved by more than the 1.35 mm reach, folia and a lobule 1.5 to 2 mm off and whole torn flaps
several millimetres off, stay where the affine puts them. The stage does not chase them, apart from the flap edge noted below,
and no part of the method is built for them: in a checkerboard of this specimen the brainstem continues across the tiles and the
torn cerebellum breaks.

The two overlays of the run, through the affine alone and through the affine and the field, were compared on 15 random MRI
planes (five per axis), shown as X and Y in random order with the outline of the method's MRI foreground, by language-model
image readers who each looked for one thing: the boundary, interior structures, the worst error, artefacts, and the four halves
of each plane. The overlay with the field was preferred 65 times and the affine 4 times, with 6 ties, and 53 to 0 among the
clear votes. The readers named the same cause throughout: along one flank of most planes the affine leaves the OCT tissue 0.5 to
1.4 mm outside the MRI boundary, and with the field it lies on it to about 0.2 mm, while fibre bundles and vessels that were 0.4
to 0.7 mm off come to overlap. Section stripes stayed straight and evenly spaced. The same comparison between a field fitted to
the rim ridge and the field of the method (three readers: boundary, interior, artefacts) gave 1 vote to 43 with 1 tie, and no
artefact in either. The readers saw a field fitted with λ 0.32, from an earlier run of this stage on the cached base grids of the
ablation prep at the affine of release 1.0. It differs from the field of the method by a median of 0.024 mm, a sixth of a
base-grid voxel, and by at most 0.125 mm.

Two flaws were introduced. In axial plane 217 a vessel that the affine places on its MRI position lies 0.9 mm off with the
field, where neighbouring block matches disagree among themselves about the displacement through the plane. And the reader for
artefacts found the torn cerebellar flap in two axial planes squeezed by about 10 % to bring its outer edge onto the MRI
boundary, within the strain limit but not anatomy. That reader measured a compression of 5 to 7 % near the corrected surfaces
and judged it smooth.

The boundary evidence of §6 is the surface edge because of what the rim does on this specimen. The OCT intensity along the MRI
boundary normals peaks 0.3 mm inside the MRI boundary and falls to half exactly at it. Taking the ridge of the rim for the
surface reads −0.77 mm on faces whose normal points along the sectioning axis, +0.43 mm on the opposite faces and −0.20 mm on
the faces across it, while the interior matches show no such pattern (against their normal component the ridge offsets follow
δ = 0.88 × interior − 0.36 mm, median disagreement 0.44 mm). The edge rule gives −0.09, +0.60 and +0.12 mm on the same faces,
six times as many usable points (20,550 against 3,393), a median |δ| of 0.30 instead of 0.58 mm and a median disagreement with
the interior of 0.27 mm. With the ridge evidence the strain limit decided between two list weights whose strains were 0.143 and
0.155, which is why the smallest weight is bisected. With the edge evidence the held-out error has a flat minimum near λ 0.1
to 0.3 and the strain limit meets it there. The folds test interpolation, which is what the field is used for: folds with a
buffer of 4.5 mm between training and held-out blocks (cells of 14 or 21 mm) predict nothing, since the field has a
correlation length of a few millimetres.

The elements of §6 were ablated one at a time on the base grids of the I58 benchmark, at the affine above, in a separate run
of the stage, so its rows can sit one digit from the run's own numbers above. Each row is scored
by the read-outs of the stage on the warped OCT and by two numbers the fit never sees: F of §5 at σ 0.3 mm, and the two-class
score of §2 (polarity × S_class) over the core 1.5 mm below both surfaces.

| variant | λ | strain | interior (mm) | boundary (mm) | within 0.3 mm | F | two-class, core | field median / max (mm) |
|---|---|---|---|---|---|---|---|---|
| no deformation | | | 0.250 | 0.285 | 51 % | 0.0781 | 0.0651 | 0 |
| §6 | 0.51 | 0.148 | 0.121 | 0.105 | 74 % | 0.0811 | 0.0766 | 0.22 / 1.16 |
| interior evidence only | 0.30 | 0.134 | 0.101 | 0.237 | 58 % | 0.0806 | 0.0614 | 0.21 / 1.02 |
| boundary evidence only | 1.02 | 0.148 | 0.495 | 0.100 | 75 % | 0.0775 | 0.0851 | 0.40 / 1.30 |
| rim ridge instead of the edge | 1.22 | 0.148 | 0.176 | 0.290 | 52 % | 0.0794 | 0.0683 | 0.26 / 0.98 |
| no support rule | 0.82 | 0.148 | 0.124 | 0.089 | 80 % | 0.0824 | 0.0712 | 0.28 / 1.05 |
| no Huber re-weighting | 0.92 | 0.148 | 0.136 | 0.117 | 72 % | 0.0808 | 0.0745 | 0.21 / 1.09 |
| lattice 7 mm | 0.30 | 0.143 | 0.140 | 0.104 | 76 % | 0.0816 | 0.0704 | 0.28 / 1.25 |
| lattice 10 mm | 0.30 | 0.123 | 0.161 | 0.111 | 75 % | 0.0812 | 0.0735 | 0.29 / 1.27 |
| strain limit 0.10 | 1.57 | 0.099 | 0.151 | 0.125 | 71 % | 0.0804 | 0.0771 | 0.19 / 0.96 |
| strain limit 0.20 | 0.30 | 0.194 | 0.112 | 0.099 | 75 % | 0.0814 | 0.0754 | 0.24 / 1.23 |

Each kind of evidence alone leaves the other kind where it was or makes it worse, so both stay. A field fitted to the ridge
offsets does not move the edges at all (0.285 to 0.290 mm). The support rule costs a little on the boundary read-out and keeps
the field out of regions without correspondence, the torn cerebellar folia and a bubble. Two parts of an earlier version had no
effect and were deleted. A second and a third round of warping, measuring again and refitting left the residuals where they
were, interior 0.111, 0.112 and 0.110 mm and boundary 0.099, 0.113 and 0.097 mm after one, two and three rounds, measured while
the stage still had them. A search over lattice spacings of 10, 7 and 5 mm chose 5 mm, which the two lattice rows confirm.

Each ablation changes one element of the method (the §1 grids and masks are computed once per mask source, and the no-§5 row
stops after §4). Pose changes are block-corner means, of the pose after §4 against the method's pose after §4 and of the final
pose against the result; the cut-face row compares the two outlines on the same cut.

| change | after §4 (mm) | final (mm) |
|---|---|---|
| intensity-threshold OCT mask instead of the texture mask | 42.0 | 42.9 |
| no outline term | 41.7 | 40.9 |
| polarity forced to +1 | 41.1 | 42.3 |
| no scale prior (λ 0, clamp 1.0): S 0.501 at the §4 pose, final scales 0.51-0.57 | 38.7 | 35.7 |
| the other handedness (OCT mirrored) | 29.5 | 30.4 |
| of the scale prior, the clamp alone (λ 0, clamp kept): final scales 0.86-0.88, at the bound | 16.9 | 38.6 |
| of the scale prior, the penalty alone (clamp 1.0, λ kept) | 0.00 | 0.00 |
| no fine-structure refinement (§5) | 0 | 2.36 |
| OCT flattening off | 2.41 | 0.00 |
| holes filled in 3-D only | 2.19 | 0.17 |
| standardised intensities instead of two-class maps | 0.81 | 0.00 |
| two-sided outline (agarose over MRI tissue counted as a mismatch) | 0.23 | 0.00 |
| MRI flattening off | 0.18 | 0.00 |
| cut face simulated by removing the mask beyond 70 % of its extent along OCT axis 1: method / two-sided outline | 0.40 / 4.68 | 0.11 / 3.87 |

The first five turn the block over or misplace it, and §5, which only refines, does not bring it back. The two rows that
split the scale prior show that all of it is the penalty: without the penalty the block collapses onto the clamp, at 0.86 to
0.88 of its length, and without the clamp the pose does not move at all. §5 itself moves the result by 2.4 mm. The next five
changes of the §4 pose lie within the reach of §5, which settles them; those steps stay for the §4 pose they give.

Two of the five move that pose by less than the 0.5 mm at which the benchmark deletes a step, and neither is kept on this
pair's evidence. The one-sided outline (0.23 mm here) is what holds the block at a cut face: with one simulated, the two-sided
version ends 3.9 mm off, in the last row of the table. MRI flattening (0.18 mm here) does nothing on this pair, whose MRI is a
small crop with little bias, and does work on a whole-hemisphere MRI. At the pose two methods agree on for DANDI sub-I55,
switching it off drops |S_class| from 0.719 to 0.672, while switching OCT flattening off changes it by 0.002 there, against
2.41 mm of §4 pose here. Each of the two flattenings is load bearing on one of the two specimens and idle on the other, which
is why one pair is the wrong place to decide either.

The two-class maps are the same story read the other way. On this pair they are the weaker feature: |S_class| is 0.11 at the
result and A4's standardised intensities reach 0.15, and the block is found by the outline. On sub-I55, where the outline
carries nothing, the two-class maps give 0.719 against 0.389, and against 24 wrong poses of the block, 12 near and 12 dropped
anywhere in the crop, their margin over the best wrong pose is 0.41 against 0.22. What the map buys is a signal where a shape
comparison has none, which is the case the method exists for.

### Beyond the target data: DANDI:000026

The Broca-area blocks of DANDI:000026 (Costantini et al. 2023) are cortex slabs cut from a hemisphere and imaged without
embedding at 12 to 35 µm, each with an ex-vivo MRI of the whole hemisphere and labels the registration never sees. They are
not the target data of this method. Nothing is embedded, so the specimen mask covers the whole block and the outline term has
little to hold on to, and one slab of cortex looks much like another. They are the hardest generality check available.

One caution on names: sub-I58 of this dataset is a cortex block from the same donor as the brainstem pair above, and a
different specimen. The subjects are written sub-IXX here and the brainstem pair is never written that way.

Nine subjects have both modalities. In sub-I56 the affine of every label file disagrees with its array, so the block region
cannot be read from it and it is left out. The other eight are run as any other pair, through `octreg register` with the voxel
spacing of the BIDS sidecar (`bench/dandi.py`). The MRI is cut to the bounding box of the subject's own Broca-area label plus
5 mm, with every side grown to at least the longest side of the OCT array: that label marks the cortical ribbon and is a thin
sheet, while the block reaches below it, and a crop that cannot hold the block is not the input the method asks for. The OCT
files carry no orientation, so §3-5 run for both handednesses.

| subject | crop (mm) | block (mm) | specimen mask of the measured OCT | S | S_class | F | layer separation: §4 pose -> final | random poses |
|---|---|---|---|---|---|---|---|---|
| sub-I38 | 62 × 62 × 47 | 47 × 21 × 18 | 1.9 of 55.1 cm³ (3 %) | 0.5211 | +0.2857 | 0.2896 | 0.047 -> 0.073 | 0.049, max 0.074 (5 of 10 scored) |
| sub-I46 | 47 × 42 × 26 | 15 × 15 × 8 | 1.5 of 1.5 cm³ (98 %) | 0.6169 | +0.4270 | 0.1884 | 0.118 -> 0.104 | 0.098, max 0.396 (7 of 10 scored) |
| sub-I48 | 53 × 52 × 40 | 42 × 37 × 26 | 30.7 of 39.9 cm³ (77 %) | 0.3227 | -0.1701 | 0.1026 | 0.014 -> 0.029 | 0.100, max 0.156 (7 of 10 scored) |
| sub-I55 | 41 × 41 × 32 | 22 × 21 × 21 | 6.9 of 9.9 cm³ (70 %) | 0.7307 | -0.6733 | 0.2205 | 0.430 -> **0.432** | 0.108, max 0.371 (8 of 10 scored) |
| sub-I57 | 71 × 71 × 48 | 60 × 37 × 27 | 29.6 of 31.2 cm³ (95 %) | 0.3639 | -0.2075 | 0.0942 | 0.070 -> 0.013 | 0.058, max 0.074 (4 of 10 scored) |
| sub-I58 | 53 × 52 × 41 | 41 × 32 × 23 | 8.7 of 21.4 cm³ (40 %) | 0.3682 | -0.1266 | 0.0680 | 0.194 -> 0.004 | 0.091, max 0.107 (3 of 10 scored) |
| sub-I61 | 52 × 52 × 39 | 42 × 32 × 21 | 20.3 of 24.6 cm³ (82 %) | 0.4130 | -0.1288 | 0.0713 | 0.015 -> 0.013 | 0.090, max 0.131 (7 of 10 scored) |
| sub-I62 | 72 × 55 × 38 | 45 × 37 × 21 | 7.5 of 24.6 cm³ (31 %) | 0.3817 | -0.1672 | 0.1047 | 0.136 -> 0.029 | 0.108, max 0.203 (4 of 10 scored) |

The crop contains the block in every row. §6 reports `deformation_not_supported` in every row.

A pose counts as found when the two ends of the cortical-layer label, brought into the registered OCT, are told apart by the
two-class map of §2 better than at any random pose of the block inside the crop, and when the outlines of that label follow the
bright and dark bands of the registered OCT (`qc_labels.png`). Both were decided before the runs. Ten random poses are drawn
per subject and the ones that drop too little of the label inside the specimen mask cannot be scored, so between three and
eight of them count; the table gives that number.

One block registers. On sub-I55, 22 mm across, the separation is 0.432 against 0.371 at the best of the eight random poses that
could be scored, and the
label outlines follow the gyri of the registered OCT in every plane. The read-out is calibrated on that block by an
independent route: the pose the earlier research pipeline of this dataset found for it, with different machinery and judged
correct by eye at the time, lies 1.4 mm from octreg's over the block corners (1.96 mm at most, 2.2°) and gives 0.436. Two
methods that share no code agree on this block, and the read-out reacts to both.
Four blocks fail outright. On sub-I48, sub-I57, sub-I58 and sub-I61 the separation at the result is 0.004 to 0.029, below the
median of the random poses, the label outlines cut across the bands of the registered OCT, and the checkerboard puts OCT cortex
on deep MRI structures. The specimen mask is not the problem on three of them, where it keeps 77 to 95 % of the measured OCT,
and on sub-I58, where it keeps 40 %, the repaired mask below does not help either.

Three are undecided rather than failed, for reasons the benchmark can name. On sub-I46 the block is 15 by 15 by 8 mm and lies
inside cortex at almost any pose, so random poses reach 0.396 and neither octreg's pose (0.104) nor the earlier pipeline's
(0.158) stands out: the read-out has no power there. On sub-I38 and sub-I62 §1 keeps 3 % and 31 % of the measured OCT, so few
labelled points land inside the mask at any pose and the earlier pipeline's pose cannot be scored at all. What the method does
with a mask like that is not a test of §2-6.

Where §1 keeps less than half of the measured OCT, the block was registered again with the OCT given as its own `--oct-mask`,
which is the specimen mask a block without embedding should have. This is a diagnostic, not a second method: the released
command with one more argument, and `bench/dandi.py --steps evaluate --tag _allmeasured` for the read-out, which writes the
eval.json every number below is taken from. It rescues none of the three. On sub-I38 the repaired mask lets the read-out work at
last, with 110,765 labelled points inside the mask instead of 2,397, and the pose is still wrong: the separation is 0.013
against a random median of 0.056, while F falls from 0.290, measured on mask fragments, to 0.107. On sub-I58 the separation
rises to 0.128 against a random maximum of 0.068, above the random range, but the label outlines cut across the gyri of the
registered OCT and the checkerboard puts cortex on white matter, so the second half of the criterion refuses the pose. A
label-free number can prefer a wrong one, which is why the criterion has two halves and why the evaluation of this method is
visual. On sub-I62 the repaired mask makes the pose worse: S_outline falls from 0.81 to 0.10, and at the new pose fewer of the
136,395 points of the two label ends land inside the specimen mask than the 500 the read-out needs, against 1,207 before, so it
cannot be computed at all.

The failures are not silent within this set. On the block that registered the two-class agreement |S_class| is 0.67 and the
fine-structure agreement F is 0.22. On the four that fail outright |S_class| is 0.13 to 0.21 and F is 0.07 to 0.10. The three
undecided rows sit between: sub-I46 reaches 0.43 and 0.19, sub-I38 0.29 and 0.29, sub-I62 0.17 and 0.10, and sub-I38's F is
measured on mask fragments and falls to 0.11 once the mask is repaired.

Those bands are not a threshold, and nothing here licenses carrying them to another specimen. The brainstem pair above is a
correct registration and its own |S_class| is 0.11, below every failing block in this table, because its class structure is
weak and its outline term, S_outline 0.63, is what finds it; on these cortex slabs the outline term carries nothing and
|S_class| is all there is. A score says how much of the evidence the method used was there to be had, which depends on the
specimen. What it cannot do is replace looking at the overlay, which is why the criterion has a second half. The scores
separate the outright failures from the one success here and
leave the undecided rows undecided, which is what a user reading result.json gets.

What fails is mostly §3, the search. Without embedding the specimen mask is the block itself, so the outline term compares a
rectangular slab with a crop full of cortex and adds almost nothing, and the two-class map of a cortical slab repeats across
the crop. On six of the eight the §4 pose is already inside the random range, so the search never gets near. The two that
are not are sub-I55, which registers, and sub-I58.

On three it is not that simple, and the §4 column of the table is there to show it. On sub-I58 the §4 pose separates the layer
ends at 0.194, above that subject's random maximum of 0.107, and §5 moves it to 0.004. On sub-I57 and sub-I62 §5 also lowers
the read-out, from 0.070 to 0.013 and from 0.136 to 0.029, although neither §4 pose clears its random range. So on cortex
without embedding the fine-structure refinement is not a safe last step either: §5 is written to refine a pose the outline has
already placed to within a millimetre or two, and it has no way of knowing that it has not been given one. On sub-I55, where
the block is found, §5 changes the read-out by 0.002, which is what refining a correct pose looks like.

The crop assumption is met here, by construction. The embedding assumption, stated at the top of this document, is
not, and these blocks are included because they are the hardest case available, not because the method claims them.

§6 does not engage here. Its interior evidence is block matches of fine structure shared by both volumes, and cortex at
0.15 mm has almost none, so every subject reports `deformation_not_supported` and writes no field. That is the gate doing its
work, and it leaves the smooth deformation validated on the brainstem pair alone.

## Parameters

The tunable constants are fields of `octreg.params.Params` (override with `--params`); the brainstem pair used the defaults. A
stage also holds a few constants of its own, at the top of its module: they set how a step is computed rather than what a
dataset needs, they do not enter the Params hash, and their values are in the text of the section that uses them.

| parameter | default | role |
|---|---|---|
| `search_mm`, `base_mm`, `fine_mm` | 0.6, 0.15, 0.04 mm | search grid, base grid, OCT grid of the specimen mask |
| `valley_ratio`, `min_component` | 0.5, 0.01 | MRI histogram valley / smaller peak; smallest MRI foreground component kept, as a fraction of the foreground |
| `texture_bandpass_mm`, `texture_window_mm` | 0.08, 0.36 mm | smoothing before, and window of, the directional coefficient of variation |
| `texture_grid_mm`, `texture_smooth_mm`, `texture_close_mm` | 0.16, 1.2, 0.48 mm | texture blocks, smoothing of log F (set on the brainstem pair), closing radius |
| `flatten_sigma_mm`, `sigmoid_std` | 10 mm, 0.25 | flattening scale; sigmoid width in foreground standard deviations |
| `n_rot`, `seed`, `topk` | 8000, 0, 24 | rotations, rotation set seed, poses refined |
| `nms_mm`, `nms_deg` | 3 mm, 10° | poses closer in both count as one |
| `iters`, `lam`, `clamp` | 200, 2.0, 0.15 | Adam iterations, prior weight λ, bound on log-scales and shears |
| `lr_rot`, `lr_t`, `lr_ls`, `lr_sh` | 0.02 rad, 0.3 mm, 0.01, 0.01 | Adam learning rates (§5: a fifth of these) |
| `ngf_sigmas_mm`, `ngf_erode_mm`, `ngf_iters` | 0.6, 0.4, 0.3 mm; 0.8 mm; 150 | gradient scales of the §5 passes, mask erosion, Adam iterations per pass |
| `df_sigma_mm`, `df_block_mm`, `df_step_mm` | 0.24, 4.5, 1.5 mm | §6: Gaussian σ of the gradients of the structure feature, edge of the matched blocks, grid step of the block centres |
| `df_z_min`, `df_erode_mm` | 4, 0.6 mm | §6: standard deviations above the mean of its score map a match needs, erosion of both masks that keeps the blocks inside |
| `df_range_mm`, `df_reach_mm`, `df_profile_mm` | 1.35, 1.35, 2.1 mm | §6: search range of the interior matches, largest distance of a surface edge from the MRI foreground surface, half length of a boundary profile |
| `df_edge_mad`, `df_support_mm` | 5, 5 mm | §6: prominence of a fall in MADs above the median of the profile derivative, largest distance from a boundary point to an interior match |
| `df_huber_mm`, `df_grid_mm` | 0.3, 5 mm | §6: Huber threshold of both residuals, spacing of the control lattice |
| `df_lams`, `df_max_strain` | 30, 10, 3, 1, 0.3; 0.15 | §6: list of membrane weights λ, to which the smallest λ that keeps the strain limit is added; the strain limit |
| `df_gain`, `df_min_interior`, `df_min_boundary` | 0.9, 100, 300 | §6: held-out score needed relative to no deformation, least interior matches and supported boundary points for a field |
