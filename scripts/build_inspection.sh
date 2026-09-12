#!/bin/bash
# Assemble ~/autodl-tmp/oct-mri-registration/results_inspection/<run>/ for manual checking:
# transforms, result.json, QC figures, NIfTI regions (open in Freeview/ITK-SNAP), plus INDEX.md.
cd ~/autodl-tmp/oct-mri-registration
R=results_inspection; mkdir -p $R
for run in I46_otsu I46_parser I55_otsu I38_otsu I56_otsu I62_otsu I48_otsu I57_otsu I61_otsu I58_otsu xiangrui_I58bs_novasc xiangrui_I58bs; do
  s=work/runs/$run
  [ -d "$s" ] || continue
  d=$R/$run; mkdir -p $d
  for f in T_oct2mri.npy T_oct2mri_structural.npy T_oct2mri.lta result.json qc_vessels.png qc_oct_space.png mri_region.nii.gz oct_in_mri_region.nii.gz mri_labels_region.nii.gz oct_class150.npy; do
    [ -f "$s/$f" ] && cp -n "$s/$f" "$d/" 
  done
done
cat > $R/INDEX.md <<'IDX'
# Registration results — manual inspection index (v1, 2026-08-21)

Per run: `T_oct2mri.npy/.lta` (OCT world -> MRI world; `_structural` = before the vascular stage),
`result.json` (all diagnostics), `qc_vessels.png` (3 depth slices: OCT | MRI via structural T | MRI via final T,
with OCT WM/GM contour cyan, OCT vessels green, manual MRI vessels yellow), and NIfTI volumes on the same grid
for Freeview/ITK-SNAP: `mri_region.nii.gz` + `oct_in_mri_region.nii.gz` (+ labels where they exist).

| run | verdict (visual QC) | note |
|---|---|---|
| I46_otsu | correct | full annotations; vessels 167 um / f150 0.47 (own seg 123/0.56), Dice 0.80/0.83, depth stretch 1.22 |
| I46_parser | correct | trained-parser comparison, same result as otsu |
| I55_otsu | correct | Dice 0.92/0.93; sidecar depth spacing correct -> vascular stage moves only 0.33 mm |
| I38_otsu | correct | 29x52x47 mm slab, gyrus-by-gyrus; labels cover 61% (low Dice is an artifact) |
| I56_otsu | correct | 33x49x44 mm slab, gyrus-by-gyrus; partial labels |
| I62_otsu | correct | no hemisphere labels (Dice 0 is an artifact) |
| I48_otsu | uncertain | no search margin (0.257 vs 0.256) |
| I57_otsu | wrong | captured by a flat MRI region (two-piece slab) |
| I61_otsu | wrong | MRI tissue mask failed (bath; mask frac 0.74) -> placed in noise |
| I58_otsu | wrong | only a 10-deg-flip MRI exists (almost no GM/WM contrast) |
| xiangrui_I58bs_novasc | coarse correct (reported) | brainstem pair; vascular stage off (seam artifacts); residual a few mm |
| xiangrui_I58bs | superseded | same but with the (harmful here) vascular stage; kept for comparison |
IDX
du -sh $R; ls $R
