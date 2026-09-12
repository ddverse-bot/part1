"""octreg — OCT-block-to-MRI registration via a contrast-agnostic structural parser.

Modules
-------
common    : I/O, physical-space affines, GPU resampling, OCT preprocessing
synth     : synthetic image generator from label maps (SynthSeg-style, with OCT-specific degradations)
parser    : 3D U-Net structural parser (bg / WM / infragranular GM / supragranular GM), training + inference
features  : registration representations: parser probabilities, MIND-SSC, Otsu tissue classes, intensity
search    : masked-NCC FFT global search over rotations x translations (no location prior)
refine    : differentiable rigid / similarity / affine refinement (multi-resolution)
evaluate  : vessel-distance, GM/WM overlap, robustness, QC figures
"""
__version__ = "0.1.0"
