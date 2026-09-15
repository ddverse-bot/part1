"""octreg: label-free affine registration of a serial-sectioning OCT block to an ex-vivo MRI cropped around it.

1. Specimen mask from isotropic texture: agarose has the intensity of tissue, but its artefacts are anisotropic.
2. One soft two-class map for both modalities, scored together with the specimen outline; the contrast polarity is the sign of
   the two-class score.
3. FFT orientation search inside the MRI crop, then one affine refinement per search pose under a scale prior.
4. Fine-structure refinement of the best pose by normalised gradient fields, which also decides the handedness of an OCT
   without an orientation header.

Method constants are in octreg.params.Params (numerical guards and fixed rule literals are documented where they are used).
Python use: `from octreg.register import register, apply, qc`; command line: `octreg register`, `octreg apply`, `octreg qc`.
"""
__version__ = "1.0.0"

from .params import Params  # noqa: E402

__all__ = ["Params", "__version__"]
