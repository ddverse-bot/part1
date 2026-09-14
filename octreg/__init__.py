"""octreg: label-free affine registration of a serial-sectioning OCT block to an ex-vivo MRI cropped around it.

1. Specimen mask from isotropic texture: agarose has the intensity of tissue, but its artefacts are anisotropic.
2. One soft two-class map for both modalities; the contrast polarity is the sign of the best FFT score.
3. Orientation search inside the MRI crop, refined by a rigid -> similarity -> affine ladder under an absolute scale prior.

Method constants are in octreg.params.Params (numerical guards and fixed rule literals are documented where they are used).
Python use: `from octreg.register import register, apply`; command line: `octreg register` and `octreg apply`.
"""
__version__ = "1.0.0"

from .params import Params  # noqa: E402

__all__ = ["Params", "__version__"]
