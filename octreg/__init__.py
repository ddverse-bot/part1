"""octreg: label-free affine registration of an OCT block to an MRI of the same specimen.

Both images are reduced to the same soft two-class map: a foreground threshold, a sectioning gain on the OCT when the data
show sectioning, 10 mm bias flattening and a sigmoid split at the Otsu threshold. An exhaustive FFT search of masked
normalised cross-correlation over rotations, translations, handedness and relative contrast polarity gives the best poses of
the four handedness x polarity hypotheses in one pass; each is refined by a rigid -> similarity -> affine ladder under an
absolute scale prior. The winner is the pose with the lowest penalised loss, and each competing choice (polarity, handedness,
a distinct runner-up) counts as decided only when a paired block jackknife inside the specimen puts it at z >= 3. Handedness
and polarity are reported with this evidence and never assumed; the status describes the evidence, not correctness.
All constants are in octreg.params.Params.
"""
__version__ = "1.0.0.dev0"

from .params import Params  # noqa: E402
from .pipeline import register  # noqa: E402

__all__ = ["Params", "register", "__version__"]
