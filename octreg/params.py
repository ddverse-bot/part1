"""The method constants and ablation switches of octreg in one frozen dataclass. Lengths in mm."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

CHOICES = {"oct_foreground": ("texture", "intensity"), "features": ("two_class", "intensity"), "polarity": ("sign", "+1", "-1")}


@dataclass(frozen=True)
class Params:
    """Method constants and ablation switches; variants via dataclasses.replace or from_dict (a partial dict)."""
    # grids (step 1)
    levels: tuple = (0.6, 0.3, 0.15)          # isotropic pyramid (mm): search at levels[0], ladder down to levels[-1] (base grid)
    fine_mm: float = 0.04                     # OCT fine grid for the specimen mask and the stripe flat field
    # foreground (step 2)
    valley_ratio: float = 0.5                 # histogram valley / smaller neighbouring peak must be below this
    min_component: float = 0.01               # keep foreground components >= this fraction of the foreground volume
    texture_bandpass_mm: float = 0.08         # Gaussian sigma of the band-pass before the directional variation
    texture_window_mm: float = 0.36           # window of the directional local coefficient of variation
    texture_grid_mm: float = 0.16             # block grid of the texture field
    texture_smooth_mm: float = 1.2            # Gaussian sigma of the log texture field before its two-class (Otsu) threshold
    texture_close_mm: float = 0.48            # closing radius of the specimen mask
    destripe: bool = True                     # section-stripe flat field of the OCT (ablation switch)
    destripe_ncc_ratio: float = 0.8           # sectioning axis iff lowest adjacent-plane NCC < ratio x second lowest
    destripe_detect_mm: float = 0.12          # in-plane high-pass sigma of the planes compared by that NCC
    destripe_smooth_mm: float = 0.48          # normalised-convolution sigma in the section plane
    destripe_highpass_mm: float = 0.24        # high-pass sigma of the flat field along the sectioning axis
    # two-class maps (step 3)
    flatten_sigma_mm: float = 10.0            # Gaussian sigma of the local foreground mean used for flattening
    sigmoid_std: float = 0.25                 # p = sigmoid((I - t) / (sigmoid_std * std of foreground values))
    # orientation search (step 4)
    n_rot: int = 8000                         # uniform rotations, R[0] = identity
    seed: int = 0                             # rotation set seed
    mirror: bool = True                       # also search every rotation composed with diag(1, 1, -1)
    topk: int = 24                            # poses kept after non-maximum suppression
    nms_mm: float = 3.0                       # same pose if centres closer than nms_mm ...
    nms_deg: float = 10.0                     # ... and rotations closer than nms_deg
    overlap_rho: float = 0.6                  # tau = overlap_rho * min(1, V_MRI / V_OCT), for search poses and along the ladder
    overlap_floor: float = 0.15               # tau never below this (flagged)
    # affine ladder (step 5), one entry per level
    keep: tuple = (8, 3, 1)                   # poses kept after each level
    iters: tuple = (120, 200, 200)            # Adam iterations per degree-of-freedom stage at each level
    lam: float = 2.0                          # L = 1 - S + lam (sum log_scale^2 + sum shear^2)
    clamp: float = 0.15                       # |log_scale|, |shear| <= clamp (absolute)
    lr_rot: float = 0.02                      # rad
    lr_t: float = 0.3                         # mm
    lr_ls: float = 0.01                       # log-scale
    lr_sh: float = 0.01                       # shear
    # ablation switches (defaults = the method)
    oct_foreground: str = "texture"           # 'texture' (isotropic texture) | 'intensity' (histogram valley)
    mri_flatten: bool = True                  # flatten the MRI before the two-class map
    oct_flatten: bool = True                  # flatten the OCT before the two-class map
    features: str = "two_class"               # 'two_class' | 'intensity' (standardised intensities)
    polarity: str = "sign"                    # 'sign' of the best score | '+1' | '-1' (forced)
    ladder: bool = True                       # rigid -> similarity -> affine; False = affine directly at levels[-1]

    def __post_init__(self):
        for k, allowed in CHOICES.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"Params {k}={getattr(self, k)!r}: expected one of {allowed}")
        if not (len(self.levels) == len(self.keep) == len(self.iters) == 3 and self.levels[0] > self.levels[1] > self.levels[2]
                and 0 < self.fine_mm <= self.levels[2]):
            raise ValueError("Params: levels, keep and iters need 3 entries, levels decreasing, 0 < fine_mm <= levels[-1]")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Params from a partial dict (e.g. JSON): lists become tuples, ints are accepted for floats; unknown keys or wrong
        types raise ValueError."""
        base, kw = cls(), {}
        for k, v in d.items():
            dv = getattr(base, k, None)
            kw[k] = v = tuple(v) if isinstance(v, list) else float(v) if isinstance(dv, float) and type(v) is int else v
            if k not in base.__dataclass_fields__ or type(v) is not type(dv):
                raise ValueError(f"Params {k}={v!r}: unknown key or not a {type(dv).__name__}")
        return cls(**kw)

    def hash(self) -> str:
        """16 hex characters of the sha256 of the sorted-key JSON of all fields (process and platform independent)."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
