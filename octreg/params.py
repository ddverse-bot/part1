"""The method constants of octreg in one frozen dataclass. Lengths in mm."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Method constants (no switches); a variant via dataclasses.replace or from_dict (a partial dict)."""
    # grids
    search_mm: float = 0.6                    # isotropic grid of the orientation search, an integer multiple of base_mm
    base_mm: float = 0.15                     # isotropic grid of the two-class maps and the affine refinement
    fine_mm: float = 0.04                     # OCT grid of the specimen mask
    # foreground and specimen mask (§1)
    valley_ratio: float = 0.5                 # MRI histogram valley / smaller neighbouring peak must be below this
    min_component: float = 0.01               # keep MRI foreground components >= this fraction of the foreground volume
    texture_bandpass_mm: float = 0.08         # Gaussian sigma of the band-pass before the directional variation
    texture_window_mm: float = 0.36           # window of the directional local coefficient of variation
    texture_grid_mm: float = 0.16             # block grid of the texture field
    texture_smooth_mm: float = 1.2            # Gaussian sigma of the log texture field before its two-class (Otsu) threshold
    texture_close_mm: float = 0.48            # closing radius of the specimen mask
    # two-class maps (§2)
    flatten_sigma_mm: float = 10.0            # Gaussian sigma of the local foreground mean used for flattening
    sigmoid_std: float = 0.25                 # p = sigmoid((I - t) / (sigmoid_std * std of foreground values))
    # orientation search (§3)
    n_rot: int = 8000                         # uniform proper rotations, R[0] = identity
    seed: int = 0                             # rotation set seed
    topk: int = 24                            # poses kept after non-maximum suppression, each one refined
    nms_mm: float = 3.0                       # same pose if centres closer than nms_mm ...
    nms_deg: float = 10.0                     # ... and rotations closer than nms_deg
    # affine refinement (§4)
    iters: int = 200                          # Adam iterations per pose
    lam: float = 2.0                          # L = 1 - S + lam (sum log_scale^2 + sum shear^2)
    clamp: float = 0.15                       # |log_scale|, |shear| <= clamp (absolute)
    lr_rot: float = 0.02                      # rad
    lr_t: float = 0.3                         # mm
    lr_ls: float = 0.01                       # log-scale
    lr_sh: float = 0.01                       # shear
    # fine-structure refinement (§5)
    ngf_sigmas_mm: tuple = (0.6, 0.4, 0.3)    # Gaussian sigma of the gradients, one pass each, coarse to fine
    ngf_erode_mm: float = 0.8                 # both masks eroded by this, so the outlines are left out
    ngf_iters: int = 150                      # Adam iterations per pass

    def __post_init__(self):
        ok = 0 < self.fine_mm <= self.base_mm < self.search_mm
        if not (ok and abs(self.search_mm / self.base_mm - round(self.search_mm / self.base_mm)) < 1e-6):
            raise ValueError("Params: need 0 < fine_mm <= base_mm < search_mm, search_mm an integer multiple of base_mm")
        if not (self.ngf_sigmas_mm and all(isinstance(x, float) and 0 < x < float("inf") for x in self.ngf_sigmas_mm)):
            raise ValueError("Params: ngf_sigmas_mm must be a non-empty tuple of positive finite floats")
        if not (self.ngf_erode_mm > 0 and self.ngf_iters >= 1):
            raise ValueError("Params: need ngf_erode_mm > 0 and ngf_iters >= 1")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Params from a partial dict (e.g. JSON); ints are accepted for floats; unknown keys or wrong types raise ValueError."""
        base, kw = cls(), {}
        for k, v in d.items():
            dv = getattr(base, k, None)
            kw[k] = v = float(v) if isinstance(dv, float) and type(v) is int else v
            if isinstance(dv, tuple) and isinstance(v, (list, tuple)):
                kw[k] = v = tuple(float(x) if type(x) is int else x for x in v)
            if k not in base.__dataclass_fields__ or type(v) is not type(dv):
                raise ValueError(f"Params {k}={v!r}: unknown key or not a {type(dv).__name__}")
        return cls(**kw)

    def hash(self) -> str:
        """16 hex characters of the sha256 of the sorted-key JSON of all fields (process and platform independent)."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
