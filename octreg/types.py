"""Data passed between the stages. Coordinates are world mm; every transform is a 4x4 world -> world matrix (numpy float64)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HANDS = ("proper", "mirrored")                      # mirrored: det(T[:3, :3]) < 0 between the two input world frames
POLARITIES = ("same", "inverted")                   # inverted: OCT classes swapped, u_c -> 1 - u_c, score -S
HYPOTHESES = tuple(f"{h}_{p}" for h in HANDS for p in POLARITIES)
STATUSES = ("ok", "flagged", "ambiguous")
FLAGS = ("foreground_no_valley", "oct_foreground_exceeds_mri", "overlap_floor", "clamp_saturated", "scale_unidentifiable",
         "nondefault_params")                       # written as 'name' or 'name:suffix' (modality 'oct'|'mri', or OCT axis 0|1|2)


class InputRefused(ValueError):
    """An input the method refuses (spec step 0). The message says why; no transform is written."""


@dataclass(eq=False)
class Volume:
    """One input image, opened lazily.

    reader: array-like in file array order (np.memmap or a plane reader with .shape, .dtype and slicing), never loaded whole.
    affine: 4x4 voxel index (i, j, k) -> world mm. spacing_mm: voxel edges along i, j, k (mm).
    frame: 'header' (NIfTI sform, else qform) | 'array' (world = diag(spacing), no physical orientation) | 'layout' (compat M1).
    handedness: 'physical' | 'unknown'. spacing_source: 'nifti' | 'ome' | 'sidecar' | 'cli'. path: the file. sha256: hex digest.
    """
    reader: Any
    affine: np.ndarray
    spacing_mm: tuple
    frame: str
    handedness: str
    spacing_source: str
    path: str
    sha256: str | None = None

    @property
    def shape(self) -> tuple:
        return tuple(int(s) for s in self.reader.shape)


@dataclass(eq=False)
class Hypothesis:
    """One pose of a handedness x polarity hypothesis.

    hand: 'proper' | 'mirrored'; polarity: 'same' | 'inverted'. T: 4x4 OCT world -> MRI world (mm).
    S: score 1/2 sum_c NCC_w (dimensionless, [-1, 1]). L: penalised loss 1 - S + lambda (sum ls^2 + sum sh^2); L = 1 - S for a
    search pose. log_scales, shears: refinement-model parameters along the OCT axes (dimensionless). overlap: fraction of the
    OCT weight on MRI foreground. search_top1, search_top2: best and second-best NMS-distinct search scores of this hypothesis;
    decisiveness: (s1 - s2) / (s1 - median of the per-orientation maxima). level_mm: grid on which S and L were measured.
    """
    hand: str
    polarity: str
    T: np.ndarray
    S: float
    L: float
    log_scales: tuple = (0.0, 0.0, 0.0)
    shears: tuple = (0.0, 0.0, 0.0)
    overlap: float | None = None
    search_top1: float | None = None
    search_top2: float | None = None
    decisiveness: float | None = None
    level_mm: float | None = None

    @property
    def name(self) -> str:
        return f"{self.hand}_{self.polarity}"


@dataclass(eq=False)
class Decision:
    """The winner W against one competitor C.

    name: 'polarity' | 'handedness' | 'runner_up'. value: what W chose ('same'/'inverted', 'proper'/'mirrored', or W's
    hypothesis name for runner_up). delta = L_C - L_W; sigma: paired block-jackknife standard error of delta; z = delta / sigma;
    B: number of blocks. decided iff z >= jk_z and B >= jk_min_blocks; reason says why not (or 'no competitor').
    """
    name: str
    value: Any
    delta: float | None
    sigma: float | None
    z: float | None
    B: int | None
    decided: bool
    reason: str = ""


@dataclass(eq=False)
class Result:
    """What register() returns and result.json records.

    status: 'ok' | 'flagged' | 'ambiguous' (descriptive, never a correctness claim). T_oct2mri: 4x4 of the winner W.
    hypotheses: best refined pose of each of the 4 hypotheses, in HYPOTHESES order. decisions: polarity, handedness,
    runner_up. flags: FLAGS entries that fired. info: everything else result.json reports (inputs, foreground, sectioning,
    search, scale_axes, boundary, handedness_conflicts_headers). timings: {stage: seconds}. alternatives: poses with z < jk_z
    against W, in rank order.
    """
    status: str
    T_oct2mri: np.ndarray
    hypotheses: list
    decisions: list
    flags: list
    info: dict
    timings: dict
    alternatives: list = field(default_factory=list)
