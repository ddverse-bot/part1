"""Every method constant of octreg, in one frozen dataclass.

Each field carries its unit, its pipeline stage and its provenance as dataclass metadata (``dataclasses.fields(Params)``);
docs/METHOD.md Table 1 is generated from it. Lengths are in mm, in units of h (the finest grid, h = max(h_min_mm, largest
MRI voxel edge)), in voxels of a named grid, or in units of a measured period. Provenance: 'spec N' = step N of
docs/design/spec_v1.0.md method_spec; 'v1 file Lnn' = the same value at tag v1.1-archive; 'a priori' = fixed before any run.

Stages: prep = steps 0-5 (cached, keyed by hash(stages=('prep',))), search = 6, refine = 7, decide = 8-9, output = files.

Params.compat holds the migration switches M1-M8 (validation_spec stage 2). All default to the release behaviour; True
reproduces the v1 behaviour of that step. Compat and its v1 constants are deleted in WP7.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("prep", "search", "refine", "decide", "output")
DOFS = ("rigid", "similarity", "affine")


def _p(default, unit, stage, why):
    assert stage in STAGES, stage
    return field(default=default, metadata={"unit": unit, "stage": stage, "provenance": why})


SWITCHES = {"M1": "v1_frames", "M2": "v1_sectioning", "M3": "v1_oct_unflattened", "M4": "v1_masked_oct_channels",
            "M5": "v1_retention", "M6": "v1_levels", "M7": "v1_cleanup", "M8": "v1_overlap"}


@dataclass(frozen=True)
class Compat:
    """Migration switches (False = release, True = v1 behaviour of that step) and the v1 constants only they use."""
    v1_frames: bool = _p(False, "switch", "prep", "M1: OCT world = v1 layout_affine(shape, spacing, v1_layout), corner origin, instead of the file header affine / array frame (v1 common.py L276, prep_subject.py L139)")
    v1_sectioning: bool = _p(False, "switch", "prep", "M2: v1 per-slice slab normalisation along array axis 0 (window v1_slab_window_um, raw threshold v1_slab_otsu_factor x Otsu, float16 gain and cap) instead of section_gain (v1 common.py L181, prep_subject.py L140-146)")
    v1_oct_unflattened: bool = _p(False, "switch", "prep", "M3: v1 OCT two-class map: no flattening, Otsu and std on all mask voxels unclipped (v1 features.py L51-65); the MRI map is unchanged")
    v1_masked_oct_channels: bool = _p(False, "switch", "prep", "M4: OCT channels multiplied by the OCT mask and one search + ladder per polarity, instead of unmasked channels and polarity from the sign of one score map (v1 features.py L63, register.py L120, L193-213)")
    v1_retention: bool = _p(False, "switch", "refine", "M5: joint top-v1_topk over both handedness, keep v1_keep per level and the v1 selection rule, instead of per-hypothesis top-K and selection by L (v1 register.py L197-213)")
    v1_levels: bool = _p(False, "switch", "prep", "M6: levels v1_levels_mm with pooling factors max(1, round(level / MRI voxel)) instead of the h rule (v1 register.py L122-128)")
    v1_cleanup: bool = _p(False, "switch", "prep", "M7: OCT mask = threshold of vol[::v1_oct_threshold_stride] + v1_oct_closing_iter closings + fill + largest component; MRI mask = raw threshold of mri[::v1_mri_threshold_stride] (v1 common.py L240-254, prep_subject.py L91-95)")
    v1_overlap: bool = _p(False, "switch", "search", "M8: tau = min(v1_overlap_cap, max(search_overlap_floor, search_rho V_M/V_O)) instead of search_rho min(1, V_M/V_O) (v1 register.py L134)")
    v1_layout: str = _p("SPR", "RAS letters", "prep", "v1 prep_subject.py L32 --oct-layout default")
    v1_slab_window_um: float = _p(1200.0, "um", "prep", "v1 prep_subject.py L144: window = max(v1_slab_window_min, round(1200 um / spacing along axis 0)) slices")
    v1_slab_window_min: int = _p(10, "slices", "prep", "v1 prep_subject.py L144")
    v1_slab_otsu_factor: float = _p(0.5, "x Otsu", "prep", "v1 prep_subject.py L143: raw threshold 0.5 x Otsu of positive finite raw[::4, ::8, ::8]")
    v1_slab_stride: tuple = _p((4, 8, 8), "voxels per array axis", "prep", "v1 prep_subject.py L143")
    v1_slab_min_voxels: int = _p(2000, "voxels", "prep", "v1 common.py L193: a slice median counts only with > 2000 voxels above the raw threshold")
    v1_f16_gain: float = _p(16384.0, "intensity", "prep", "v1 common.py L204: slab reference median stored at 16384 in float16")
    v1_f16_cap: float = _p(60000.0, "intensity", "prep", "v1 common.py L204: float16 clip after the gain")
    v1_levels_mm: tuple = _p((0.6, 0.3, 0.15), "mm", "prep", "v1 register.py L122")
    v1_oct_closing_iter: int = _p(2, "iterations", "prep", "v1 prep_subject.py L185 oct_tissue_mask(closing_iter=2), 3x3x3 cross")
    v1_oct_threshold_stride: int = _p(2, "voxels", "prep", "v1 common.py L245: threshold from vol[::2, ::2, ::2]")
    v1_mri_threshold_stride: int = _p(4, "voxels", "prep", "v1 prep_subject.py L91: threshold from positive mri[::4, ::4, ::4]")
    v1_topk: int = _p(24, "poses", "search", "v1 register.py L32 --topk, joint over both handedness, per polarity pass")
    v1_keep: tuple = _p((8, 3, 3), "poses per level", "refine", "v1 register.py L199-201 at 0.6 / 0.3 / 0.15 mm")
    v1_overlap_cap: float = _p(0.85, "fraction", "search", "v1 register.py L32 --min-overlap")

    @classmethod
    def v1(cls, *switches) -> "Compat":
        """Compat with the named switches ('M1'..'M8') at v1; no argument = all eight (regression R0)."""
        bad = [s for s in switches if s not in SWITCHES]
        if bad:
            raise ValueError(f"unknown compat switches {bad}; known {list(SWITCHES)}")
        return cls(**{SWITCHES[s]: True for s in (switches or SWITCHES)})


@dataclass(frozen=True)
class Params:
    """All method constants (see module docstring). Frozen; derive variants with dataclasses.replace."""
    # step 0-1: refusals and grids
    min_fg_voxels: int = _p(1000, "voxels at h", "prep", "spec 0: refuse below this in either image; a priori, Otsu and NCC need >= 1e3 samples")
    min_oct_span_vox: int = _p(8, "voxels at 4h", "prep", "spec 0: refuse an OCT foreground narrower than this along any axis; a priori, smallest template the 4h search resolves")
    h_min_mm: float = _p(0.15, "mm", "prep", "spec 1: h = max(h_min_mm, largest MRI voxel edge); v1 prep_subject.py L35 --target-mm")
    levels_h: tuple = _p((4, 2, 1), "h", "prep", "spec 1: pyramid 4h / 2h / h (search at levels_h[0]); 0.6 / 0.3 / 0.15 mm = v1 register.py L122 when h = 0.15")
    gauss_truncate: float = _p(3.0, "sigma", "prep", "Gaussian kernels truncated at 3 sigma; v1 synth.py L53, destripe.py L21")
    # step 2: foreground threshold (same rule for both modalities)
    fg_sample_stride: int = _p(4, "voxels per axis", "prep", "spec 2: threshold sample = every 4th voxel per axis (1/64 of the values), positive finite only; v1 prep_subject.py L91 mri[::4, ::4, ::4]")
    fg_clip_percentile: float = _p(99.5, "percentile", "prep", "spec 2; v1 common.py L223")
    fg_bins: int = _p(256, "bins", "prep", "spec 2; v1 common.py L214")
    fg_smooth_bins: int = _p(5, "bins", "prep", "spec 2: running mean of the histogram; v1 common.py L224")
    fg_prominence: float = _p(0.05, "x max count", "prep", "spec 2: peak prominence; v1 common.py L214")
    fg_peak_distance_bins: int = _p(5, "bins", "prep", "spec 2: minimum peak distance; v1 common.py L225")
    fg_valley_ratio: float = _p(0.5, "ratio", "prep", "spec 2: valley / smaller neighbouring peak must be below this; v1 common.py L214 deep=0.5")
    fg_fallback_percentile: float = _p(1.0, "percentile", "prep", "spec 2: no valley -> min(p1, 0.5 x first multi-Otsu cut), flag foreground_no_valley; v1 common.py L228")
    fg_fallback_otsu_factor: float = _p(0.5, "x Otsu cut", "prep", "spec 2; v1 common.py L228")
    fg_fallback_otsu_classes: int = _p(3, "classes", "prep", "spec 2; v1 common.py L226")
    # step 3: sectioning gain (OCT only)
    section_detect_mm: float = _p(0.04, "mm", "prep", "spec 3: detection grid pooled to >= 40 um per axis; v1.1 P3 detection level on I58 (docs/archive/v11 probe_P3)")
    section_hp_sigma_vox: float = _p(3.0, "voxels of the detection grid", "prep", "spec 3: plane high-pass = plane - Gaussian; v1 destripe.py L87")
    section_n_pairs: int = _p(40, "plane pairs", "prep", "spec 3: adjacent-plane pairs per axis; v1 destripe.py L87")
    section_skip_frac: float = _p(0.05, "fraction of the axis", "prep", "spec 3: outer 5 % of planes skipped at each end; v1 destripe.py L97")
    section_pair_min_voxels: int = _p(5000, "voxels", "prep", "a pair counts only with >= min(5000, 0.05 x plane size) voxels in both planes; v1 destripe.py L103")
    section_pair_min_frac: float = _p(0.05, "fraction of a plane", "prep", "see section_pair_min_voxels; v1 destripe.py L103")
    section_ncc_ratio: float = _p(0.8, "ratio", "prep", "spec 3: axis decisive iff lowest median NCC < 0.8 x second lowest; v1 destripe.py L87")
    section_detrend_mm: float = _p(2.4, "mm", "prep", "spec 3: running-mean detrend of the plane profile; v1 destripe.py L22")
    section_period_mm: tuple = _p((0.1, 2.0), "mm", "prep", "spec 3: FFT period search range; a priori, brackets the measured 0.30-0.39 mm sections")
    section_snr: float = _p(3.0, "x spectrum median", "prep", "spec 3: period accepted iff peak >= 3 x median power; v1 destripe.py L116")
    section_window_periods: float = _p(3.0, "periods", "prep", "spec 3: g_k = m_k / running mean over 3P; reproduces v1 windows on I46 (100 slices) and I56 (34)")
    section_min_plane_voxels: int = _p(2000, "voxels", "prep", "spec 3: planes with fewer foreground voxels get g = 1; v1 destripe.py L23, common.py L193")
    # step 4: foreground mask at h
    cleanup_closing_h: float = _p(2.0, "h", "prep", "spec 4: closing radius 2h (v1: 2 voxel iterations at 0.15 mm, common.py L248)")
    cleanup_min_component: float = _p(0.01, "fraction of foreground volume", "prep", "spec 4: keep every component >= 1 %; no largest-component rule (I57 two-piece block)")
    # step 5: two-class maps
    flat_sigma_mm: float = _p(10.0, "mm", "prep", "spec 5: G10 flattening I / max(G(I M) / G(M), floor); v1 features.py L69")
    flat_floor_factor: float = _p(0.5, "x percentile", "prep", "spec 5: floor = 0.5 x p5 of the field; v1 features.py L84")
    flat_floor_percentile: float = _p(5.0, "percentile", "prep", "spec 5; v1 features.py L84")
    flat_support_min: float = _p(0.5, "mask fraction", "prep", "p5 of the field taken where the pooled mask fraction > 0.5; v1 features.py L84")
    flat_level_h: int = _p(4, "h", "prep", "field computed on the 4h grid, trilinear back to h; v1 features.py L80 pool 4 at 0.15 mm (sigma 10 mm = 17 voxels there)")
    class_blur_h: float = _p(1.0, "h", "prep", "spec 5: blur sigma = h (v1: 1 voxel at 0.15 mm, features.py L51)")
    class_clip_percentile: float = _p(99.5, "percentile", "prep", "spec 5: Otsu of foreground values clipped at p99.5; v1 features.py L88 (MRI)")
    class_sigmoid_std: float = _p(0.25, "x std", "prep", "spec 5: p = sigmoid((I - t) / (0.25 std)); v1 features.py L60, L90")
    class_sample_stride: int = _p(3, "voxels", "prep", "Otsu and std from foreground voxels of [::3, ::3, ::3]; v1 features.py L87 (MRI)")
    # step 6: global search
    search_n_rot: int = _p(8000, "rotations", "search", "spec 6: uniform rotations, each as R and R diag(1, 1, -1); v1 register.py L32")
    search_seed: int = _p(0, "seed", "search", "spec 6: fixed rotation set, R[0] = I; v1 register.py L37")
    search_template_pad_vox: int = _p(3, "voxels at 4h", "search", "spec 6: template = bounding sphere + 3 voxels; v1 search.py L44")
    search_var_floor: float = _p(0.02, "x foreground variance", "search", "spec 6: floor of the local MRI variance; v1 search.py L59")
    search_rho: float = _p(0.8, "fraction", "search", "spec 6: tau = 0.8 min(1, V_M/V_O); v1 register.py L134 factor 0.8")
    search_overlap_floor: float = _p(0.15, "fraction", "search", "spec 6: tau < 0.15 -> 0.15 and flag overlap_floor; v1 register.py L134")
    search_topk: int = _p(12, "poses per hypothesis", "search", "spec 6: top 12 per hypothesis after NMS (v1 24 joint over both handedness)")
    search_nms_mm: float = _p(3.0, "mm", "search", "spec 6: same pose if centres < 3 mm apart AND rotations < 10 deg; v1 search.py L120")
    search_nms_deg: float = _p(10.0, "deg", "search", "spec 6; v1 search.py L120")
    prior_radius_mm: float = _p(10.0, "mm", "search", "spec 6 --init: translations within 10 mm of the prior block centre; a priori")
    prior_angle_deg: float = _p(30.0, "deg", "search", "spec 6 --init: rotations within 30 deg of the prior; a priori, the v1 restart range U(5, 30) deg")
    # step 7: refinement
    refine_lambda: float = _p(2.0, "per log-scale^2", "refine", "spec 7: L = 1 - S + lambda (sum ls^2 + sum sh^2); v1 refine.py L112, load-bearing (HANDOFF 5.3.3)")
    refine_clamp: float = _p(0.15, "log-scale / shear", "refine", "spec 7: |ls|, |sh| <= 0.15 absolute (scale 0.86-1.16); v1 refine.py L112")
    refine_lr_rot: float = _p(0.02, "rad", "refine", "spec 7: Adam learning rate, cosine schedule; v1 refine.py L110")
    refine_lr_t: float = _p(0.3, "mm", "refine", "spec 7; v1 refine.py L110")
    refine_lr_ls: float = _p(0.01, "log-scale", "refine", "spec 7; v1 refine.py L111")
    refine_lr_sh: float = _p(0.01, "shear", "refine", "spec 7; v1 refine.py L111")
    refine_mask_min: float = _p(0.05, "OCT weight", "refine", "OCT points with sampled weight > 0.05 enter the loss; v1 refine.py L85")
    refine_crop_mm: float = _p(6.0, "mm", "refine", "spec 7: MRI crop = block radius + 6 mm at every level finer than the search level; v1 register.py L156")
    refine_ladder: tuple = _p(((4, (("rigid", 120), ("similarity", 120)), 4), (2, (("rigid", 200), ("affine", 200)), 2), (1, (("affine", 200),), 2)),
                              "(level h, ((dof, iterations), ...), keep)", "refine", "spec 7 ladder; iterations and dofs as v1 register.py L199-201, keep 4/2/2 per hypothesis")
    # step 8: decisions
    distinct_mm: float = _p(2.0, "mm", "decide", "spec 8 c3: runner-up distinct iff mean mask-point displacement >= max(2 mm, 0.1 R_g); v1 register.py L220 2 mm")
    distinct_rg_frac: float = _p(0.1, "x R_g", "decide", "spec 8 c3; a priori")
    jk_block_min_mm: float = _p(2.0, "mm", "decide", "spec 8: jackknife cube edge max(2 mm, (V_fg / jk_target_blocks)^(1/3)); a priori")
    jk_target_blocks: int = _p(27, "blocks", "decide", "spec 8; a priori (A11 reports 8 / 27 / 64)")
    jk_merge_frac: float = _p(0.25, "x median block count", "decide", "spec 8: cubes holding fewer points merge into a neighbour; a priori")
    jk_z: float = _p(3.0, "z", "decide", "spec 8: decided iff z >= 3; a priori, checked on synthetic controls only (R6)")
    jk_min_blocks: int = _p(8, "blocks", "decide", "spec 8: decided needs B >= 8; a priori")
    scale_offsets: tuple = _p((0.05, 0.10), "log-scale", "decide", "spec 8: S evaluated at ls_k +- each offset; identifiable iff some paired z >= jk_z; a priori")
    scale_saturated: float = _p(0.14, "log-scale", "decide", "spec 8: flag clamp_saturated:k iff |ls_k| >= 0.14 (clamp 0.15; I38 depth exp(-0.15))")
    boundary_face_margin_h: float = _p(2.0, "h", "decide", "spec 8: boundary voxels >= 2h from the OCT FOV faces count as real outline; a priori")
    boundary_min_interior: float = _p(0.5, "fraction", "decide", "spec 8: boundary agreement reported only if >= 50 % of boundary voxels are real outline; a priori")
    # outputs
    out_margin_mm: float = _p(8.0, "mm", "output", "oct_in_mri.nii.gz box = block radius + 8 mm; v1 register.py L145")
    compat: Compat = field(default_factory=Compat, metadata={"unit": "", "stage": "prep", "provenance": "migration switches M1-M8, see Compat"})

    def __post_init__(self):
        lv = self.levels_h
        if not (lv and all(isinstance(k, int) and k >= 1 for k in lv) and list(lv) == sorted(set(lv), reverse=True) and lv[-1] == 1):
            raise ValueError(f"levels_h must be decreasing integers ending at 1, got {lv}")
        if not self.refine_ladder or self.refine_ladder[0][0] != lv[0]:
            raise ValueError("refine_ladder must start at the search level levels_h[0]")
        for level, steps, keep in self.refine_ladder:
            if level not in lv or keep < 1 or not steps or any(d not in DOFS or n < 1 for d, n in steps):
                raise ValueError(f"bad refine_ladder entry {(level, steps, keep)}")

    def to_json(self) -> str:
        """JSON text of all fields (compat nested), sorted keys."""
        return json.dumps(dataclasses.asdict(self), sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, src) -> "Params":
        """Params from JSON text, a path to a JSON file, or a dict. Missing keys keep their defaults (a bench ablation file is a
        diff); unknown keys and values of the wrong type raise ValueError. Lists become tuples; ints are accepted for floats."""
        if isinstance(src, Path) or (isinstance(src, str) and not src.lstrip().startswith("{")):
            src = Path(src).read_text()
        return _build(cls, json.loads(src) if isinstance(src, str) else src, "")

    def hash(self, stages=None) -> str:
        """16 hex characters of the sha256 of the canonical JSON of all fields (or only those of the given stages, e.g.
        ('prep',) for the steps 0-5 cache). Independent of process, platform and PYTHONHASHSEED."""
        bad = [s for s in (stages or ()) if s not in STAGES]
        if bad:
            raise ValueError(f"unknown stages {bad}; known {STAGES}")
        items = {k: v for k, v, st in _flat(self) if stages is None or st in stages}
        return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def diff_from_default(self) -> dict:
        """{dotted field name: {'default': d, 'value': v}} for every field that differs from Params(); {} = default run."""
        d = {k: v for k, v, _ in _flat(type(self)())}
        return {k: {"default": d[k], "value": v} for k, v, _ in _flat(self) if v != d[k]}


def _flat(obj, prefix=""):
    """(dotted name, value, stage) for every leaf field, nested dataclasses expanded."""
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            yield from _flat(v, prefix + f.name + ".")
        else:
            yield prefix + f.name, v, f.metadata["stage"]


def _build(cls, d, prefix):
    if not isinstance(d, dict):
        raise ValueError(f"{prefix or cls.__name__}: expected an object, got {type(d).__name__}")
    base = cls()
    bad = sorted(set(d) - {f.name for f in dataclasses.fields(cls)})
    if bad:
        raise ValueError(f"unknown {cls.__name__} keys {[prefix + k for k in bad]}")
    kw = {}
    for k, v in d.items():
        dv = getattr(base, k)
        kw[k] = _build(type(dv), v, prefix + k + ".") if dataclasses.is_dataclass(dv) else _coerce(dv, v, prefix + k)
    return cls(**kw)


def _coerce(default, v, name):
    """v converted to the type of default (element-wise for tuples, the first element as prototype beyond its length)."""
    if isinstance(default, tuple):
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"{name}: expected a list, got {v!r}")
        return tuple(_coerce(default[i] if i < len(default) else default[0], x, name) if default else x for i, x in enumerate(v))
    if isinstance(default, bool) or isinstance(v, bool):
        ok = isinstance(default, bool) and isinstance(v, bool)
    elif isinstance(default, float):
        ok = isinstance(v, (int, float))
        v = float(v) if ok else v
    else:
        ok = type(v) is type(default)
    if not ok:
        raise ValueError(f"{name}: expected {type(default).__name__}, got {v!r}")
    return v
