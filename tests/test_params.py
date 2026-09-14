"""Params: hash stable across processes, diff_from_default, JSON round trip, complete metadata, compat switches."""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from octreg.params import DOFS, STAGES, SWITCHES, Compat, Params, _flat

ROOT = Path(__file__).resolve().parents[1]


def _in_subprocess(expr, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    code = f"import dataclasses; from octreg.params import Params, Compat; print({expr})"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def test_hash_stable_across_processes():
    p = Params()
    q = dataclasses.replace(p, search_rho=0.85, refine_lambda=1.0, compat=Compat.v1("M8"))
    for seed in (0, 1, 4242):
        assert _in_subprocess("Params().hash()", seed) == p.hash()
        assert _in_subprocess("Params().hash(stages=('prep',))", seed) == p.hash(stages=("prep",))
    expr = "dataclasses.replace(Params(), search_rho=0.85, refine_lambda=1.0, compat=Compat.v1('M8')).hash()"
    assert _in_subprocess(expr, 7) == q.hash() != p.hash()


def test_hash_by_stage():
    p = Params()
    assert len(p.hash()) == 16 and p.hash() == Params().hash()
    search_only = dataclasses.replace(p, search_topk=24)
    assert search_only.hash() != p.hash() and search_only.hash(stages=("prep",)) == p.hash(stages=("prep",))
    prep = dataclasses.replace(p, flat_sigma_mm=5.0)
    assert prep.hash(stages=("prep",)) != p.hash(stages=("prep",)) and prep.hash(stages=("search",)) == p.hash(stages=("search",))
    levels = dataclasses.replace(p, compat=Compat.v1("M6"))
    assert levels.hash(stages=("prep",)) != p.hash(stages=("prep",))
    with pytest.raises(ValueError):
        p.hash(stages=("nope",))


def test_diff_from_default():
    assert Params().diff_from_default() == {}
    q = dataclasses.replace(Params(), search_rho=0.85, section_period_mm=(0.2, 1.0), compat=Compat.v1("M2", "M7"))
    assert q.diff_from_default() == {
        "search_rho": {"default": 0.8, "value": 0.85},
        "section_period_mm": {"default": (0.1, 2.0), "value": (0.2, 1.0)},
        "compat.v1_sectioning": {"default": False, "value": True},
        "compat.v1_cleanup": {"default": False, "value": True}}
    json.dumps(q.diff_from_default())                                   # result.json-ready


def test_json_round_trip(tmp_path):
    ladder = ((4, (("rigid", 60), ("similarity", 60)), 6), (1, (("affine", 100), ("affine", 50)), 1))
    for p in (Params(), dataclasses.replace(Params(), refine_ladder=ladder, scale_offsets=(0.1,), compat=Compat.v1())):
        text = p.to_json()
        q = Params.from_json(text)
        assert q == p and q.hash() == p.hash() and q.to_json() == text
        f = tmp_path / "params.json"
        f.write_text(text)
        assert Params.from_json(f) == p and Params.from_json(str(f)) == p and Params.from_json(json.loads(text)) == p


def test_from_json_partial_and_strict():
    q = Params.from_json('{"refine_lambda": 1, "compat": {"v1_overlap": true}}')   # a diff; int accepted for a float
    assert q == dataclasses.replace(Params(), refine_lambda=1.0, compat=Compat(v1_overlap=True))
    assert q.hash() == dataclasses.replace(Params(), refine_lambda=1.0, compat=Compat(v1_overlap=True)).hash()
    for bad in ('{"rho": 0.8}', '{"compat": {"M8": true}}', '{"search_topk": 12.5}', '{"search_topk": true}',
                '{"compat": {"v1_levels": 1}}', '{"levels_h": 4}', '{"compat": true}', '{"levels_h": [4, 3]}'):
        with pytest.raises(ValueError):
            Params.from_json(bad)


def test_metadata_complete():
    names = [k for k, _, _ in _flat(Params())]
    assert len(names) == len(set(names))
    for obj in (Params, Compat):
        for f in dataclasses.fields(obj):
            m = f.metadata
            assert m["stage"] in STAGES and isinstance(m["unit"], str) and len(m["provenance"]) > 5, f.name
            assert f.name == "compat" or m["unit"], f.name


def test_compat_switches():
    assert len(SWITCHES) == 8 and sorted(SWITCHES) == [f"M{i}" for i in range(1, 9)]
    c = Compat()
    assert all(getattr(c, n) is False for n in SWITCHES.values())
    assert all(getattr(Compat.v1(), n) is True for n in SWITCHES.values())
    assert [n for n in SWITCHES.values() if getattr(Compat.v1("M3"), n)] == ["v1_oct_unflattened"]
    with pytest.raises(ValueError):
        Compat.v1("M9")


def test_structural_checks():
    p = Params()
    assert p.refine_ladder[0][0] == p.levels_h[0] and all(d in DOFS for _, steps, _ in p.refine_ladder for d, _ in steps)
    for kw in ({"levels_h": (2, 4, 1)}, {"levels_h": (4, 2)}, {"refine_ladder": ((2, (("affine", 10),), 1),)},
               {"refine_ladder": ((4, (("bogus", 10),), 1),)}):
        with pytest.raises(ValueError):
            dataclasses.replace(p, **kw)
