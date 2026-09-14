"""Command line.

    octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--device cuda|cpu] [--params F]
    octreg apply --run OUT --moving X --reference Y -o Z [--inverse]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _zyx(text):
    try:
        v = tuple(float(x) for x in text.split(","))
    except ValueError:
        v = ()
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"expected Z,Y,X in um, got {text!r}")
    return v


def main(argv=None) -> int:
    """Run a subcommand (argv default sys.argv[1:]). -> exit code: 0 done, 2 input refused (ValueError, message on stderr)."""
    ap = argparse.ArgumentParser(prog="octreg", description="Label-free affine registration of an OCT block to an MRI crop.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register", help="register OCT to MRI and write the transform, overlays, qc.png and result.json")
    r.add_argument("oct", help="OCT volume: NIfTI, or TIFF / OME-TIFF / NPY with spacing")
    r.add_argument("mri", help="MRI NIfTI, already cropped to a region containing the block")
    r.add_argument("-o", "--out", required=True, help="output directory")
    r.add_argument("--oct-spacing-um", type=_zyx, metavar="Z,Y,X",
                   help="OCT (and OCT mask) voxel spacing (um) along the numpy axes of a TIFF / NPY (NIfTI axes k, j, i)")
    r.add_argument("--oct-mask", help="OCT specimen mask file (voxels > 0, in the OCT world) instead of the texture mask")
    r.add_argument("--mri-mask", help="MRI foreground mask file (voxels > 0) instead of the histogram foreground")
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    r.add_argument("--params", type=Path, metavar="F", help="JSON file with Params overrides, e.g. the params block of a result.json "
                   "(result.json flags nondefault_params)")
    a = sub.add_parser("apply", help="resample a volume through a registration")
    a.add_argument("--run", required=True, help="output directory of octreg register")
    a.add_argument("--moving", required=True, help="volume in the OCT frame (MRI frame with --inverse)")
    a.add_argument("--reference", required=True, help="volume whose grid the output takes")
    a.add_argument("-o", "--out", required=True, help="output NIfTI")
    a.add_argument("--inverse", action="store_true", help="MRI -> OCT")
    args = ap.parse_args(argv)
    from .params import Params
    from .register import apply, register
    try:
        if args.command == "register":
            params = Params.from_dict(json.loads(args.params.read_text())) if args.params else Params()
            res = register(args.oct, args.mri, args.out, args.oct_spacing_um, args.oct_mask, args.mri_mask, params, args.device)
            pose, s = res["pose"], res["search"]
            print(f"octreg: S {pose['S']:.4f}, polarity {pose['polarity']:+d}, search top1/top2 {s['top1']:.4f}/{s['top2'] or 0:.4f}, "
                  f"flags {res['flags'] or 'none'} -> {args.out}")
        else:
            print(f"octreg: wrote {apply(args.run, args.moving, args.reference, args.out, args.inverse)}")
    except ValueError as e:
        print(f"octreg: error: {e}", file=sys.stderr)
        return 2
    return 0
