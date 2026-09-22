"""Command line.

    octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--device cuda|cpu] [--params F]
    octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--affine-only] [--oct-spacing-um Z,Y,X]
    octreg qc --run OUT --oct OCT --mri MRI [--T F] [-o PREFIX] [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F]
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
    ap = argparse.ArgumentParser(prog="octreg", description="Label-free registration of an OCT block to an MRI crop: an affine and a "
                                 "small smooth deformation on top.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register", help="register OCT to MRI and write the transform, overlays, visual QC and result.json")
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
    a.add_argument("--inverse", action="store_true", help="MRI -> OCT (through the affine alone: the deformation field is not inverted)")
    a.add_argument("--affine-only", action="store_true", help="leave out the deformation field oct2mri_warp.nii.gz of the run")
    a.add_argument("--oct-spacing-um", type=_zyx, metavar="Z,Y,X",
                   help="as for register, for the file in the OCT frame (default: the run's)")
    q = sub.add_parser("qc", help="render the visual QC of a run for its transform or another one, without registering")
    q.add_argument("--run", required=True, help="output directory of octreg register")
    q.add_argument("--oct", required=True, help="the run's OCT volume")
    q.add_argument("--mri", required=True, help="the run's MRI")
    q.add_argument("--T", metavar="F", help="4x4 OCT world -> MRI world, text as T_oct2mri.txt or .npy (default: the run's)")
    q.add_argument("-o", "--out", metavar="PREFIX", help="write PREFIX.png, PREFIX_montage.png (default RUN/qc, RUN/qc_<F stem>)")
    q.add_argument("--oct-spacing-um", type=_zyx, metavar="Z,Y,X", help="as for register (default: the run's)")
    q.add_argument("--oct-mask", help="as for register (default: the run's)")
    q.add_argument("--mri-mask", help="as for register (default: the run's)")
    args = ap.parse_args(argv)
    from .params import Params
    from .register import apply, deform_label, qc, register
    try:
        if args.command == "register":
            if args.params is not None and not args.params.is_file():
                raise ValueError(f"{args.params}: no such file")
            params = Params.from_dict(json.loads(args.params.read_text())) if args.params else Params()
            res = register(args.oct, args.mri, args.out, args.oct_spacing_um, args.oct_mask, args.mri_mask, params, args.device)
            pose, s = res["pose"], res["search"]
            print(f"octreg: S {pose['S']:.4f}, polarity {pose['polarity']:+d}, search top1/top2 {s['top1']:.4f}/{s['top2'] or 0:.4f}, "
                  f"NGF {pose['NGF_start']:.4f} -> {pose['NGF']:.4f}, {deform_label(res['deform'])}, "
                  f"flags {res['flags'] or 'none'} -> {args.out}")
        elif args.command == "apply":
            print(f"octreg: wrote {apply(args.run, args.moving, args.reference, args.out, args.inverse, args.affine_only, args.oct_spacing_um)}")
        else:
            paths = qc(args.run, args.oct, args.mri, args.T, args.out, args.oct_spacing_um, args.oct_mask, args.mri_mask)
            print(f"octreg: wrote {paths[0]} and {paths[1]}")
    except ValueError as e:
        print(f"octreg: error: {e}", file=sys.stderr)
        return 2
    return 0
