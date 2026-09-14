"""Command line: the whole public surface.

    octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--init T.txt] [--device cuda|cpu]
    octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--nearest]
    octreg qc OUT

Hidden: `register --params FILE.json` (bench ablations only; Params.from_json, sets flag nondefault_params).
"""
from __future__ import annotations


def main(argv=None) -> int:
    """Parse argv (default sys.argv[1:]) and run the subcommand. -> exit code: 0 done, 2 input refused (message on stderr)."""
    raise NotImplementedError
