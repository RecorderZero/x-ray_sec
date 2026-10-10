#!/usr/bin/env python3
"""Refuse to overwrite tracked result files unless ``--force`` is given (AF-029).

Scripts that write into results/, paper_assets/, reports/, image/ or appendix/
call ``add_force_flag(parser)`` and then ``refuse_overwrite(paths, args.force)``
before doing any work, so an accidental invocation (including ``--help`` on a
script without argparse, as happened in AUD-20261011-01) cannot silently replace
a committed result.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


def add_force_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing tracked outputs (refused by default)",
    )


def refuse_overwrite(paths: Iterable[Path | str], force: bool) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise SystemExit(
            "refusing to overwrite existing tracked outputs (pass --force to replace): "
            + ", ".join(existing)
        )
