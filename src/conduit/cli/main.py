"""The ``conduit`` console script: checks the cli extra is installed, then runs the command.

The entry point imports nothing from the extra until it knows the extra is there, so a
``conduit`` run without it says which extra to install instead of failing on an import.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence

MISSING = (
    "The conduit command needs the cli extra: install cognisn-conduit[cli], or run "
    'uvx --from "cognisn-conduit[cli]" conduit\n'
)


def main(argv: Sequence[str] | None = None) -> int:
    if importlib.util.find_spec("rich") is None:
        sys.stderr.write(MISSING)
        return 2
    from conduit.cli.app import main as run_main

    return run_main(argv)
