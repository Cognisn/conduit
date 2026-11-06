"""``python -m conduit.cli``: the same as the ``conduit`` command."""

from conduit.cli.main import main

if __name__ == "__main__":  # imported (by the tripwires' module walk), it does nothing
    raise SystemExit(main())
