"""conduit: one AI integration layer for Cognisn products."""

from importlib.resources import files

__all__ = ["__version__"]

#: The package version, read from ``_version.txt`` (written only by scripts/stamp_version.py).
__version__: str = files(__name__).joinpath("_version.txt").read_text(encoding="utf-8").strip()
