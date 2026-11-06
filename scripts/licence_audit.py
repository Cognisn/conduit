"""Enforce conduit's licence policy on its runtime dependencies.

    python scripts/licence_audit.py                    # the core install
    python scripts/licence_audit.py --extra bedrock    # the core plus an extra
    python scripts/licence_audit.py --all-extras       # every extra but dev

Walks the installed runtime dependency tree of ``cognisn-conduit`` (the core,
plus any extras named; environment markers evaluated for this interpreter) and
reads each distribution's licence metadata. The extras named must be installed.
The policy is warden's (W8):

- No strong copyleft (GPL, AGPL, SSPL, EUPL, OSL) in a runtime dependency,
  ever. Naming such a package as an exception does not admit it.
- Weak copyleft (LGPL, MPL, EPL, CDDL) only as a named, reviewed exception
  for a package used unmodified and installed separately.
- A package that declares no licence is refused unless it is named.

For a dual licence the most permissive alternative counts.

Every conduit layer beyond the core is an optional extra, so an extra's
dependencies are runtime dependencies for the products that install it, and
they are held to the same policy. The ``dev`` extra is never audited.

Exceptions are listed below, each with the reason it was agreed. Adding one
needs explicit agreement and a licence check, recorded in the project's
knowledge base.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = "cognisn-conduit"
NOT_AUDITED = frozenset({"dev"})

EXCEPTIONS: dict[str, str] = {
    # Agreed 2026-11-04 (Skuld Q0016): Mozilla's CA bundle, required by httpx and
    # requests (the ollama, gemini, openai and mcp extras); used unmodified and
    # installed as its own package.
    "certifi": "MPL-2.0 CA bundle required by httpx and requests; used unmodified, "
    "installed separately (Q0016)",
    # Agreed 2026-11-05 (Skuld Q0032): the progress-bar library required by
    # sentence-transformers, transformers and huggingface_hub (the local-embeddings
    # extra); conduit never shows its bars. Used unmodified and installed as its own
    # package.
    "tqdm": "MPL-2.0 AND MIT progress bars required by sentence-transformers, transformers "
    "and huggingface_hub; used unmodified, installed separately (Q0032)",
    # Agreed 2026-11-05 (Skuld Q0036): the PostgreSQL driver required by cognisn-warden
    # (the warden extra), whose audit recorder imports it. conduit never imports it; used
    # unmodified and installed as its own package.
    "psycopg": "LGPL-3.0 PostgreSQL driver required by cognisn-warden (the warden extra); "
    "used unmodified, installed separately (Q0036)",
}

_STRONG = re.compile(
    r"\b(?:A?GPL|GPLv\d|SSPL|EUPL|OSL)\b"
    r"|GNU (?:Affero )?General Public|Server Side Public|European Union Public"
    r"|Open Software Licen[cs]e",
    re.IGNORECASE,
)
_WEAK = re.compile(
    r"\b(?:LGPL|LGPLv\d|MPL|EPL|CDDL)\b"
    r"|GNU (?:Lesser|Library)|Mozilla Public|Eclipse Public"
    r"|Common Development and Distribution",
    re.IGNORECASE,
)
_RANK = {"permissive": 0, "weak copyleft": 1, "strong copyleft": 2}
_UNKNOWN = {"", "unknown", "none", "license :: osi approved"}


@dataclass(frozen=True)
class Package:
    name: str
    licences: list[str]
    requires: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)


@dataclass
class Report:
    checked: list[Package] = field(default_factory=list)
    violations: list[Package] = field(default_factory=list)
    excepted: list[Package] = field(default_factory=list)


def _one(text: str) -> str:
    if _STRONG.search(text):
        return "strong copyleft"
    if _WEAK.search(text):
        return "weak copyleft"
    return "permissive"


def classify(licences: list[str]) -> str:
    """``permissive``, ``weak copyleft``, ``strong copyleft`` or ``unknown``.

    Each licence string is judged by its most permissive ``OR`` alternative;
    the distribution by its least permissive string.
    """
    known = [text.strip() for text in licences if text.strip().lower() not in _UNKNOWN]
    if not known:
        return "unknown"
    verdicts = [
        min((_one(alt) for alt in re.split(r"\s+OR\s+", text)), key=_RANK.__getitem__)
        for text in known
    ]
    return max(verdicts, key=_RANK.__getitem__)


def installed(name: str) -> Package:
    dist = metadata.distribution(name)
    meta = dist.metadata
    licence_field = (meta.get("License") or "").strip()
    licences = [
        meta.get("License-Expression") or "",
        licence_field.splitlines()[0] if licence_field else "",
        *[c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")],
    ]
    return Package(
        name=canonicalize_name(meta["Name"]),
        licences=licences,
        requires=dist.requires or [],
        extras=list(meta.get_all("Provides-Extra") or []),
    )


def _wanted(requirement: Requirement, extras: Iterable[str]) -> bool:
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras))


def audit(
    root: str = ROOT,
    lookup: Callable[[str], Package] = installed,
    exceptions: dict[str, str] | None = None,
    extras: Iterable[str] = (),
) -> Report:
    """Audit ``root`` installed with ``extras``, and everything it requires."""
    allowed = EXCEPTIONS if exceptions is None else exceptions
    report = Report()
    seen: set[tuple[str, frozenset[str]]] = set()
    checked: dict[str, Package] = {}
    pending: list[tuple[str, frozenset[str]]] = [(root, frozenset(extras))]
    while pending:
        raw_name, wanted_extras = pending.pop()
        name = canonicalize_name(raw_name)
        if (name, wanted_extras) in seen:
            continue
        seen.add((name, wanted_extras))
        package = checked.get(name) or lookup(name)
        if name not in checked:
            checked[name] = package
            if name != canonicalize_name(root):
                verdict = classify(package.licences)
                if verdict == "strong copyleft":
                    report.violations.append(package)
                elif verdict != "permissive":
                    (report.excepted if name in allowed else report.violations).append(package)
        for spec in package.requires:
            requirement = Requirement(spec)
            if _wanted(requirement, wanted_extras):
                pending.append((requirement.name, frozenset(requirement.extras)))
    report.checked = sorted(checked.values(), key=lambda p: p.name)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--extra", action="append", default=[], help="an extra to audit")
    parser.add_argument("--all-extras", action="store_true", help="audit every extra but dev")
    args = parser.parse_args(argv)
    extras: list[str] = list(args.extra)
    if args.all_extras:
        extras += [e for e in installed(ROOT).extras if e not in NOT_AUDITED]
    refused = sorted(set(extras) & NOT_AUDITED)
    if refused:
        print(f"refusing to audit {', '.join(refused)}: not a runtime extra", file=sys.stderr)
        return 2

    report = audit(extras=extras)
    for package in report.checked:
        shown = "; ".join(text for text in package.licences if text) or "(none declared)"
        print(f"{package.name:<24} {classify(package.licences):<16} {shown}")
    for package in report.excepted:
        print(f"EXCEPTION {package.name}: {EXCEPTIONS[package.name]}")
    if report.violations:
        for package in report.violations:
            print(f"REFUSED {package.name}: {classify(package.licences)} licence", file=sys.stderr)
        return 1
    scope = f" (extras: {', '.join(sorted(set(extras)))})" if extras else ""
    print(f"licence audit passed: {len(report.checked) - 1} runtime dependencies{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
