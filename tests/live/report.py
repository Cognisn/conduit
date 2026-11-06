"""The live run's report: what each check observed, as verdicts and numbers only.

Each check records one line: the area (``chat``, ``tools``, ``caching``, ``catalogue``
and so on), the subject (provider, route and model), the check, a verdict and a detail.
The detail is written by the suite itself (counts, limits, region names, a failure
class or an exception's class name) and never holds provider text, a key, a signature
or a response body.

Verdicts:

- ``match``: the observation agrees with what conduit or its catalogue states.
- ``mismatch``: it disagrees. A mismatch fails the test that found it.
- ``unobserved``: the provider gave nothing to compare (a model that chose not to call
  the tool, a cache the provider did not report, a limit no API exposes). Never a
  failure; the report says what could not be seen.
- ``finding``: a fact for a decision (X.AI's reasoning arithmetic, whether Ollama
  reports cached prompt tokens), neither a match nor a mismatch.

The report is printed at the end of the run (``conftest.pytest_terminal_summary``) and,
when ``CONDUIT_LIVE_REPORT`` names a file, written there as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class Verdict(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNOBSERVED = "unobserved"
    FINDING = "finding"


@dataclass(frozen=True, slots=True)
class Line:
    area: str
    subject: str
    check: str
    verdict: Verdict
    detail: str = ""


class Report:
    """Every line recorded in this run, in order."""

    def __init__(self) -> None:
        self.lines: list[Line] = []

    def record(
        self, area: str, subject: str, check: str, verdict: Verdict, detail: str = ""
    ) -> Line:
        line = Line(area, subject, check, verdict, detail)
        self.lines.append(line)
        return line

    def mismatches(self, subject: str | None = None) -> list[Line]:
        return [
            line
            for line in self.lines
            if line.verdict is Verdict.MISMATCH and (subject is None or line.subject == subject)
        ]

    def render(self) -> list[str]:
        """The report as text lines: one per check, grouped by area."""
        if not self.lines:
            return ["no live check ran (every live test was skipped or deselected)"]
        rendered: list[str] = []
        for area in sorted({line.area for line in self.lines}):
            rendered.append(f"[{area}]")
            for line in self.lines:
                if line.area == area:
                    detail = f" ({line.detail})" if line.detail else ""
                    rendered.append(
                        f"  {line.verdict.value:<10} {line.subject} :: {line.check}{detail}"
                    )
        counts = {verdict: 0 for verdict in Verdict}
        for line in self.lines:
            counts[line.verdict] += 1
        rendered.append(", ".join(f"{verdict.value} {counts[verdict]}" for verdict in Verdict))
        return rendered

    def write_json(self, path: Path) -> None:
        payload = [asdict(line) | {"verdict": line.verdict.value} for line in self.lines]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


#: The run's report. One per pytest process.
REPORT = Report()
