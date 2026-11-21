"""Recorded-response replay, so provider tests never need a network.

A ``Recording`` is a named list of interactions: a request (as the adapter sends it to
its SDK) and the response it got (a payload, or a list of stream chunks). ``Replay``
plays them back in order and raises ``ReplayMismatch`` when a request differs from the
recorded one, so a test fails on a changed request rather than passing on stale data.
Provider adapter tests give their SDK a fake client that answers from ``Replay.next``.

Recordings are committed to a public repository, so ``save_recording`` and
``load_recording`` refuse anything that looks unscrubbed: cloud access keys and API
keys, bearer tokens, email addresses outside example.com, example.org and example.net,
IPv4 addresses outside the documentation ranges and loopback, IPv6 addresses outside
2001:db8::/32 and loopback, and any 12-digit number (a possible AWS account id, in an
ARN or on its own) other than the documentation account 123456789012.

``ReplayProvider`` replays conduit stream events, for upper-layer tests recorded from a
real provider.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from conduit.core.errors import ConfigurationError, ReplayMismatch
from conduit.core.provider import BaseProvider
from conduit.core.settings import failed_fields
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: Any
    response: Any


class Recording(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=64)
    interactions: tuple[Interaction, ...]


_SECRETS = [
    ("cloud access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("bearer token", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
]
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
# Candidate IPv6 addresses: two to seven colons between hex groups, not inside a longer
# token. Each candidate is confirmed by ipaddress, so times such as 12:30:45 pass.
_IPV6 = re.compile(
    r"(?<![0-9A-Za-z:.])((?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4})(?![0-9A-Za-z:])"
)
# Any bare 12-digit number is a possible AWS account id (in an ARN or on its own).
_ACCOUNT = re.compile(r"(?<![\w.])(\d{12})(?![\w.])")
_SYNTHETIC_DOMAINS = ("example.com", "example.org", "example.net")
_DOCUMENTATION_NETWORKS = [
    ipaddress.ip_network(n)
    for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "127.0.0.0/8", "0.0.0.0/32")
]
_DOCUMENTATION_NETWORKS_V6 = [
    ipaddress.ip_network(n) for n in ("2001:db8::/32", "::1/128", "::/128")
]
_DOCUMENTATION_ACCOUNT = "123456789012"


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def _problems(text: str) -> Iterator[str]:
    for label, pattern in _SECRETS:
        if pattern.search(text):
            yield label
    for domain in _EMAIL.findall(text):
        if not domain.lower().endswith(_SYNTHETIC_DOMAINS):
            yield "email address outside the example domains"
    for found in _IPV4.findall(text):
        try:
            address = ipaddress.ip_address(found)
        except ValueError:
            continue
        if not any(address in network for network in _DOCUMENTATION_NETWORKS):
            yield "IPv4 address outside the documentation ranges"
    for found in _IPV6.findall(text):
        try:
            address6 = ipaddress.IPv6Address(found)
        except ValueError:
            continue
        if not any(address6 in network for network in _DOCUMENTATION_NETWORKS_V6):
            yield "IPv6 address outside 2001:db8::/32"
    for account in _ACCOUNT.findall(text):
        if account != _DOCUMENTATION_ACCOUNT:
            yield "AWS account number other than 123456789012"


def check_scrubbed(recording: Recording) -> None:
    """Raise ``ConfigurationError`` naming what looks unscrubbed (never the value)."""
    found = sorted(
        {
            problem
            for interaction in recording.interactions
            for text in _strings(interaction.model_dump(mode="json"))
            for problem in _problems(text)
        }
    )
    if found:
        raise ConfigurationError(
            f"recording {recording.name!r} is not scrubbed: {'; '.join(found)}"
        )


def save_recording(recording: Recording, path: str | Path) -> None:
    check_scrubbed(recording)
    text = json.dumps(recording.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def load_recording(path: str | Path) -> Recording:
    fields: list[str] | None = None
    try:
        recording = Recording.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValidationError as error:
        fields = failed_fields(error)
    if fields is not None:
        raise ConfigurationError(
            f"recording {path} cannot be loaded: invalid at {', '.join(fields)}"
        )
    check_scrubbed(recording)
    return recording


def _normal(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        found: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                found.append(f"{path}{key}")
            else:
                found.extend(_differences(expected[key], actual[key], f"{path}{key}."))
        return found
    return [] if expected == actual else [path.rstrip(".") or "<request>"]


class Replay:
    """Plays a recording's interactions in order."""

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self._next = 0

    def next(self, request: Any) -> Any:
        """The recorded response, if ``request`` matches the next recorded request."""
        if self._next >= len(self.recording.interactions):
            raise ReplayMismatch(f"recording {self.recording.name!r} has no more interactions")
        interaction = self.recording.interactions[self._next]
        expected, actual = _normal(interaction.request), _normal(request)
        if expected != actual:
            fields = ", ".join(_differences(expected, actual))
            raise ReplayMismatch(
                f"recording {self.recording.name!r} interaction {self._next}: request differs "
                f"at {fields}"
            )
        self._next += 1
        return interaction.response

    def assert_exhausted(self) -> None:
        left = len(self.recording.interactions) - self._next
        if left:
            raise ReplayMismatch(f"recording {self.recording.name!r} has {left} unplayed")


_EVENTS: TypeAdapter[list[StreamEvent]] = TypeAdapter(list[StreamEvent])


def _request_key(request: ModelRequest) -> Any:
    return request.model_dump(mode="json", exclude={"run_id", "correlation_id", "task_class"})


class ReplayProvider(BaseProvider):
    """A provider that replays recorded conduit stream events."""

    kind = "replay"

    def __init__(self, recording: Recording) -> None:
        self.replay = Replay(recording)

    @staticmethod
    def record(request: ModelRequest, events: Iterable[StreamEvent], *, name: str) -> Recording:
        return ReplayProvider.record_many([(request, list(events))], name=name)

    @staticmethod
    def record_many(
        calls: Sequence[tuple[ModelRequest, Sequence[StreamEvent]]], *, name: str
    ) -> Recording:
        return Recording(
            name=name,
            provider="conduit",
            interactions=tuple(
                Interaction(
                    request=_request_key(request),
                    response=_EVENTS.dump_python(list(events), mode="json"),
                )
                for request, events in calls
            ),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        for event in _EVENTS.validate_python(self.replay.next(_request_key(request))):
            yield event
