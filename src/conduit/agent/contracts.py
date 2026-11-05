"""Output contracts: the pydantic model a run's final answer must satisfy.

The runner checks the final answer with ``parse_output``. An answer that fails is retried
once, with a message naming what failed; a second failure ends the run with the typed
failure ``RunFailure.OUTPUT_INVALID``. An invalid answer is never returned.

Parsing is strict on purpose: the answer must be exactly one JSON object (optionally in a
single ``json`` fence), with no repeated keys. Errors name the field and pydantic's error
type, never the value, because the value may echo content an attacker influenced; the
retry message is built from those errors alone. A contract's model should forbid extra
fields, so an unexpected field is an error rather than silently dropped.

Carried across from VOR's ``vor.ai.output`` and generalised: the contract is the
product's (its model, name and retry wording), and conduit keeps no registry of them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from conduit.core.errors import ConfigurationError
from conduit.core.identifiers import IDENTIFIER_PATTERN

_FENCE = re.compile(r"^```(?:json)?[ \t]*\n(?P<body>.*)\n```$", re.DOTALL)
_NAME = re.compile(IDENTIFIER_PATTERN)

#: What the retry tells the model when the answer was cut off at the output limit.
CUT_OFF = "the reply was cut off at the output limit"


def default_retry_prompt(name: str, errors: tuple[str, ...]) -> str:
    return (
        f"Your reply did not match the output contract {name}: {'; '.join(errors)}. "
        "Reply again with only the JSON object."
    )


@dataclass(frozen=True, slots=True)
class OutputContract:
    """A final answer's contract. ``structured`` asks the model for structured output.

    With ``structured`` set, the model's JSON Schema is sent as the request's
    ``output_schema`` when the assigned model declares structured output; the answer is
    still checked here either way. ``retry_prompt`` builds the retry message from the
    errors; the default names the contract and the errors.
    """

    model: type[BaseModel]
    name: str
    structured: bool = False
    retry_prompt: Callable[[tuple[str, ...]], str] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not (isinstance(self.model, type) and issubclass(self.model, BaseModel)):
            raise ConfigurationError("an output contract needs a pydantic model")
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ConfigurationError("an output contract's name must be an identifier")

    def schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()

    def retry_message(self, errors: tuple[str, ...]) -> str:
        if self.retry_prompt is not None:
            return self.retry_prompt(errors)
        return default_retry_prompt(self.name, errors)


@dataclass(frozen=True, slots=True)
class ContractCheck:
    """The outcome of a check: the validated value, or the errors (never both)."""

    value: BaseModel | None
    errors: tuple[str, ...]


class _RepeatedKey(ValueError):
    pass


def _no_repeats(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _RepeatedKey
        out[key] = value
    return out


def parse_output(contract: OutputContract, text: str) -> ContractCheck:
    """``text`` checked against ``contract``. Errors name fields and types only."""
    body = text.strip()
    fenced = _FENCE.fullmatch(body)
    if fenced:
        body = fenced.group("body").strip()
    problem: str | None = None
    value: Any = None
    try:
        value = json.loads(body, object_pairs_hook=_no_repeats)
    except _RepeatedKey:
        problem = "the reply repeats a key"
    except ValueError:
        problem = "the reply is not exactly one JSON object"
    if problem is not None:
        return ContractCheck(None, (problem,))
    if not isinstance(value, dict):
        return ContractCheck(None, ("the reply is not a JSON object",))
    issues: tuple[str, ...] = ()
    try:
        return ContractCheck(contract.model.model_validate(value), ())
    except ValidationError as error:
        issues = tuple(
            f"{'.'.join(str(part) for part in issue['loc']) or '(root)'}: {issue['type']}"
            for issue in error.errors(include_input=False, include_url=False)
        )
    return ContractCheck(None, issues)
