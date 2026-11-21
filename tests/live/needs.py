"""What each live test needs from the environment, and the skip when it is absent.

Every live test names its needs with the ``live_needs`` marker; the autouse gate in
``tests/live/conftest.py`` checks them against the process environment before the test
body runs, and skips with a reason that names the variables, never a value. A need is
met when every variable of one of its groups is set and non-empty, so a need can offer
alternatives (Bedrock's credential chain or its access keys, for example).

Every variable starts with ``CONDUIT_LIVE_``. Credentials are read only through
``EnvironmentSecretResolver(prefix=PREFIX)`` from ``secret://<name>`` references, so a
provider SDK's own variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``) are never the
source of a key, and the settings never hold one.

``Need.check`` takes the environment as a mapping, so the skip guard
(``test_skip_reasons.py``) proves every test skips with a stated reason against an empty
environment, without running any live test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pytest

#: The prefix of every variable the live suite reads.
PREFIX = "CONDUIT_LIVE_"


class LiveEnv(Mapping[str, str]):
    """The live variables, as a mapping whose ``repr`` and ``str`` never show a value.

    pytest prints frame arguments in long tracebacks and locals under ``-l``, so the
    suite never passes ``os.environ`` or a dict of values around: every helper that reads
    a variable takes a ``LiveEnv``. Only ``CONDUIT_LIVE_`` variables are kept.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = {k: v for k, v in values.items() if k.startswith(PREFIX)}

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"LiveEnv(<{len(self._values)} variables, values redacted>)"

    __str__ = __repr__


def live_env() -> LiveEnv:
    """A redacted snapshot of the process's ``CONDUIT_LIVE_`` variables."""
    return LiveEnv(os.environ)


def var(name: str) -> str:
    """The full variable name for ``name`` (``ANTHROPIC_API_KEY`` gives the prefixed one)."""
    return PREFIX + name


def present(environ: Mapping[str, str], name: str) -> bool:
    """Whether the prefixed variable ``name`` is set and non-empty."""
    return bool(environ.get(var(name), "").strip())


def value(environ: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    """A non-secret setting's value (a region, a model id, a URL), or ``default``."""
    found = environ.get(var(name), "").strip()
    return found or default


@dataclass(frozen=True, slots=True)
class Need:
    """A named requirement: met when every variable of any one group is set."""

    label: str
    groups: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.label or not self.groups or not all(self.groups):
            raise ValueError("a need has a label and at least one non-empty group")

    @classmethod
    def all_of(cls, label: str, *names: str) -> Need:
        return cls(label, (tuple(names),))

    @classmethod
    def any_of(cls, label: str, *groups: tuple[str, ...]) -> Need:
        return cls(label, tuple(groups))

    def met(self, environ: Mapping[str, str]) -> bool:
        return any(all(present(environ, name) for name in group) for group in self.groups)

    def reason(self) -> str:
        """Why a test skips: the variables to set. Names only, never a value."""
        options = " or ".join(" and ".join(var(name) for name in group) for group in self.groups)
        return f"{self.label}: set {options}"

    def variables(self) -> frozenset[str]:
        return frozenset(var(name) for group in self.groups for name in group)


def check(needs: tuple[Need, ...], environ: Mapping[str, str]) -> None:
    """Skip, naming every unmet need's variables, unless all of ``needs`` are met."""
    if not needs:
        # A live test with no stated need could reach a live service: refuse it.
        pytest.fail("a live test must state its needs with the live_needs marker")
    unmet = [need.reason() for need in needs if not need.met(environ)]
    if unmet:
        pytest.skip("; ".join(unmet))


# Bedrock (bedrock-runtime). The region is always needed; each authentication mode has
# its own need, so a run covers the modes whose credentials are set.
BEDROCK_REGION = Need.all_of("Bedrock region", "BEDROCK_REGION")
BEDROCK_CHAIN = Need.all_of("Bedrock credential chain", "BEDROCK_CREDENTIAL_CHAIN")
BEDROCK_ACCESS_KEYS = Need.all_of(
    "Bedrock access keys", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
)
BEDROCK_LONG_TERM_KEY = Need.all_of("Bedrock long-term API key", "BEDROCK_API_KEY")
#: AWS credentials of either kind (for short-term keys, and the control-plane probes).
AWS_CREDENTIALS = Need.any_of(
    "AWS credentials",
    ("BEDROCK_CREDENTIAL_CHAIN",),
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
)
#: Any way of calling Bedrock at all.
BEDROCK_ANY_AUTH = Need.any_of(
    "Bedrock credentials",
    ("BEDROCK_CREDENTIAL_CHAIN",),
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    ("BEDROCK_API_KEY",),
)

# bedrock-mantle: short-term keys from AWS credentials, or a long-term key.
MANTLE_REGION = Need.all_of("bedrock-mantle region", "BEDROCK_MANTLE_REGION")
MANTLE_ANY_AUTH = Need.any_of(
    "bedrock-mantle credentials",
    ("BEDROCK_CREDENTIAL_CHAIN",),
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    ("BEDROCK_API_KEY",),
)

# Anthropic, direct and in Microsoft Foundry.
ANTHROPIC_DIRECT = Need.all_of("Anthropic API", "ANTHROPIC_API_KEY")
FOUNDRY_ANTHROPIC_KEY = Need.all_of(
    "Claude in Foundry with an API key", "FOUNDRY_ANTHROPIC_BASE_URL", "FOUNDRY_ANTHROPIC_API_KEY"
)
FOUNDRY_ANTHROPIC_ENTRA = Need.all_of(
    "Claude in Foundry with Entra ID", "FOUNDRY_ANTHROPIC_BASE_URL", "FOUNDRY_ENTRA_ID"
)

# The OpenAI-compatible services.
OPENAI = Need.all_of("OpenAI API", "OPENAI_API_KEY")
XAI = Need.all_of("X.AI API", "XAI_API_KEY")
FOUNDRY_OPENAI_KEY = Need.all_of(
    "Foundry OpenAI v1 with an API key",
    "FOUNDRY_OPENAI_ENDPOINT",
    "FOUNDRY_OPENAI_API_KEY",
    "FOUNDRY_OPENAI_DEPLOYMENT",
)
FOUNDRY_OPENAI_ENTRA = Need.all_of(
    "Foundry OpenAI v1 with Entra ID",
    "FOUNDRY_OPENAI_ENDPOINT",
    "FOUNDRY_ENTRA_ID",
    "FOUNDRY_OPENAI_DEPLOYMENT",
)
FOUNDRY_GROK = Need.all_of("a Grok deployment in Foundry", "FOUNDRY_GROK_DEPLOYMENT")
FOUNDRY_EMBEDDING = Need.all_of("a Foundry embedding deployment", "FOUNDRY_EMBEDDING_DEPLOYMENT")
FOUNDRY_ANY_AUTH = Need.any_of(
    "Foundry credentials", ("FOUNDRY_OPENAI_API_KEY",), ("FOUNDRY_ENTRA_ID",)
)
FOUNDRY_ENDPOINT = Need.all_of("the Foundry OpenAI v1 endpoint", "FOUNDRY_OPENAI_ENDPOINT")

# Gemini, Ollama and local embeddings.
GEMINI = Need.all_of("Gemini API", "GEMINI_API_KEY")
OLLAMA = Need.all_of("a local Ollama server", "OLLAMA_URL")
LOCAL_EMBEDDINGS = Need.all_of(
    "local embeddings", "LOCAL_EMBEDDING_MODEL", "LOCAL_EMBEDDING_DIMENSIONS"
)

#: The default catalogue verification pass (Q0002) is opt-in on top of credentials.
CATALOGUE_PASS = Need.all_of("the catalogue verification pass", "CATALOGUE")

#: Every need the suite defines, for the runbook and the skip guard.
ALL_NEEDS: tuple[Need, ...] = (
    BEDROCK_REGION,
    BEDROCK_CHAIN,
    BEDROCK_ACCESS_KEYS,
    BEDROCK_LONG_TERM_KEY,
    AWS_CREDENTIALS,
    BEDROCK_ANY_AUTH,
    MANTLE_REGION,
    MANTLE_ANY_AUTH,
    ANTHROPIC_DIRECT,
    FOUNDRY_ANTHROPIC_KEY,
    FOUNDRY_ANTHROPIC_ENTRA,
    OPENAI,
    XAI,
    FOUNDRY_OPENAI_KEY,
    FOUNDRY_OPENAI_ENTRA,
    FOUNDRY_GROK,
    FOUNDRY_EMBEDDING,
    FOUNDRY_ANY_AUTH,
    FOUNDRY_ENDPOINT,
    GEMINI,
    OLLAMA,
    LOCAL_EMBEDDINGS,
    CATALOGUE_PASS,
)
