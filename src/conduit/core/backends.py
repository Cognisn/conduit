"""Protocols for the backends products supply: token store, skill source, approvals.

All are async (C11). The transcript store is in ``conduit.core.transcript``, the event
sink in ``conduit.core.events`` and the secret resolver in ``conduit.core.secrets``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class TokenSet(_Model):
    """OAuth tokens for one MCP server (C14). Tokens are hidden in repr and dumps."""

    access_token: SecretStr
    refresh_token: SecretStr | None = None
    token_type: str = "Bearer"  # noqa: S105 - the OAuth token type, not a secret
    expires_at: datetime | None = None
    scope: str | None = None

    def to_storage(self) -> dict[str, Any]:
        """A JSON-ready mapping with the token values revealed, for a store to persist."""
        data = self.model_dump(mode="json")
        data["access_token"] = self.access_token.get_secret_value()
        data["refresh_token"] = (
            self.refresh_token.get_secret_value() if self.refresh_token is not None else None
        )
        return data

    @classmethod
    def from_storage(cls, data: Mapping[str, Any]) -> TokenSet:
        return cls.model_validate(dict(data))


@runtime_checkable
class TokenStore(Protocol):
    """Holds MCP OAuth tokens by key (for example ``mcp:<server>``). Supplied by the product."""

    async def get(self, key: str) -> TokenSet | None: ...

    async def put(self, key: str, tokens: TokenSet) -> None: ...

    async def delete(self, key: str) -> None: ...


class SkillSummary(_Model):
    """What progressive disclosure shows the model before a skill is loaded."""

    name: str = Field(min_length=1, max_length=128)
    description: str
    version: str | None = None


class Skill(_Model):
    summary: SkillSummary
    body: str  # the SKILL.md body
    resources: Mapping[str, str] = Field(default_factory=dict)  # name to text


@runtime_checkable
class SkillSource(Protocol):
    """Where skills come from (a folder, a database). ``load_skill`` raises ``LookupError``."""

    async def list_skills(self) -> Sequence[SkillSummary]: ...

    async def load_skill(self, name: str) -> Skill: ...


class ApprovalRequest(_Model):
    """A tool call that needs a person's approval before it runs."""

    request_id: str = Field(min_length=1, max_length=256)
    run_id: str | None = None
    task_class: str | None = None
    tool: str
    arguments: dict[str, Any]
    reason: str = ""


class ApprovalDecision(_Model):
    approved: bool
    reason: str = ""
    decided_by: str | None = None


@runtime_checkable
class ApprovalHandler(Protocol):
    """Decides approval requests (a prompt in a terminal, a queue in a web UI)."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...
