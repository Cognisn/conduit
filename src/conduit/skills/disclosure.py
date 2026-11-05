"""Progressive disclosure: list skills for the model, load them when it asks.

A ``SkillSession`` serves one conversation. The model first sees only the skills'
summaries (name, version and description), in the system prompt. When it calls the
``use_skill`` tool (``conduit.skills.tool``) with a skill's name, the session loads
that skill's body from the ``SkillSource``; asked again with a resource path, it
returns that resource's text as the tool result.

Cache placement (C19): ``system(base)`` gives the request's system blocks: the
product's own ``base`` blocks, then the skill listing, then each loaded skill in the
order it was loaded, then one ``CacheBreakpoint``. Loaded skills therefore sit in the
stable prefix with the system prompt and the tool definitions, and are cached with
them: loading a skill rewrites the cached prefix once, and every later call reads it.
A skill's body is never sent in the messages; the tool result only says it is loaded.
A resource, read on request, goes once into the messages as a tool result, which
leaves the prefix unchanged. Rendering is deterministic, so the same skills loaded in
the same order give byte-identical blocks (across sessions and after ``restore``), and
the listing is a snapshot taken on first use, unchanged until ``refresh()``.

Markers and events (C19): each load, use (a repeat request or a resource read) and
unload is a ``SkillUse`` transcript marker, appended to the product's transcript store
when one is given and kept in ``markers``, and a skill event carrying names, versions
and sizes only. ``restore(transcript)`` reloads a resumed session's skills in their
original order without writing new markers.

Limits are guards: at most ``max_loaded`` skills and ``max_loaded_bytes`` of bodies
at once, at most ``max_resource_bytes`` per resource read, only skills the session
lists (and the product allows), and a resource only once its skill is loaded. The
session holds a lock, so concurrent tool calls cannot pass a limit or load a skill
twice.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypedDict

from conduit.core.backends import SkillSource, SkillSummary
from conduit.core.errors import ConfigurationError
from conduit.core.events import Event, EventSink
from conduit.core.tools import ToolCallContext
from conduit.core.transcript import SkillUse, Transcript, TranscriptStore
from conduit.core.types import CacheBreakpoint, SystemBlock, TextBlock
from conduit.skills.errors import (
    SkillError,
    SkillInvalid,
    SkillNotFound,
    SkillRefused,
    SkillResourceRefused,
)
from conduit.skills.events import (
    SkillLoadedEvent,
    SkillRefusedEvent,
    SkillUnloadedEvent,
    SkillUsedEvent,
)
from conduit.skills.format import is_skill_name, is_skill_version, resource_path_refusal

if TYPE_CHECKING:
    from conduit.skills.tool import UseSkillTool

__all__ = [
    "DisclosureLimits",
    "LoadedSkill",
    "SkillSession",
    "render_listing",
    "render_skill",
]

logger = logging.getLogger("conduit.skills")

Action = Literal["loaded", "used", "unloaded"]


class _Ids(TypedDict):
    run_id: str | None
    correlation_id: str | None
    task_class: str | None


@dataclass(frozen=True, slots=True)
class DisclosureLimits:
    """What one session discloses. Each limit is inclusive."""

    max_loaded: int = 8
    max_loaded_bytes: int = 256 * 1024
    max_resource_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for name in self.__slots__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError(f"disclosure limit {name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """A skill loaded into a session."""

    name: str
    version: str | None
    body: str
    resources: Mapping[str, str]

    @property
    def body_size(self) -> int:
        """The body's size in UTF-8 bytes."""
        return len(self.body.encode("utf-8"))

    @property
    def resource_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.resources))


def render_listing(summaries: Sequence[SkillSummary]) -> str:
    """The default skill listing for the system prompt."""
    lines = [
        "# Skills",
        "",
        "These skills are available. Before a task one of them covers, call use_skill "
        "with the skill's name to load its instructions; they then appear in this system "
        "prompt. Read a loaded skill's resources with use_skill, giving the skill's name "
        "and the resource path.",
        "",
    ]
    for summary in summaries:
        version = f" (version {summary.version})" if summary.version else ""
        lines.append(f"- {summary.name}{version}: {summary.description}")
    return "\n".join(lines)


def render_skill(loaded: LoadedSkill) -> str:
    """The default rendering of one loaded skill in the system prompt."""
    version = f' version="{loaded.version}"' if loaded.version else ""
    parts = [f'<skill name="{loaded.name}"{version}>', loaded.body.rstrip("\n")]
    if loaded.resources:
        parts += ["", "Resources: " + ", ".join(loaded.resource_paths)]
    parts.append("</skill>")
    return "\n".join(parts)


class SkillSession:
    """One conversation's skills: listing, loading, resources, markers and events."""

    def __init__(
        self,
        source: SkillSource,
        *,
        limits: DisclosureLimits | None = None,
        allowed: Iterable[str] | None = None,
        events: EventSink | None = None,
        transcript: TranscriptStore | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        correlation_id: str | None = None,
        task_class: str | None = None,
        render_listing: Callable[[Sequence[SkillSummary]], str] = render_listing,
        render_skill: Callable[[LoadedSkill], str] = render_skill,
    ) -> None:
        if transcript is not None and not session_id:
            raise ConfigurationError("a skill session writing to a transcript needs a session id")
        self._source = source
        self.limits = limits or DisclosureLimits()
        self._allowed = frozenset(allowed) if allowed is not None else None
        self._events = events
        self._transcript = transcript
        self._session_id = session_id
        self._run_id = run_id
        self._correlation_id = correlation_id
        self._task_class = task_class
        self._render_listing = render_listing
        self._render_skill = render_skill
        self._lock = asyncio.Lock()
        self._summaries: tuple[SkillSummary, ...] | None = None
        self._loaded: dict[str, LoadedSkill] = {}
        self._markers: list[SkillUse] = []
        self._started = False

    # State.

    @property
    def loaded(self) -> tuple[LoadedSkill, ...]:
        """The loaded skills, in the order they were loaded."""
        return tuple(self._loaded.values())

    @property
    def loaded_size(self) -> int:
        """The loaded bodies' total size in UTF-8 bytes."""
        return sum(s.body_size for s in self._loaded.values())

    @property
    def markers(self) -> tuple[SkillUse, ...]:
        """Every marker this session recorded, as stored when a transcript is given."""
        return tuple(self._markers)

    def tool(self, *, task_classes: Iterable[str]) -> UseSkillTool:
        """The ``use_skill`` tool for this session, serving ``task_classes``."""
        from conduit.skills.tool import UseSkillTool

        return UseSkillTool(self, task_classes=task_classes)

    # Listing.

    async def summaries(self) -> tuple[SkillSummary, ...]:
        """The skills the model may load, sorted by name: a snapshot taken on first use."""
        if self._summaries is None:
            self._summaries = await self._fetch_summaries()
        return self._summaries

    async def refresh(self) -> tuple[SkillSummary, ...]:
        """Take a new snapshot of the source's skills. Changes the prefix if they changed."""
        async with self._lock:
            self._summaries = await self._fetch_summaries()
            return self._summaries

    async def _fetch_summaries(self) -> tuple[SkillSummary, ...]:
        kept: dict[str, SkillSummary] = {}
        left_out = 0
        for summary in await self._source.list_skills():
            valid = is_skill_name(summary.name) and (
                summary.version is None or is_skill_version(summary.version)
            )
            if not valid or summary.name in kept:
                left_out += 1
                continue
            if self._allowed is not None and summary.name not in self._allowed:
                continue
            kept[summary.name] = summary
        if left_out:
            logger.warning(
                "%d skill summaries left out: invalid or repeated names or versions", left_out
            )
        return tuple(kept[name] for name in sorted(kept))

    async def _listed(self) -> frozenset[str]:
        return frozenset(s.name for s in await self.summaries())

    # The request prefix.

    async def system(self, base: Sequence[SystemBlock] = ()) -> tuple[SystemBlock, ...]:
        """``base``, the listing and the loaded skills, closed by one cache breakpoint."""
        summaries = await self.summaries()
        blocks: list[SystemBlock] = list(base)
        added = False
        if summaries:
            blocks.append(TextBlock(text=self._render_listing(summaries)))
            added = True
        for loaded in self._loaded.values():
            blocks.append(TextBlock(text=self._render_skill(loaded)))
            added = True
        if blocks and (added or not isinstance(blocks[-1], CacheBreakpoint)):
            blocks.append(CacheBreakpoint())
        return tuple(blocks)

    # Requests.

    async def load(self, name: str, *, context: ToolCallContext | None = None) -> LoadedSkill:
        """Load ``name`` into the session, or record a use if it is loaded already."""
        async with self._lock:
            loaded = self._loaded.get(name)
            if loaded is not None:
                await self._mark(loaded.name, loaded.version, "used")
                await self._emit(
                    SkillUsedEvent(skill=loaded.name, version=loaded.version, **self._ids(context))
                )
                return loaded
            await self._require_listed(name, context)
            if len(self._loaded) >= self.limits.max_loaded:
                await self._refuse("too_many_loaded", name, context)
                raise SkillRefused("too_many_loaded", skill=name)
            loaded = await self._acquire(name, context)
            await self._mark(loaded.name, loaded.version, "loaded")
            self._loaded[loaded.name] = loaded
            self._started = True
            await self._emit_loaded(loaded, context)
            return loaded

    async def read_resource(
        self, name: str, path: str, *, context: ToolCallContext | None = None
    ) -> str:
        """The text of one resource of a loaded skill."""
        async with self._lock:
            loaded = self._loaded.get(name)
            if loaded is None:
                await self._refuse("skill_not_loaded", name, context)
                listed = name in await self._listed()
                raise SkillRefused("skill_not_loaded", skill=name if listed else None)
            text, size, reason = self._resource(loaded, path)
            if reason is not None:
                await self._refuse(reason, name, context)
                raise SkillResourceRefused(reason, skill=name)
            await self._mark(loaded.name, loaded.version, "used")
            await self._emit(
                SkillUsedEvent(
                    skill=loaded.name,
                    version=loaded.version,
                    resource=path,
                    resource_size=size,
                    **self._ids(context),
                )
            )
            return text

    async def unload(self, name: str, *, context: ToolCallContext | None = None) -> None:
        """Take a loaded skill out of the session (and the prefix)."""
        async with self._lock:
            loaded = self._loaded.get(name)
            if loaded is None:
                listed = name in await self._listed()
                raise SkillRefused("skill_not_loaded", skill=name if listed else None)
            await self._mark(loaded.name, loaded.version, "unloaded")
            del self._loaded[name]
            await self._emit(
                SkillUnloadedEvent(
                    skill=loaded.name,
                    version=loaded.version,
                    body_size=loaded.body_size,
                    loaded_count=len(self._loaded),
                    loaded_size=self.loaded_size,
                    **self._ids(context),
                )
            )

    async def restore(self, transcript: Transcript) -> None:
        """Reload a resumed session's skills, in the order they were first loaded.

        Writes no marker for a skill restored; a skill that can no longer be loaded is
        left out and marked unloaded. Only a session that has not started may restore.
        """
        async with self._lock:
            if self._started or self._loaded or self._markers:
                raise SkillRefused("already_started")
            order: dict[str, str | None] = {}
            for entry in transcript.entries:
                if isinstance(entry, SkillUse):
                    if entry.action == "unloaded":
                        order.pop(entry.skill, None)
                    elif entry.skill not in order:
                        order[entry.skill] = entry.version
            for name, version in order.items():
                loaded: LoadedSkill | None = None
                try:
                    if len(self._loaded) < self.limits.max_loaded:
                        loaded = await self._acquire(name, None)
                except SkillError:
                    loaded = None
                if loaded is None:
                    await self._mark(name, version, "unloaded")
                    continue
                self._loaded[name] = loaded
                await self._emit_loaded(loaded, None)
            self._started = True

    # Internals (called with the lock held).

    async def _acquire(self, name: str, context: ToolCallContext | None) -> LoadedSkill:
        """Load ``name`` from the source after every check; nothing is committed."""
        await self._require_listed(name, context)
        missing = False
        try:
            skill = await self._source.load_skill(name)
        except SkillError:
            raise
        except LookupError:
            missing = True
        if missing:
            await self._refuse("skill_unknown", name, context)
            raise SkillNotFound(name)
        if skill.summary.name != name:
            await self._refuse("skill_invalid", name, context)
            raise SkillInvalid("name_mismatch", skill=name)
        version = skill.summary.version
        if version is not None and not is_skill_version(version):
            await self._refuse("skill_invalid", name, context)
            raise SkillInvalid("version_invalid", skill=name)
        loaded = LoadedSkill(
            name=name,
            version=version,
            body=skill.body,
            resources=MappingProxyType(dict(skill.resources)),
        )
        if self.loaded_size + loaded.body_size > self.limits.max_loaded_bytes:
            await self._refuse("loaded_too_large", name, context)
            raise SkillRefused("loaded_too_large", skill=name)
        return loaded

    def _resource(self, loaded: LoadedSkill, path: str) -> tuple[str, int, str | None]:
        """The resource's text and size, or the reason it is refused."""
        reason = resource_path_refusal(path)
        if reason is not None:
            return "", 0, reason
        text = loaded.resources.get(path)
        if text is None:
            return "", 0, "resource_missing"
        size = len(text.encode("utf-8"))
        if size > self.limits.max_resource_bytes:
            return "", 0, "resource_too_large"
        return text, size, None

    async def _require_listed(self, name: str, context: ToolCallContext | None) -> None:
        if not is_skill_name(name) or name not in await self._listed():
            await self._refuse("skill_unknown", None, context)
            raise SkillNotFound()

    async def _mark(self, skill: str, version: str | None, action: Action) -> None:
        marker = SkillUse(skill=skill, version=version, action=action)
        if self._transcript is not None and self._session_id is not None:
            stored = await self._transcript.append(self._session_id, [marker])
            first = stored[0] if stored else marker
            marker = first if isinstance(first, SkillUse) else marker
        self._markers.append(marker)

    def _ids(self, context: ToolCallContext | None) -> _Ids:
        return {
            "run_id": context.run_id if context is not None else self._run_id,
            "correlation_id": (context.correlation_id if context is not None else None)
            or self._correlation_id,
            "task_class": context.task_class if context is not None else self._task_class,
        }

    async def _emit(self, event: Event) -> None:
        if self._events is not None:
            await self._events.emit(event)

    async def _emit_loaded(self, loaded: LoadedSkill, context: ToolCallContext | None) -> None:
        await self._emit(
            SkillLoadedEvent(
                skill=loaded.name,
                version=loaded.version,
                body_size=loaded.body_size,
                resource_count=len(loaded.resources),
                loaded_count=len(self._loaded),
                loaded_size=self.loaded_size,
                **self._ids(context),
            )
        )

    async def _refuse(self, reason: str, name: str | None, context: ToolCallContext | None) -> None:
        listed = name is not None and is_skill_name(name) and name in await self._listed()
        await self._emit(
            SkillRefusedEvent(skill=name if listed else None, reason=reason, **self._ids(context))
        )
