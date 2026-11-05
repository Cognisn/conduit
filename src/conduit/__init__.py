"""conduit: one AI integration layer for Cognisn products."""

from importlib.resources import files

from conduit.core.backends import ApprovalHandler, SkillSource, TokenStore
from conduit.core.catalogue import (
    Catalogue,
    CatalogueOverlay,
    ModelEntry,
    ModelKey,
    default_catalogue,
)
from conduit.core.errors import (
    AssignmentRefused,
    CapabilityRefused,
    ConduitError,
    ConfigurationError,
    ModelNotAssigned,
    ProviderError,
    ProviderFailure,
    ResidencyRefused,
    SettingsRefused,
    UndeclaredModel,
)
from conduit.core.events import (
    CaptureAs,
    CapturePolicy,
    Event,
    EventSink,
    FanOutSink,
    ToolContent,
)
from conduit.core.facade import (
    SyncFacade,
    SyncProvider,
    SyncRuntime,
    SyncTaskBinding,
    start_sync,
    sync,
)
from conduit.core.provider import BaseProvider, Provider
from conduit.core.registry import MeteredProvider, ProviderFactory, ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings, Runtime, TaskBinding, start
from conduit.core.secrets import (
    EnvironmentSecretResolver,
    LiteralSecretResolver,
    SecretRef,
    SecretResolver,
)
from conduit.core.settings import AssignmentSettings, ProviderSettings
from conduit.core.streaming import (
    ProviderStateReported,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    UsageReported,
    collect,
)
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.tools import Tool, ToolCatalogue, ToolRegistry
from conduit.core.transcript import Transcript, TranscriptStore
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderStateBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

__all__ = [
    "ApprovalHandler",
    "AssignmentRefused",
    "AssignmentSettings",
    "BaseProvider",
    "CacheBreakpoint",
    "CapabilityFloor",
    "CapabilityRefused",
    "CaptureAs",
    "CapturePolicy",
    "Catalogue",
    "CatalogueOverlay",
    "ConduitError",
    "ConduitSettings",
    "ConfigurationError",
    "EnvironmentSecretResolver",
    "Event",
    "EventSink",
    "FanOutSink",
    "ImageBlock",
    "LiteralSecretResolver",
    "Message",
    "MeteredProvider",
    "ModelEntry",
    "ModelKey",
    "ModelNotAssigned",
    "ModelRequest",
    "ModelResponse",
    "Provider",
    "ProviderError",
    "ProviderFactory",
    "ProviderFailure",
    "ProviderRegistry",
    "ProviderSettings",
    "ProviderStateBlock",
    "ProviderStateReported",
    "ResidencyPolicy",
    "ResidencyRefused",
    "ResidencyScope",
    "Runtime",
    "SecretRef",
    "SecretResolver",
    "SettingsRefused",
    "SkillSource",
    "StopReason",
    "StreamStopped",
    "SyncFacade",
    "SyncProvider",
    "SyncRuntime",
    "SyncTaskBinding",
    "TaskBinding",
    "TaskClass",
    "TextBlock",
    "TextDelta",
    "TokenStore",
    "Tool",
    "ToolCallFinished",
    "ToolCallStarted",
    "ToolCatalogue",
    "ToolContent",
    "ToolRegistry",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Transcript",
    "TranscriptStore",
    "UndeclaredModel",
    "Usage",
    "UsageReported",
    "__version__",
    "collect",
    "default_catalogue",
    "start",
    "start_sync",
    "sync",
]

#: The package version, read from ``_version.txt`` (written only by scripts/stamp_version.py).
__version__: str = files(__name__).joinpath("_version.txt").read_text(encoding="utf-8").strip()
