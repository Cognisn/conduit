"""Errors raised by conduit.

Provider error text can echo prompt content, so a ``ProviderError`` carries only its
failure class and a fixed message: provider text never reaches a log, an exception or
an event. Refusals raised at start-up (``AssignmentRefused`` and its subclasses) name
the task class, provider and model, which are configuration, never content.
"""

from __future__ import annotations

from enum import StrEnum


class ConduitError(Exception):
    """Base class for every error conduit raises on purpose."""


class ProviderFailure(StrEnum):
    UNAVAILABLE = "unavailable"  # connection failure, throttling, server error
    TIMEOUT = "timeout"
    REJECTED = "rejected"  # the provider refused the request (bad request, auth, model)
    REFUSED = "refused"  # the model declined to answer (safety refusal, guardrail)
    PROTOCOL = "protocol"  # the stream broke conduit's event contract


class ProviderError(ConduitError):
    """A provider call failed. Carries only the failure class, never provider text."""

    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(f"model provider call failed: {failure.value}")
        self.failure = failure


class ConfigurationError(ConduitError, ValueError):
    """Settings, a catalogue or an overlay cannot be accepted."""


class UndeclaredModel(ConfigurationError, LookupError):
    """A model is not declared in the catalogue for the provider asked for."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(f"model {model!r} is not declared in the catalogue for {provider!r}")
        self.provider = provider
        self.model = model


class AssignmentRefused(ConfigurationError):
    """A task-class assignment cannot be accepted. Raised at start-up, never mid-task."""


class CapabilityRefused(AssignmentRefused):
    """The assigned model is below the task class's capability floor."""


class ResidencyRefused(AssignmentRefused):
    """The assignment would breach the deployment's residency policy."""


class SecretUnavailable(ConduitError, LookupError):
    """A secret reference could not be resolved. The message names the reference only."""


class UnknownTool(ConduitError, LookupError):
    """A tool name is not in the task's registry."""


class ToolRegistrationError(ConfigurationError):
    """A tool was declared wrongly. Raised when the catalogue is built."""


class FacadeError(ConduitError, RuntimeError):
    """The sync facade was used where it would deadlock, or after it was closed."""


class ReplayMismatch(ConduitError, AssertionError):
    """A recorded-response replay was asked for something it did not record."""
