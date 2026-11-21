"""Untrusted-content framing, as the agent runner uses it: the core's.

The runner frames every tool result (but those a tool declares it frames itself) and
every ``UntrustedText`` in an ``AgentTask`` with ``conduit.core.untrusted``, the one
sanitiser and frame every layer shares. The names are re-exported here and from
``conduit.agent`` so a product reaches them where it reaches the runner.
"""

from __future__ import annotations

from conduit.core.untrusted import (
    BASE64_RUN_THRESHOLD,
    DEFAULT_MAX_LENGTH,
    ESCAPED_DELIMITER,
    FRAMING_NOTE,
    UntrustedBlock,
    UntrustedText,
    frame_untrusted,
    sanitise_untrusted,
    untrusted_text_block,
)

__all__ = [
    "BASE64_RUN_THRESHOLD",
    "DEFAULT_MAX_LENGTH",
    "ESCAPED_DELIMITER",
    "FRAMING_NOTE",
    "UntrustedBlock",
    "UntrustedText",
    "frame_untrusted",
    "sanitise_untrusted",
    "untrusted_text_block",
]
