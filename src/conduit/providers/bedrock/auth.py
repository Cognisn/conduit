"""Bedrock API keys sent as bearer tokens: short-term generation and refresh.

A short-term Bedrock API key is a SigV4 presigned URL for the ``CallWithBearerToken``
action on ``bedrock.amazonaws.com``, without its scheme, with ``&Version=1`` appended,
base64-encoded and prefixed ``bedrock-api-key-``. It is valid for the presigned
lifetime (at most 12 hours), or until the signing credentials expire if that is sooner.

This module holds no SDK import. The presigning itself is done with botocore's signer
in the adapter factory's ``create_client`` and handed to ``ShortTermKeySource`` as an
issuing function, so the refresh logic is tested without botocore or a network.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

SHORT_TERM_PREFIX = "bedrock-api-key-"
TOKEN_VERSION = "&Version=1"  # noqa: S105 - a format suffix, not a credential
KEY_HOST = "bedrock.amazonaws.com"
KEY_URL = f"https://{KEY_HOST}/"
KEY_ACTION = "CallWithBearerToken"
KEY_SIGNING_NAME = "bedrock"


def encode_short_term_key(presigned_url: str) -> str:
    """The short-term API key for a presigned ``CallWithBearerToken`` URL."""
    if not presigned_url.startswith("https://"):
        raise ValueError("a presigned key URL must use https")
    token = presigned_url.removeprefix("https://") + TOKEN_VERSION
    return SHORT_TERM_PREFIX + base64.b64encode(token.encode("utf-8")).decode("ascii")


@dataclass(frozen=True, slots=True)
class IssuedKey:
    """A key and the time (``time.time()`` seconds) it stops being valid."""

    value: str
    expires_at: float

    def __repr__(self) -> str:
        return f"IssuedKey(<hidden>, expires_at={self.expires_at})"


class KeySource(Protocol):
    """Hands out the bearer key to send with the next request."""

    def current(self) -> str: ...


class StaticKeySource:
    """A long-term key, handed out unchanged."""

    def __init__(self, key: str) -> None:
        self._key = key

    def current(self) -> str:
        return self._key

    def __repr__(self) -> str:
        return "StaticKeySource(<hidden>)"


class ShortTermKeySource:
    """Short-term keys, issued on first use and replaced before they expire.

    ``issue(lifetime_s)`` returns a new key and its expiry. A key is replaced once the
    clock is within ``refresh_before_s`` of its expiry, so an expired key is never
    handed out. Safe to call from several threads: one refresh serves them all.
    """

    def __init__(
        self,
        issue: Callable[[int], IssuedKey],
        *,
        lifetime_s: int,
        refresh_before_s: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._issue = issue
        self._lifetime_s = lifetime_s
        self._refresh_before_s = refresh_before_s
        self._clock = clock
        self._key: IssuedKey | None = None
        self._lock = threading.Lock()

    def _fresh(self, key: IssuedKey | None) -> bool:
        return key is not None and self._clock() < key.expires_at - self._refresh_before_s

    def current(self) -> str:
        with self._lock:
            key = self._key
            if key is None or not self._fresh(key):
                key = self._issue(self._lifetime_s)
                if not self._fresh(key):
                    raise ValueError(
                        "a new short-term Bedrock API key expires within its refresh margin; "
                        "the signing credentials are about to expire"
                    )
                self._key = key
            return key.value

    def __repr__(self) -> str:
        return f"ShortTermKeySource(lifetime_s={self._lifetime_s})"
