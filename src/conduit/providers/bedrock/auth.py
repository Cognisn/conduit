"""Bedrock API keys sent as bearer tokens: short-term generation, refresh and logging.

A short-term Bedrock API key is a SigV4 presigned URL for the ``CallWithBearerToken``
action on ``bedrock.amazonaws.com``, without its scheme, with ``&Version=1`` appended,
base64-encoded and prefixed ``bedrock-api-key-``. It is valid for the presigned
lifetime (at most 12 hours), or until the signing credentials expire if that is sooner.

``short_term_keys`` is the one presign step the bedrock and bedrock-mantle adapters both
use. Each adapter's ``create_client`` resolves the signing credentials (a session is a
client construction, so it stays there) and hands the step a function that returns them.
The step:

- installs conduit's record filters on botocore's loggers, once per process
  (``keep_keys_out_of_logs``);
- presigns each key with botocore's SigV4 query signer for the configured lifetime
  (``presign_short_term_key``), capping its expiry at the signing credentials' expiry;
- hands keys out through ``ShortTermKeySource``, which replaces a key once the clock is
  within the refresh margin of that expiry;
- presigns the first key at once, so missing credentials are a ``ConfigurationError``
  at start-up that names only the error's class.

botocore is imported only when a key is presigned, so importing this module imports no
SDK, and ``ShortTermKeySource`` is tested without botocore or a network.

Keys and log records
--------------------
botocore's SigV4 signer logs the canonical request, the string to sign and the
signature of everything it signs, at DEBUG on the ``botocore.auth`` logger. For a
presigned short-term key those records are the key: whoever reads them can rebuild it
and call Bedrock with it until it expires. A product enables those records simply by
running its root logger at DEBUG. ``SignatureRecordFilter`` therefore drops every
``botocore.auth`` record whose message starts with ``Signature:``, and only those.

botocore's endpoint logs each request it sends at DEBUG on ``botocore.endpoint``, as
the request object itself, and the bedrock adapter sets the bearer key on that object
afterwards, in a ``before-send`` handler. A handler that formats the record at once never
sees the key; one that keeps the record and formats it later (a ``MemoryHandler``, any
buffering or deferred handler) would. ``RenderedRecordFilter`` renders every
``botocore.endpoint`` record's message when it is logged and drops none.

Both filters are process-wide. They sit on botocore's loggers themselves, so they also
apply to every other botocore client in the process, the product's own among them: its
signature records are dropped too, and its endpoint records are rendered when logged. A
product that removes ``SignatureRecordFilter`` (to debug signing, say) gets every
signature back, and with it every short-term Bedrock key conduit presigns from then on,
in whatever its handlers write. One that removes ``RenderedRecordFilter`` and formats
records late gets every bearer key the bedrock adapter sends. Remove them only for local
debugging with throwaway credentials, and put them back after.

conduit cannot keep every credential out of botocore's DEBUG records. When botocore
refreshes temporary credentials (assume-role through STS, SSO, a container or instance
role), it can log the responses that carry them, on loggers these filters do not touch;
conduit neither makes those calls nor can tell such records from any other. Products
keep the ``botocore`` logger above DEBUG in production.
"""

from __future__ import annotations

import base64
import importlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from conduit.core.errors import ConfigurationError

SHORT_TERM_PREFIX = "bedrock-api-key-"
TOKEN_VERSION = "&Version=1"  # noqa: S105 - a format suffix, not a credential
KEY_HOST = "bedrock.amazonaws.com"
KEY_URL = f"https://{KEY_HOST}/"
KEY_ACTION = "CallWithBearerToken"
KEY_SIGNING_NAME = "bedrock"


class KeyRefreshRefused(ValueError):
    """A new short-term key would expire within its refresh margin (credentials expiring)."""


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
                    raise KeyRefreshRefused(
                        "a new short-term Bedrock API key expires within its refresh margin; "
                        "the signing credentials are about to expire"
                    )
                self._key = key
            return key.value

    def __repr__(self) -> str:
        return f"ShortTermKeySource(lifetime_s={self._lifetime_s})"


#: The logger botocore's SigV4 signer writes its signatures to.
SIGNER_LOGGER = "botocore.auth"

#: The logger botocore's endpoint writes each request it sends to.
ENDPOINT_LOGGER = "botocore.endpoint"


class SignatureRecordFilter(logging.Filter):
    """Drops botocore's ``Signature:`` debug records; every other record passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (isinstance(record.msg, str) and record.msg.startswith("Signature:"))


class RenderedRecordFilter(logging.Filter):
    """Renders each record's message when it is logged; every record passes.

    botocore's endpoint logs the request object itself, and the bearer key is set on
    that object afterwards. Rendered at once, the record keeps the request as it was
    when logged, whenever a handler formats it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            rendered: str | None = None
            try:
                rendered = record.getMessage()
            except Exception:  # left for the handler to report, as logging does
                rendered = None
            if rendered is not None:
                record.msg = rendered
                record.args = ()
        return True


#: Serialises the check-then-add below, so concurrent clients install each filter once.
_FILTERS_LOCK = threading.Lock()


def keep_keys_out_of_logs() -> None:
    """Install ``SignatureRecordFilter`` and ``RenderedRecordFilter`` on botocore's
    signer and endpoint loggers, each once per process, even from several threads."""
    with _FILTERS_LOCK:
        for name, kind in (
            (SIGNER_LOGGER, SignatureRecordFilter),
            (ENDPOINT_LOGGER, RenderedRecordFilter),
        ):
            logger = logging.getLogger(name)
            if not any(isinstance(f, kind) for f in logger.filters):
                logger.addFilter(kind())


def presign_short_term_key(
    credentials: Any,
    *,
    region: str,
    lifetime_s: int,
    clock: Callable[[], float] = time.time,
) -> IssuedKey:
    """One short-term key, presigned with botocore's SigV4 query signer.

    ``credentials`` are botocore credentials (refreshable ones are read afresh). The key
    expires after ``lifetime_s``, or when the credentials do if that is sooner.
    """
    botocore_auth = importlib.import_module("botocore.auth")
    awsrequest = importlib.import_module("botocore.awsrequest")
    frozen = credentials.get_frozen_credentials()
    request = awsrequest.AWSRequest(
        method="POST",
        url=KEY_URL,
        headers={"host": KEY_HOST},
        params={"Action": KEY_ACTION},
    )
    signer = botocore_auth.SigV4QueryAuth(frozen, KEY_SIGNING_NAME, region, expires=lifetime_s)
    signer.add_auth(request)
    expires_at = clock() + lifetime_s
    credentials_expiry = getattr(credentials, "_expiry_time", None)
    if credentials_expiry is not None:
        expires_at = min(expires_at, credentials_expiry.timestamp())
    return IssuedKey(value=encode_short_term_key(request.url), expires_at=expires_at)


def short_term_keys(
    credentials: Callable[[], Any],
    *,
    region: str,
    lifetime_s: int,
    refresh_before_s: int,
    clock: Callable[[], float] = time.time,
) -> ShortTermKeySource:
    """Short-term keys presigned for ``region``, replaced before they expire.

    ``credentials()`` resolves the signing credentials (``None`` if there are none); it
    is called once, when the first key is presigned. Installs the record filters first
    (``keep_keys_out_of_logs``), then presigns the first key at once: a key that cannot
    be had is a ``ConfigurationError`` naming only the error's class, raised after the
    handler so nothing is chained to it.
    """
    keep_keys_out_of_logs()
    resolved: Any = None

    def issue(lifetime: int) -> IssuedKey:
        nonlocal resolved
        if resolved is None:
            resolved = credentials()
            if resolved is None:
                raise importlib.import_module("botocore.exceptions").NoCredentialsError()
        return presign_short_term_key(resolved, region=region, lifetime_s=lifetime, clock=clock)

    keys = ShortTermKeySource(
        issue, lifetime_s=lifetime_s, refresh_before_s=refresh_before_s, clock=clock
    )
    refused: str | None = None
    try:
        keys.current()
    except Exception as error:
        refused = type(error).__name__
    if refused is not None:
        raise ConfigurationError(f"a short-term Bedrock API key cannot be generated: {refused}")
    return keys
