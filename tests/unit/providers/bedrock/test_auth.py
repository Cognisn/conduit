"""Bedrock API keys: the key format, refresh before expiry, presigning and log records."""

from __future__ import annotations

import base64
import importlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

import pytest

from conduit.core.errors import ConfigurationError, ProviderFailure
from conduit.providers.bedrock.auth import (
    ENDPOINT_LOGGER,
    SHORT_TERM_PREFIX,
    SIGNER_LOGGER,
    IssuedKey,
    KeyRefreshRefused,
    RenderedRecordFilter,
    ShortTermKeySource,
    SignatureRecordFilter,
    StaticKeySource,
    encode_short_term_key,
    keep_keys_out_of_logs,
    presign_short_term_key,
    short_term_keys,
)
from conduit.providers.bedrock.errors import classify

# The documentation example access key pair, never a real credential.
ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
MARKER = "SECRET-MARKER-5d1e"

PRESIGNED = (
    "https://bedrock.amazonaws.com/?Action=CallWithBearerToken"
    "&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=43200&X-Amz-Signature=00ff"
)


def decode(key: str) -> dict[str, list[str]]:
    assert key.startswith(SHORT_TERM_PREFIX)
    url = base64.b64decode(key.removeprefix(SHORT_TERM_PREFIX)).decode()
    host, _, query = url.partition("/?")
    assert host == "bedrock.amazonaws.com"
    return parse_qs(query)


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Issuer:
    """Issues numbered keys that expire ``lifetime`` seconds after the clock's time."""

    def __init__(self, clock: Clock, *, cap: float | None = None) -> None:
        self.clock = clock
        self.cap = cap
        self.issued = 0
        self.lifetimes: list[int] = []

    def __call__(self, lifetime_s: int) -> IssuedKey:
        self.issued += 1
        self.lifetimes.append(lifetime_s)
        expires = self.clock() + lifetime_s
        if self.cap is not None:
            expires = min(expires, self.cap)
        return IssuedKey(value=f"key-{self.issued}", expires_at=expires)


class TestEncoding:
    def test_a_short_term_key_is_the_prefixed_base64_presigned_url_with_its_version(self) -> None:
        key = encode_short_term_key(PRESIGNED)
        assert key.startswith(SHORT_TERM_PREFIX)
        decoded = base64.b64decode(key[len(SHORT_TERM_PREFIX) :]).decode()
        assert decoded == PRESIGNED.removeprefix("https://") + "&Version=1"

    def test_a_url_that_is_not_https_is_refused(self) -> None:
        with pytest.raises(ValueError, match="https"):
            encode_short_term_key("http://bedrock.amazonaws.com/?Action=CallWithBearerToken")


class TestRefresh:
    def make(
        self, *, lifetime: int = 3_600, margin: int = 300, cap: float | None = None
    ) -> tuple[ShortTermKeySource, Issuer, Clock]:
        clock = Clock()
        issuer = Issuer(clock, cap=cap)
        source = ShortTermKeySource(
            issuer, lifetime_s=lifetime, refresh_before_s=margin, clock=clock
        )
        return source, issuer, clock

    def test_the_first_key_is_issued_on_first_use_with_the_configured_lifetime(self) -> None:
        source, issuer, _ = self.make()
        assert issuer.issued == 0
        assert source.current() == "key-1"
        assert issuer.lifetimes == [3_600]

    def test_a_key_is_reused_while_it_is_outside_the_refresh_margin(self) -> None:
        source, issuer, clock = self.make()
        source.current()
        clock.now += 3_600 - 300 - 1
        assert source.current() == "key-1"
        assert issuer.issued == 1

    def test_a_key_is_replaced_once_it_enters_the_refresh_margin(self) -> None:
        source, issuer, clock = self.make()
        source.current()
        clock.now += 3_600 - 300
        assert source.current() == "key-2"
        assert issuer.issued == 2

    def test_an_expired_key_is_never_handed_out(self) -> None:
        source, _, clock = self.make()
        source.current()
        clock.now += 10_000
        assert source.current() == "key-2"

    def test_a_key_capped_by_its_credentials_expiry_is_refreshed_by_that_expiry(self) -> None:
        # Temporary credentials expiring sooner than the key's lifetime end the key early.
        source, issuer, clock = self.make(cap=1_000.0 + 900)
        source.current()
        clock.now += 900 - 300 - 1
        assert source.current() == "key-1"
        clock.now += 1
        issuer.cap = clock.now + 900  # the credentials were refreshed meanwhile
        assert source.current() == "key-2"
        assert issuer.issued == 2

    def test_concurrent_callers_share_one_refresh(self) -> None:
        clock = Clock()
        gate = threading.Barrier(8)
        issued: list[int] = []
        lock = threading.Lock()

        def slow_issue(lifetime_s: int) -> IssuedKey:
            with lock:
                issued.append(lifetime_s)
            time.sleep(0.05)  # long enough for every caller to arrive while it runs
            return IssuedKey(value="shared", expires_at=clock() + lifetime_s)

        source = ShortTermKeySource(slow_issue, lifetime_s=3_600, refresh_before_s=300, clock=clock)

        def call() -> str:
            gate.wait()
            return source.current()

        with ThreadPoolExecutor(8) as pool:
            keys = list(pool.map(lambda _: call(), range(8)))
        assert keys == ["shared"] * 8
        assert len(issued) == 1

    def test_a_key_that_would_be_issued_already_inside_its_margin_is_refused(self) -> None:
        source, _, _ = self.make(cap=1_000.0 + 100)
        with pytest.raises(KeyRefreshRefused, match="expires"):
            source.current()
        assert classify(KeyRefreshRefused("x")) is ProviderFailure.REJECTED


class TestStatic:
    def test_a_static_key_is_handed_out_unchanged(self) -> None:
        assert StaticKeySource("long-term-example").current() == "long-term-example"

    def test_its_repr_never_shows_the_key(self) -> None:
        assert "long-term-example" not in repr(StaticKeySource("long-term-example"))


# The record filter that keeps botocore's signatures out of the logs.


def _record(msg: str, *args: Any) -> logging.LogRecord:
    return logging.LogRecord(SIGNER_LOGGER, logging.DEBUG, __file__, 1, msg, args, None)


def test_the_filter_drops_signature_records_only() -> None:
    record_filter = SignatureRecordFilter()
    assert not record_filter.filter(_record("Signature:\n%s", "abc123"))
    assert record_filter.filter(_record("Calculating signature using v4 auth."))
    assert record_filter.filter(_record("StringToSign:\n%s", "AWS4-HMAC-SHA256"))


class Mutable:
    """Renders as its current text: a request whose headers change after it is logged."""

    def __init__(self) -> None:
        self.text = "headers={}"

    def __str__(self) -> str:
        return self.text


def test_the_rendering_filter_fixes_a_record_when_it_is_logged() -> None:
    request = Mutable()
    record = logging.LogRecord(
        ENDPOINT_LOGGER, logging.DEBUG, __file__, 1, "Sending http request: %s", (request,), None
    )
    assert RenderedRecordFilter().filter(record)
    request.text = "headers={'Authorization': 'Bearer later'}"
    assert record.getMessage() == "Sending http request: headers={}"


def test_the_rendering_filter_passes_a_record_it_cannot_render_unchanged() -> None:
    record = logging.LogRecord(ENDPOINT_LOGGER, logging.DEBUG, __file__, 1, "%d", ("x",), None)
    assert RenderedRecordFilter().filter(record)
    assert (record.msg, record.args) == ("%d", ("x",))


def test_each_filter_is_installed_on_its_logger_once(monkeypatch: pytest.MonkeyPatch) -> None:
    signer = logging.getLogger(SIGNER_LOGGER)
    endpoint = logging.getLogger(ENDPOINT_LOGGER)
    monkeypatch.setattr(signer, "filters", [])  # restored afterwards
    monkeypatch.setattr(endpoint, "filters", [])
    keep_keys_out_of_logs()
    keep_keys_out_of_logs()
    assert [type(f) for f in signer.filters] == [SignatureRecordFilter]
    assert [type(f) for f in endpoint.filters] == [RenderedRecordFilter]


class SlowFilters(list[Any]):
    """A logger's filter list whose membership check is slow, so racing installers
    all pass the check before any of them adds a filter, unless they are serialised."""

    def __iter__(self) -> Any:
        seen = list(super().__iter__())  # what the check sees
        time.sleep(0.05)  # ...while another thread may add
        return iter(seen)


def test_concurrent_installs_leave_exactly_one_of_each_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer = logging.getLogger(SIGNER_LOGGER)
    endpoint = logging.getLogger(ENDPOINT_LOGGER)
    monkeypatch.setattr(signer, "filters", SlowFilters())
    monkeypatch.setattr(endpoint, "filters", SlowFilters())
    threads = 8
    barrier = threading.Barrier(threads)

    def install() -> None:
        barrier.wait()  # all start together
        keep_keys_out_of_logs()

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for future in [pool.submit(install) for _ in range(threads)]:
            future.result()
    assert [type(f) for f in list.__iter__(signer.filters)] == [SignatureRecordFilter]
    assert [type(f) for f in list.__iter__(endpoint.filters)] == [RenderedRecordFilter]


class TestPresigning:
    """The shared presign step, with botocore's own signer and synthetic credentials."""

    @pytest.fixture(autouse=True)
    def _botocore(self) -> None:
        pytest.importorskip("botocore", reason="needs the bedrock or bedrock-mantle extra")

    @staticmethod
    def credentials(expiry: float | None = None) -> Any:
        read_only = importlib.import_module("botocore.credentials").ReadOnlyCredentials

        class Credentials:
            def __init__(self) -> None:
                self.reads = 0
                if expiry is not None:
                    self._expiry_time = datetime.fromtimestamp(expiry, tz=UTC)

            def get_frozen_credentials(self) -> Any:
                self.reads += 1
                return read_only(ACCESS_KEY_ID, SECRET_ACCESS_KEY, None)

        return Credentials()

    def test_a_key_is_presigned_for_its_lifetime_and_region(self) -> None:
        issued = presign_short_term_key(
            self.credentials(), region="ap-southeast-2", lifetime_s=3_600, clock=Clock(1_000.0)
        )
        query = decode(issued.value)
        assert query["Action"] == ["CallWithBearerToken"]
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert query["X-Amz-Expires"] == ["3600"]
        assert query["X-Amz-Credential"][0].startswith(ACCESS_KEY_ID + "/")
        assert query["X-Amz-Credential"][0].endswith("/ap-southeast-2/bedrock/aws4_request")
        assert query["Version"] == ["1"]
        assert "X-Amz-Signature" in query
        assert issued.expires_at == 4_600.0

    def test_its_expiry_is_capped_by_the_credentials_expiry(self) -> None:
        issued = presign_short_term_key(
            self.credentials(expiry=1_700.0), region="eu-west-2", lifetime_s=900, clock=Clock()
        )
        assert issued.expires_at == 1_700.0
        assert decode(issued.value)["X-Amz-Expires"] == ["900"]

    def test_the_step_installs_both_filters_and_issues_the_first_key_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signer = logging.getLogger(SIGNER_LOGGER)
        endpoint = logging.getLogger(ENDPOINT_LOGGER)
        monkeypatch.setattr(signer, "filters", [])
        monkeypatch.setattr(endpoint, "filters", [])
        credentials = self.credentials()
        resolved: list[int] = []

        def resolve() -> Any:
            resolved.append(1)
            return credentials

        keys = short_term_keys(
            resolve, region="eu-west-2", lifetime_s=3_600, refresh_before_s=600, clock=Clock()
        )
        assert credentials.reads == 1  # the first key, before any request
        assert [type(f) for f in signer.filters] == [SignatureRecordFilter]
        assert [type(f) for f in endpoint.filters] == [RenderedRecordFilter]
        assert keys.current().startswith(SHORT_TERM_PREFIX)
        assert credentials.reads == 1 and resolved == [1]

    def test_a_key_is_replaced_within_its_margin_of_the_credentials_expiry(self) -> None:
        clock = Clock(1_000.0)
        credentials = self.credentials(expiry=1_700.0)
        resolved: list[int] = []

        def resolve() -> Any:
            resolved.append(1)
            return credentials

        keys = short_term_keys(
            resolve,
            region="eu-west-2",
            lifetime_s=900,
            refresh_before_s=60,
            clock=clock,
        )
        keys.current()
        clock.now = 1_630.0  # 70 s before the credentials expire: still fresh
        keys.current()
        assert credentials.reads == 1
        credentials._expiry_time = datetime.fromtimestamp(5_000.0, tz=UTC)  # refreshed
        clock.now = 1_650.0  # inside the margin, though the key's lifetime runs to 1 900
        keys.current()
        # Presigned again. (botocore signs with the wall clock, so a key presigned in
        # the same second can be identical; the presign count is what shows it.)
        assert credentials.reads == 2
        # Resolved once: refreshable credentials renew themselves when read.
        assert resolved == [1]

    def test_missing_credentials_are_refused_at_once_naming_only_the_class(self) -> None:
        with pytest.raises(ConfigurationError, match="NoCredentialsError") as raised:
            short_term_keys(lambda: None, region="eu-west-2", lifetime_s=900, refresh_before_s=60)
        assert raised.value.__cause__ is None and raised.value.__context__ is None

    def test_a_failure_to_resolve_credentials_names_only_its_class(self) -> None:
        def resolve() -> Any:
            raise LookupError(f"profile {MARKER} has {SECRET_ACCESS_KEY}")

        with pytest.raises(ConfigurationError) as raised:
            short_term_keys(resolve, region="eu-west-2", lifetime_s=900, refresh_before_s=60)
        error = raised.value
        assert "LookupError" in str(error)
        for text in (str(error), repr(error), repr(error.args)):
            assert MARKER not in text and SECRET_ACCESS_KEY not in text
        assert error.__cause__ is None and error.__context__ is None

    def test_an_interrupt_is_not_turned_into_a_refusal(self) -> None:
        def resolve() -> Any:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            short_term_keys(resolve, region="eu-west-2", lifetime_s=900, refresh_before_s=60)
