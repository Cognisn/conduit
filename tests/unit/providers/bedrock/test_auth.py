"""Bedrock API keys: the short-term key format, and refresh before expiry."""

from __future__ import annotations

import base64
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from conduit.providers.bedrock.auth import (
    SHORT_TERM_PREFIX,
    IssuedKey,
    ShortTermKeySource,
    StaticKeySource,
    encode_short_term_key,
)

PRESIGNED = (
    "https://bedrock.amazonaws.com/?Action=CallWithBearerToken"
    "&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=43200&X-Amz-Signature=00ff"
)


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
        with pytest.raises(ValueError, match="expires"):
            source.current()


class TestStatic:
    def test_a_static_key_is_handed_out_unchanged(self) -> None:
        assert StaticKeySource("long-term-example").current() == "long-term-example"

    def test_its_repr_never_shows_the_key(self) -> None:
        assert "long-term-example" not in repr(StaticKeySource("long-term-example"))
