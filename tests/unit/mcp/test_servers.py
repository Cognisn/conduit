"""The loopback test servers' ports (D0087).

With ``CONDUIT_MCP_TEST_PORT_BASE`` set, each server takes its fixed offset in the
worktree's slot. Without it, each takes a port the operating system assigns, kept for
the whole test session, so suites running at once never collide on a fixed default.
"""

from __future__ import annotations

import socket

import pytest

from tests.unit.mcp import servers
from tests.unit.mcp.servers import OFFSETS, PortPlan

LEGACY_BLOCK = range(22400, 22500)


def test_a_set_base_gives_the_slot_s_fixed_offsets() -> None:
    plan = PortPlan("22200")
    assert plan.port("http") == 22210
    assert plan.port("callback") == 22220
    assert {plan.port(name) for name in OFFSETS} == {22200 + o for o in OFFSETS.values()}


@pytest.mark.parametrize("unset", [None, ""])
def test_without_a_base_every_server_takes_a_free_port(unset: str | None) -> None:
    plan = PortPlan(unset)
    taken = {name: plan.port(name) for name in OFFSETS}
    assert len(set(taken.values())) == len(OFFSETS)  # one port per server
    assert not any(number in LEGACY_BLOCK for number in taken.values())
    assert {name: plan.port(name) for name in OFFSETS} == taken  # kept for the session
    for number in taken.values():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", number))


def test_an_unknown_server_name_is_refused() -> None:
    with pytest.raises(KeyError):
        PortPlan(None).port("nonesuch")
    with pytest.raises(KeyError):
        PortPlan("22200").port("nonesuch")


def test_a_port_handed_out_twice_is_not_given_to_two_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handed = iter([30001, 30001, 30001, 30002])
    monkeypatch.setattr(servers, "free_port", lambda: next(handed))
    plan = PortPlan(None)
    assert (plan.port("http"), plan.port("sse")) == (30001, 30002)
