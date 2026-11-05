"""The client-construction scanner, shown synthetic sources it must catch and must allow."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.tripwires.clientscan import (
    CLIENT_CONSTRUCTORS,
    construction_violations,
    factory_call_violations,
    metered_violations,
    scan_source,
)

PATH = Path("synthetic.py")


def _constructions(source: str, module: str) -> list[str]:
    scan = scan_source(textwrap.dedent(source), module, path=PATH)
    return [str(f) for f in construction_violations(scan)]


def _metered(source: str, module: str) -> list[str]:
    scan = scan_source(textwrap.dedent(source), module, path=PATH)
    return [str(f) for f in metered_violations(scan.metered)]


def _factory_calls(source: str, module: str) -> list[str]:
    scan = scan_source(textwrap.dedent(source), module, path=PATH)
    return [str(f) for f in factory_call_violations(scan.factory_calls)]


@pytest.mark.parametrize("constructor", sorted(CLIENT_CONSTRUCTORS))
def test_every_constructor_is_caught_outside_a_factory(constructor: str) -> None:
    module = constructor.rpartition(".")[0]
    source = f"""
    import {module}

    def build():
        return {constructor}()
    """
    found = _constructions(source, "conduit.providers.example")
    assert len(found) == 1 and constructor in found[0]


def test_inside_create_client_in_a_provider_layer_is_allowed() -> None:
    source = """
    import anthropic

    class Factory:
        def create_client(self, settings, secrets):
            return anthropic.AsyncAnthropic(api_key=secrets["api_key"])
    """
    assert _constructions(source, "conduit.providers.anthropic") == []
    assert _constructions(source, "conduit.embeddings.remote") == []


def test_inside_create_client_outside_the_provider_layers_is_caught() -> None:
    source = """
    import openai

    def create_client(settings, secrets):
        return openai.AsyncOpenAI()
    """
    assert len(_constructions(source, "conduit.agent.runner")) == 1


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            from anthropic import AsyncAnthropic as Client
            client = Client()
            """,
            id="from-import-alias",
        ),
        pytest.param(
            """
            import boto3 as b
            def make():
                return b.client("bedrock-runtime")
            """,
            id="module-alias",
        ),
        pytest.param(
            """
            import importlib
            sdk = importlib.import_module("anthropic")
            def make():
                return sdk.AsyncAnthropic()
            """,
            id="import-module-assigned",
        ),
        pytest.param(
            """
            import importlib
            def make():
                return importlib.import_module("openai").AsyncOpenAI()
            """,
            id="import-module-inline",
        ),
        pytest.param(
            """
            from google import genai
            def make():
                return genai.Client()
            """,
            id="from-package-import-module",
        ),
        pytest.param(
            """
            def make(session):
                return session.client("bedrock-runtime", region_name="ap-southeast-2")
            """,
            id="session-client",
        ),
        pytest.param(
            """
            import httpx
            def stream():
                return httpx.AsyncClient()
            """,
            id="http-client-in-a-provider",
        ),
    ],
)
def test_every_import_form_is_resolved(source: str) -> None:
    assert len(_constructions(source, "conduit.providers.example")) == 1


def test_an_http_client_outside_the_provider_layers_is_not_a_provider_client() -> None:
    source = """
    import httpx
    def connect():
        return httpx.AsyncClient()
    """
    assert _constructions(source, "conduit.mcp.client") == []


def test_unrelated_calls_named_like_constructors_are_ignored() -> None:
    source = """
    class Client:
        pass
    def make(table):
        return Client(), table.client("other")
    """
    assert _constructions(source, "conduit.providers.example") == []


def test_create_client_may_be_called_only_by_the_registry() -> None:
    source = """
    def build(factory, settings):
        return factory.create_client(settings, {})
    """
    assert _factory_calls(source, "conduit.core.registry") == []
    assert len(_factory_calls(source, "conduit.agent.runner")) == 1
    assert len(_factory_calls(source, "conduit.providers.bedrock")) == 1


def test_a_metered_provider_may_be_made_only_by_the_registry() -> None:
    source = """
    from conduit.core.registry import MeteredProvider as Metered

    def wrap(inner, settings, catalogue, events):
        return Metered(inner, name="p", settings=settings, catalogue=catalogue, events=events)
    """
    assert _metered(source, "conduit.core.registry") == []
    found = _metered(source, "conduit.agent.runner")
    assert len(found) == 1 and "MeteredProvider" in found[0]
    plain = """
    import conduit.core.registry as registry
    def wrap(inner):
        return registry.MeteredProvider(inner)
    """
    assert len(_metered(plain, "conduit.core.runtime")) == 1


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            def make(session):
                return session.client(service_name="bedrock-runtime")
            """,
            id="session-client-service-name-keyword",
        ),
        pytest.param(
            """
            from aiobotocore.session import get_session
            def make():
                return get_session()
            """,
            id="aiobotocore-get-session",
        ),
        pytest.param(
            """
            import aiobotocore.session
            def make():
                return aiobotocore.session.AioSession()
            """,
            id="aiobotocore-aiosession",
        ),
        pytest.param(
            """
            def make(session):
                return session.create_client("bedrock-runtime", region_name="ap-southeast-2")
            """,
            id="session-create-client",
        ),
        pytest.param(
            """
            def make(session):
                return session.create_client(service_name="bedrock-runtime")
            """,
            id="session-create-client-service-name-keyword",
        ),
        pytest.param(
            """
            import anthropic
            def make():
                return getattr(anthropic, "AsyncAnthropic")()
            """,
            id="getattr-literal",
        ),
        pytest.param(
            """
            import importlib
            def make():
                return getattr(importlib.import_module("openai"), "OpenAI")()
            """,
            id="getattr-on-import-module",
        ),
        pytest.param(
            """
            import ollama as sdk
            def make(name):
                return getattr(sdk, name)()
            """,
            id="getattr-dynamic-name-on-an-sdk-module",
        ),
    ],
)
def test_every_construction_form_is_caught(source: str) -> None:
    assert len(_constructions(source, "conduit.providers.example")) == 1


def test_an_sdk_session_create_client_is_not_the_factory_method() -> None:
    source = """
    def make(session):
        return session.create_client("bedrock-runtime")
    """
    assert _factory_calls(source, "conduit.providers.bedrock") == []


def test_getattr_on_unrelated_objects_is_ignored() -> None:
    source = """
    import json
    def make(obj, name):
        return getattr(obj, "Client")(), getattr(json, "dumps")({}), getattr(obj, name)()
    """
    assert _constructions(source, "conduit.providers.example") == []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            from anthropic.lib.bedrock import AsyncAnthropicBedrockMantle
            client = AsyncAnthropicBedrockMantle()
            """,
            id="mantle-from-lib",
        ),
        pytest.param(
            """
            import importlib
            sdk = importlib.import_module("anthropic.lib.bedrock")
            client = sdk.AsyncAnthropicBedrockMantle()
            """,
            id="mantle-import-module",
        ),
        pytest.param(
            """
            import importlib
            from conduit.providers.bedrock_mantle.auth import bearer_client_class
            sdk = importlib.import_module("anthropic.lib.bedrock")
            client = bearer_client_class(sdk.AsyncAnthropicBedrockMantle, keys)(region="x")
            """,
            id="mantle-keyed-class",
        ),
    ],
)
def test_a_bedrock_mantle_client_outside_a_factory_is_caught(source: str) -> None:
    assert _constructions(source, "conduit.providers.bedrock_mantle.stray") != []


def test_a_bedrock_mantle_client_inside_create_client_is_allowed() -> None:
    source = """
    import importlib
    from conduit.providers.bedrock_mantle.auth import bearer_client_class

    class Factory:
        def create_client(self, settings, secrets):
            sdk = importlib.import_module("anthropic.lib.bedrock")
            client_class = bearer_client_class(sdk.AsyncAnthropicBedrockMantle, secrets)
            return client_class(region=settings.region)
    """
    assert _constructions(source, "conduit.providers.bedrock_mantle.factory") == []
