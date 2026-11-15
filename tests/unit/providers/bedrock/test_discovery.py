"""Bedrock discovery: foundation models and inference profiles on the control plane.

The listing logic is tested against recorded control-plane answers through a fake
control client; the control client itself is a real boto3 client from the factory,
answered locally (``before-send``), so its service, Region, endpoint and signing are
the real ones.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.embeddings.bedrock import BedrockEmbeddingFactory
from conduit.providers.bedrock import BedrockFactory, BedrockSettings
from conduit.providers.bedrock.factory import BedrockClients
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

from .test_errors import ClientError

REGION = recordings.REGION
SETTINGS = {"kind": "bedrock", "region": REGION}
# The documentation example access key pair, never a real credential.
ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
SECRETS = {"bedrock/id": ACCESS_KEY_ID, "bedrock/secret": SECRET_ACCESS_KEY}
ACCESS_KEYS = {
    "mode": "access_keys",
    "access_key_id": "secret://bedrock/id",
    "secret_access_key": "secret://bedrock/secret",
}


class FakeControl:
    """A control-plane client answering from a recording, or failing with ``error``."""

    def __init__(self, replay: Replay | None = None, error: Exception | None = None) -> None:
        self.replay = replay
        self.error = error

    def _answer(self, operation: str, params: dict[str, Any]) -> Any:
        if self.error is not None:
            raise self.error
        assert self.replay is not None
        return self.replay.next({"operation": operation, "params": params})["body"]

    def list_foundation_models(self, **params: Any) -> Any:
        return self._answer("ListFoundationModels", params)

    def list_inference_profiles(self, **params: Any) -> Any:
        return self._answer("ListInferenceProfiles", params)


def _registry() -> ProviderRegistry:
    return ProviderRegistry(secrets=LiteralSecretResolver(SECRETS), factories=[BedrockFactory()])


async def _discover(control: Any, settings: Mapping[str, Any] = SETTINGS) -> DiscoveryResult:
    clients = BedrockClients(runtime=object(), control=control)
    return await _registry().discover("bedrock_sydney", settings, client=clients)


def _replay(name: str) -> Replay:
    return Replay(load_recording(recordings.ROOT / name))


def test_the_bedrock_factory_discovers() -> None:
    assert isinstance(BedrockFactory(), DiscoveringFactory)


class TestListing:
    async def test_foundation_models_and_profiles_are_listed_and_joined(self) -> None:
        replay = _replay("bedrock/discovery-listing.json")
        result = await _discover(FakeControl(replay))
        replay.assert_exhausted()  # both profile pages were read
        assert result.supported
        assert {m.model_id: (m.outcome, m.label) for m in result.models} == {
            "amazon.nova-lite-v1:0": (DiscoveryOutcome.ALLOWED, None),
            "amazon.titan-embed-text-v2:0": (DiscoveryOutcome.ALLOWED, None),
            "example.text-model-v1:0": (DiscoveryOutcome.UNDECLARED, None),
            "anthropic.claude-opus-5-5": (DiscoveryOutcome.REFUSED, None),
            "au.anthropic.claude-opus-5-5": (DiscoveryOutcome.ALLOWED, "SYSTEM_DEFINED"),
            "global.anthropic.claude-opus-5-5": (DiscoveryOutcome.ALLOWED, "SYSTEM_DEFINED"),
            "example-application-1": (DiscoveryOutcome.UNDECLARED, "APPLICATION"),
        }

    async def test_a_model_without_text_or_embedding_output_is_left_out(self) -> None:
        result = await _discover(FakeControl(_replay("bedrock/discovery-listing.json")))
        assert "example.image-model-v1:0" not in {m.model_id for m in result.models}

    async def test_the_region_refusal_is_the_start_up_reason(self) -> None:
        result = await _discover(FakeControl(_replay("bedrock/discovery-listing.json")))
        [refused] = result.refused
        assert refused.reason == (
            f"bedrock_sydney: model 'anthropic.claude-opus-5-5' is not offered in region {REGION}"
        )

    async def test_an_empty_account(self) -> None:
        replay = _replay("bedrock/discovery-empty.json")
        result = await _discover(FakeControl(replay))
        replay.assert_exhausted()
        assert result.supported and result.models == ()

    @pytest.mark.parametrize(
        ("code", "status", "failure"),
        [
            ("AccessDeniedException", 403, ProviderFailure.REJECTED),
            ("UnrecognizedClientException", 403, ProviderFailure.REJECTED),
            ("ThrottlingException", 429, ProviderFailure.UNAVAILABLE),
            ("InternalServerException", 500, ProviderFailure.UNAVAILABLE),
        ],
    )
    async def test_a_failed_listing_is_a_provider_failure(
        self, code: str, status: int, failure: ProviderFailure
    ) -> None:
        error = ClientError(code, status, message="denied for arn:aws:iam::123456789012:user/x")
        with pytest.raises(ProviderError) as raised:
            await _discover(FakeControl(error=error))
        assert raised.value.failure is failure
        assert raised.value.__cause__ is None and raised.value.__context__ is None

    @pytest.mark.parametrize(
        "body",
        [{}, {"modelSummaries": "x"}, {"modelSummaries": [{"modelArn": "a"}]}],
    )
    async def test_an_answer_that_is_not_the_documented_shape_is_a_protocol_failure(
        self, body: Any
    ) -> None:
        class Odd(FakeControl):
            def list_foundation_models(self, **params: Any) -> Any:
                return body

        with pytest.raises(ProviderError) as raised:
            await _discover(Odd())
        assert raised.value.failure is ProviderFailure.PROTOCOL

    async def test_a_profile_listing_that_never_ends_is_a_protocol_failure(self) -> None:
        class Endless(FakeControl):
            def list_foundation_models(self, **params: Any) -> Any:
                return {"modelSummaries": []}

            def list_inference_profiles(self, **params: Any) -> Any:
                return {"inferenceProfileSummaries": [], "nextToken": "again"}

        with pytest.raises(ProviderError) as raised:
            await _discover(Endless())
        assert raised.value.failure is ProviderFailure.PROTOCOL


class TestUnsupported:
    async def test_a_replaced_runtime_endpoint_is_reported_unsupported(self) -> None:
        settings = {**SETTINGS, "endpoint_url": "https://bedrock-runtime.example.com"}
        result = await _registry().discover("bedrock_sydney", settings)
        assert not result.supported
        assert result.reason is not None and "endpoint_url" in result.reason


class Closing:
    """A client that records its close, and fails it with ``error`` when given one."""

    def __init__(self, error: Exception | None = None) -> None:
        self.closed = False
        self.close_error = error

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class ClosingControl(FakeControl):
    """A control-plane client that also records its close."""

    def __init__(self, replay: Replay | None = None, error: Exception | None = None) -> None:
        super().__init__(replay, error)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestClosing:
    def test_closing_the_bundle_closes_both_clients(self) -> None:
        runtime, control = Closing(), Closing()
        BedrockClients(runtime=runtime, control=control).close()
        assert runtime.closed and control.closed

    def test_without_a_control_client_the_runtime_client_is_closed(self) -> None:
        runtime = Closing()
        BedrockClients(runtime=runtime, control=None).close()
        assert runtime.closed

    def test_a_failed_runtime_close_still_closes_the_control_client(self) -> None:
        runtime, control = Closing(RuntimeError("runtime close")), Closing()
        with pytest.raises(RuntimeError, match="runtime close") as raised:
            BedrockClients(runtime=runtime, control=control).close()
        assert control.closed
        assert raised.value.__cause__ is None and raised.value.__context__ is None

    def test_a_failed_control_close_is_raised_once_both_are_closed(self) -> None:
        runtime, control = Closing(), Closing(RuntimeError("control close"))
        with pytest.raises(RuntimeError, match="control close") as raised:
            BedrockClients(runtime=runtime, control=control).close()
        assert runtime.closed
        assert raised.value.__cause__ is None and raised.value.__context__ is None

    def test_when_both_closes_fail_the_first_is_raised_with_nothing_chained(self) -> None:
        runtime = Closing(RuntimeError("runtime close"))
        control = Closing(OSError("control close"))
        with pytest.raises(RuntimeError, match="runtime close") as raised:
            BedrockClients(runtime=runtime, control=control).close()
        assert runtime.closed and control.closed
        assert raised.value.__cause__ is None and raised.value.__context__ is None

    async def test_discovery_closes_both_clients(self) -> None:
        runtime = Closing()
        control = ClosingControl(_replay("bedrock/discovery-empty.json"))
        clients = BedrockClients(runtime=runtime, control=control)
        await _registry().discover("bedrock_sydney", SETTINGS, client=clients)
        assert runtime.closed and control.closed

    async def test_a_failed_discovery_closes_both_clients(self) -> None:
        runtime = Closing()
        control = ClosingControl(error=ClientError("AccessDeniedException", 403, message="no"))
        clients = BedrockClients(runtime=runtime, control=control)
        with pytest.raises(ProviderError):
            await _registry().discover("bedrock_sydney", SETTINGS, client=clients)
        assert runtime.closed and control.closed

    async def test_a_built_provider_closes_both_clients(self) -> None:
        runtime, control = Closing(), Closing()
        clients = BedrockClients(runtime=runtime, control=control)
        provider = await _registry().build("bedrock_sydney", SETTINGS, client=clients)
        await provider.aclose()
        assert runtime.closed and control.closed


class TestCopying:
    def test_a_copy_does_not_recurse_and_keeps_both_clients(self) -> None:
        runtime, control = object(), object()
        copied = copy.copy(BedrockClients(runtime=runtime, control=control))
        assert copied.runtime is runtime and copied.control is control

    def test_a_deep_copy_does_not_recurse(self) -> None:
        copied = copy.deepcopy(BedrockClients(runtime=Closing(), control=None))
        assert isinstance(copied.runtime, Closing) and copied.control is None

    def test_a_half_built_bundle_forwards_nothing(self) -> None:
        bare = object.__new__(BedrockClients)
        with pytest.raises(AttributeError):
            _ = bare.runtime
        with pytest.raises(AttributeError):
            _ = bare.converse


boto3 = pytest.importorskip("boto3", reason="needs the bedrock extra")


class Answer:
    """Answers each control-plane request locally and keeps what was sent."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, request: Any, **_: Any) -> Any:
        aws_response = importlib.import_module("botocore.awsrequest").AWSResponse
        self.requests.append(request)
        path = request.url.split("amazonaws.com", 1)[1]
        if path.startswith("/foundation-models"):
            body: dict[str, Any] = {"modelSummaries": []}
        else:
            body = {"inferenceProfileSummaries": []}
        data = json.dumps(body).encode()

        class Raw:
            def stream(self, **_: Any) -> Iterator[bytes]:
                yield data

        return aws_response(request.url, 200, {"Content-Type": "application/json"}, Raw())


class AnsweringFactory(BedrockFactory):
    def __init__(self, answer: Answer) -> None:
        super().__init__()
        self.answer = answer

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        clients = super().create_client(settings, secrets)
        clients.control.meta.events.register("before-send.bedrock", self.answer)
        return clients


def _settings(auth: dict[str, Any] | None = None, **fields: Any) -> BedrockSettings:
    return BedrockSettings.model_validate(
        {"region": "eu-west-2", **({"auth": auth} if auth else {}), **fields}
    )


_SECRETS = {
    f"auth.{prefix}{name}": SecretStr(value)
    for prefix in ("", "source.")
    for name, value in (
        ("access_key_id", ACCESS_KEY_ID),
        ("secret_access_key", SECRET_ACCESS_KEY),
        ("api_key", "long-term-example-key"),
    )
}


class TestControlClient:
    def test_create_client_makes_a_control_plane_client_beside_the_runtime_client(
        self,
    ) -> None:
        clients = BedrockFactory().create_client(_settings(ACCESS_KEYS), _SECRETS)
        assert isinstance(clients, BedrockClients)
        assert clients.runtime.meta.service_model.service_name == "bedrock-runtime"
        assert clients.control.meta.service_model.service_name == "bedrock"
        assert clients.control.meta.region_name == "eu-west-2"
        assert clients.control.meta.endpoint_url == "https://bedrock.eu-west-2.amazonaws.com"

    def test_the_clients_behave_as_the_runtime_client(self) -> None:
        clients = BedrockFactory().create_client(_settings(ACCESS_KEYS), _SECRETS)
        assert clients.meta is clients.runtime.meta
        assert clients.converse == clients.runtime.converse

    def test_no_control_client_for_a_replaced_runtime_endpoint(self) -> None:
        settings = _settings(ACCESS_KEYS, endpoint_url="https://bedrock-runtime.example.com")
        clients = BedrockFactory().create_client(settings, _SECRETS)
        assert clients.control is None

    async def test_access_keys_sign_the_listing_with_sigv4_for_bedrock(self) -> None:
        answer = Answer()
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver(SECRETS), factories=[AnsweringFactory(answer)]
        )
        result = await registry.discover(
            "bedrock_london", {"kind": "bedrock", "region": "eu-west-2", "auth": ACCESS_KEYS}
        )
        assert result.supported and result.models == ()
        assert [r.url for r in answer.requests] == [
            "https://bedrock.eu-west-2.amazonaws.com/foundation-models?byInferenceType=ON_DEMAND",
            "https://bedrock.eu-west-2.amazonaws.com/inference-profiles?maxResults=1000",
        ]
        for request in answer.requests:
            authorisation = request.headers["Authorization"].decode()
            assert authorisation.startswith("AWS4-HMAC-SHA256 Credential=" + ACCESS_KEY_ID)
            assert "/eu-west-2/bedrock/aws4_request" in authorisation

    async def test_a_long_term_key_is_sent_as_a_bearer_token_to_the_control_plane(
        self,
    ) -> None:
        answer = Answer()
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({"bedrock/key": "long-term-example-key"}),
            factories=[AnsweringFactory(answer)],
        )
        await registry.discover(
            "bedrock_london",
            {
                "kind": "bedrock",
                "region": "eu-west-2",
                "auth": {"mode": "long_term_api_key", "api_key": "secret://bedrock/key"},
                "allow_long_term_api_keys": True,
            },
        )
        assert answer.requests
        for request in answer.requests:
            header = request.headers["Authorization"]
            text = header.decode() if isinstance(header, bytes) else header
            assert text == "Bearer long-term-example-key"


def _spy_on_close(clients: BedrockClients, closed: list[str]) -> None:
    """Record each real botocore client's close, then close it."""
    for role, client in (("runtime", clients.runtime), ("control", clients.control)):
        if client is None:
            continue
        real = client.close

        def close(role: str = role, real: Any = real) -> None:
            closed.append(role)
            real()

        client.close = close


class SpyingFactory(AnsweringFactory):
    """Real clients from ``create_client``, answered locally, with their closes recorded."""

    def __init__(self, answer: Answer) -> None:
        super().__init__(answer)
        self.closed: list[str] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        clients = super().create_client(settings, secrets)
        _spy_on_close(clients, self.closed)
        return clients


class SpyingEmbeddingFactory(BedrockEmbeddingFactory):
    def __init__(self) -> None:
        super().__init__()
        self.closed: list[str] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        clients = super().create_client(settings, secrets)
        _spy_on_close(clients, self.closed)
        return clients


_LONDON = {"kind": "bedrock", "region": "eu-west-2", "auth": ACCESS_KEYS}


class TestRealClientsAreClosed:
    async def test_discovery_closes_the_runtime_and_control_clients(self) -> None:
        factory = SpyingFactory(Answer())
        registry = ProviderRegistry(secrets=LiteralSecretResolver(SECRETS), factories=[factory])
        await registry.discover("bedrock_london", _LONDON)
        assert sorted(factory.closed) == ["control", "runtime"]

    async def test_a_built_provider_closes_the_runtime_and_control_clients(self) -> None:
        factory = SpyingFactory(Answer())
        registry = ProviderRegistry(secrets=LiteralSecretResolver(SECRETS), factories=[factory])
        provider = await registry.build("bedrock_london", _LONDON)
        await provider.aclose()
        assert sorted(factory.closed) == ["control", "runtime"]

    async def test_a_built_embedder_closes_the_runtime_and_control_clients(self) -> None:
        factory = SpyingEmbeddingFactory()
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver(SECRETS), embedding_factories=[factory]
        )
        embedder = await registry.build_embedder("bedrock_london", _LONDON)
        await embedder.aclose()
        assert sorted(factory.closed) == ["control", "runtime"]
