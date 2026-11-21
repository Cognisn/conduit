"""Product-declared task classes, capability floors and residency, enforced at start-up."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import (
    AssignmentRefused,
    CapabilityRefused,
    ConfigurationError,
    ResidencyRefused,
    UndeclaredModel,
    UnknownTaskClass,
)
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.settings import AssignmentSettings, ProviderSettings
from conduit.core.tasks import CapabilityFloor, TaskClass, resolve_assignments

AU = frozenset({"ap-southeast-2", "ap-southeast-4"})


def _entry(model_id: str, **overrides: Any) -> ModelEntry:
    values: dict[str, Any] = {
        "model_id": model_id,
        "provider": "bedrock",
        "residency_scope": "in_region",
        "regions": ["ap-southeast-2", "us-east-1"],
        "capabilities": {
            "tools": True,
            "vision": True,
            "prompt_caching": True,
            "structured_output": True,
        },
        "context_window": 200_000,
        "output_limit": 64_000,
    }
    values.update(overrides)
    return ModelEntry.model_validate(values)


CATALOGUE = Catalogue(
    [
        _entry("in-region-large"),
        _entry("in-region-small", context_window=32_000, capabilities={"tools": False}),
        _entry("au-profile", residency_scope="cross_region", regions=sorted(AU)),
        _entry("us-profile", residency_scope="cross_region", regions=["us-east-1", "us-west-2"]),
        _entry("global-profile", residency_scope="global", regions=[]),
        _entry("labelled", labels={"tier:advanced"}),
        _entry(
            "embedder",
            kind="embedding",
            output_limit=None,
            embedding_dimensions=1024,
            context_window=8_192,
        ),
        _entry("local", provider="stub", residency_scope="local", regions=[]),
    ],
    version="test",
)

PROVIDERS = {
    "sydney": ProviderSettings(kind="bedrock", region="ap-southeast-2"),
    "virginia": ProviderSettings(kind="bedrock", region="us-east-1"),
    "oregon": ProviderSettings(kind="bedrock", region="us-west-2"),
    "nowhere": ProviderSettings(kind="bedrock"),
    "stub": ProviderSettings(kind="stub"),
}

CHAT = TaskClass(name="chat", floor=CapabilityFloor(tools=True, min_context_window=100_000))


def resolve(
    model: str,
    *,
    provider: str = "sydney",
    task: TaskClass = CHAT,
    residency: ResidencyPolicy | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    return resolve_assignments(
        [task],
        {
            task.name: AssignmentSettings(
                provider=provider, model=model, max_output_tokens=max_output_tokens
            )
        },
        PROVIDERS,
        CATALOGUE,
        residency or ResidencyPolicy(),
    )


class TestResolution:
    def test_a_capable_assignment_resolves(self) -> None:
        assignments = resolve("in-region-large")
        resolved = assignments.for_task("chat")
        assert resolved.model == "in-region-large"
        assert resolved.provider == "sydney"
        assert resolved.kind == "bedrock"
        assert resolved.model_id == "bedrock:in-region-large"
        assert resolved.max_output_tokens == 64_000  # the model's limit by default
        assert resolved.task_class == CHAT

    def test_an_undeclared_model_is_refused(self) -> None:
        with pytest.raises(UndeclaredModel):
            resolve("not-in-the-catalogue")

    def test_a_missing_provider_is_refused(self) -> None:
        with pytest.raises(AssignmentRefused, match="no provider named"):
            resolve("in-region-large", provider="absent")

    def test_an_assignment_for_an_undeclared_task_class_is_refused(self) -> None:
        with pytest.raises(AssignmentRefused, match="not declared"):
            resolve_assignments(
                [CHAT],
                {
                    "chat": AssignmentSettings(provider="sydney", model="in-region-large"),
                    "triage": AssignmentSettings(provider="sydney", model="in-region-large"),
                },
                PROVIDERS,
                CATALOGUE,
            )

    def test_a_declared_task_class_without_an_assignment_is_refused(self) -> None:
        other = TaskClass(name="triage")
        with pytest.raises(AssignmentRefused, match="triage"):
            resolve_assignments(
                [CHAT, other],
                {"chat": AssignmentSettings(provider="sydney", model="in-region-large")},
                PROVIDERS,
                CATALOGUE,
            )
        # Unless the product says not every class must be assigned.
        resolved = resolve_assignments(
            [CHAT, other],
            {"chat": AssignmentSettings(provider="sydney", model="in-region-large")},
            PROVIDERS,
            CATALOGUE,
            require_all=False,
        )
        assert set(resolved) == {"chat"}

    def test_duplicate_task_class_names_are_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            resolve_assignments([CHAT, CHAT], {}, PROVIDERS, CATALOGUE)

    def test_a_model_not_offered_in_the_provider_region_is_refused(self) -> None:
        with pytest.raises(AssignmentRefused, match="not offered in region"):
            resolve("in-region-large", provider="oregon")
        resolve("in-region-large", provider="virginia")  # offered in us-east-1
        # A cross-region profile is called from one of its own regions.
        with pytest.raises(AssignmentRefused, match="not offered in region"):
            resolve("us-profile", provider="sydney")

    def test_unknown_task_class_at_run_time(self) -> None:
        with pytest.raises(UnknownTaskClass):
            resolve("in-region-large").for_task("triage")

    def test_task_class_names_are_identifiers(self) -> None:
        with pytest.raises(ValidationError):
            TaskClass(name="Has Spaces")


class TestFloors:
    def test_each_shortfall_is_refused_and_named(self) -> None:
        with pytest.raises(CapabilityRefused) as raised:
            resolve("in-region-small")
        message = str(raised.value)
        assert "tools" in message and "context window" in message

    @pytest.mark.parametrize(
        ("floor", "named"),
        [
            (CapabilityFloor(vision=True), "vision"),
            (CapabilityFloor(prompt_caching=True), "prompt caching"),
            (CapabilityFloor(structured_output=True), "structured output"),
            (CapabilityFloor(min_output_limit=100_000), "output limit"),
            (CapabilityFloor(required_labels=frozenset({"tier:advanced"})), "tier:advanced"),
            (CapabilityFloor(kind="embedding"), "embedding"),
        ],
    )
    def test_every_capability_is_part_of_the_floor(
        self, floor: CapabilityFloor, named: str
    ) -> None:
        bare = _entry("bare", capabilities={})
        catalogue = Catalogue([bare], version="t")
        task = TaskClass(name="t", floor=floor)
        with pytest.raises(CapabilityRefused, match=named):
            resolve_assignments(
                [task],
                {"t": AssignmentSettings(provider="sydney", model="bare")},
                PROVIDERS,
                catalogue,
            )

    def test_labels_meet_a_label_floor(self) -> None:
        task = TaskClass(
            name="t", floor=CapabilityFloor(required_labels=frozenset({"tier:advanced"}))
        )
        assert resolve("labelled", task=task).for_task("t").model == "labelled"

    def test_an_embedding_model_cannot_serve_a_chat_task(self) -> None:
        with pytest.raises(CapabilityRefused, match="embedding"):
            resolve("embedder", task=TaskClass(name="chat"))

    def test_an_output_budget_above_the_model_limit_is_refused(self) -> None:
        with pytest.raises(CapabilityRefused, match="output"):
            resolve("in-region-large", max_output_tokens=64_001)
        assert (
            resolve("in-region-large", max_output_tokens=4_096).for_task("chat").max_output_tokens
            == 4_096
        )


class TestResidency:
    IN_REGION_AU = ResidencyPolicy(scopes=frozenset({ResidencyScope.IN_REGION}), regions=AU)

    def test_in_region_in_an_allowed_region_passes(self) -> None:
        resolve("in-region-large", residency=self.IN_REGION_AU)

    def test_a_cross_region_profile_under_an_in_region_policy_is_refused(self) -> None:
        # The Termograph lesson: privileged data stays in region, so an inference profile
        # is refused at start-up even though its regions are all Australian.
        with pytest.raises(ResidencyRefused, match="cross_region"):
            resolve("au-profile", residency=self.IN_REGION_AU)

    def test_a_cross_region_profile_within_the_allowed_regions_passes(self) -> None:
        policy = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION}), regions=AU
        )
        resolve("au-profile", residency=policy)

    def test_a_cross_region_profile_reaching_outside_the_regions_is_refused(self) -> None:
        policy = ResidencyPolicy(regions=frozenset({"us-east-1"}))
        with pytest.raises(ResidencyRefused, match="us-west-2"):
            resolve("us-profile", provider="virginia", residency=policy)

    def test_an_in_region_call_from_a_disallowed_region_is_refused(self) -> None:
        with pytest.raises(ResidencyRefused, match="us-east-1"):
            resolve("in-region-large", provider="virginia", residency=self.IN_REGION_AU)

    def test_an_in_region_model_needs_a_provider_region_under_a_region_policy(self) -> None:
        with pytest.raises(ResidencyRefused, match="no region"):
            resolve("in-region-large", provider="nowhere", residency=self.IN_REGION_AU)

    def test_global_is_refused_whenever_regions_are_restricted(self) -> None:
        with pytest.raises(ResidencyRefused, match="global"):
            resolve("global-profile", residency=ResidencyPolicy(regions=AU))
        resolve("global-profile")  # no policy: allowed

    def test_a_disallowed_scope_is_refused_without_regions(self) -> None:
        policy = ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL}))
        with pytest.raises(ResidencyRefused):
            resolve("in-region-large", residency=policy)

    def test_local_models_satisfy_any_region_policy(self) -> None:
        policy = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.LOCAL, ResidencyScope.IN_REGION}), regions=AU
        )
        resolve_assignments(
            [TaskClass(name="t")],
            {"t": AssignmentSettings(provider="stub", model="local")},
            PROVIDERS,
            CATALOGUE,
            policy,
        )


class TestStartUpOnly:
    def test_resolved_assignments_do_not_change_when_the_catalogue_does(self) -> None:
        assignments = resolve("in-region-large")
        from conduit.core.catalogue import CatalogueOverlay, ModelKey

        CATALOGUE.apply(
            CatalogueOverlay(withdraw=(ModelKey(provider="bedrock", model_id="in-region-large"),))
        )
        assert assignments.for_task("chat").entry.context_window == 200_000

    def test_assignments_are_read_only(self) -> None:
        assignments = resolve("in-region-large")
        with pytest.raises(TypeError):
            assignments["chat"] = assignments["chat"]
