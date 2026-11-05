"""Output contracts: strict parsing of a final answer against a pydantic model."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from conduit.agent.contracts import OutputContract, parse_output
from conduit.core.errors import ConfigurationError


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    score: int


CONTRACT = OutputContract(model=Verdict, name="verdict")


def test_a_json_object_matching_the_model_is_accepted() -> None:
    check = parse_output(CONTRACT, '{"label": "benign", "score": 3}')
    assert check.value == Verdict(label="benign", score=3)
    assert check.errors == ()


def test_a_single_json_fence_is_accepted() -> None:
    check = parse_output(CONTRACT, '```json\n{"label": "benign", "score": 3}\n```')
    assert check.value == Verdict(label="benign", score=3)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("not json", "the reply is not exactly one JSON object"),
        ('{"label": "a", "score": 1} trailing', "the reply is not exactly one JSON object"),
        ("[1, 2]", "the reply is not a JSON object"),
        ('{"label": "a", "label": "b", "score": 1}', "the reply repeats a key"),
    ],
)
def test_a_reply_that_is_not_one_object_is_refused(text: str, error: str) -> None:
    check = parse_output(CONTRACT, text)
    assert check.value is None
    assert check.errors == (error,)


def test_errors_name_the_field_and_type_never_the_value() -> None:
    check = parse_output(CONTRACT, '{"label": "zz secret value 991 zz", "score": "high"}')
    assert check.value is None
    assert check.errors == ("score: int_parsing",)
    assert "zz secret value" not in " ".join(check.errors)
    assert "high" not in " ".join(check.errors)


def test_a_missing_and_an_extra_field_are_both_named() -> None:
    check = parse_output(CONTRACT, '{"label": "a", "extra": 1}')
    assert set(check.errors) == {"score: missing", "extra: extra_forbidden"}


def test_a_contract_needs_a_pydantic_model() -> None:
    with pytest.raises(ConfigurationError, match="pydantic model"):
        OutputContract(model=dict, name="verdict")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "has space", "x" * 300])
def test_a_contract_name_must_be_an_identifier(name: str) -> None:
    with pytest.raises(ConfigurationError, match="name"):
        OutputContract(model=Verdict, name=name)


def test_the_retry_message_names_the_errors() -> None:
    message = CONTRACT.retry_message(("score: missing",))
    assert "verdict" in message
    assert "score: missing" in message


def test_a_product_may_supply_its_own_retry_message() -> None:
    contract = OutputContract(
        model=Verdict, name="verdict", retry_prompt=lambda errors: "again: " + ",".join(errors)
    )
    assert contract.retry_message(("a: missing", "b: missing")) == "again: a: missing,b: missing"


def test_the_schema_is_the_models_json_schema() -> None:
    assert CONTRACT.schema() == Verdict.model_json_schema()
