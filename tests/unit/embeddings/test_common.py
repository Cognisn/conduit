"""Reading provider values: vectors and token counts."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.embeddings._common import batches, read_count, read_vector


def test_a_vector_of_numbers_is_read_as_floats() -> None:
    assert read_vector([1, 0.5, -2]) == (1.0, 0.5, -2.0)


@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), 0.1],
        [float("inf")],
        [0.1, float("-inf")],
        [True, 0.1],
        [0.1, False],
        ["0.1"],
        [None],
        [],
        "0.1",
        None,
        {"a": 0.1},
    ],
)
def test_anything_else_is_not_a_vector(values: Any) -> None:
    assert read_vector(values) is None


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "3", None])
def test_a_count_is_a_non_negative_integer(value: Any) -> None:
    assert read_count(value) is None


def test_counts_are_read() -> None:
    assert read_count(0) == 0
    assert read_count(42) == 42


def test_batches_split_in_order() -> None:
    assert list(batches(("a", "b", "c"), 2)) == [("a", "b"), ("c",)]
