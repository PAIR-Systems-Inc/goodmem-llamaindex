"""Compare translated SQL's NULL behavior with LlamaIndex's native evaluator."""

import sqlite3

import pytest
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.utils import build_metadata_filter_fn

from llama_index.tools.goodmem.filters import filter_expression

FILTERS = (
    [
        MetadataFilters(
            filters=[MetadataFilter(key="deleted", operator=operator, value=value)],
            condition=condition,
        )
        for operator, value in [("==", "yes"), ("!=", "yes"), ("in", ["yes"]), ("nin", ["yes"])]
        for condition in ["and", "not"]
    ]
    + [
        MetadataFilters(filters=[MetadataFilter(key="rank", operator=operator, value=1)])
        for operator in [">", ">=", "<", "<="]
    ]
    + [
        MetadataFilters(
            filters=[
                MetadataFilter(key="deleted", value="yes"),
                MetadataFilter(key="rank", value=1),
            ],
            condition=condition,
        )
        for condition in ["and", "or"]
    ]
)


@pytest.mark.parametrize("filters", FILTERS)
def test_sql_null_behavior_matches_framework(filters):
    rows = [
        {},
        {"deleted": None, "rank": None},
        {"deleted": "yes", "rank": 1},
        {"deleted": "no", "rank": 0},
        {"rank": 2},
    ]
    expression = filter_expression(filters)
    for row in rows:
        expected = build_metadata_filter_fn(lambda _: row, filters)("node")
        # SQLite executes the same three-valued boolean/comparison semantics as
        # PostgreSQL here. val returns the scalar produced by GoodMem's JSON_VALUE.
        with sqlite3.connect(":memory:") as database:
            database.create_function("val", 1, lambda path: row.get(path[2:]))
            actual = database.execute(f"SELECT {expression}").fetchone()[0]
        assert actual is not None, (row, expression)
        assert bool(actual) == expected, (row, expression)
