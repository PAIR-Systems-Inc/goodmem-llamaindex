"""Translate supported LlamaIndex metadata filters to native GoodMem expressions."""

import math
import re

from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters


def _literal(value):
    if isinstance(value, str):
        if "\0" in value:
            raise ValueError("Filter strings cannot contain NUL")
        # The server decodes escapes. Separate backslashes to preserve literal \\n.
        parts = []
        for part in value.split("\\"):
            escaped = part.replace("'", "\\'").replace("\n", "\\n")
            escaped = escaped.replace("\r", "\\r").replace("\t", "\\t")
            parts.append("'" + escaped + "'")
        return "(" + " || '\\\\' || ".join(parts) + ")", "TEXT"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Filter numbers must be finite")
        return str(value), "NUMERIC"
    raise ValueError("Filter values must be strings or finite numbers")


def filter_expression(filters: MetadataFilters | MetadataFilter | None) -> str | None:
    """Compile comparisons, IN/NOT IN and nested AND/OR/NOT conditions.

    Unsupported operators raise before any request. Keys are simple top-level
    field names; use the retriever's explicit native expression for other paths.
    An empty filter means no constraint and never sends bare TRUE to the server.
    """
    if filters is None:
        return None
    if isinstance(filters, MetadataFilters):
        parts = [filter_expression(f) for f in filters.filters]
        if not parts:
            return None
        if any(p is None for p in parts):
            raise ValueError("Empty nested filter groups are ambiguous")
        condition = filters.condition or "and"
        if condition == "not":
            if len(parts) != 1:
                raise ValueError("NOT requires exactly one filter")
            return f"NOT ({parts[0]})"
        return "(" + f" {condition.upper()} ".join(parts) + ")"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", filters.key):
        raise ValueError("Metadata filter keys must be simple top-level field names")
    operator = filters.operator
    if operator in {"in", "nin"}:
        if not isinstance(filters.value, list) or not filters.value:
            raise ValueError("IN/NOT IN requires a nonempty list")
        comparisons = [
            filter_expression(MetadataFilter(key=filters.key, value=v)) for v in filters.value
        ]
        expression = "(" + " OR ".join(comparisons) + ")"
        return f"NOT {expression}" if operator == "nin" else expression
    operators = {"==": "=", "!=": "!=", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
    if operator not in operators:
        raise ValueError(f"Unsupported metadata filter operator: {operator}")
    literal, type_ = _literal(filters.value)
    return f"CAST(val('$.{filters.key}') AS {type_}) {operators[operator]} {literal}"
