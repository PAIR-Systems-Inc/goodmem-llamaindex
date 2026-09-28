"""GoodMem resource IDs are UUIDs; anything else is refused before a request.

The official SDK interpolates IDs into URL paths without escaping them, and httpx
resolves dot segments before sending. An ID such as "../spaces/<uuid>" would
therefore address a different resource, so every ID is checked here first.
"""

import re
from typing import Annotated

from pydantic import Field

UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
_UUID = re.compile(UUID_PATTERN)

# Tool schemas tell the model an ID is a UUID with the standard JSON-schema format.
# LlamaIndex keeps only json_schema_extra from a top-level annotation and prints the
# annotation in each tool description; a pattern there would push retrieve_memories
# past OpenAI's 1024-character description limit. require_uuid is the real guard.
UuidStr = Annotated[str, Field(json_schema_extra={"format": "uuid"})]


def require_uuid(value: object, field: str) -> str:
    """Return the canonical lowercase UUID, or raise ValueError naming the field."""
    # fullmatch, unlike match with "$", also rejects a trailing newline.
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise ValueError(
            f"{field} must be a UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx); no request was sent"
        )
    return value.lower()
