"""Persist LlamaIndex metadata visibility settings alongside user metadata."""

_DOCUMENT_METADATA = "_llamaindex_goodmem"
_EXCLUSIONS = ("excluded_llm_metadata_keys", "excluded_embed_metadata_keys")


def document_metadata(document):
    """Encode formatting settings without overwriting a user metadata field."""
    if _DOCUMENT_METADATA in document.metadata:
        raise ValueError(f"Metadata key {_DOCUMENT_METADATA} is reserved for LlamaIndex settings")
    return dict(document.metadata) | {
        _DOCUMENT_METADATA: {key: list(getattr(document, key)) for key in _EXCLUSIONS}
    }


def stored_metadata(metadata):
    """Separate stored settings, rejecting malformed exclusions before rendering."""
    metadata = dict(metadata or {})
    options = metadata.pop(_DOCUMENT_METADATA, {})
    if not isinstance(options, dict):
        raise ValueError(f"Invalid {_DOCUMENT_METADATA} metadata settings")
    exclusions = {}
    for key in _EXCLUSIONS:
        value = options.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"Invalid stored {key}: expected a list of strings")
        exclusions[key] = list(value)
    return metadata, exclusions
