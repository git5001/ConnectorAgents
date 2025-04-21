from typing import Any

from util.PydanticSerializer import PydanticSerializer


# Use these as replacements for encode_payload and decode_payload
def encode_payload(payload: Any) -> Any:
    """
    Replacement for original encode_payload that handles nested Pydantic models correctly.
    """
    return PydanticSerializer.serialize(payload)


def decode_payload(payload: Any) -> Any:
    """
    Replacement for original decode_payload that handles nested Pydantic models correctly.
    """
    return PydanticSerializer.deserialize(payload)