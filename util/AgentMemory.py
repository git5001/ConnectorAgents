from __future__ import annotations

""" AgentMemory implementation.

Key improvements
----------------
* Inherits from ``pydantic.BaseModel`` so instances are *natively* serialisable
  via ``model_dump*`` / ``model_convert*`` helpers – no custom ``dump`` / ``load``
  methods required.
* Clear, concise public API: ``add_message``, ``new_turn``, ``get_history``.
* Overflow handled with slice‑assignment for speed and clarity.
* Convenience ``to_json`` / ``from_json`` helpers for explicit JSON round‑trips.
* Fully‑typed, minimal surface – easy to understand, extend, and test.

The class is drop‑in compatible with the original but safer and more pythonic.
"""

import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field



class Message(BaseModel):
    """A single chat message."""

    role: str
    content: BaseModel
    turn_id: Optional[str] = None

    # Allow arbitrary (user‑defined) subclasses of ``BaseModel``
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentMemory(BaseModel):
    """In‑memory chat transcript manager.

    Parameters
    ----------
    max_messages: int | ``None``
        Maximum number of messages to keep. If ``None`` the history is unbounded.
    """

    history: List[Message] = Field(default_factory=list)
    max_messages: Optional[int] = None
    current_turn_id: Optional[str] = None

    # Allow arbitrary ``BaseIOSchema`` subclasses inside the model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def initialize_turn(self) -> str:
        """Start a *new* turn and return its UUID."""
        self.current_turn_id = uuid.uuid4().hex
        return self.current_turn_id

    def new_turn(self) -> None:
        """Explicitly end the current turn (next write will create a fresh one)."""
        self.current_turn_id = None

    def add_message(self, role: str, content: BaseModel) -> None:
        """Append a message to the transcript.

        A fresh turn‑ID is generated automatically when necessary.
        """
        if self.current_turn_id is None:
            self.initialize_turn()

        self.history.append(
            Message(role=role, content=content, turn_id=self.current_turn_id)
        )
        self._manage_overflow()

    def get_history(
            self,
            n: Optional[int] = None,
            *,
            as_dict: bool = True
    ) -> List[Union[dict, Message]]:
        """
        Return the last `n` messages (or all if None) in chronological order (oldest→newest).
        """
        # Determine slice: last n messages or all
        if n is not None and n > 0:
            recent = self.history[-n:]
        else:
            recent = list(self.history)

        if not as_dict:
            return recent

        records: List[dict] = []
        for msg in recent:
            content_obj = msg.content
            if getattr(content_obj, "images", None):
                # multimodal
                text_val = None
                for field in content_obj.model_fields:
                    if field.endswith("text"):
                        text_val = getattr(content_obj, field)
                        break
                text_val = text_val or str(content_obj)
                blocks = [{"type": "text", "text": text_val}]
                for url in content_obj.images:
                    blocks.append({"type": "image_url", "url": url})
                records.append({"role": msg.role, "content": blocks})
            else:
                # plain text
                print("content ",type(content_obj))
                to_text_method = getattr(content_obj, "to_text", None)
                if callable(to_text_method):
                    text_val = to_text_method()
                else:
                    text_val = getattr(content_obj, "text", None) or content_obj.model_dump_json()
                records.append({"role": msg.role, "content": text_val})
        return records

    def format_history(
            history: List[dict]
    ) -> str:
        """
        Convert a list of OpenAI-style history dicts into a human-readable string.

        Each message is shown on its own lines; images are shown as URLs below their text.
        """
        lines: List[str] = []
        for entry in history:
            role = entry.get("role", "")
            content = entry["content"]
            if isinstance(content, list):
                # multimodal
                text_block = next((b for b in content if b.get("type") == "text"), {})
                text = text_block.get("text", "")
                lines.append(f"{role.capitalize()}: {text}")
                for block in content:
                    if block.get("type") == "image_url":
                        lines.append(f"(Image) {block.get('url')}")
            else:
                lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)


    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401 (simple return value)
        """Return number of stored messages."""
        return len(self.history)

    def copy(self) -> "AgentMemory":
        """Deep copy the memory (using Pydantic's built‑in copier)."""
        return self.model_copy(deep=True)

    # Round‑trip JSON helpers --------------------------------------------------
    def to_json(self, **kwargs) -> str:  # noqa: D401
        """Serialise the entire ``AgentMemory`` as a JSON string."""
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, raw: str) -> "AgentMemory":  # noqa: D401
        """Re‑create an ``AgentMemory`` from JSON produced by :py:meth:`to_json`."""
        return cls.model_validate_json(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manage_overflow(self) -> None:
        """Enforce the ``max_messages`` cap – drop *oldest* messages first."""
        if self.max_messages is not None and len(self.history) > self.max_messages:
            # Keep only the *newest* ``max_messages`` elements.
            self.history[:] = self.history[-self.max_messages:]
