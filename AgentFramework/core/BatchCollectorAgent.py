from __future__ import annotations

from AgentFramework.core.ListCollectionAgent import ListModel

"""Batch‑collector agent
~~~~~~~~~~~~~~~~~~~~~~~~
Buffers incoming messages until a user‑defined threshold (`amount`) is
reached and then emits them as a :class:`ListModel`.

The implementation mirrors the structure of ``IdentityAgent``—same IO
schemas, lightweight ``run``—but introduces a configurable batching
mechanism and a minimal internal state.
"""

from typing import List, Optional, TypeVar

from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig
from AgentFramework.core.ConnectedAgent import ConnectedAgent


T = TypeVar("T", bound=BaseModel)


class BatchCollectorAgentConfig(BaseToolConfig):
    """Configuration for :class:`BatchCollectorAgent`.

    Parameters
    ----------
    amount : int
        Number of messages to gather before emitting a batch (``> 0``).
    """

    amount: int = Field(..., gt=0, description="Number of messages to collect before emitting a batch")


class BatchCollectorAgentState(BaseModel):
    """Internal buffer holding messages that haven't been emitted yet."""

    buffer: List[BaseIOSchema] = Field(default_factory=list, description="Buffered messages awaiting emission")


class BatchCollectorAgent(ConnectedAgent):
    """A simple batching agent.

    It stores incoming messages until *amount* messages have been received.
    Once the threshold is met, it returns a :class:`ListModel` containing
    those messages (after passing each through :meth:`run`).  Remaining
    messages stay in the buffer so multiple batches can be emitted over
    time.
    """

    # --- I/O Schemas -----------------------------------------------------
    input_schema = BaseIOSchema  # accept anything that conforms to the base IO schema
    output_schema = ListModel   # runtime generics aren’t enforced, but conceptually this is ListModel[BaseIOSchema]

    # --- State -----------------------------------------------------------
    state_schema = BatchCollectorAgentState
    _state: BatchCollectorAgentState

    # --------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------
    def __init__(self, config: BatchCollectorAgentConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._state = BatchCollectorAgentState()  # fresh buffer for each instance

    # --------------------------------------------------------------------
    # Core functionality
    # --------------------------------------------------------------------
    def process(self, params: BaseIOSchema, parents: List[str], unique_id:str = None) -> Optional[ListModel]:
        """Buffer *params* and emit a batch once the threshold is hit.

        Parameters
        ----------
        params
            The incoming payload (ignored by this method beyond buffering).
        parents
            Parent message identifiers (unused in this agent).

        Returns
        -------
        Optional[ListModel]
            • ``None`` until *amount* messages have accumulated.
            • A :class:`ListModel` **containing exactly *amount* messages** once
              the buffer reaches or exceeds the threshold.  Surplus messages
              remain for the next batch.
        """
        # 1―stash the new message
        self._state.buffer.append(params)

        # 2―if we’ve hit the threshold, slice off one batch and emit it
        if len(self._state.buffer) >= self.config.amount:
            batch = self._state.buffer[: self.config.amount]
            self._state.buffer = self._state.buffer[self.config.amount :]  # keep leftovers for the future

            processed: List[BaseIOSchema] = [self.call_advanced_run(msg, unique_id) for msg in batch]
            return ListModel(data=processed)

        # not enough yet → nothing to return
        return None

    # --------------------------------------------------------------------
    # Hook for subclasses
    # --------------------------------------------------------------------
    def run(self, params: BaseIOSchema) -> BaseIOSchema:  # noqa: D401 (simple return)
        """Identity transform.

        Override to transform each message *before* it goes into the batch.
        """
        return params
