from typing import List, Optional
from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.NullSchema import NullSchema


class FIFOStateAgentConfig(BaseToolConfig):
    """Configuration for FIFOStateAgent."""
    pass


class FIFOState(BaseModel):
    """
    State object that stores all received messages in a FIFO queue.

    Attributes:
        data (List[BaseModel]): Queue of messages (FIFO order).
    """

    data: List[BaseModel] = Field(
        default_factory=list,
        description="The message queue (FIFO) holding all received BaseModel instances.",
    )

    def push(self, item: BaseModel) -> None:
        """Append a new message to the back of the queue."""
        self.data.append(item)

    def pop(self) -> Optional[BaseModel]:
        """Remove and return the oldest message (front of the queue)."""
        if self.data:
            return self.data.pop(0)
        return None

    def peek(self) -> Optional[BaseModel]:
        """Return the oldest message without removing it from the queue."""
        if self.data:
            return self.data[0]
        return None

    def __len__(self) -> int:
        """Return the number of messages in the queue."""
        return len(self.data)


class FIFOStateAgent(ConnectedAgent):
    """
    An agent that stores all incoming messages in a FIFO queue without producing output.

    Typically used as a sink or buffer, this agent supports peeking at, popping, and
    listing the stored messages. It is useful for observing and debugging message
    flows or deferring message handling.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Always :class:`NullSchema`, indicating no output.
    """

    input_schema = BaseIOSchema
    output_schema = NullSchema
    _state: Optional[FIFOState] = None

    def __init__(self, config: FIFOStateAgentConfig = FIFOStateAgentConfig(), **kwargs) -> None:
        """Initialize the FIFOStateAgent with the given configuration and UUID."""
        super().__init__(config, **kwargs)

    def _ensure_state(self) -> None:
        """Ensure that the internal state is initialized."""
        if self._state is None:
            self._state = FIFOState()

    def push(self, item: BaseModel) -> None:
        """Push a message into the internal FIFO queue."""
        self._ensure_state()
        self._state.push(item)

    def pop(self) -> Optional[BaseModel]:
        """Pop and return the oldest message from the queue, if any."""
        if not self._state:
            return None
        return self._state.pop()

    def peek(self) -> Optional[BaseModel]:
        """Return the oldest message without removing it, if any."""
        if not self._state:
            return None
        return self._state.peek()

    def size(self) -> int:
        """Return the number of stored messages."""
        if not self._state:
            return 0
        return len(self._state)

    def run(self, params: BaseModel) -> BaseModel:
        """
        Accept input data and store it in the FIFO queue.

        Args:
            params (BaseModel): Input data.

        Returns:
            NullSchema: No output is produced.
        """
        self.push(params)
        return NullSchema()

    def getData(self):
        """Return a list of all stored messages in FIFO order."""
        if not self._state:
            return []
        return list(self._state.data)

    __len__ = size
    __iter__ = lambda self: iter(self.getData())
