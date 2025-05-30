from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.NullSchema import NullSchema
from util.LLMSupport import LLMReply, ChatMessage


class Msg(BaseModel):
    """
    A simple test message schema.
    """
    text: str

    model_config = {
        "title": "Msg",
        "description": "A message with one text field."
    }

class CounterState(BaseModel):
    """
    Tracks how many times this agent ran.
    """
    count: int = 0

    model_config = {
        "title": "CounterState",
        "description": "Tracks how many times the agent has run."
    }


class EchoAgent(ConnectedAgent):
    input_schema = Msg
    output_schema = Msg
    state_schema = CounterState

    def __init__(self, uuid: str):
        super().__init__(config=BaseToolConfig(), uuid=uuid)

    def run(self, params: Msg) -> Msg:
        if self._state is None:
            self._state = CounterState()
        self._state.count += 1
        return params


# ---------------------------------------------------------------------------
# CheckAgent – consumes a single‑word answer ("correct" / "incorrect") from LLM‑2
# and stores a Boolean flag internally.  It returns NullSchema so nothing flows
# further down the graph, but its state can be inspected or gathered with
# get_final_outputs().
# ---------------------------------------------------------------------------
class CheckAgentState(BaseModel):
    """
    just check
    """
    is_correct: bool = False
    count: int = 0

class CheckAgent(ConnectedAgent):
    input_schema = LLMReply
    output_schema = NullSchema
    state_schema = CheckAgentState

    def __init__(self, uuid: str):
        super().__init__(config=BaseToolConfig(), uuid=uuid)

    def run(self, params: LLMReply) -> NullSchema:
        """Save whether the upstream LLM said 'correct' in lowercase."""
        message: ChatMessage = params.reply
        answer = message.text.strip().lower()
        print("Answer is ",answer)
        if self._state is None:
            self._state = CheckAgentState()
        self._state.count += 1
        self._state.is_correct = answer == "correct"
        # Returning NullSchema keeps the value internal but still lets the
        # scheduler collect an object if required.
        return NullSchema()
