from typing import Type, List, Optional

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from openai.types import CompletionUsage
from pydantic import Field, BaseModel

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from util.AgentMemory import AgentMemory
from agent_logging import logger
from util.ChatSupport import ChatSupport, CallStep, Prompt
from util.LLMSupport import LLMAgentConfig, LLMModel, LLMRequest, LLMReply, ChatMessage


class LLmAgentState(BaseModel):
    """
    Agent state, keep memory
    """
    memory: AgentMemory = Field(..., description="The agent memory.")


class LLMAgent(ConnectedAgent):
    """
    An agent that calls OpenAI's LLMs to summarize and condense news articles.

    Attributes:
        input_schema (Type[BaseIOSchema]): Expected input schema.
        output_schema (Type[BaseIOSchema]): Expected output schema.
    """
    input_schema = LLMRequest
    output_schema = LLMReply
    state_schema =LLmAgentState
    _state: LLmAgentState = None

    def __init__(self, config: LLMAgentConfig) -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config)
        self._state = LLmAgentState(memory=AgentMemory())
        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()
        self._chat = ChatSupport({CallStep.DEFAULT:self.model})

    def getHistory(self, n: Optional[int] = None) -> List[dict]:
        """
        Obtain the state history of the agent
        :return: The history as list of dict
        """
        if not self._state:
            return None
        if not self._state.memory:
            return None
        return self._state.memory.get_history(n)

    def getFormattedHistory(self, n: Optional[int] = None):
        history = self.getHistory(n)
        return '\n'.join(f"{msg['role']}: {msg['content']}" for msg in history)

    def run(self, user_input: LLMRequest) -> LLMReply:
        """
        Processes the user input and returns a structured summary.
        If `DUMMY_LLM` is enabled, returns dummy data.

        Args:
            user_input (Optional[BaseIOSchema], optional): The input data. Defaults to None.

        Returns:
            BaseIOSchema: The processed response from the LLM.
        """

        idata = Prompt(step=CallStep.DEFAULT,
                       sysText=user_input.system,
                       prompt=user_input.user,
                       title="Test LLM call",
                       userMessage=None,
                       temperature=0.5)
        logger.info(f"Running llm ... {self.model.name()}")
        history = None
        if self.config.use_memory:
            history = self._state.memory.get_history()
        try:
            if self.config.use_response:
                result_object, fixed_dict, usage = self._chat.generate_json_pydantic(idata, history, None, "ChatResult", ChatMessage)
                result = LLMReply(usage=usage, reply=result_object, error=None)
                logger.info(f"LLM structured result {result.reply}")
            else:
                text,  usage = self._chat.generate_chat(idata, history)
                result = LLMReply(usage=usage, reply=ChatMessage(text=text), error=None)
                logger.info(f"LLM plain result {result.reply}")
        except Exception as e:
            logger.error(f"LLM Error {e}")
            result = LLMReply(usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0), reply=None, error=f"Error:{e}")

        # Memory?
        if self.config.use_memory:
            self._state.memory.add_message(role="user", content=ChatMessage(text=user_input.user))
            self._state.memory.add_message(role="assistant", content=result.reply)
            # h = self._state.memory.get_history()
            # print("History",h)


        return result

