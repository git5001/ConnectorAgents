from typing import List, Union
from pydantic import BaseModel



class DebugInterface:
    """
    Abstract interface for debugging agents in a pipeline.
    Implement this interface to receive callbacks on agent state and message flow.
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def init_debugger(self, timeout:int =30) -> None:
        """
        Init debugger, e.g. starts tcp thread and wait a bit.
        :param timeout: Optional timneout in init
        :return:
        """
        pass

    def exit_debugger(self) -> None:
        """
        Close debugger
        :return:
        """
        pass

    def no_input(self, agent: "ConnectedAgent") -> None:
        """
        Called when no input message is passed to the debugger.

        :param agent: The agent instance about to run.
        """
        pass

    def input(self, agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        """
        Called when an input message is passed to the debugger.

        :param agent: The agent instance about to run.
        :param msg: The incoming message, as a Pydantic BaseModel instance.
        :param parents: List of parent agent UUIDs for tracing message origin.
        """
        pass

    def output(self, agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        """
        Called when an output message is produced by an agent.

        :param agent: The agent instance about to run.
        :param msg: The outgoing message, as a Pydantic BaseModel instance.
        :param parents: List of parent agent UUIDs for tracing message flow.
        """
        pass

    def start_agent(self, agent: "ConnectedAgent", step_count: int) -> None:
        """
        Called just before an agent begins execution.

        :param agent: The agent instance about to run.
        :param step_count: The current scheduler step count.
        """
        pass

    def finished_agent(self, agent: "ConnectedAgent", step_count: int, did_run: bool) -> None:
        """
        Called after an agent finishes or idles.

        :param agent: The agent instance that just finished.
        :param step_count: The scheduler step count when finished.
        :param did_run: True if the agent executed, False if it idled.
        """
        pass

    def error_agent(self, agent: "ConnectedAgent", step_count: int, exception: Exception) -> None:
        """
        Called when an agent raises an exception during execution.

        :param agent: The agent instance that errored.
        :param step_count: The scheduler step count at error time.
        :param exception: The exception raised by the agent.
        """
        pass

    def is_pause(self, pause_count: int, step_counter: int) -> bool:
        """
        Determine whether the pipeline should pause at a given step.

        :param pause_count: The number of pauses already requested.
        :param step_counter: The current scheduler step count.
        :return: True if execution should pause, False otherwise.
        """
        pass

    def user_message(self,
                     name: str,
                     agent: "ConnectedAgent",
                     data: Union[BaseModel, str]) -> None:
        """
        Called when a user message is sent or received in the conversation.

        :param name: Identifier for the user message (e.g., 'id' or custom name).
        :param agent: The agent handling the user message.
        :param data: The message content, as a Pydantic BaseModel or raw string.
        """
        pass
