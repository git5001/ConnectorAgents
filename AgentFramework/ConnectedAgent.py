from typing import Type, List, Optional
from rich.console import Console
from pydantic import BaseModel

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

from AgentFramework.NullSchema import NullSchema
from AgentFramework.ToolPort import ToolPort

rich_console = Console()

class ConnectedAgent(BaseTool):
    """
    A specialized agent that extends the BaseTool framework by adding input and output ports.
    This allows agents to be connected, enabling automatic information flow between them.

    Attributes:
        input_schema (Type[BaseModel]): Defines the expected input schema for the agent.
        output_schema (Type[BaseModel]): Defines the expected output schema for the agent.
        input_port (ToolPort): Handles incoming messages.
        output_port (ToolPort): Handles outgoing messages.
    """
    input_schema: Type[BaseModel] = None
    output_schema: Type[BaseModel] = None

    def __init__(self, config: BaseToolConfig = BaseToolConfig()) -> None:
        """
        Initializes a ConnectedAgent instance with input and output ports.

        Args:
            config (BaseToolConfig, optional): Configuration for the agent. Defaults to BaseToolConfig().

        Raises:
            TypeError: If `input_schema` or `output_schema` is not defined in a subclass.
        """
        super().__init__(config)
        if not self.input_schema or not self.output_schema:
            raise TypeError("Each ConnectedAgent subclass must define `input_schema` and `output_schema`.")

        self.input_port: ToolPort = ToolPort(ToolPort.Direction.INPUT, self.input_schema)
        self.output_port: ToolPort = ToolPort(ToolPort.Direction.OUTPUT, self.output_schema)

    def feed(self, message: BaseIOSchema) -> None:
        """
        Feeds an agent explictely with an input message.

        Args:
            message (BaseIOSchemw): The message to feed.
        """
        self.input_port.receive(message,[])

    def get_final_outputs(self) -> List[BaseModel]:
        """
        Retrieves and clears all stored outputs from the output port.

        Returns:
            List[BaseModel]: A list of all final output messages.
        """
        return self.output_port.get_final_outputs()

    def get_one_output(self) -> Optional[BaseModel]:
        """
        Retrieves and removes one message from the output port.

        Returns:
            Optional[BaseModel]: A single output message, or None if no messages are available.
        """
        return self.output_port.get_one_output()

    def step(self) -> bool:
        """
        Processes one message from the input queue and sends output if applicable.
        If no input is available, returns False, allowing a scheduler to halt execution.

        Returns:
            bool: True if processing occurred, False otherwise.
        """
        if self.input_port.queue:
            parents, input_msg = self.input_port.queue.popleft()
            rich_console.print(f"[blue]Running agent {self.__class__.__name__}[/blue] parents={parents}")
            output_msg = self.process(input_msg, parents)
            if output_msg and not isinstance(output_msg, NullSchema):
                self.output_port.send(output_msg, parents)
            return True
        return False

    def process(self, params: BaseIOSchema, parents: List[str]) -> BaseIOSchema:
        """
        Processes an input message by calling the `run` method, which subclasses should override.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.
            parents (List[str]): A list of parent agent identifiers for tracking message flow.

        Returns:
            BaseIOSchema: The processed output message.
        """
        return self.run(params)