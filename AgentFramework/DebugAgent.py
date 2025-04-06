from pydantic import BaseModel
from rich.console import Console

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.NullSchema import NullSchema

rich_console = Console()


class DebugAgentConfig(BaseToolConfig):
    """
    Configuration for DebugAgent.
    """
    pass


class DebugAgent(ConnectedAgent):
    """
    A debug agent that receives data from an output port but does not send any data out.
    Acts as a sink for debugging purposes.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Always NullSchema, indicating no output.
    """
    input_schema = BaseIOSchema
    output_schema = NullSchema

    def __init__(self, config: DebugAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes the DebugAgent.

        Args:
            config (DebugAgentConfig): Configuration for the agent.
        """
        super().__init__(config, uuid)

    def run(self, params: BaseModel) -> BaseModel:
        """
        Runs the debug agent, logging received data but not producing any output.

        Args:
            params (BaseModel): Input data received from the connected output port.

        Returns:
            NullSchema: Always returns a NullSchema to indicate no output.
        """
        rich_console.print("[bold red]Running debug agent[/bold red]")
        rich_console.print(f"[green]Debug data:[/green]\n[red]  - Type:[/red] [blue]{params.__class__.__name__}[/blue]\n[red]  - Data:[/red]",params)
        return NullSchema()  # Do not create or store output
