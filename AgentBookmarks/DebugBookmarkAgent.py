from pydantic import BaseModel
from rich.console import Console

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentBookmarks.FirefoxBookmarkAgent import FirefoxBookmarksOutput
from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.NullSchema import NullSchema
from agent_logging import rich_console


class DebugBookmarkAgentConfig(BaseToolConfig):
    """
    Configuration for DebugAgent.
    """
    pass


class DebugBookmarkAgent(ConnectedAgent):
    """
    A debug agent

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Always NullSchema, indicating no output.
    """
    input_schema = FirefoxBookmarksOutput
    output_schema = NullSchema

    def __init__(self, config: DebugBookmarkAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes the DebugAgent.

        Args:
            config (DebugAgentConfig): Configuration for the agent.
        """
        super().__init__(config, uuid)

    def run(self, params: FirefoxBookmarksOutput) -> BaseModel:
        """
        Runs the debug agent, logging received data but not producing any output.

        Args:
            params (BaseModel): Input data received from the connected output port.

        Returns:
            NullSchema: Always returns a NullSchema to indicate no output.
        """
        length = len(params.bookmarks)
        rich_console.print(f"[bold red]Output of bookmark debug agent:[/bold red] [blue]{params.__class__.__name__}[/blue]")
        rich_console.print(f"[red]   - Data: len bookmarks = {length}[/red]")
        return NullSchema()  # Do not create or store output
