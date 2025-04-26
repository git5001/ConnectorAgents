import os
from typing import List

from pydantic import BaseModel, TypeAdapter
from rich.console import Console

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.NullSchema import NullSchema
from agent_logging import rich_console


class SaveJsonAgentConfig(BaseToolConfig):
    """
    Configuration for SaveJsonAgent.
    Stores the filename where the JSON will be saved.
    """
    filename: str
    use_uuid: bool = False


class SaveJsonAgent(ConnectedAgent):
    """
    An agent that saves the received JSON data (a pydantic model)
    to a specified file. It uses a filename provided via its config.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Always NullSchema, indicating no output.
    """
    input_schema = BaseIOSchema
    output_schema = NullSchema

    def __init__(self, config: SaveJsonAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes the SaveJsonAgent with a configuration that
        includes the filename for saving JSON.
        """
        super().__init__(config, uuid)
        self.filename = config.filename
        self._data = None  # to store the last received data
        self._use_uuid = config.use_uuid

    def process(self, params: BaseIOSchema, parents: List[str]) -> BaseIOSchema:
        """
        Processes an input message by calling the `run` method, which subclasses should override.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.
            parents (List[str]): A list of parent agent identifiers for tracking message flow.

        Returns:
            BaseIOSchema: The processed output message.
        """
        return self.run_parents(params, parents)

    def run_parents(self, params: BaseModel, parents: List[str]) -> BaseModel:
        """
        Saves the incoming pydantic object to a JSON file.

        Args:
            params (BaseModel): Input data received from the connected output port.

        Returns:
            NullSchema: Indicates no further output.
        """
        # Store the data internally
        self._data = params

        if self._use_uuid:
            # cleaned_list = [item.split(':')[0] if ':' in item else item for item in parents]
            # parent = cleaned_list[-1]
            parent = parents[-1] or ""
            parent = parent.replace(":","_")
            name, ext = os.path.splitext(self.filename)
            new_filename = f"{name}_{parent}{ext}"
        else:
            new_filename = self.filename
        # Save the data to the specified JSON file
        self._save(new_filename)
        rich_console.print(f"[green]Saved JSON to {new_filename}[/green]")
        return NullSchema()

    def _save(self, filepath: str) -> None:
        """
        Save the agent's internal state (parsed data) to a JSON file.

        Args:
            filepath (str): The path to the JSON file.
        """
        if self._data is not None:
            # Ensure the directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path:  # skip if filepath is just a filename in the current directory
                os.makedirs(dir_path, exist_ok=True)

            # Create a TypeAdapter for the type of the data received
            adapter = TypeAdapter(type(self._data))
            # dump_json returns bytes; decode to get a string
            json_bytes = adapter.dump_json(self._data, indent=2)
            json_str = json_bytes.decode('utf-8') if isinstance(json_bytes, bytes) else json_bytes
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)


