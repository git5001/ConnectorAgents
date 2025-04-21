from typing import Type

from pydantic import BaseModel, TypeAdapter
from rich.console import Console

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from agent_logging import rich_console


class LoadJsonAgentConfig(BaseToolConfig):
    """
    Configuration for LoadJsonAgent.
    Stores the filename from which the JSON will be loaded.
    """
    filename: str
    model_class: Type[BaseModel]


class LoadJsonAgent(ConnectedAgent):
    """
    An agent that loads JSON data from a specified file and returns the deserialized
    pydantic model. It ignores any input provided to run().

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): The pydantic model loaded from JSON.
    """
    input_schema = BaseIOSchema
    output_schema = BaseModel  # This indicates that run() returns a pydantic BaseModel instance.

    def __init__(self, config: LoadJsonAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes the LoadJsonAgent with the provided configuration.

        Args:
            config (LoadJsonAgentConfig): Contains the filename to load from.
        """
        super().__init__(config, uuid)
        self.filename = config.filename
        self.model_class = config.model_class

    def run(self, params: BaseModel) -> BaseModel:
        """
        Loads the JSON data from the specified file and returns the deserialized model.
        The input parameter is ignored.

        Args:
            params (BaseModel): Input data (ignored).

        Returns:
            BaseModel: The pydantic model loaded from the JSON file.
        """
        loaded_data = self.load(self.filename, self.model_class)
        rich_console.print(f"[green]Loaded JSON from {self.filename}[/green]")
        print("Loaded ",loaded_data.__class__)
        return loaded_data

    def load(self, filepath: str, model_class: Type[BaseModel]) -> BaseModel:
        """
        Load and deserialize the JSON using the provided model class.

        Args:
            filepath (str): Path to the JSON file.
            model_class (Type[BaseModel]): Pydantic model to load into.

        Returns:
            BaseModel: The deserialized model instance.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = f.read()

        adapter = TypeAdapter(model_class)
        return adapter.validate_json(json_data)