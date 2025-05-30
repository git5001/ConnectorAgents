import os
from pydantic import BaseModel

from atomic_agents.lib.base.base_tool import BaseToolConfig
from AgentFramework.core.ConnectedAgent import ConnectedAgent


class LoadTextInput(BaseModel):
    """
    Input schema with the file path to read from.
    """
    file_path: str


class LoadTextOutput(BaseModel):
    """
    Output schema containing the loaded text.
    """
    text: str


class LoadTextAgent(ConnectedAgent):
    """
    Agent that loads text from a file given in the input.

    Attributes:
        input_schema (Type[BaseModel]): Input with file path.
        output_schema (Type[BaseModel]): Output with loaded text.
    """
    input_schema = LoadTextInput
    output_schema = LoadTextOutput

    def __init__(self, config: BaseToolConfig = BaseToolConfig(), **kwargs) -> None:
        super().__init__(config, **kwargs)

    def run(self, params: LoadTextInput) -> LoadTextOutput:
        """
        Reads text from the file path provided in input.

        Args:
            params (LoadTextInput): Input with file_path.

        Returns:
            LoadTextOutput: Output containing the text.
        """
        if not os.path.exists(params.file_path):
            raise FileNotFoundError(f"File not found: {params.file_path}")

        with open(params.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return LoadTextOutput(text=text)
