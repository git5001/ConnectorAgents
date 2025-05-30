import os
from typing import List, Optional, Callable

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.NullSchema import NullSchema


class SaveTextConfig(BaseToolConfig):
    """
    Configuration for SaveTextAgent.

    Attributes:
        file_path (str): File path where text will be saved.
        to_text_callback (Optional[Callable[[BaseIOSchema], str]]): Optional function to convert params to text.
    """
    file_path: str = Field(..., description="File path where text will be saved")
    to_text_callback: Optional[Callable[[BaseIOSchema], str]] = Field(
        default=None,
        description="Optional function to convert params to text"
    )
    use_counter: Optional[int] = Field(default=None,description="Enable automatic file counting"
    )

class SaveTextAgentState(BaseModel):
    """
    The local state of the worker.
    """
    counter: int = Field(..., description="File counter")

class SaveTextAgent(ConnectedAgent):
    """
    Agent that saves a text to a file.

    Attributes:
        input_schema (Type[BaseModel]): Expected input schema.
        output_schema (Type[BaseModel]): Output schema (null).
        file_path (str): Path to file for saving text.
    """
    input_schema = BaseIOSchema  # Accepts any BaseModel with a `to_text()` method
    output_schema = NullSchema
    _state: SaveTextAgentState = None

    def __init__(self, config: SaveTextConfig,  **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.file_path = config.file_path
        self._state = SaveTextAgentState(counter=0)

    def run(self, params: BaseIOSchema) -> NullSchema:
        """
        Writes the result of `params.to_text()` to the file path defined in config.

        Args:
            params (BaseModel): Input object that must implement `to_text()` method.

        Returns:
            NullSchema: Empty output schema.
        """

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        if self.config.to_text_callback:
            text = self.config.to_text_callback(params)
        else:
            if not hasattr(params, "to_text") or not callable(params.to_text):
                raise ValueError(f"Input {type(params)} must implement a 'to_text()' method or a callback must be provided")
            text = params.to_text()

        # Adjust the file path if counter is enabled
        file_path = self.file_path
        if self.config.use_counter is not None:
            base, ext = os.path.splitext(self.file_path)
            file_path = f"{base}_{self._state.counter}{ext}"
            self._state.counter += 1

        # Write the text to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        return NullSchema()
