from pydantic import BaseModel

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent

class IdentityAgent(ConnectedAgent):
    """
    An agent that forwards input to output as is.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Always NullSchema, indicating no output.
    """
    input_schema = BaseIOSchema
    output_schema = BaseIOSchema

    def __init__(self, config: BaseToolConfig = BaseToolConfig(),  **kwargs) -> None:
        """
        Initializes the IdentityAgent.
        """
        super().__init__(config, **kwargs)


    def run(self, params: BaseModel) -> BaseModel:
        """
        Outputs the input.

        Args:
            params (BaseModel): The input parameters for the message.

        Returns:
            BaseModel: The output containing the message.
        """
        return params


