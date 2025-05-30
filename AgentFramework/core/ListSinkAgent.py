from pydantic import BaseModel

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent


class ListSinkAgent(ConnectedAgent):
    """
    An agent that collets a complete list input and forwards it once collected.

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
        super().__init__(config, collect_input=True, **kwargs)


    def run(self, params: BaseModel, unique_id:str = None ) -> BaseModel:
        """
        Outputs the input.

        Args:
            params (BaseModel): The input parameters for the message.

        Returns:
            BaseModel: The output containing the message.
        """
        return params


