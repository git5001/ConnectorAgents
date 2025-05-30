from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel, Field

from AgentFramework.support.IfAgent import IfAgent, IfAgentConfig


class MyIfAgentConfig(BaseToolConfig):
    """
    Configuration for IfAgent.
    """
    pass

class MyIfAgentInputSchema(BaseModel):
    """
    Input schema for MyIfAgent.
    """
    condition: str = Field(..., description="A string used to decide the output schema.")
    message: str = Field(..., description="The message payload.")

class MyIfAgent(IfAgent):
    input_schema = MyIfAgentInputSchema
    output_schema = None

    def __init__(self, config: MyIfAgentConfig, **kwargs):
        self._myconfig = config
        schema_names = ["PositiveResponse", "NegativeResponse", "NeutralResponse"]
        config = IfAgentConfig(schemas=schema_names)
        super().__init__(config=config, **kwargs)


    def choice(self, params: MyIfAgentInputSchema) -> str:
        """
        Decide schema name based on the condition string.
        """
        print("CONT",params.condition, params.message)
        if params.condition == "positive":
            return "PositiveResponse"
        elif params.condition == "negative":
            return "NegativeResponse"
        else:
            return "NeutralResponse"
