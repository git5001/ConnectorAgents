import os
from typing import List, Dict

from pydantic import BaseModel, TypeAdapter, Field
from rich.console import Console

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ListCollectionAgent import ListModel
from AgentFramework.NullSchema import NullSchema
from agent_logging import rich_console

class CountNumbersAgentConfig(BaseToolConfig):
    """
    Configuration for CountNumbers from to to from-1.
    """
    from_number: int = Field(..., description="Start number.",)
    to_number: int = Field(..., description="Stop number (exclusive).",)


class CountNumbersAgentSchema(BaseIOSchema):
    """
    Output number, the counted bumber.
    """
    number: int = Field(..., description="Current number.",)



class CountNumbersAgent(ConnectedAgent):
    """
    An agent that just creates a sequence of numbers from from..to-1.
    Use for debugging. It needs no input and autostarts. it stops
    as inactive immediately after creating numbers.


    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Output schema.
    """
    input_schema = NullSchema
    output_schema = CountNumbersAgentSchema

    def __init__(self, config: BaseToolConfig, uuid:str = 'default') -> None:
        """
        Initializes the Agent with a configuration.
        """
        super().__init__(config, uuid)
        self._config = config


    def run(self, params: NullSchema) -> CountNumbersAgentSchema:
        """
        Runs the agent.

        """
        result = []
        # Create list of numbers between from and to-1, return as list to
        # create parallel qeuque
        for cnt in range(self._config.from_number, self._config.to_number):
            result.append( CountNumbersAgentSchema(number = cnt))
        self.is_active = False
        return result