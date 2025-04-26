import os
from typing import List, Dict, Union, Type, Tuple

from pydantic import BaseModel, TypeAdapter, Field
from rich.console import Console

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ListCollectionAgent import ListModel
from AgentFramework.NullSchema import NullSchema
from agent_logging import rich_console

class MultiOutSimpleAgentConfig(BaseToolConfig):
    """
    Configuration for CountNumbers from to to from-1.
    """
    pass

class MultiOutSimpleAgentSchema1(BaseIOSchema):
    """
    Output number, the counted bumber.
    """
    number: int = Field(..., description="Current number.",)

class MultiOutSimpleAgentSchema2(BaseIOSchema):
    """
    Output number, the counted bumber.
    """
    id: str = Field(..., description="Current string.",)

class MultiOutSimpleAgentState(BaseModel):
    """
    State of the list agent.
    """
    count: int = Field(..., description="Dummy counter")


class MultiOutSimpleAgent2(ConnectedAgent):
    """
    An agent that just creates a sequence of numbers from from..to-1.
    Use for debugging. It needs no input and autostarts. it stops
    as inactive immediately after creating numbers.


    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Output schema.
    """
    input_schema = MultiOutSimpleAgentSchema1
    output_schemas = [MultiOutSimpleAgentSchema1, MultiOutSimpleAgentSchema2]
    state_schema = MultiOutSimpleAgentState

    def __init__(self, config: BaseToolConfig, uuid:str = 'default') -> None:
        """
        Initializes the Agent with a configuration.
        """
        super().__init__(config, uuid)
        self._config = config
        self._state = MultiOutSimpleAgentState(count=-1)



    def run(self, params: MultiOutSimpleAgentSchema1) ->  Union[
                                                MultiOutSimpleAgentSchema1,
                                                MultiOutSimpleAgentSchema2,
                                                Tuple[MultiOutSimpleAgentSchema1, MultiOutSimpleAgentSchema2]]:
        """
        Runs the agent.

        """
        print("Called multi2 ",type(params), self._state.count)
        if self._state.count < 0:
            self._state.count = params.number
            print("Start counter ",self._state.count)
        elif self._state.count > 0:
            self._state.count -= 1
        cnt = self._state.count
        print("Multi2 agent putput ",cnt)
        if cnt > 0:
            return MultiOutSimpleAgentSchema1(number=cnt)
        else:
            print("Loop done")
            return MultiOutSimpleAgentSchema2(id=f"{cnt+10}")
