from typing import Union, Type, Tuple

from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.NullSchema import NullSchema


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


class MultiOutSimpleAgent(ConnectedAgent):
    """
    An agent that just creates a sequence of numbers from from..to-1.
    Use for debugging. It needs no input and autostarts. it stops
    as inactive immediately after creating numbers.


    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Output schema.
    """
    input_schema = NullSchema
    output_schemas = [MultiOutSimpleAgentSchema1, MultiOutSimpleAgentSchema2]
    state_schema = MultiOutSimpleAgentState

    def __init__(self, config: BaseToolConfig, uuid:str = 'default') -> None:
        """
        Initializes the Agent with a configuration.
        """
        super().__init__(config, uuid)
        self._config = config
        self._state = MultiOutSimpleAgentState(count=0)



    def run(self, params: NullSchema) ->  Union[
                                                MultiOutSimpleAgentSchema1,
                                                MultiOutSimpleAgentSchema2,
                                                Tuple[MultiOutSimpleAgentSchema1, MultiOutSimpleAgentSchema2]]:
        """
        Runs the agent.

        """
        cnt = self._state.count
        self._state.count += 1
        if cnt == 0:
            return MultiOutSimpleAgentSchema1(number=cnt)
        elif cnt == 1:
            return MultiOutSimpleAgentSchema2(id=f"{cnt}")
        else:
            return (MultiOutSimpleAgentSchema1(number=cnt), MultiOutSimpleAgentSchema2(id=f"{cnt}"))
