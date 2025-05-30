from typing import List, Dict

from pydantic import Field

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.CollectorPort import ListModel
from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.NullSchema import NullSchema


class CounterAgentConfig(BaseToolConfig):
    """
    Configuration for CounterAgent.
    Specifies which fields should be counted., e.g. 'count', 'overall'
    """
    counter_fields: List[str] = Field(..., description="List of field names to sum from the input data.")


class CounterSchema(BaseIOSchema):
    """
    Input schema for the counter. :List of counter names, e.g. count,overall.
    These fields will be generated and counter.
    """
    counts: Dict[str,int] = Field(..., description="Current count.",)

class CounterAgent(ConnectedAgent):
    """
    An agent that just counts a list input of counter types.
    Tyically a transformer sets the values of a coutner state and
    this agents sums them all up.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Output.
    """
    input_schema = BaseIOSchema
    output_schema = NullSchema

    def __init__(self, config: CounterAgentConfig, **kwargs) -> None:
        """
        Initializes the Agent with a configuration.
        """
        super().__init__(config, **kwargs)
        self._config = config


    def run(self, params: ListModel[CounterSchema]) -> CounterSchema:
        """
        Runs the agent.

        Args:
            params (MessageInput): The input parameters for the message.

        Returns:
            MessageOutput: The output containing success status and message.
        """
        current_counts = {field: 0 for field in self._config.counter_fields}
        for data in params.data:
            for field in self._config.counter_fields:
                current_counts[field] += data.counts.get(field, 0)
        return CounterSchema(counts = current_counts)