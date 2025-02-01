from typing import Type, List, Dict, Optional
from pydantic import BaseModel

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ToolPort import ToolPort
from AgentFramework.listutil import compare_lists, longest_common_sublist


class MultiPortAggregatorAgent(ConnectedAgent):
    """
    An agent that receives inputs from multiple ports and combines related messages into a single output.
    Identifies messages from the same parent (based on UUIDs) and processes them together.
    Note: How this collects list items to inputs is a bit counterintuitive and might need a better solution!

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Defines multiple input schemas.
        output_schema (Type[BaseModel]): Defines the expected output schema.
        input_ports (Dict[str, ToolPort]): Manages multiple input ports.
        output_port (ToolPort): Handles outgoing processed messages.
    """
    input_schemas: Dict[str, Type[BaseModel]] = {}
    output_schema: Type[BaseModel] = None
    input_schema: BaseModel = BaseIOSchema()

    def __init__(self, config: BaseToolConfig = BaseToolConfig()) -> None:
        """
        Initializes the MultiPortAggregatorAgent.

        Args:
            config (BaseToolConfig): Configuration settings for the agent.

        Raises:
            TypeError: If input_schemas or output_schema are not defined.
        """
        super().__init__(config)

        if not self.input_schemas or not self.output_schema:
            raise TypeError("CollectorAgent must define `input_schemas` and `output_schema`.")

        # Create input ports dynamically based on `input_schemas`
        self.input_ports: Dict[str, ToolPort] = {
            name: ToolPort(ToolPort.Direction.INPUT, schema)
            for name, schema in self.input_schemas.items()
        }

        # Single output port (in parent)
        # self.output_port = ToolPort(ToolPort.Direction.OUTPUT, self.output_schema)

    def step(self) -> bool:
        """
        Processes messages only when all input ports have received data.
        Identifies matching parent messages and combines them into one output.

        Returns:
            bool: True if processing occurred, False otherwise.
        """
        if any(not port.queue for port in self.input_ports.values()):
            return False  # Exit if any queue is empty

        for name in self.input_ports:
            last_parent, last_model = self.input_ports[name].queue[-1]

            match = {}
            remove = {}
            parent_map = {}

            for work_name in self.input_ports:
                if work_name == name:
                    continue # Skip the same queue

                for idx, (parents, model) in enumerate(self.input_ports[work_name].queue):
                    if compare_lists(last_parent, parents):
                        match[work_name] = model
                        remove[work_name] = idx
                        parent_map[work_name] = parents
                        break

            if len(match) == len(self.input_ports) - 1:
                match[name] = last_model
                parent_map[name] = last_parent
                remove[name] = len(self.input_ports[name].queue) - 1

                for del_name, del_idx in remove.items():
                    del self.input_ports[del_name].queue[del_idx]

                join_parents = longest_common_sublist(parent_map)

                output_msg = self.run(match)

                if output_msg:
                    self.output_port.send(output_msg, join_parents)

                return True

        return False
