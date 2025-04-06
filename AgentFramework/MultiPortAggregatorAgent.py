from typing import Type, List, Dict, Optional, Callable, Union
from pydantic import BaseModel
from rich.console import Console
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ToolPort import ToolPort
from AgentFramework.listutil import compare_lists, longest_common_sublist
from util.SchedulerException import SchedulerException

rich_console = Console()

class MultiPortAggregatorAgent(ConnectedAgent):
    """
    An agent that receives inputs from multiple ports and combines related messages into a single output.
    Identifies messages from the same parent (based on UUIDs) and processes them together.
    Note: How this collects list items to inputs is a bit counterintuitive and might need a better solution!

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Defines multiple input schemas.
        output_schema (Type[BaseModel]): Defines the expected output schema.
        _input_ports (Dict[str, ToolPort]): Manages multiple input ports.
        output_port (ToolPort): Handles outgoing processed messages.
    """
    input_schemas: Dict[str, Type[BaseModel]] = {}
    output_schema: Type[BaseModel] = None
    input_schema: BaseModel = BaseIOSchema()

    def __init__(self, config: BaseToolConfig = BaseToolConfig(), uuid:str = 'default') -> None:
        """
        Initializes the MultiPortAggregatorAgent.

        Args:
            config (BaseToolConfig): Configuration settings for the agent.

        Raises:
            TypeError: If input_schemas or output_schema are not defined.
        """
        super().__init__(config, uuid=uuid, create_ports=False)

        if not self.input_schemas or not self.output_schema:
            raise TypeError("CollectorAgent must define `input_schemas` and `output_schema`.")

        # Create input ports dynamically based on `input_schemas`
        self._input_ports: Dict[str, ToolPort] = {
            name: ToolPort(ToolPort.Direction.INPUT, schema, self.__class__.__name__)
            for name, schema in self.input_schemas.items()
        }
        self.output_port: ToolPort = ToolPort(ToolPort.Direction.OUTPUT, self.output_schema, self.__class__.__name__)

    @property
    def input_port(self):
        raise AttributeError("Use 'input_ports' (plural) instead of 'input_port' in multi agents")

    @input_port.setter
    def input_port(self, value):
        raise AttributeError("Setting 'input_port' in multi agents is not allowed. Use 'input_ports' instead.")

    def _find_input_port(self, source_port_name:str = None):

        """
        Finds a port with a given name or the default port if no name
        :param source_port_name: Name of port or none
        :return: The port
        """
        if not source_port_name:
            raise NotImplementedError("Multi port must provide source port.")
        return self._input_ports[source_port_name]


    def step(self) -> bool:
        """
        Processes messages only when all input ports have received data.
        Identifies matching parent messages and combines them into one output.

        Returns:
            bool: True if processing occurred, False otherwise.
        """
        if any(not port.queue for port in self._input_ports.values()):
            return False  # Exit if any queue is empty
        rich_console.print(f"[blue]Running multi agent  {self.__class__.__name__}[/blue]")

        # We'll track every (del_name, del_idx, del_item) here, so we can restore if needed
        staged_removals = []  # will hold (port_name, index, removed_item) for each port

        for port_name in self._input_ports:
            last_parent, last_model = self._input_ports[port_name].queue[-1]

            match = {}
            removed_indices = {}
            parent_map = {}

            for current_port_name in self._input_ports:
                if current_port_name == port_name:
                    continue # Skip the same queue

                for idx, (parents, model) in enumerate(self._input_ports[current_port_name].queue):
                    # rich_console.print(f"[blue]Running multi agent {current_port_name} {self.__class__.__name__}[/blue] parents={len(parents)}")

                    if compare_lists(last_parent, parents):
                        match[current_port_name] = model
                        removed_indices[current_port_name] = idx
                        parent_map[current_port_name] = parents
                        break

            if len(match) == len(self._input_ports) - 1:
                match[port_name] = last_model
                parent_map[port_name] = last_parent
                removed_indices[port_name] = len(self._input_ports[port_name].queue) - 1

                for del_name, del_idx in removed_indices.items():
                    removed_item = self._input_ports[del_name].queue[del_idx]  # safely get BEFORE delete
                    staged_removals.append((del_name, del_idx, removed_item))
                    del self._input_ports[del_name].queue[del_idx]

                # Sort by descending index (so removing one doesn’t shift the others)
                # staged_removals.sort(key=lambda x: x[1], reverse=True)
                # for del_name, del_idx, del_item in staged_removals:
                #     del self._input_ports[del_name].queue[del_idx]

                join_parents = longest_common_sublist(parent_map)

                try:
                    output_msg = self.run(match)
                except Exception as e:
                    for del_name, del_idx, removed_item in staged_removals:
                        self._input_ports[del_name].queue.insert(del_idx, removed_item)
                    # Raise a scheduler-specific exception for higher-level handling
                    raise SchedulerException(self.__class__.__name__, "Processing multi step failed", e)

                if output_msg:
                    self.output_port.send(output_msg, join_parents)

                return True

        return False

    def _gather_ports(self) -> Dict[str, ToolPort]:
        """
        Override to return all input ports plus the single output port
        under whichever keys we prefer. Something like:

        {
          "<input_port_name>": <ToolPort object>,
          ...
          "output_port": <ToolPort>
        }
        """
        all_ports = {}
        # Add each input port by its dictionary key
        for in_name, port in self._input_ports.items():
            all_ports[f"input_{in_name}"] = port  # or just use in_name directly
        # Add the single output port
        all_ports["output_port"] = self.output_port
        return all_ports

    def load_state(self, state_dict: dict):
        """
        Optionally override load_state() if you want fine-grained control of
        which schema is used for each input port.

        By default, we'll rely on the parent's load_state but
        we need a consistent way to pick the correct schema for each input port.
        """
        # 1) restore internal state
        if state_dict.get("state") and self.state_schema:
            self._state = self.state_schema(**state_dict["state"])

        # 2) For each port, load from state but pick the right schema
        for port_name, port_obj in self._gather_ports().items():
            port_state = state_dict.get("ports", {}).get(port_name, {})

            # Figure out which schema to use. If it's an input, we parse out the original in_name.
            if port_name.startswith("input_"):
                in_name = port_name.replace("input_", "")
                schema = self.input_schemas.get(in_name)
            else:
                # Must be the output port
                schema = self.output_schema

            self._load_port(port_obj, port_state, schema)