from typing import Type, List, Dict, Optional, Callable, Union, Tuple
from pydantic import BaseModel
from rich.console import Console
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ToolPort import ToolPort
from AgentFramework.listutil import compare_lists, longest_common_sublist
from agent_logging import rich_console
from util.SchedulerException import SchedulerException
import re

class MultiPortAggregatorAgent(ConnectedAgent):
    """
    An agent that receives inputs from multiple ports and combines related messages into a single output.
    Identifies messages from the same parent (based on UUIDs) or same run index and processes them together.
    It is a bit heuristic to gather the correct inputs together. Especially if they stem from different parents.

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Defines multiple input schemas.
        output_schema (Type[BaseModel]): Defines the expected output schema.
        _input_ports (Dict[str, ToolPort]): Manages multiple input ports.
        output_port (ToolPort): Handles outgoing processed messages.
    """
    input_schemas: Dict[str, Type[BaseModel]] = {}
    output_schema: Type[BaseModel] = None
    input_schema: BaseModel = BaseIOSchema

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
            name: ToolPort(ToolPort.Direction.INPUT, schema, f"{uuid}:{self.__class__.__name__}")
            for name, schema in self.input_schemas.items()
        }
        self.output_port: ToolPort = ToolPort(ToolPort.Direction.OUTPUT, self.output_schema, f"{uuid}:{self.__class__.__name__}")

    @property
    def input_port(self):
        raise AttributeError("Use 'input_ports' (plural) instead of 'input_port' in multi agents")

    @input_port.setter
    def input_port(self, value):
        raise AttributeError("Setting 'input_port' in multi agents is not allowed. Use 'input_ports' instead.")

    def queque_size(self):
        return ", ".join(str(len(port.queue)) for name, port in self._input_ports.items())


    def _find_input_port(self, source_port_name:str = None):

        """
        Finds a port with a given name or the default port if no name
        :param source_port_name: Name of port or none
        :return: The port
        """
        if not source_port_name:
            raise NotImplementedError("Multi port must provide source port.")
        return self._input_ports[source_port_name]

    @staticmethod
    def extract_parents_with_suffix(parents: str) -> Dict[str,Tuple[str]]:
        """
        Extract UUIDs from a list of strings like UUID:idx1:idx2,
        where idx2 > 1. Return a dict mapping UUID -> "idx1:idx2".
        """
        result = {}
        pattern = re.compile(r"^(.*):(\d+):(\d+)$")

        for p in parents:
            match = pattern.match(p)
            if match:
                uuid, idx1, idx2 = match.groups()
                if int(idx2) > 1:
                    result[p] = (uuid,idx1,idx2)

        return result

    # DEPRECATED
    def _find_parent_indices_1(self):
        parent_indices = None
        # Loop all elements of the first port queque
        first_port_name = next(iter(self._input_ports))
        for anchor_index, (anchor_parents, _) in enumerate(self._input_ports[first_port_name].queue):
            # Get all multi items like xxx:1:127, returns parent:(uuid,idx,len) with parent = uuid:idx:lex
            chain_parents: Dict[str, Tuple[str]] = MultiPortAggregatorAgent.extract_parents_with_suffix(anchor_parents)

            parent_indices = []
            # Loop all other ports and try to find matching indices
            for port_name in self._input_ports:
                if port_name == first_port_name:
                    parent_indices.append(anchor_index)
                    continue
                parent_idx = None
                # Loop the whole queque of the  search port
                for idx, (parents, models) in enumerate(self._input_ports[port_name].queue):
                    matching_parents = 0
                    # Loop all uuids of the anchor
                    for chain_parent, chain_parent_tuple in chain_parents.items():
                        chain_uuid, chain_idx, chain_len = chain_parent_tuple
                        search = f":{chain_len}"
                        # Check if any parent of the seach ends with the search len - we just match the same length
                        # if we match uuid as well then we can never combine queques of different parents. Here we match
                        # same sized elements
                        for parent in parents:
                            if parent.endswith(search):
                                matching_parents += 1
                    # Ideally we have only one collection item in anchor and search queque but it could be more
                    if matching_parents == len(chain_parents):
                        parent_idx = idx
                        break
                # We store which index we found in teh current queque, we take the first matching one
                if parent_idx is not None:
                    parent_indices.append(parent_idx)
            if len(parent_indices) == len(self._input_ports):
                break
        return parent_indices

    def _find_parent_indices_2(self):
        parent_indices = None

        first_port_name = next(iter(self._input_ports))
        first_port = self._input_ports[first_port_name]

        # Iterate over candidates in the first port's queue.
        for anchor_index, (anchor_parents, _) in enumerate(first_port.queue):
            # Extract the chain parents from the anchor and precompute search keys.
            chain_parents = MultiPortAggregatorAgent.extract_parents_with_suffix(anchor_parents)
            search_keys = {f":{chain_idx}:{chain_len}" for _, (_, chain_idx, chain_len) in chain_parents.items()}

            parent_indices = []

            # Loop through each port to find a matching candidate.
            for port_name, port in self._input_ports.items():
                # For the first port, simply use the current anchor index.
                if port_name == first_port_name:
                    parent_indices.append(anchor_index)
                    continue

                found_idx = None
                # Iterate through the current port's queue.
                for idx, (parents, models) in enumerate(port.queue):
                    # Precompute the set of suffixes from each parent in the candidate.
                    # candidate_suffixes = {parent[parent.rfind(":"):] for parent in parents if ":" in parent}
                    candidate_suffixes = {
                        ":" + ":".join(parent.split(":")[1:])
                        for parent in parents if parent.count(":") >= 2
                    }

                    # Check if candidate has all required search keys.
                    if search_keys.issubset(candidate_suffixes):
                        found_idx = idx
                        break
                if found_idx is not None:
                    parent_indices.append(found_idx)
                else:
                    # If any port doesn't provide a matching candidate, move on to the next anchor.
                    break

            # If a matching candidate is found in all ports, exit the search.
            if len(parent_indices) == len(self._input_ports):
                break
        return parent_indices

    def step(self) -> bool:
        """
        Processes messages only when all input ports have received data.
        Identifies matching parent messages and combines them into one output.

        Returns:
            bool: True if processing occurred, False otherwise.
        """

        if any(not port.queue for port in self._input_ports.values()):
            return False  # Exit if any queue is empty
        # rich_console.print(f"[blue]Running multi agent  {self.__class__.__name__}[/blue]")


        # Look up queque indices for all ports
        parent_indices = self._find_parent_indices_2()



        if len(parent_indices) != len(self._input_ports):
            return False

        parent_map = {}
        model_map = {}
        staged_removals = []

        for port_name, index in zip(self._input_ports, parent_indices):
            port = self._input_ports[port_name]
            parents, model = port.queue[index]

            # Store in the map for later processing
            parent_map[port_name] = parents
            model_map[port_name] = model

            # Stage the removal in case we need to roll back
            staged_removals.append((port_name, index, (parents, model)))

            # Remove from queue
            del port.queue[index]

        #  Finds the longest common prefix among lists stored in a dictionary. Return a list of the matching per port
        join_parents = longest_common_sublist(parent_map)


        try:
            output_msg = self.run(model_map)
        except Exception as e:
            for del_name, del_idx, removed_item in staged_removals:
                self._input_ports[del_name].queue.insert(del_idx, removed_item)
            # Raise a scheduler-specific exception for higher-level handling
            raise SchedulerException(self.__class__.__name__, "Processing multi step failed", e)

        if output_msg:
            self.output_port.send(output_msg, join_parents)

        return True




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