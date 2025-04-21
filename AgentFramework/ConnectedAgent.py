import traceback
from typing import Type, List, Optional, Dict, Callable, Union

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from pydantic import BaseModel

from AgentFramework.NullSchema import NullSchema
from AgentFramework.ToolPort import ToolPort
from util.SchedulerException import SchedulerException
from util.SerializeHelper import decode_payload, encode_payload


class ConnectedAgent(BaseTool):
    """
    A specialized agent that extends the BaseTool framework by adding input and output ports.
    This allows agents to be connected, enabling automatic information flow between them.

    Attributes:
        input_schema (Type[BaseModel]): Defines the expected input schema for the agent.
        output_schema (Type[BaseModel]): Defines the expected output schema for the agent.
        input_port (ToolPort): Handles incoming messages.
        output_port (ToolPort): Handles outgoing messages.
    """
    # Static
    input_schema: Type[BaseModel] = None
    output_schema: Type[BaseModel] = None
    state_schema: Optional[Type[BaseModel]] = None
    is_active: bool = True
    uuid: str = "default"
    _debug:bool = False

    # Dynamic
    _state: Optional[BaseModel] = None
    _input_port: ToolPort
    _output_port: ToolPort

    def __init__(self, config: BaseToolConfig = BaseToolConfig(), uuid = 'default', create_ports:bool = True) -> None:
        """
        Initializes a ConnectedAgent instance with input and output ports.

        Args:
            config (BaseToolConfig, optional): Configuration for the agent. Defaults to BaseToolConfig().

        Raises:
            TypeError: If `input_schema` or `output_schema` is not defined in a subclass.
        """
        super().__init__(config)
        self._debug = False
        if not self.input_schema or not self.output_schema:
            raise TypeError("Each ConnectedAgent subclass must define `input_schema` and `output_schema`.")
        self.uuid = uuid

        if create_ports:
            self._input_port: ToolPort = ToolPort(ToolPort.Direction.INPUT, self.input_schema, f"{uuid}:{self.__class__.__name__}")
            self._output_port: ToolPort = ToolPort(ToolPort.Direction.OUTPUT, self.output_schema, f"{uuid}:{self.__class__.__name__}")


    @property
    def input_port(self):
        return self._input_port

    @input_port.setter
    def input_port(self, value):
        self._input_port = value

    @property
    def output_port(self):
        return self._output_port

    @output_port.setter
    def output_port(self, value):
        self._output_port = value

    def queque_size(self):
        return f"{len(self._input_port.queue)}"


    def connectTo(self,
                  target_agent: "ConnectedAgent",
                  transformer: Optional[Callable[[BaseModel], Union[BaseModel, List[BaseModel]]]] = None,
                  target_port_name:str = None,
                  condition: Optional[Callable[[BaseModel], bool]] = None,
                  ) -> None:
        """
        Connects an output port to an input port with an optional transformer.

        Args:
            target_agent (ConenctedAgent): The target agent.
            transformer (Optional[Callable[[BaseModel], Union[BaseModel, List[BaseModel]]]], optional):
                A function to transform messages before sending. Defaults to None.
            target_port_name(str): optional target port name, for multi port agents
            condition(Optional[Callable[BaseModel]]): A condition function

        Raises:
            ValueError: If an invalid port direction is used.
        """

        input_port = target_agent._find_input_port(target_port_name)
        output_port = self._find_output_port()
        output_port.connect(input_port, transformer=transformer, source=self, target=target_agent, condition=condition)

    def _find_input_port(self, source_port_name:str = None):

        """
        Finds a port with a given name or the default port if no name
        :param source_port_name: Name of port or none
        :return: The port
        """
        if not source_port_name:
            return self._input_port
        raise NotImplementedError("Multi port not implemented ")

    def _find_output_port(self):

        """
        Finds a port with a given name
        :return: The port
        """
        return self.output_port

    def feed(self, message: BaseIOSchema) -> None:
        """
        Feeds an agent explictely with an input message.

        Args:
            message (BaseIOSchemw): The message to feed.
        """
        self.input_port.receive(message,[])

    def clear_final_outputs(self) -> None:
        """
        Clear remaining data
        """
        self.output_port.unconnected_outputs.clear()

    def get_final_outputs(self) -> List[BaseModel]:
        """
        Retrieves and clears all stored outputs from the output port.

        Returns:
            List[BaseModel]: A list of all final output messages.
        """
        return self.output_port.get_final_outputs()

    def get_one_output(self) -> Optional[BaseModel]:
        """
        Retrieves and removes one message from the output port.

        Returns:
            Optional[BaseModel]: A single output message, or None if no messages are available.
        """
        return self.output_port.get_one_output()

    def step(self) -> bool:
        """
        Processes one message from the input queue and sends output if applicable.
        If no input is available, returns False, allowing a scheduler to halt execution.

        Returns:
            bool: True if processing occurred, False otherwise.
        """
        # Check if we have a full queque or we are an inital agent which acts as source
        if self.input_port.queue or self.input_schema is NullSchema:
            if self.input_schema is NullSchema:
                parents, input_msg = [], NullSchema()
            else:
                parents, input_msg = self.input_port.queue.popleft()
            # rich_console.print(f"   [blue]Running connected agent {self.__class__.__name__}[/blue] parents={len(parents)}")
            try:
                output_msg = self.process(input_msg, parents)
            except Exception as e:
                # Push the message back to the front of the queue to preserve order
                self.input_port.queue.appendleft((parents, input_msg))
                # Raise a scheduler-specific exception for higher-level handling
                raise SchedulerException(self.__class__.__name__, "Processing step failed", e)

            if output_msg and not isinstance(output_msg, NullSchema):
                result_parents = self.output_port.send(output_msg, parents)
            return True
        else:
            # rich_console.print(f"   [grey53]No input for connected agent {self.__class__.__name__}[/grey53]")
            pass
        return False

    def process(self, params: BaseIOSchema, parents: List[str]) -> BaseIOSchema:
        """
        Processes an input message by calling the `run` method, which subclasses should override.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.
            parents (List[str]): A list of parent agent identifiers for tracking message flow.

        Returns:
            BaseIOSchema: The processed output message.
        """
        return self.run(params)


    def save_state_to_file(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.save_state(), f, indent=2)

    def load_state_from_file(self, path: str):
        import json
        with open(path, "r") as f:
            self.load_state(json.load(f))

    def _gather_ports(self) -> Dict[str, ToolPort]:
        """
        Return a dict of all ports this agent has.
        Default: one input port and one output port.
        Child classes can override to return more or fewer ports.
        Keys in the returned dict will be used as keys in the saved state.
        Ports that are None will be excluded.
        """
        ports = {}
        if hasattr(self, "_input_port") and self._input_port is not None:
            ports["input_port"] = self._input_port
        if hasattr(self, "_output_port") and self._output_port is not None:
            ports["output_port"] = self._output_port
        return ports



    def save_state(self) -> dict:
        """
        Save the agent's state to a dictionary, including all ports returned by _gather_ports().
        """
        state_dict = {
            "state": encode_payload(self._state) if self._state else None,
            "ports": {}
        }


        # Dump each port the agent knows about
        for port_name, port_obj in self._gather_ports().items():
            state_dict["ports"][port_name] = self._dump_port(port_obj)

        return state_dict


    def load_state(self, state_dict: dict):
        """
        Load the agent's state from a dictionary, including all ports returned by _gather_ports().
        """
        # Restore the agent's private state
        if state_dict.get("state") and self.state_schema:
            try:
                real_data = decode_payload(state_dict["state"])  # might become a dict or a raw object
            except Exception as e:
                print("Failed to decode state", e)
                print("  Error state", state_dict["state"])
                raise SchedulerException(self.__class__.__name__, "Failed to decode state", e)
            if isinstance(real_data, dict) and self.state_schema is not None:
                # Attempt to parse with pydantic
                try:
                    self._state = self.state_schema(**real_data)
                except Exception as e:
                    print(f"[safe_model_load] Could not parse schema: {e}")
                    return (None, None)
            else:
                self._state = real_data

        # Load data for each known port
        for port_name, port_obj in self._gather_ports().items():
            port_state = state_dict.get("ports", {}).get(port_name, {})
            # Figure out which schema to use for this particular port
            #   (some child classes might use multiple schemas)
            # By default, assume input port uses self.input_schema, output port uses self.output_schema
            if port_name == "input_port":
                schema = self.input_schema
            elif port_name == "output_port":
                schema = self.output_schema
            else:
                # Could be a child-defined custom port
                # Child class might override load_state() or we can
                # detect the schema from the child's dictionary
                schema = None

            self._load_port(port_obj, port_state, schema)


    def _dump_port(self, port: Optional['ToolPort']) -> dict:
        """
        Helper that serializes a single ToolPort instance.
        port.queue is a deque of (List[str], BaseModel) or (msg_ids, msg).
        """
        if not port:
            return {"queue": [], "unconnected_outputs": []}

        def safe_model_dump(item):
            """
            item: (msg_ids, msg)
            We pass the raw BaseModel to `encode_payload` so it
            can mark it as _pydantic:True if needed.
            """
            try:
                msg_ids, msg = item
            except ValueError:
                return (None, None)

            if not msg:
                return (msg_ids, None)

            try:
                encoded = encode_payload(msg)
                return (msg_ids, encoded)
            except Exception as e:
                print(f"[safe_model_dump] Failed to encode message: {e}")
                traceback.print_exc()
                return (None, None)

        return {
            "queue": [safe_model_dump(item) for item in getattr(port, "queue", []) or []],
            "unconnected_outputs": [
                safe_model_dump(item) for item in getattr(port, "unconnected_outputs", []) or []
            ],
        }

    def _load_port(self, port: Optional['ToolPort'], port_state: dict, schema: Optional[Type[BaseModel]]) -> None:
        """
        Helper that deserializes a single ToolPort instance from its state.
        """
        if not port:
            return

        def safe_model_load(msg_ids, payload, schema):
            """
            - decode_payload(...) returns either a dict, a Pydantic model,
              or something else (like a pickled object).
            - If it's a dict and `schema` is not None, try building a new Pydantic model.
            """
            if not payload:
                return (msg_ids, None)

            real_data = decode_payload(payload)  # might become a dict or a raw object

            if isinstance(real_data, dict) and schema is not None:
                # Attempt to parse with pydantic
                try:
                    return (msg_ids, schema(**real_data))
                except Exception as e:
                    print(f"[safe_model_load] Could not parse schema: {e}")
                    return (None, None)
            else:
                # It's either a fully rebuilt Pydantic model already,
                # or an unpickled object, or a plain type
                return (msg_ids, real_data)

        # Clear existing port data
        port.queue.clear()
        port.unconnected_outputs.clear()

        # Reload queue
        for msg_ids, payload in port_state.get("queue", []):
            loaded = safe_model_load(msg_ids, payload, schema)
            if loaded != (None, None):
                port.queue.append(loaded)

        # Reload unconnected outputs
        for msg_ids, payload in port_state.get("unconnected_outputs", []):
            loaded = safe_model_load(msg_ids, payload, schema)
            if loaded != (None, None):
                port.unconnected_outputs.append(loaded)