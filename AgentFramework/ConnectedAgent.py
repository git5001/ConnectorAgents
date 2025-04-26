import traceback
from typing import Type, List, Optional, Dict, Callable, Union, Tuple

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from pydantic import BaseModel

from AgentFramework.InfiniteSchema import InfiniteSchema
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
        output_schemas (Type[BaseModel]): Defines the expected output schemas for the agent.
        input_port (ToolPort): Handles incoming messages.
        output_port (ToolPort): Handles outgoing messages.
    """
    # Static
    input_schema: Type[BaseModel] = None
    output_schema: Type[BaseModel] = None
    output_schemas: List[Type[BaseModel]] = None
    state_schema: Optional[Type[BaseModel]] = None
    is_active: bool = True
    uuid: str = "default"
    _debug:bool = False
    _debugger:"DebugInterface" = None
    _global_state: Optional["GlobalState"] = None  # Saved by the scheduler

    # Dynamic
    _state: Optional[BaseModel] = None
    _input_port: ToolPort
    _output_ports: Dict[Type[BaseModel], ToolPort]

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
        if not self.input_schema or not (self.output_schema or self.output_schemas):
            raise TypeError("Each ConnectedAgent subclass must define `input_schema` and `output_schema`.")
        self.uuid = uuid
        self._config = config

        if create_ports:
            self._output_ports = {}
            self._input_port: ToolPort = ToolPort(ToolPort.Direction.INPUT, self.input_schema, f"{uuid}:{self.__class__.__name__}")

            if self.output_schema is not None:
                self._output_ports[self.output_schema]: ToolPort = ToolPort(ToolPort.Direction.OUTPUT, self.output_schema, f"{uuid}:{self.__class__.__name__}")
            else:
                port_cnt = 0
                for output_schema in self.output_schemas:
                    self._output_ports[output_schema]: ToolPort = ToolPort(ToolPort.Direction.OUTPUT, output_schema,f"{uuid}:{self.__class__.__name__}#{port_cnt}")
                    port_cnt += 1


    @property
    def global_state(self) -> BaseModel:
        return self._global_state

    @global_state.setter
    def global_state(self, global_state: BaseModel) -> None:
        self._global_state = global_state

    @property
    def state(self) -> BaseModel:
        return self._state

    @state.setter
    def state(self, state: BaseModel) -> None:
        self._state = state

    @property
    def debugger(self) -> "DebugInterface":
        return self._debugger

    @debugger.setter
    def debugger(self, debugger: "DebugInterface") -> None:
        self._debugger = debugger

    @property
    def input_port(self):
        return self._input_port

    @input_port.setter
    def input_port(self, value):
        self._input_port = value

    @property
    def output_ports(self):
        return self._output_ports

    @output_ports.setter
    def output_ports(self, value):
        self._output_ports = value

    def queque_size(self):
        return f"{len(self._input_port.queue)}"


    def connectTo(self,
                  target_agent: "ConnectedAgent",
                  transformer: Optional[Callable[[BaseModel], Union[BaseModel, List[BaseModel]]]] = None,
                  target_port_name:str = None,
                  output_schema:Type[BaseModel] = None,
                  condition: Optional[Callable[[BaseModel], bool]] = None,
                  ) -> None:
        """
        Connects an output port to an input port with an optional transformer.

        Args:
            target_agent (ConenctedAgent): The target agent.
            transformer (Optional[Callable[[BaseModel], Union[BaseModel, List[BaseModel]]]], optional):
                A function to transform messages before sending. Defaults to None.
            target_port_name(str): optional target port name, for multi port agents
            output_schema(Type[BaseModel]): Optional the schema to connect to (only if more than one)
            condition(Optional[Callable[BaseModel]]): A condition function

        Raises:
            ValueError: If an invalid port direction is used.
        """

        input_port = target_agent._find_input_port(target_port_name)
        # If we have only one schema we can use it directly
        if not output_schema:
            output_schema = self.output_schema
        output_port = self._find_output_port(output_schema)
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

    def _find_output_port(self, schema: Type[BaseModel] = None) -> ToolPort:

        """
        Finds a port with a given name
        :param schema Output port schema
        :return: The port
        """
        return self.output_ports.get(schema, None)

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
        for port in self.output_ports.values():
            port.unconnected_outputs.clear()

    def get_final_outputs(self) -> List[BaseModel]:
        """
        Retrieves and clears all stored outputs from the output port.

        Returns:
            List[BaseModel]: A list of all final output messages.
        """
        result: List[BaseModel] = []
        for port in self.output_ports.values():
            result.extend(port.get_final_outputs())
        return result

    def get_one_output(self) -> Optional[BaseModel]:
        """
        Retrieves and removes one message from the output port.

        Returns:
            Optional[BaseModel]: A single output message, or None if no messages are available.
        """
        for port in self.output_ports.values():
            value = port.get_one_output()
            if value is not  None:
                return value
        return None

    def step(self) -> bool:
        """
        Processes one message from the input queue and sends output if applicable.
        If no input is available, returns False, allowing a scheduler to halt execution.

        Returns:
            bool: True if processing occurred, False otherwise.
        """
        # Check if we have a full queque or we are an inital agent which acts as source
        if self.input_port.queue or self.input_schema is InfiniteSchema:
            if self.input_schema is InfiniteSchema:
                parents, input_msg = [], InfiniteSchema()
            else:
                parents, input_msg = self.input_port.queue.popleft()
            # rich_console.print(f"   [blue]Running connected agent {self.__class__.__name__}[/blue] parents={len(parents)}")
            try:
                if self.debugger:
                    self.debugger.input(self, input_msg, parents)
                output_msg = self.process(input_msg, parents)
                if self.debugger:
                    self.debugger.output(self, output_msg, parents)
            except Exception as e:
                # Push the message back to the front of the queue to preserve order
                self.input_port.queue.appendleft((parents, input_msg))
                # Raise a scheduler-specific exception for higher-level handling
                raise SchedulerException(self.__class__.__name__, "Processing step failed", e)

            # Send message to port
            self._send_output_msg(output_msg, parents)
            return True
        else:
            if self.debugger:
                self.debugger.no_input(self)
            # rich_console.print(f"   [grey53]No input for connected agent {self.__class__.__name__}[/grey53]")
            pass
        return False

    def _send_output_msg(self, output_msg_var:Optional[Union[BaseModel, List[BaseModel], Tuple[BaseModel, ...]]], parents:List[str]) -> None:
        """
        Send the output message if not null or NullSchema to the output port.
        :param output_msg: The message
        :param parents: The paremnts
        """
        if output_msg_var and not isinstance(output_msg_var, NullSchema):
            if isinstance(output_msg_var, tuple):
                output_msgs = list(output_msg_var)
            elif isinstance(output_msg_var, list):
                output_msgs = [output_msg_var] # WE need to double wrap list since we remove first list in loop here
            else:
                output_msgs = [output_msg_var]
            # Loop tuple or single message, treat list asw one element item too to defer processing
            for output_msg in output_msgs:
                port_type = None
                if isinstance(output_msg, list):
                    if len(output_msg) > 0:
                        port_type = type(output_msg[0])
                else:
                    port_type = type(output_msg)
                port = None
                if port_type:
                    port = self.output_ports.get(port_type, None)
                # Exactly one port we take it for backward compat
                if port_type and not port and len(self.output_ports) == 1:
                    port = next(iter(self.output_ports.values()))
                if port:
                    result_parents = port.send(output_msg, parents)
                else:
                    print("Error in port finding ", self.__class__.__name__, type(output_msg))
                    raise SchedulerException(self.__class__.__name__, f"No output port for schame {type(output_msg)} found", None)

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

    # --------------------------------------------------------------------------- #
    # 1.  _gather_ports                                                           #
    # --------------------------------------------------------------------------- #
    def _gather_ports(self) -> Dict[str, "ToolPort"]:
        """
        Return **all** ports the agent owns as a flat
            {key → ToolPort}
        mapping.

        ── Keys ────────────────────────────────────────────────────────────────
        input_port                       – the single input port (as before)

        output_ports:<SchemaName>        – one key per *output* schema if the
                                           agent uses the new multi‑output design
                                           (e.g. "output_ports:UserEvent").

        output_port                      – the legacy single output port key
                                           kept for forward compatibility.

        Ports that resolve to *None* are skipped.
        """
        ports: Dict[str, "ToolPort"] = {}

        # ---- input (always at most one) ------------------------------------
        if getattr(self, "_input_port", None):
            ports["input_port"] = self._input_port

        # ---- new multi‑output mapping --------------------------------------
        if getattr(self, "_output_ports", None):
            for schema, port in self._output_ports.items():
                if port is not None:
                    ports[f"output_ports:{schema.__name__}"] = port

        # ---- legacy single output ------------------------------------------
        # (keep so we can still restore really old checkpoints)
        if getattr(self, "_output_port", None):
            ports["output_port"] = self._output_port

        return ports

    # --------------------------------------------------------------------------- #
    # 2.  save_state                                                              #
    # --------------------------------------------------------------------------- #
    def save_state(self) -> dict:
        """
        Serialise the agent’s private state **plus every port** returned by
        :meth:`_gather_ports`.  Each port entry is stored under the exact key
        produced by `_gather_ports`, so the structure is stable across versions.
        """
        state_dict = {
            "state": encode_payload(self._state) if self._state else None,
            "ports": {},
        }

        for port_key, port_obj in self._gather_ports().items():
            state_dict["ports"][port_key] = self._dump_port(port_obj)

        return state_dict

    # --------------------------------------------------------------------------- #
    # 3.  load_state                                                               #
    # --------------------------------------------------------------------------- #
    def load_state(self, state_dict: dict):
        """
        Restore the agent’s private state **and** every port snapshot previously
        produced by :meth:`save_state`.

        The schema for each port is determined like this:

        • input_port                    → `self.input_schema`
        • output_ports:<SchemaName>     → resolved from `<SchemaName>`
        • output_port (legacy)          → `self.output_schema`
        • anything else                 → fallback to `port_obj.schema`
        """
        # ── 3‑a  rebuild the private “state” blob ─────────────────────────────
        if state_dict.get("state") and self.state_schema:
            try:
                raw = decode_payload(state_dict["state"])
            except Exception as e:
                raise SchedulerException(
                    self.__class__.__name__, "Failed to decode state", e
                )

            if isinstance(raw, dict) and self.state_schema:
                try:
                    self._state = self.state_schema(**raw)
                except Exception as e:
                    print(f"[safe_model_load] Could not parse schema: {e}")
                    self._state = None
            else:
                self._state = raw

        # ── 3‑b  restore every port we currently expose ──────────────────────
        port_snapshots: dict = state_dict.get("ports", {})

        for port_key, port_obj in self._gather_ports().items():

            # find the saved snapshot (may be missing if the graph changed)
            snapshot = port_snapshots.get(port_key, {})

            # -------- decide which schema to use for this port ---------------
            if port_key == "input_port":
                schema = self.input_schema

            elif port_key == "output_port":  # legacy single output
                schema = self.output_schema

            elif port_key.startswith("output_ports:"):
                # extract <SchemaName> and match against the agent’s declared
                # output schema(s)
                schema_name = port_key.split(":", 1)[1]
                candidates = {}

                if self.output_schema is not None:
                    candidates[self.output_schema.__name__] = self.output_schema

                if self.output_schemas:
                    candidates.update({s.__name__: s for s in self.output_schemas})

                schema = candidates.get(schema_name)

            else:
                # last resort: ask the port itself (works if ToolPort stores it)
                schema = getattr(port_obj, "schema", None)

            # finally reload the port
            self._load_port(port_obj, snapshot, schema)

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