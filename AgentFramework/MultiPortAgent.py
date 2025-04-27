from typing import Dict, List, Type, Tuple, Optional
from pydantic import BaseModel

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig
from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ToolPort import ToolPort
from util.SchedulerException import SchedulerException


class MultiPortAgent(ConnectedAgent):
    """
    ConnectedAgent with N independent input ports.
    On each `step()` it pops **one** message – chosen round-robin – and
    processes it with `self.process(input_msg, parents)`.
    """

    # ----- class-level contracts --------------------------------------------
    input_schemas: List[Type[BaseModel]] = []
    output_schema: Optional[Type[BaseModel]] = None
    output_schemas: Optional[List[Type[BaseModel]]] = None

    # single-port compatibility (unused directly)
    input_schema: Type[BaseModel] = BaseIOSchema

    # ------------------------------------------------------------------------
    def __init__(self,
                 config: BaseToolConfig = BaseToolConfig(),
                 uuid: str = "default") -> None:

        super().__init__(config, uuid=uuid, create_ports=False)

        if not self.input_schemas:
            raise TypeError("MultiPortAgent requires `input_schemas`.")
        if self.output_schema is None and not self.output_schemas:
            raise TypeError("Define `output_schema` or `output_schemas`.")

        # ---- INPUT ports -----------------------------------------------------
        self._input_ports: Dict[Type[BaseModel], ToolPort] = {
            schema: ToolPort(ToolPort.Direction.INPUT, schema,
                           f"{uuid}:{self.__class__.__name__}")
            for schema in self.input_schemas
        }

        # ---- OUTPUT port(s) ---------------------------------------------------
        self._output_ports: Dict[Type[BaseModel], ToolPort] = {}
        if self.output_schema:
            self._output_ports[self.output_schema] = ToolPort(
                ToolPort.Direction.OUTPUT,
                self.output_schema,
                f"{uuid}:{self.__class__.__name__}")
        else:
            for n, schema in enumerate(self.output_schemas):
                self._output_ports[schema] = ToolPort(
                    ToolPort.Direction.OUTPUT,
                    schema,
                    f"{uuid}:{self.__class__.__name__}#{n}"
                )

        # round-robin pointer
        self._rr_idx: int = -1

    # ------------------------------------------------------------------------
    @property
    def input_port(self):
        raise AttributeError("Use multiple input ports (. _input_ports).")

    @input_port.setter
    def input_port(self, _):
        raise AttributeError("Setting single .input_port is not allowed.")

    # ------------------------------------------------------------------------
    def _find_input_port(self, source_port_schema:Type[BaseModel] = None):

        """
        Finds a port with a given name or the default port if no name
        :param source_port_schema: Schema of port or none
        :return: The port
        """
        if not source_port_schema:
            raise NotImplementedError("Multi port must provide source port.")
        if not source_port_schema in self._input_ports:
            raise ValueError(f"Input port {source_port_schema} is not defined in {self.__class__.__name__}.")
        found = self._input_ports[source_port_schema]
        return found
    # ------------------------------------------------------------------------
    def step(self) -> bool:
        """
        Pops **one** message from the next non-empty input queue (round-robin),
        forwards it to `self.process`, and handles rollback on error.
        """

        port_schemas = list(self._input_ports.keys())
        if not port_schemas:
            return False

        n_ports = len(port_schemas)

        # --- find the next queue that has data -------------------------------
        for i in range(n_ports):
            probe_idx = (self._rr_idx + 1 + i) % n_ports
            port_schema = port_schemas[probe_idx]
            probe_port = self._input_ports[port_schema]
            if probe_port.queue:
                self._rr_idx = probe_idx
                break
        else:  # every queue empty
            if self.debugger:
                self.debugger.no_input(self)
            return False

        # --- dequeue one message --------------------------------------------
        parents, input_msg = probe_port.queue.popleft()

        # --- process ---------------------------------------------------------
        try:
            if self.debugger:
                self.debugger.input(self, input_msg, parents)
            input_map ={port_schema:input_msg}
            output_msg = self.process(input_map, parents)
            if self.debugger:
                self.debugger.output(self, output_msg, parents)
        except Exception as e:
            probe_port.queue.appendleft((parents, input_msg))  # rollback
            raise SchedulerException(self.__class__.__name__,"Processing step failed", e)

        # --- emit downstream -------------------------------------------------
        self._send_output_msg(output_msg, parents)
        return True

    # ------------------------------------------------------------------------
    def _gather_ports(self) -> Dict[str, ToolPort]:
        """
        Expose ports for checkpoint / debug.

        Keys must stay identical across save- and load-cycles,
        so we index inputs numerically in list order.
        """
        ports: Dict[str, ToolPort] = {
            f"input_{idx}": self._input_ports[schema]
            for idx, schema in enumerate(self.input_schemas)
        }

        for schema, port in self._output_ports.items():
            ports[f"output_ports:{schema.__name__}"] = port

        if getattr(self, "_output_port", None):
            ports["output_port"] = self._output_port

        return ports

    # ------------------------------------------------------------------------
    def load_state(self, state_dict: dict):
        if state_dict.get("state") and self.state_schema:
            self._state = self.state_schema(**state_dict["state"])

        for pname, pobj in self._gather_ports().items():
            pstate = state_dict.get("ports", {}).get(pname, {})

            if pname.startswith("input_"):
                idx = int(pname[len("input_"):])
                schema = self.input_schemas[idx]
            else:  # outputs stay exactly as before
                if self.output_schema:
                    schema = self.output_schema
                else:
                    schema_name = pname.split(":", 1)[1]
                    schema = next(
                        s for s in self._output_ports if s.__name__ == schema_name
                    )

            self._load_port(pobj, pstate, schema)

