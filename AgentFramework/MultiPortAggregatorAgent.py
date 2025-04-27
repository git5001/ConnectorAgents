from typing import Type, List, Dict, Optional, Tuple
from pydantic import BaseModel
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ToolPort import ToolPort
from AgentFramework.listutil import longest_common_sublist
from util.SchedulerException import SchedulerException
import re


class MultiPortAggregatorAgent(ConnectedAgent):
    """
    Aggregates messages coming in on multiple independent input ports.

    Key design alignment with `MultiPortAgent` (April 2025 revision):
      • `input_schemas` is now a **list** of schemas; the schema object itself
        identifies a port, so we no longer need arbitrary string names.
      • `_input_ports` is keyed by schema, matching the new model.
      • Checkpoint keys for input ports are numerical (input_0, input_1, …)
        to remain stable across refactors.
    """

    # ------------------------------------------------------------------
    input_schemas: List[Type[BaseModel]] = []
    output_schema: Optional[Type[BaseModel]] = None
    output_schemas: Optional[List[Type[BaseModel]]] = None

    # single‑port compatibility (unused directly)
    input_schema: Type[BaseModel] = BaseIOSchema

    # ------------------------------------------------------------------
    def __init__(self, config: BaseToolConfig = BaseToolConfig(), uuid: str = "default") -> None:
        super().__init__(config, uuid=uuid, create_ports=False)

        if not self.input_schemas:
            raise TypeError("MultiPortAggregatorAgent requires `input_schemas`.")
        if self.output_schema is None and not self.output_schemas:
            raise TypeError("Define `output_schema` or `output_schemas`.")

        # ---- INPUT ports -------------------------------------------------
        self._input_ports: Dict[Type[BaseModel], ToolPort] = {
            schema: ToolPort(ToolPort.Direction.INPUT, schema, f"{uuid}:{self.__class__.__name__}")
            for schema in self.input_schemas
        }

        # ---- OUTPUT port(s) ----------------------------------------------
        self._output_ports: Dict[Type[BaseModel], ToolPort] = {}
        if self.output_schema:
            self._output_ports[self.output_schema] = ToolPort(
                ToolPort.Direction.OUTPUT,
                self.output_schema,
                f"{uuid}:{self.__class__.__name__}",
            )
        else:
            for n, schema in enumerate(self.output_schemas):
                self._output_ports[schema] = ToolPort(
                    ToolPort.Direction.OUTPUT,
                    schema,
                    f"{uuid}:{self.__class__.__name__}#{n}",
                )

    # ------------------------------------------------------------------
    @property
    def input_port(self):  # type: ignore[override]
        raise AttributeError("Use multiple input ports (`_input_ports`).")

    @input_port.setter  # type: ignore[override]
    def input_port(self, _):
        raise AttributeError("Setting single .input_port is not allowed.")

    # ------------------------------------------------------------------
    def queue_size(self) -> str:
        """Return comma‑separated queue sizes for debug / monitoring."""
        return ", ".join(str(len(p.queue)) for p in self._input_ports.values())

    # ------------------------------------------------------------------
    def _find_input_port(self, source_port_schema: Type[BaseModel] = None):
        if source_port_schema is None:
            raise NotImplementedError("Multi port must provide source port.")
        if source_port_schema not in self._input_ports:
            raise ValueError(f"Input port {source_port_schema} is not defined in {self.__class__.__name__}.")
        return self._input_ports[source_port_schema]

    # ------------------------------------------------------------------
    @staticmethod
    def extract_parents_with_suffix(parents: List[str]) -> Dict[str, Tuple[str, str, str]]:
        """Extract UUID, idx1, idx2 for strings of form UUID:idx1:idx2 (idx2 > 1)."""
        pattern = re.compile(r"^(.*):(\d+):(\d+)$")
        result: Dict[str, Tuple[str, str, str]] = {}
        for p in parents:
            m = pattern.match(p)
            if m:
                uuid, idx1, idx2 = m.groups()
                if int(idx2) > 1:
                    result[p] = (uuid, idx1, idx2)
        return result

    # ------------------------------------------------------------------
    def _find_parent_indices_2(self) -> List[int]:
        """Align queues so that matching chain‑length messages can be processed."""
        if not isinstance(self.input_schemas, list):
            raise ValueError(
                f"Expected input_schemas to be a list, got {type(self.input_schemas).__name__} in {self.__class__.__name__}"
            )
        first_port_schema = self.input_schemas[0]
        first_port = self._input_ports[first_port_schema]

        for anchor_idx, (anchor_parents, _) in enumerate(first_port.queue):
            chain_parents = self.extract_parents_with_suffix(anchor_parents)
            search_keys = {f":{idx1}:{idx2}" for _, (_, idx1, idx2) in chain_parents.items()}

            candidate_indices: List[int] = []

            for port_schema, port in self._input_ports.items():
                if port_schema == first_port_schema:
                    candidate_indices.append(anchor_idx)
                    continue

                found = None
                for idx, (parents, _) in enumerate(port.queue):
                    candidate_suffixes = {":" + ":".join(p.split(":")[1:]) for p in parents if p.count(":") >= 2}
                    if search_keys.issubset(candidate_suffixes):
                        found = idx
                        break
                if found is None:
                    break
                candidate_indices.append(found)

            if len(candidate_indices) == len(self._input_ports):
                return candidate_indices

        return []  # No alignment yet

    # ------------------------------------------------------------------
    def step(self) -> bool:
        if any(not p.queue for p in self._input_ports.values()):
            return False

        parent_indices = self._find_parent_indices_2()
        if len(parent_indices) != len(self._input_ports):
            return False

        parent_map: Dict[Type[BaseModel], List[str]] = {}
        model_map: Dict[Type[BaseModel], BaseModel] = {}
        staged: List[Tuple[Type[BaseModel], int, Tuple[List[str], BaseModel]]] = []

        for port_schema, idx in zip(self.input_schemas, parent_indices):
            port = self._input_ports[port_schema]
            parents, model = port.queue[idx]

            parent_map[port_schema] = parents
            model_map[port_schema] = model

            staged.append((port_schema, idx, (parents, model)))
            del port.queue[idx]

        join_parents = longest_common_sublist(parent_map)

        try:
            if self.debugger:
                for data in model_map.values():
                    self.debugger.input(self, data, join_parents)
            output_msg = self.run(model_map)  # type: ignore[attr-defined]
            if self.debugger:
                self.debugger.output(self, output_msg, join_parents)
        except Exception as e:
            for schema, idx, item in staged:
                self._input_ports[schema].queue.insert(idx, item)
            raise SchedulerException(self.__class__.__name__, "Processing multi step failed", e)

        self._send_output_msg(output_msg, join_parents)
        return True

    # ------------------------------------------------------------------
    def _gather_ports(self) -> Dict[str, ToolPort]:
        ports: Dict[str, ToolPort] = {
            f"input_{idx}": self._input_ports[schema]
            for idx, schema in enumerate(self.input_schemas)
        }
        for schema, port in self._output_ports.items():
            ports[f"output_ports:{schema.__name__}"] = port
        if getattr(self, "_output_port", None):
            ports["output_port"] = self._output_port
        return ports

    # ------------------------------------------------------------------
    def load_state(self, state_dict: dict):
        if state_dict.get("state") and self.state_schema:
            self._state = self.state_schema(**state_dict["state"])

        for pname, pobj in self._gather_ports().items():
            pstate = state_dict.get("ports", {}).get(pname, {})

            if pname.startswith("input_"):
                idx = int(pname[len("input_"):])
                schema = self.input_schemas[idx]
            else:
                if self.output_schema:
                    schema = self.output_schema
                else:
                    schema_name = pname.split(":", 1)[1]
                    schema = next(s for s in self._output_ports if s.__name__ == schema_name)

            self._load_port(pobj, pstate, schema)
