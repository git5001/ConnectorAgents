from __future__ import annotations

import json
from typing import Dict, List, Type, Optional, Tuple
import re

from pydantic import BaseModel

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.IdWrapper import IdWrapper
from AgentFramework.core.ToolPort import ToolPort
from AgentFramework.core.listutil import longest_common_sublist

from util.SchedulerException import SchedulerException

class MultiPortPayload(BaseIOSchema):
    """
    Wrapper class for a dictionary of BaseIOSchema instances.
    """
    payload: Dict[Type[BaseModel], BaseModel]
    ids: Dict[Type[BaseModel], str]
    def to_text(self, indent: int = 4) -> str:
        """
        Convert the payload into a JSON-formatted string.

        Args:
            indent (int): Number of spaces for indentation in JSON output.

        Returns:
            str: JSON representation of the payload.
        """
        serializable_payload = {
            model_class.__name__: model_instance.model_dump()
            for model_class, model_instance in self.payload.items()
        }
        return json.dumps(serializable_payload, indent=indent)


class MultiPortAgent(ConnectedAgent):
    """Agent with *N* independent input ports.

    Parameters
    ----------
    aggregate: bool, default ``True``
        If *True*, the agent waits until it has *matching* messages on **all**
        input ports (same *chain‑length* parent IDs) before forwarding them to
        :py:meth:`run`.  When *False*, it processes exactly **one** message per
        :py:meth:`step`, chosen round‑robin from the non‑empty input queues.
    """

    # ----- class‑level contracts -------------------------------------------
    input_schemas: List[Type[BaseModel]] = []
    output_schema: Optional[Type[BaseModel]] = None
    output_schemas: Optional[List[Type[BaseModel]]] = None

    # single‑port compatibility (unused directly but required elsewhere)
    input_schema: Type[BaseModel] = BaseIOSchema

    # ---------------------------------------------------------------------
    def __init__(
        self,
        config: BaseToolConfig = BaseToolConfig(),
        *,
        aggregate: bool = True,
        schemas: Optional[List[Type[BaseModel]]] = None,
        **kwargs,
    ) -> None:
        if schemas is not None:
            self.input_schemas = schemas
        if not self.output_schemas and not self.output_schema:
            self.output_schema = MultiPortPayload
        super().__init__(config, create_ports=False, **kwargs)

        if not self.input_schemas:
            raise TypeError("MultiPortAgent requires `input_schemas`.")
        if self.output_schema is None and not self.output_schemas:
            raise TypeError("Define `output_schema` or `output_schemas`.")

        self.aggregate: bool = aggregate

        # ---- INPUT ports --------------------------------------------------
        self._input_ports: Dict[Type[BaseModel], ToolPort] = {
            schema: ToolPort(ToolPort.Direction.INPUT, schema, f"{self.uuid}:{self.__class__.__name__}")
            for schema in self.input_schemas
        }

        # ---- OUTPUT port(s) ----------------------------------------------
        self._output_ports: Dict[Type[BaseModel], ToolPort] = {}
        if self.output_schema:
            self._output_ports[self.output_schema] = ToolPort(
                ToolPort.Direction.OUTPUT,
                self.output_schema,
                f"{self.uuid}:{self.__class__.__name__}",
            )
        else:
            for n, schema in enumerate(self.output_schemas):
                self._output_ports[schema] = ToolPort(
                    ToolPort.Direction.OUTPUT,
                    schema,
                    f"{self.uuid}:{self.__class__.__name__}#{n}",
                )

        # round‑robin pointer – only used in non‑aggregate mode
        self._rr_idx: int = -1

    # ------------------------------------------------------------------
    #                               helpers
    # ------------------------------------------------------------------
    @property
    def input_port(self):  # type: ignore[override]
        raise AttributeError("Use multiple input ports (`_input_ports`).")

    @input_port.setter  # type: ignore[override]
    def input_port(self, _):
        raise AttributeError("Setting single .input_port is not allowed.")

    # ..................................................................
    def queue_size(self) -> str:
        """Return comma‑separated queue sizes for debug / monitoring."""
        return ", ".join(str(len(p.queue)) for p in self._input_ports.values())

    # ..................................................................
    def _find_input_port(self, source_port_schema: Type[BaseModel] | None = None):
        if source_port_schema is None:
            raise NotImplementedError("Multi port must provide source port.")
        if source_port_schema not in self._input_ports:
            raise ValueError(
                f"Input port {source_port_schema} is not defined in {self.__class__.__name__}."
            )
        return self._input_ports[source_port_schema]

    # ..................................................................
    # ---- aggregator‑mode helpers (copied verbatim) --------------------
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

    # ..................................................................
    #                               core step
    # ..................................................................
    def step(self) -> bool:  # noqa: C901 – complexity comes from two modes
        """Perform one scheduling step (either aggregated or round‑robin)."""
        if self.aggregate:
            # ---------- aggregated (synchronising) mode -------------------
            if any(not p.queue for p in self._input_ports.values()):
                return False  # at least one queue empty

            parent_indices = self._find_parent_indices_2()
            if len(parent_indices) != len(self._input_ports):
                return False  # alignment not ready

            parent_map: Dict[Type[BaseModel], List[str]] = {}
            model_map: Dict[Type[BaseModel], BaseModel] = {}
            id_map: Dict[Type[BaseModel], str] = {}
            staged: List[Tuple[Type[BaseModel], int, Tuple[List[str], BaseModel]]] = []

            # --- dequeue one synchronised message from each port --------
            for port_schema, idx in zip(self.input_schemas, parent_indices):
                port = self._input_ports[port_schema]
                parents, timestamp, unique_id, model = port.queue[idx]

                parent_map[port_schema] = parents
                model_map[port_schema] = model
                id_map[port_schema] = unique_id

                staged.append((port_schema, idx, (parents, timestamp, unique_id, model)))
                del port.queue[idx]

            join_parents = longest_common_sublist(parent_map)

            # --- process synchronised batch -----------------------------
            try:
                if self.debugger:
                    for data in model_map.values():
                        self.debugger.input(self, data, join_parents)
                payload_wrapper = MultiPortPayload(payload=model_map, ids=id_map)
                # Have no id for multi message must use the warpped map
                output_msg = self.process(payload_wrapper, join_parents, None)  # type: ignore[attr-defined]
                ids_str = ":".join(dict.fromkeys(id_map.values()))
                output_msg, ids = self.unwrap_id(output_msg, ids_str)
                if self.debugger:
                    self.debugger.output(self, output_msg, join_parents)
            except Exception as e:
                # rollback: re‑insert messages at their original indices
                for schema, idx, item in staged:
                    self._input_ports[schema].queue.insert(idx, item)
                raise SchedulerException(self.__class__.__name__, "Processing multi step failed", e)

            # --- emit downstream ---------------------------------------

            self._send_output_msg(output_msg, join_parents, ids)
            return True

        # ----------------------------------------------------------------
        # ------- non‑aggregating (round‑robin) mode ----------------------
        port_schemas = list(self._input_ports.keys())
        if not port_schemas:
            return False

        n_ports = len(port_schemas)

        # --- find the next queue that has data --------------------------
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

        # --- dequeue one message ---------------------------------------
        parents, timestamp, unique_id, input_msg = probe_port.queue.popleft()

        # --- process ----------------------------------------------------
        try:
            if self.debugger:
                self.debugger.input(self, input_msg, parents)
            input_map = {port_schema: input_msg}
            id_map = {port_schema: unique_id}
            payload_wrapper = MultiPortPayload(payload=input_map, ids=id_map)
            output_msg = self.process(payload_wrapper, parents, unique_id)  # type: ignore[attr-defined]
            output_msg, ids = self.unwrap_id(output_msg, unique_id)
            if self.debugger:
                self.debugger.output(self, output_msg, parents)
        except Exception as e:
            probe_port.queue.appendleft((parents, timestamp, unique_id, input_msg))  # rollback
            raise SchedulerException(self.__class__.__name__, "Processing step failed", e)

        # --- emit downstream -------------------------------------------
        self._send_output_msg(output_msg, parents, ids)
        return True

    # ------------------------------------------------------------------
    #                               ports & state
    # ------------------------------------------------------------------
    def _gather_ports(self) -> Dict[str, ToolPort]:
        """Expose ports for checkpoint / debug."""
        ports: Dict[str, ToolPort] = {
            f"input_{idx}": self._input_ports[schema]
            for idx, schema in enumerate(self.input_schemas)
        }
        for schema, port in self._output_ports.items():
            ports[f"output_ports:{schema.__name__}"] = port
        if getattr(self, "_output_port", None):
            ports["output_port"] = self._output_port
        return ports

    # ..................................................................
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

    # ------------------------------------------------------------------
    #                        default implementation hooks
    # ------------------------------------------------------------------
    def run(self, params: MultiPortPayload) -> MultiPortPayload:  # noqa: D401
        """Default *aggregated* handler – simply wrap inputs into a payload.
        """
        return params

