import time
import datetime
from typing import Dict, Optional, List, Set, Iterable
from pydantic import BaseModel, Field
import os
import json

from AgentFramework.core import ConnectedAgent
from AgentFramework.core.InfiniteSchema import InfiniteSchema
from AgentFramework.core.MultiPortAgent import MultiPortAgent
from AgentFramework.core.Schedulable import Schedulable
from AgentFramework.core.ToolPort import ToolPort
from agent_logging import rich_console
from util.SchedulerException import SchedulerException

class AgentSchedulerState(BaseModel):
    """
    Schema for the scheduler state.

    """
    agent_idx: int = Field(..., description="Agent index running")
    step_counter: int = Field(..., description="Step counter")
    all_done_counter : int = Field(..., description="Track if all agents are done")

class AgentScheduler(Schedulable):
    """
    Scheduler for managing and executing agent tasks.

    This class maintains a list of agents, runs them step by step until
    they have no more tasks, and retrieves final outputs from agents
    that serve as sinks for messages.

    Attributes:
        agents (List[ConnectedAgent]): List of agents managed by the scheduler.
        save_dir: Store agents there or None
        error_dir: Store agents there if we had an error or None
        save_step: After how many steps shall we save
    """
    # Static by init
    debugger: "DebugInterface"
    agents: List["ConnectedAgent"]
    save_dir: str
    error_dir: str
    save_step: int
    uuid: str


    # Dynamic
    state: AgentSchedulerState
    _global_state: BaseModel
    is_active: bool

    def __init__(self,
                 save_dir:str = None,
                 error_dir:str = None,
                 save_step:int = 1,
                 global_state:BaseModel=None,
                 debugger:"DebugInterface" = None,
                 uuid:str = 'Scheduler') -> None:
        """Initializes an empty agent scheduler."""
        self.agents = []
        self.save_dir = save_dir
        self.error_dir = error_dir
        self.save_step = save_step
        self._global_state = global_state
        self.debugger = debugger
        self.is_active = True
        self.uuid = uuid
        if self.debugger:
            self.debugger.init_debugger()
        self.state = AgentSchedulerState(agent_idx=0, step_counter=0, all_done_counter=0)

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

    @property
    def global_state(self) -> BaseModel:
        return self._global_state

    def close(self):
        """
        Close debugger etc
        :return:
        """
        if self.debugger:
            self.debugger.exit_debugger()

    def add_agent(self, agent: ConnectedAgent, skipAgent=False) -> None:
        """
        Adds an agent to the scheduler.

        Args:
            agent (ConnectedAgent): The agent instance to add.
        """
        if not skipAgent:
            if self.debugger:
                agent.debugger = self.debugger
            if self._global_state is not None:
                agent.global_state = self._global_state
            self.agents.append(agent)


    def queque_sizes(self):
        """
        Get queque size.
        :return: the size
        """
        sizes = {}
        for agent in self.agents:
            sizes[f"{agent.__class__.__name__}#{id(agent)}"] = agent.queque_size()
        return sizes

    # ------------------------------------------------------------------ #
    # 1) do the actual work for exactly ONE agent ---------------------- #
    # ------------------------------------------------------------------ #
    def _step_one_agent(self) -> bool:
        """
        Low‑level helper: run the agent that `self.state.agent_idx` points at,
        advance all indices, and return True if that agent did any work.
        """
        # lazy‑initialise
        if self.state.agent_idx is None:
            self.state.agent_idx = 0
        if self.state.step_counter is None:
            self.state.step_counter = 0

        # wrap‑around
        if self.state.agent_idx >= len(self.agents):
            self.state.agent_idx = 0

        idx = self.state.agent_idx
        agent = self.agents[idx]

        rich_console.print(f"[green]#{idx}: running {agent}, active={agent.is_active}[/green]")
        old_counter = self.state.step_counter
        if self.debugger:
            self.debugger.start_agent(agent, old_counter)
        # prepare state for next call *before* we actually step
        self.state.agent_idx = (self.state.agent_idx + 1) % len(self.agents)
        self.state.step_counter += 1

        # skip inactive
        if not agent.is_active:
            rich_console.print(f"   [grey53]#{idx}: {agent} is inactive[/grey53]")
            return False

        # run
        try:
            did_run = bool(agent.step())
            if did_run:
                agent_log = agent.run_log_string()
                self.log_run_agent(type(agent), agent.uuid,  self.state.step_counter, self.state.agent_idx, agent_log)
            if self.debugger:
                self.debugger.finished_agent(agent, old_counter, did_run)
        except SchedulerException as e:
            rich_console.print(
                f"[red]#{idx}: [ERROR] {e.agent_name} failed in step "
                f"{self.state.step_counter} with: {e.original_exception}[/red]"
            )
            if self.debugger:
                self.debugger.error_agent(agent, self.state.step_counter, e)
            if self.error_dir:
                self.save_scheduler(self.error_dir)
            raise

        if did_run:
            rich_console.print(f"   [orange3]#{idx}: step finished (running)[/orange3]")
        # else: silent idle

        return did_run

    def all_done(self):
        """
        Check if all agents ran
        :return:  True if all done
        """
        if self.state.all_done_counter >= len(self.agents):
            return True
        return False

    # ------------------------------------------------------------------ #
    # 2) public “tick” -------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def step(self) -> bool:
        """
        Run *one* scheduler tick (i.e. potentially one agent) **and** all
        the associated bookkeeping.
        Returns **True** as long as there might still be work left,
        **False** when a full round produced only idle agents.
        """
        if not self.agents:
            return False  # nothing to do

        # show round‑header once per round‑robin
        round_idx = self.state.step_counter // len(self.agents)
        if self.state.agent_idx == 0:
            rich_console.print(
                f"[red]Executing scheduler '{self.uuid}' step {round_idx}|{self.state.agent_idx} "
                f"/ done={self.state.all_done_counter}/{len(self.agents)}"
                f"  -------------------------------------[/red]"
            )

        did_run = self._step_one_agent()

        # update consecutive‑idle counter
        if did_run:
            # Reset the idle counter on any run true
            self.state.all_done_counter = 0
        else:
            self.state.all_done_counter += 1
            if self.state.all_done_counter >= len(self.agents):
                rich_console.print(
                    f"[red]No active agent found scheduler {self.uuid} step "
                    f"{round_idx} / {self.state.all_done_counter} / {len(self.agents)}[/red]"
                )
                return False  # <- we are done

        # auto‑save at end of a round
        if self.state.agent_idx == 0 and self.save_dir and round_idx % self.save_step == 0:
            path = f"{self.save_dir}/step_{round_idx}"
            self.save_scheduler(path)

        return True

    # ------------------------------------------------------------------ #
    # 3) “run‑to‑completion” wrapper ----------------------------------- #
    # ------------------------------------------------------------------ #
    def step_all(self, clear_previous_outputs:bool = False, validate_pipeline=False) -> int:
        """
        Keep calling `step()` until it reports that every agent was idle
        for a whole round.  Returns the final `step_counter`.
        :param clear_previous_outputs: Calls clear final outputs before running
        """
        rich_console.print(
            f"[red]Start scheduler at step {self.state.step_counter} "
            f"agents={len(self.agents)}[/red]"
        )

        if validate_pipeline:
            roots = list(self.find_entry_agents())
            missing = self.add_all_from_pipeline(roots, check_only=True)
            if missing:
                print("[WARNING] [Pipeline Check] Missing agents:")
                for a in missing:
                    print(f"  - {a.__class__.__name__} (uuid={a.uuid})")
                raise RuntimeError("Pipeline is incomplete – missing agents were not added.")

        if clear_previous_outputs:
            self.clear_final_outputs()

        self.clear_log()

        self.state.all_done_counter = 0  # fresh run

        while self.step():
            if self.debugger:
                pause_cnt = 0
                pause_state = self.debugger.is_pause(pause_cnt, self.state.step_counter)
                while pause_state:
                    time.sleep(0.25)
                    pause_cnt += 1
                    pause_state = self.debugger.is_pause(pause_cnt, self.state.step_counter)
            pass  # everything is handled inside

        return self.state.step_counter

    def get_final_outputs(self) -> Dict["ConnectedAgent", List[BaseModel]]:
        """
        Retrieves and clears final outputs from all agents that serve as sinks.

        Returns:
            Dict["ConnectedAgent", List[BaseModel]]: A dictionary where keys are agent
            and values are lists of final output data from those agents.
        """
        return {agent: agent.get_final_outputs() for agent in self.agents}

    def pop_one_output_per_agent(self) -> Dict["ConnectedAgent", Optional[BaseModel]]:
        """
        Retrieves and removes one message per agent if available.

        Returns:
            Dict["ConnectedAgent", Optional[BaseModel]]: A dictionary where keys are agents
            and values are a single output message from each agent, if available.
        """
        return {agent: agent.pop_one_output() for agent in self.agents}

    def clear_final_outputs(self) -> None:
        """
        Clear remaining data
        """
        for a in self.agents:
            a.clear_final_outputs()

    def pop_one_output_for_agent(self, agent: "ConnectedAgent") -> Optional[BaseModel]:
        """
        Retrieves and removes one message for the given agent if available.

        Args:
            agent (ConnectedAgent): The agent for whom to retrieve the message.

        Returns:
            Optional[BaseModel]: The output message from the specified agent, if available.
        """
        for a in self.agents:
            if a == agent:
                return a.pop_one_output()
        return None

    def pop_all_outputs(self) -> list[BaseModel]:
        """
        Retrieves and removes all available output messages from all agents.

        Returns:
            list[BaseModel]: A list of all output messages from all agents.
        """
        all_outputs = []
        for agent in self.agents:
            while True:
                output = agent.pop_one_output()
                if output is None:
                    break
                all_outputs.append(output)
        return all_outputs

    def save_state(self) -> dict:
        """
        Return a *pure-Python* snapshot describing:
            • the scheduler’s own counters
            • the optional `global_state`
            • one nested snapshot *per agent* (keyed by `uuid`)
        No filesystem access here – that keeps the snapshot re-usable for
        nested schedulers and unit tests.
        """
        return {
            "is_active": self.is_active,
            "scheduler_state": self.state.model_dump(),
            "global_state": (
                self._global_state.model_dump()
                if isinstance(self._global_state, BaseModel)
                else self._global_state
            ),
            "global_state_cls": (
                f"{self._global_state.__class__.__module__}."
                f"{self._global_state.__class__.__qualname__}"
                if self._global_state is not None else None
            ),
            "agents": {a.uuid: a.save_state() for a in self.agents},
        }

    def load_state(self, snapshot: dict) -> None:
        """
        Recreate internal state from a snapshot produced by `save_state`.
        Works for both plain agents and nested schedulers.
        """
        # -- core counters ------------------------------------------------
        self.is_active = snapshot.get("is_active", True)
        self.state = AgentSchedulerState.model_validate(snapshot["scheduler_state"])

        # -- global -------------------------------------------------------
        gs_data = snapshot.get("global_state")
        if gs_data is None:
            self._global_state = None

        elif snapshot.get("global_state_cls"):
            module_name, _, cls_name = snapshot["global_state_cls"].rpartition(".")
            module = __import__(module_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            self._global_state = (
                cls.model_validate(gs_data) if issubclass(cls, BaseModel) else gs_data
            )
        else:
            self._global_state = gs_data

        # -- agents -------------------------------------------------------
        #  Match snapshots by uuid; ignore unknown or missing safely.
        uuid2agent = {a.uuid: a for a in self.agents}
        for uuid, agent_state in snapshot.get("agents", {}).items():
            agent = uuid2agent.get(uuid)
            if agent:
                agent.load_state(agent_state)
            else:
                rich_console.print(f"[yellow] orphan snapshot for uuid '{uuid}' ignored[/yellow]")

        # -- FILE HELPERS -----------------------------------------------------

    def save_scheduler(self, path: str) -> None:
        """
        Persist the *entire* scheduler tree into `<path>/state.json`.
        This is the only on-disk format we keep; per-agent files are no longer
        written (they are redundant).  Existing tools that called
        `save_scheduler()` still work – they just look at a different file.
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(self.save_state(), f, indent=2)

    def load_scheduler(self, path: str) -> None:
        """
        Restore the scheduler from `<path>/state.json`.
        """
        with open(os.path.join(path, "state.json"), "r") as f:
            snapshot = json.load(f)
        self.load_state(snapshot)


    @staticmethod
    def collect_pipeline_agents(roots: "ConnectedAgent|Iterable[ConnectedAgent]") \
            -> Set["ConnectedAgent"]:
        """
        Return the *union* of `gather_pipeline_agents()` for one or many roots.
        """
        if not isinstance(roots, Iterable):
            roots = [roots]

        found: Set[ConnectedAgent] = set()
        for root in roots:
            found.update(root.gather_pipeline_agents())
        return found

    def add_all_from_pipeline(self,
                              roots: "ConnectedAgent|Iterable[ConnectedAgent]",
                              *,
                              check_only: bool = False
                              ) -> Set["ConnectedAgent"]:
        """
        • Collect every agent reachable from `roots`
        • Compute *missing* = required − already-scheduled
        • If `check_only=True`  → just return *missing*
          else                 → auto-add every missing agent.

        The return value is the (possibly empty) *missing* set.
        """
        required = self.collect_pipeline_agents(roots)
        scheduled = set(self.agents)                          # by *identity*
        missing   = required - scheduled

        if not check_only:
            for a in missing:
                self.add_agent(a)

        return missing

    def find_entry_agents(self) -> Set["ConnectedAgent"]:
        """
        Automatically detects root (entry) agents even if they are multi-port.
        An agent is considered an entry if:
        - It has InfiniteSchema as input, OR
        - None of its input ports are targeted by other agents' output ports.
        """
        input_ports_to_agents: Dict[ToolPort, ConnectedAgent] = {}
        input_ports_with_sources = set()

        for agent in self.agents:
            if isinstance(agent, MultiPortAgent):
                for port in agent._input_ports.values():
                    input_ports_to_agents[port] = agent
            else:
                try:
                    port = agent.input_port
                    input_ports_to_agents[port] = agent
                except AttributeError:
                    print(f"[warn] Agent {agent.__class__.__name__} has no accessible input_port.")
                    continue

        # Find all input ports that are the target of some output port
        for agent in self.agents:
            for port in getattr(agent, "output_ports", {}).values():
                for (target_port, _, _, _, _) in port.connections:
                    input_ports_with_sources.add(target_port)

        # Now check which agents have no upstream connection
        entry_agents = set()
        for port, agent in input_ports_to_agents.items():
            # InfiniteSchema agents are always entries
            if getattr(agent, "input_schema", None) is InfiniteSchema:
                entry_agents.add(agent)
            elif port not in input_ports_with_sources:
                entry_agents.add(agent)

        return entry_agents


    def clear_log(self) -> None:
        """
        Clears the scheduler log file (`scheduler.log`) in the `save_dir` directory.

        If `save_dir` is not set, this method does nothing.

        This is typically called at the beginning of a run to reset the log file.
        """
        if self.save_dir:
            log_path = os.path.join(self.save_dir, "scheduler.log")
            with open(log_path, "w") as f:
                f.write("")  # clear file contents

    def log_run_agent(self, agent_type: type, agent_uuid: str, counter: int, index: int, agent_log: str = None) -> None:
        """
        Appends a log entry for an agent run to the `scheduler.log` file in `save_dir`.

        Each log entry includes:
            - Current date and time (UTC)
            - Step counter
            - Agent type and UUID
            - Agent index in the agents list

        Args:
            agent_type (type): The class/type of the agent.
            agent_uuid (str): UUID of the agent.
            counter (int): Current step counter.
            index (int): Index of the agent in the agents list.
        """
        if self.save_dir:
            als = f": {agent_log}" if agent_log else ""
            log_path = os.path.join(self.save_dir, "scheduler.log")
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            log_entry = f"[{timestamp}] step={counter} agent={agent_type.__name__} ({agent_uuid}) index={index}: {als}\n"
            with open(log_path, "a") as f:
                f.write(log_entry)
