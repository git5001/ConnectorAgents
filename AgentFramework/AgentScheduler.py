import time
import os
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from rich.console import Console
import os
import json

from AgentFramework import ConnectedAgent
from agent_logging import rich_console
from util.SchedulerException import SchedulerException

class AgentSchedulerState(BaseModel):
    """
    Schema for the scheduler state.

    """
    agent_idx: int = Field(..., description="Agent index running")
    step_counter: int = Field(..., description="Step counter")
    all_done_counter : int = Field(..., description="Track if all agents are done")

class AgentScheduler:
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

    # Dynamic
    state: AgentSchedulerState
    _global_state: BaseModel

    def __init__(self,
                 save_dir:str = None,
                 error_dir:str = None,
                 save_step:int = 1,
                 global_state:BaseModel=None,
                 debugger:"DebugInterface" = None) -> None:
        """Initializes an empty agent scheduler."""
        self.agents = []
        self.save_dir = save_dir
        self.error_dir = error_dir
        self.save_step = save_step
        self._global_state = global_state
        self.debugger = debugger
        if self.debugger:
            self.debugger.init_debugger()
        self.state = AgentSchedulerState(agent_idx=0, step_counter=0, all_done_counter=0)

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)


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
                print("Agent is ",type(agent))
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
                f"[red]Executing scheduler step {round_idx}|{self.state.agent_idx} "
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
                    f"[red]No active agent found scheduler step "
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
    def step_all(self) -> int:
        """
        Keep calling `step()` until it reports that every agent was idle
        for a whole round.  Returns the final `step_counter`.
        """
        rich_console.print(
            f"[red]Start scheduler at step {self.state.step_counter} "
            f"agents={len(self.agents)}[/red]"
        )
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

    def get_final_outputs(self) -> Dict[str, List[BaseModel]]:
        """
        Retrieves and clears final outputs from all agents that serve as sinks.

        Returns:
            Dict[str, List[BaseModel]]: A dictionary where keys are agent class names
            and values are lists of final output data from those agents.
        """
        return {agent: agent.get_final_outputs() for agent in self.agents}

    def get_one_output_per_agent(self) -> Dict[str, Optional[BaseModel]]:
        """
        Retrieves and removes one message per agent if available.

        Returns:
            Dict[str, Optional[BaseModel]]: A dictionary where keys are agent class names
            and values are a single output message from each agent, if available.
        """
        return {agent: agent.get_one_output() for agent in self.agents}


    def save_agents(self, directory: str) -> None:
        """
        Saves the scheduler's agents into `directory`.
        Creates one JSON file per agent (or fallback to pickle-JSON).
        """
        os.makedirs(directory, exist_ok=True)

        rich_console.print(f"[blue]Saving {len(self.agents)} agents into {directory}.[/blue]")
        for idx, agent in enumerate(self.agents):
            uuid = agent.uuid
            filename = f"agent_{uuid}_{agent.__class__.__name__}.json"
            path = os.path.join(directory, filename)
            # Each agent uses its own save method:
            agent.save_state_to_file(path)
            # rich_console.print(f"   [blue]Saving agent {agent.__class__.__name__}[/blue] to {path}")



    def load_agents(self, directory: str) -> None:
        """
        Loads each agent's state from JSON files in `directory`.
        Agents must already be in `self.agents` in the same order/class
        as they were during `save_scheduler(...)`.
        """
        for idx, agent in enumerate(self.agents):
            uuid = agent.uuid
            filename = f"agent_{uuid}_{agent.__class__.__name__}.json"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                agent.load_state_from_file(filepath)
            else:
                rich_console.print(f"[red]Agent not found {agent.__class__.__name__}[/red] from {filepath}")

            rich_console.print(f"[blue]Loaded agent {agent.__class__.__name__}[/blue] from {filepath}")
        rich_console.print(f"[blue]Loaded {len(self.agents)} agents from '{directory}'.[/blue]")

    def save_state(self, dir: str) -> None:
        """
        Save scheduler + global state into <dir>/scheduler_state.json
        """
        path = os.path.join(dir, "scheduler_state.json")

        # Anything that’s a pydantic model serialises cleanly via .model_dump()
        payload = {
            "scheduler_state": self.state.model_dump(),
            "global_state": (
                self._global_state.model_dump()
                if isinstance(self._global_state, BaseModel)
                else self._global_state  # could be None or a plain dict
            ),
            # Optional but handy – lets you restore into the right class later
            "global_state_cls": (
                f"{self._global_state.__class__.__module__}."
                f"{self._global_state.__class__.__qualname__}"
                if self._global_state is not None else None
            ),
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_state(self, dir: str) -> None:
        """
        Load scheduler + global state from <dir>/scheduler_state.json
        """
        path = os.path.join(dir, "scheduler_state.json")
        with open(path, "r") as f:
            payload = json.load(f)

        # Scheduler core
        self.state = AgentSchedulerState.model_validate(payload["scheduler_state"])

        # Global state – three common cases
        gs_data = payload.get("global_state")
        if gs_data is None:
            self._global_state = None

        elif payload.get("global_state_cls"):
            # Rebuild the exact model class it was saved from
            module_name, _, cls_name = payload["global_state_cls"].rpartition(".")
            module = __import__(module_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            if issubclass(cls, BaseModel):
                self._global_state = cls.model_validate(gs_data)
            else:
                # Fallback – just leave it as the raw dict
                self._global_state = gs_data
        else:
            # It was a plain dict (or something already JSON-serialisable)
            self._global_state = gs_data

    def save_scheduler(self, path: str) -> None:
        """
        save all agents and scheduler.
        :param dir: The dir
        """
        os.makedirs(path, exist_ok=True)
        self.save_state(path)
        self.save_agents(path)

    def load_scheduler(self, path:str) -> None:
        """
        load all agents and scheduler.
        :param dir: The dir
        """
        self.load_agents(path)
        self.load_state(path)