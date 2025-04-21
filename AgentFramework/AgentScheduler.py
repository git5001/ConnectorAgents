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

    def __init__(self, save_dir:str = None, error_dir:str = None, save_step:int = 1) -> None:
        """Initializes an empty agent scheduler."""
        self.agents: List[ConnectedAgent] = []
        self.save_dir = save_dir
        self.error_dir = error_dir
        self.save_step = save_step
        self.state = AgentSchedulerState(agent_idx=0, step_counter=0, all_done_counter=0)

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

    def add_agent(self, agent: ConnectedAgent, skipAgent=False) -> None:
        """
        Adds an agent to the scheduler.

        Args:
            agent (ConnectedAgent): The agent instance to add.
        """
        if not skipAgent:
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

    def step_all(self) -> int:
        """
        Loop over agents until all are done. We call `step()` repeatedly, which
        runs exactly ONE agent, then move to the next. Once every agent returns
        False consecutively, we break.

        :return: The step counter
        """
        # Initialize a new persistent counter if it doesn't exist yet
        self.state.all_done_counter = 0

        rich_console.print(f"[red]Start scheduler at step {self.state.step_counter} agents={len(self.agents)}[/red]")

        while True:
            round = int(self.state.step_counter // len(self.agents))
            if self.state.agent_idx == 0:
                rich_console.print(f"[red]Executing scheduler step {round}|{self.state.agent_idx} / done={self.state.all_done_counter}/{len(self.agents)}  -------------------------------------[/red]")
            #rich_console.print(f"Queue sizes:\n{json.dumps(self.queque_sizes(), indent=4)}")

            # Execute exactly one agent and increase agent_idx
            did_run = self.step()

            if not did_run:
                self.state.all_done_counter += 1

                # If we've seen `False` for every agent in a row, we are done
            if self.state.all_done_counter >= len(self.agents):
                rich_console.print(f"[red]No active agent found scheduler step {round} / {self.state.all_done_counter} / {len(self.agents)} [/red]")
                break

            # Save progress after one round robin
            if self.state.agent_idx == 0:
                self.state.all_done_counter = 0
                if self.save_dir is not None and round % self.save_step == 0:
                    dir = f"{self.save_dir}/step_{round}"
                    os.makedirs(dir, exist_ok=True)
                    self.save_state(dir)
                    self.save_agents(dir)



        return self.state.step_counter

    def step(self) -> bool:
        """
        Runs one step for exactly ONE agent (the next in sequence).

        Returns:
            bool: True if that agent performed work, False otherwise.
        """
        if self.state.agent_idx is None:
            self.state.agent_idx = 0

        if self.state.step_counter is None:
            self.state.step_counter = 0

        # Make sure we wrap around if index >= len(agents)
        if self.state.agent_idx >= len(self.agents):
            self.state.agent_idx = 0

        idx = self.state.agent_idx
        agent = self.agents[idx]

        rich_console.print(f"[green]#{idx}:Running Agent {agent}, active={agent.is_active}[/green]")

        # Advance agent_idx for the next call, wrapping around
        self.state.agent_idx = (self.state.agent_idx + 1) % len(self.agents)
        self.state.step_counter += 1

        # Skip inactive agents (always returns False if inactive)
        if not agent.is_active:
            rich_console.print(f"   [grey53]#{idx}:Agent {agent} is inactive[/grey53]")
            did_run = False
        else:
            try:
                # rich_console.print(f"   [blue]#{idx}:stepping Agent {agent} is active[/blue]")
                result = agent.step()
            except SchedulerException as e:
                rich_console.print(
                    f"[red]#{idx}:[ERROR] Agent {e.agent_name} #{self.state.agent_idx} failed in step {self.state.step_counter} with: {e.original_exception}[/red]"
                )
                if self.error_dir:
                    self.save_agents(self.error_dir)
                    self.save_state(self.error_dir)
                raise

            # Show the result in logs
            if result:
                rich_console.print(f"   [orange3]#{idx}:Agent {agent} step result finished (running)[/orange3]")
                did_run = True
            else:
                # rich_console.print(f"   [grey53]#{idx}:Agent {agent} step result finished (idle)[/grey53]")
                did_run = False



        return did_run

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



    def save_state(self, dir:str) -> None:
        """
        Save all agents and scheulder to dir.
        :param dir:  save dir
        """
        path = os.path.join(dir, "scheduler_state.json")
        with open(path, "w") as f:
            json.dump(self.state.model_dump(), f, indent=2)

    def load_state(self, dir:str) -> None:
        """
        Load all agents and sschulder.
        :param dir: The dir
        """
        path = os.path.join(dir, "scheduler_state.json")
        with open(path, "r") as f:
            data = json.load(f)
        self.state = AgentSchedulerState.model_validate(data)

