import time
import os
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from rich.console import Console
import os
import json

from AgentFramework import ConnectedAgent
from util.SchedulerException import SchedulerException

rich_console = Console()

class AgentSchedulerState(BaseModel):
    """
    Schema for the scheduler

    """
    agent_idx: int = Field(..., description="Agent index running")
    step_counter: int = Field(..., description="Step counter")

class AgentScheduler:
    """
    Scheduler for managing and executing agent tasks.

    This class maintains a list of agents, runs them step by step until
    they have no more tasks, and retrieves final outputs from agents
    that serve as sinks for messages.

    Attributes:
        agents (List[ConnectedAgent]): List of agents managed by the scheduler.
    """

    def __init__(self, save_dir=None, error_dir=None) -> None:
        """Initializes an empty agent scheduler."""
        self.agents: List[ConnectedAgent] = []
        self.save_dir = save_dir
        self.error_dir = error_dir
        self.state = AgentSchedulerState(agent_idx=0, step_counter=0)

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

    def step_all(self) -> int:
        """
        Loop all agents until all are done
        :return: The step counter
        """
        # Loop all
        self.state.step_counter = 0
        while self.step():
            rich_console.print(f"[red]Executing scheduler step {self.state.step_counter+1} -------------------------------------[/red]")
            self.state.step_counter += 1
            time.sleep(0.01)
        return self.state.step_counter



    def step(self) -> bool:
        """
        Runs one step for all agents.

        Returns:
            bool: True if any agent performed work, False otherwise.
        """
        did_run = False
        start_index = self.state.agent_idx or 0
        for idx in range(start_index, len(self.agents)):
            agent = self.agents[idx]
            rich_console.print(f"[green]#{idx}:Running Agent {agent} , active={agent.is_active}[/green]")
            self.state.agent_idx = idx
            # Omit idle agents
            if not agent.is_active:
                continue
            try:
                result = agent.step()
            except SchedulerException as e:
                rich_console.print(f"[red]#{idx}:[ERROR] Agent {e.agent_name} failed with: {e.original_exception}[/red]")
                if self.error_dir:
                    self.save_agents(self.error_dir)
                    self.save_state(self.error_dir)
                raise
            rich_console.print(f"   [grey53]#{idx}:Agent {agent} step result finished: {result}[/grey53]")
            if result:
                did_run = True

        # reset state
        self.state.agent_idx = 0

        # Save progress
        if self.save_dir is not None:
            dir = f"{self.save_dir}/step_{self.state.step_counter}"
            os.makedirs(dir, exist_ok=True)
            self.save_state(dir)
            self.save_agents(dir)


        return did_run


    def get_final_outputs(self) -> Dict[str, List[BaseModel]]:
        """
        Retrieves and clears final outputs from all agents that serve as sinks.

        Returns:
            Dict[str, List[BaseModel]]: A dictionary where keys are agent class names
            and values are lists of final output data from those agents.
        """
        return {agent.__class__.__name__: agent.get_final_outputs() for agent in self.agents}

    def get_one_output_per_agent(self) -> Dict[str, Optional[BaseModel]]:
        """
        Retrieves and removes one message per agent if available.

        Returns:
            Dict[str, Optional[BaseModel]]: A dictionary where keys are agent class names
            and values are a single output message from each agent, if available.
        """
        return {agent.__class__.__name__: agent.get_one_output() for agent in self.agents}


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
            rich_console.print(f"   [blue]Saving agent {agent.__class__.__name__}[/blue] to {path}")



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
            agent.load_state_from_file(filepath)
            rich_console.print(f"[blue]Loaded agent {agent.__class__.__name__}[/blue] from {filepath}")
        rich_console.print(f"[blue]Loaded {len(self.agents)} agents from '{directory}'.[/blue]")



    def save_state(self, dir):
        path = os.path.join(dir, "scheduler_state.json")
        with open(path, "w") as f:
            json.dump(self.state.model_dump(), f, indent=2)

    def load_state(self, dir) -> AgentSchedulerState:
        path = os.path.join(dir, "scheduler_state.json")
        with open(path, "r") as f:
            data = json.load(f)
        self.state = AgentSchedulerState.model_validate(data)

