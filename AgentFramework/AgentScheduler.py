import time
from typing import Dict, Optional, List
from pydantic import BaseModel

from AgentFramework import ConnectedAgent


class AgentScheduler:
    """
    Scheduler for managing and executing agent tasks.

    This class maintains a list of agents, runs them step by step until
    they have no more tasks, and retrieves final outputs from agents
    that serve as sinks for messages.

    Attributes:
        agents (List[ConnectedAgent]): List of agents managed by the scheduler.
    """

    def __init__(self) -> None:
        """Initializes an empty agent scheduler."""
        self.agents: List[ConnectedAgent] = []

    def add_agent(self, agent: ConnectedAgent) -> None:
        """
        Adds an agent to the scheduler.

        Args:
            agent (ConnectedAgent): The agent instance to add.
        """
        self.agents.append(agent)

    def step(self) -> bool:
        """
        Runs one step for all agents.

        Returns:
            bool: True if any agent performed work, False otherwise.
        """
        return any(agent.step() for agent in self.agents)

    def step_all(self) -> None:
        """
        Continuously runs agent steps until all agents report no work remaining.
        """
        while self.step():
            time.sleep(0.1)  # Optional delay to prevent busy-waiting

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
