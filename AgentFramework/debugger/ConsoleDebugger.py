import os
from pathlib import Path
from typing import List, Union, Optional
from pydantic import BaseModel
from datetime import datetime

from AgentFramework.core.DebugInterface import DebugInterface


class ConsoleDebugger(DebugInterface):
    """
    A console-based debugger that prints and optionally logs agent events and messages.
    """
    def __init__(self, print_console: bool = False, show_parents: bool = False, log_dir: Optional[Union[str, Path]] = None):
        """
        :param show_parents: If True, prints parent UUIDs for messages.
        :param log_dir: Optional directory to save a 'debug.log' file. If provided,
                        the log file is reset on init_debugger and appended thereafter.
        """
        self.show_parents = show_parents
        self.print_console = print_console
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / 'console_debug.log'
        else:
            self.log_file = None

    def _timestamp(self) -> str:
        return datetime.now().isoformat()

    def _log(self, message: str) -> None:
        """Prints to console and appends to log file if configured."""
        if self.print_console:
            print(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"[{self._timestamp()}] Failed to write log: {e}")

    def init_debugger(self, timeout: int = 30) -> None:
        """Initialize debugger, clearing log file if needed."""
        header = f"[{self._timestamp()}] Debugger initialized with timeout={timeout}s"
        if self.log_file:
            # Reset log file
            try:
                self.log_file.unlink(missing_ok=True)
            except Exception:
                pass
        self._log(header)

    def exit_debugger(self) -> None:
        self._log(f"[{self._timestamp()}] Debugger exited")

    def no_input(self, agent: "ConnectedAgent") -> None:
        self._log(f"[{self._timestamp()}] Agent {type(agent).__name__}({agent.uuid}) has no input and will idle.")

    def transmission(self, from_agent: "ConnectedAgent", to_agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        line = (f"[{self._timestamp()}] Transmission from {type(from_agent).__name__}({from_agent.uuid}) "
                f"to {type(to_agent).__name__}({to_agent.uuid}): {type(msg).__name__}")
        if self.show_parents and parents:
            line += f" | parents={parents}"
        self._log(line)

    def input(self, agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        line = f"[{self._timestamp()}] Input to {type(agent).__name__}({agent.uuid}): {type(msg).__name__}"
        if self.show_parents and parents:
            line += f" | parents={parents}"
        self._log(line)

    def output(self, agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        line = f"[{self._timestamp()}] Output from {type(agent).__name__}({agent.uuid}): {type(msg).__name__}"
        if self.show_parents and parents:
            line += f" | parents={parents}"
        self._log(line)

    def start_agent(self, agent: "ConnectedAgent", step_count: int) -> None:
        self._log(f"[{self._timestamp()}] Starting agent {type(agent).__name__}({agent.uuid}) at step {step_count}")

    def finished_agent(self, agent: "ConnectedAgent", step_count: int, did_run: bool) -> None:
        status = "ran" if did_run else "idled"
        self._log(f"[{self._timestamp()}] Finished agent {type(agent).__name__}({agent.uuid}) at step "
                  f"{step_count} (status={status})")

    def error_agent(self, agent: "ConnectedAgent", step_count: int, exception: Exception) -> None:
        self._log(f"[{self._timestamp()}] Error in agent {type(agent).__name__}({agent.uuid}) "
                  f"at step {step_count}: {exception}")

    def is_pause(self, pause_count: int, step_counter: int) -> bool:
        # Never pause by default
        return False

    def user_message(self,
                     name: str,
                     agent: "ConnectedAgent",
                     data: Union[BaseModel, str]) -> None:
        content = data if isinstance(data, str) else data.model_dump_json()
        self._log(f"[{self._timestamp()}] User message '{name}' on "
                  f"agent {type(agent).__name__}({agent.uuid}): {content}")
