"""
TCP Debugger implementation for agent pipelines.

This module defines a generic ``DebugInterface`` plus a concrete ``TCPDebugger``
that transmits debugging events over a TCP socket and listens for control
commands (``pause`` / ``continue``).

Design highlights
-----------------
* **Non‑blocking**: All socket work lives in a background thread so the agent
  loop never blocks.
* **Stateless send**: Debug events are *fire‑and‑forget*. If the UI is not
  connected, we silently drop the data.
* **Latest‑state pause control**: A single boolean flag ``pause_requested``
  is updated whenever a control message arrives; ``is_pause`` simply returns
  that flag. No message queue or buffering required.
* **Minimal dependencies**: Only Python stdlib + Pydantic.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import List, Optional, Union, Any

from pydantic import BaseModel

from AgentFramework.core.DebugInterface import DebugInterface

# ======================================================================
# Concrete implementation: TCPDebugger
# ======================================================================
class TCPDebugger(DebugInterface):
    """Debugger that streams JSON events over TCP and listens for control."""

    _PAUSE_ACTIONS = {"pause", "stop"}
    _CONTINUE_ACTIONS = {"continue", "resume"}
    _STEP_ACTIONS = {"step", "single"}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5005,
        reconnect_interval: float = 0.5,
        start_paused: bool = True,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval

        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None

        self.connected: bool = False
        self._step_once: bool = False
        self._start_paused = start_paused
        self._pause_requested = self._start_paused

        self._pause_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Life-cycle --------------------------------------------------------
    # ------------------------------------------------------------------
    def init_debugger(self, timeout: int = 120) -> None:
        if self._thread and self._thread.is_alive():
            return

        print(f"[TCPDebugger] listening on {self.host}:{self.port} (timeout {timeout}s)")
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(timeout)

        self._thread = threading.Thread(target=self._tcp_loop, daemon=True)
        self._thread.start()

        start = time.time()
        while not self.connected and (time.time() - start) < timeout:
            time.sleep(0.1)

    def exit_debugger(self) -> None:
        self._pause_requested = self._start_paused
        self._stop_event.set()
        try:
            if self._client_socket:
                self._client_socket.shutdown(socket.SHUT_RDWR)
                self._client_socket.close()
        except OSError:
            pass
        try:
            if self._server_socket:
                self._server_socket.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Event callbacks ---------------------------------------------------
    # ------------------------------------------------------------------
    def no_input(self, agent: "ConnectedAgent") -> None:
        self._send({
            "event": "no_input",
            "timestamp": time.time(),
            "agent_type": type(agent).__name__,
            "active":False,
            "agent_uuid": getattr(agent, "uuid", "default"),
        })

    def input(self, agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        self._send({
            "event": "input",
            "timestamp": time.time(),
            "parents": parents,
            "agent_type": type(agent).__name__,
            "active":agent.is_active,
            "agent_uuid": getattr(agent, "uuid", "default"),
            "payload": self._dump(msg),
        })

    def output(self, agent: "ConnectedAgent", msg: BaseModel, parents: List[str]) -> None:
        self._send({
            "event": "output",
            "timestamp": time.time(),
            "parents": parents,
            "agent_type": type(agent).__name__,
            "active":agent.is_active,
            "agent_uuid": getattr(agent, "uuid", "default"),
            "payload": self._dump(msg),
        })

    def start_agent(self, agent: "ConnectedAgent", step_count: int) -> None:
        self._send({
            "event": "start_agent",
            "timestamp": time.time(),
            "step": step_count,
            "agent_type": type(agent).__name__,
            "active":agent.is_active,
            "agent_uuid": getattr(agent, "uuid", "default"),
        })

    def finished_agent(
        self,
        agent: "ConnectedAgent",
        step_count: int,
        did_run: bool,
    ) -> None:
        self._send({
            "event": "finished_agent",
            "timestamp": time.time(),
            "step": step_count,
            "did_run": did_run,
            "agent_type": type(agent).__name__,
            "active":agent.is_active,
            "agent_uuid": getattr(agent, "uuid", "default"),
        })

    def error_agent(
        self,
        agent: "ConnectedAgent",
        step_count: int,
        exception: Exception,
    ) -> None:
        self._send({
            "event": "error_agent",
            "timestamp": time.time(),
            "step": step_count,
            "agent_type": type(agent).__name__,
            "active":agent.is_active,
            "agent_uuid": getattr(agent, "uuid", "default"),
            "error": {
                "type": type(exception).__name__,
                "msg": str(exception),
            },
        })

    def is_pause(self, pause_count: int, step_counter: int) -> bool:
        with self._pause_lock:
            if self._step_once:
                # consume the "step once" token → pause again next time
                self._step_once = False
                return False  # allow exactly one iteration
            return self._pause_requested

    def user_message(
        self,
        name: str,
        agent: "ConnectedAgent",
        data: Union[BaseModel, str],
    ) -> None:
        self._send({
            "event": "user_message",
            "timestamp": time.time(),
            "name": name,
            "agent_type": type(agent).__name__,
            "active":agent.is_active,
            "agent_uuid": getattr(agent, "uuid", "default"),
            "payload": self._dump(data),
        })

    # ------------------------------------------------------------------
    # Background thread -------------------------------------------------
    # ------------------------------------------------------------------
    def _tcp_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.connected:
                try:
                    self._server_socket.settimeout(self.reconnect_interval)
                    client, _ = self._server_socket.accept()
                    client.settimeout(0.1)
                    self._client_socket = client
                    self.connected = True
                except socket.timeout:
                    continue
                except OSError:
                    time.sleep(self.reconnect_interval)
                    continue

            try:
                data = self._client_socket.recv(4096)
                if not data:
                    self._drop_connection()
                    continue
                for line in data.splitlines():
                    self._handle_control(line)
            except socket.timeout:
                pass
            except OSError:
                self._drop_connection()

        self._drop_connection()

    # ------------------------------------------------------------------
    # Control message handling -----------------------------------------
    # ------------------------------------------------------------------
    def _handle_control(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        action = msg.get("action", "").lower()

        with self._pause_lock:
            if action in self._PAUSE_ACTIONS:
                self._pause_requested = True

            elif action in self._CONTINUE_ACTIONS:
                self._pause_requested = False

            elif action in self._STEP_ACTIONS:  # now handled like the others
                self._step_once = True

    def _drop_connection(self) -> None:
        self.connected = False
        self._pause_requested = self._start_paused
        try:
            if self._client_socket:
                self._client_socket.close()
        except OSError:
            pass
        self._client_socket = None

    # ------------------------------------------------------------------
    # Sending utilities -------------------------------------------------
    # ------------------------------------------------------------------
    def _send(self, payload: dict) -> None:
        if not self.connected or not self._client_socket:
            return
        try:
            try:
                jpload = json.dumps(payload)
            except TypeError as e:
                print("Cannot serialize object ",type(payload),e)
                print("payload",payload)
            self._client_socket.sendall((jpload + "\n").encode())
        except OSError as e:
            print("Json error ",e)
            self._drop_connection()

    @staticmethod
    def _dump(obj: Any) -> Any:
        # If it's a Pydantic model, dump it to a dict
        if isinstance(obj, BaseModel):
            return obj.model_dump()

        # If it's a dict, recurse on its values
        if isinstance(obj, dict):
            return {key: TCPDebugger._dump(val) for key, val in obj.items()}

        # If it's a list or tuple, recurse on the elements
        if isinstance(obj, (list, tuple)):
            dumped = [TCPDebugger._dump(item) for item in obj]
            # preserve tuple vs list
            return tuple(dumped) if isinstance(obj, tuple) else dumped

        # otherwise assume it's already JSON serializable (str, int, etc.)
        return obj
