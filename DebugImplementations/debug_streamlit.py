#!/usr/bin/env python3
"""
Streamlit UI for the TCPDebugger used in the agent pipeline.

 • Connects to the debugger at 127.0.0.1:5005 (override in sidebar)
 • Streams all events in real time (auto-refresh every second)
 • Pause / Continue / Step control
 • Background colour = even/odd step number
 • Header includes agent class and UUID
 • Active agents: green (even) / blue (odd)
 • Inactive agents: pale orange (even) / pale red (odd)
 • NEW: “Clear view” button to flush the current log
"""

from __future__ import annotations

import datetime as dt
import html
import json
import socket
import threading
import time
from queue import Empty, Queue
from typing import List
import json
from typing import Any, Dict, Union

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ───────────────────────────────────────────────────────────────
# TCP client thread
# ───────────────────────────────────────────────────────────────
class TCPClient(threading.Thread):
    """Background thread that maintains the TCP connection."""

    def __init__(self, host: str, port: int, queue: Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.queue, self.stop = queue, stop
        self.sock: socket.socket | None = None

    def run(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(0.5)
                backoff = 1.0
                self.queue.put({"event": "_status", "state": "connected"})
                buf = b""
                while not self.stop.is_set():
                    try:
                        chunk = self.sock.recv(4096)
                        if not chunk:
                            raise ConnectionResetError
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            try:
                                evt = json.loads(line.decode())
                            except Exception as exc:
                                evt = {"event": "_error", "error": str(exc), "raw": line.decode()}
                            self.queue.put(evt)
                    except socket.timeout:
                        continue
            except (ConnectionRefusedError, ConnectionResetError, OSError):
                self.queue.put({"event": "_status", "state": "disconnected"})
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
            finally:
                if self.sock:
                    try:
                        self.sock.close()
                    except OSError:
                        pass
                self.sock = None
        self.queue.put({"event": "_status", "state": "stopped"})

    def send_control(self, action: str) -> None:
        if self.sock:
            try:
                self.sock.sendall(json.dumps({"action": action}).encode() + b"\n")
            except OSError:
                pass


# ───────────────────────────────────────────────────────────────
# Helpers: colour constants
# ───────────────────────────────────────────────────────────────
_STYLE_GREEN            = "background-color:#e8ffe8;"   # active, even
_STYLE_BLUE             = "background-color:#e8f1ff;"   # active, odd
_STYLE_INACTIVE_EVEN    = "background-color:#fff3e0;"   # inactive, even
_STYLE_INACTIVE_ODD     = "background-color:#ffecec;"   # inactive, odd


def parse_json_string(value: Any) -> Union[Dict, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {k: parse_json_string(v) for k, v in parsed.items()}
            return parsed
        except json.JSONDecodeError:
            return value
    elif isinstance(value, dict):
        return {k: parse_json_string(v) for k, v in value.items()}
    else:
        return value

def build_html_tree(obj: Any, indent: int = 0) -> str:
    html_out = ""
    indent_px = indent * 16
    if isinstance(obj, dict):
        for key, value in obj.items():
            html_out += (
                f"<p style='margin:0;padding:0;margin-left:{indent_px}px;'>"
                f"<strong>{html.escape(str(key))}</strong>:</p>"
            )
            html_out += build_html_tree(value, indent + 1)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            html_out += (
                f"<p style='margin:0;padding:0;margin-left:{indent_px}px;'>"
                f"[{idx}]</p>"
            )
            html_out += build_html_tree(item, indent + 1)
    else:
        # leaf: escape and replace newlines
        raw = str(obj)
        # first collapse any double-CR into a single gap
        collapsed = raw.replace("\r\n", "\n").replace("\n\n", "\n")
        # now escape, then convert single \n to <br>
        safe = html.escape(collapsed).replace("\n", "<br/>")
        html_out += (
            f"<p style='margin:0;padding:0;margin-left:{indent_px}px;'>"
            f"{safe}</p>"
        )
    return html_out

def _render_event(evt: Dict[str, Any], style: str) -> None:
    ts      = dt.datetime.fromtimestamp(evt.get("timestamp", 0)).strftime("%H:%M:%S")
    et      = evt.get("event", "<event>").upper()
    step    = evt.get("step")
    agent   = evt.get("agent_type", "<agent>")
    uuid    = evt.get("agent_uuid", "")
    parts: List[str] = [et, f"{agent}:{uuid}"]
    if step is not None:
        parts.append(f"step {step}")
    parts.append(ts)
    header = " | ".join(parts)

    st.markdown(
        f"<div style='{style} padding:6px 10px; border-radius:6px;'>"
        f"<strong>{header}</strong></div>",
        unsafe_allow_html=True,
    )
    st.json(evt, expanded=False)
    parsed_evt = parse_json_string(evt.get("payload", {}))
    tree_html = build_html_tree(parsed_evt)
    with st.expander("Event Details", expanded=False):
        st.markdown(tree_html, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# Streamlit app
# ───────────────────────────────────────────────────────────────
st.set_page_config("Agent TCP Debugger", layout="wide")

# Session-persistent objects
if "queue"          not in st.session_state: st.session_state.queue          = Queue()
if "stop_event"     not in st.session_state: st.session_state.stop_event     = threading.Event()
if "client_thread"  not in st.session_state: st.session_state.client_thread  = None  # type: ignore
if "events"         not in st.session_state: st.session_state.events         = []    # type: List[Dict[str, Any]]
if "connected"      not in st.session_state: st.session_state.connected      = False
if "desired_pause"  not in st.session_state: st.session_state.desired_pause  = True  # start paused

# ───────── Sidebar ─────────
st.sidebar.title("Debugger Connection")
host = st.sidebar.text_input("Host", "127.0.0.1")
port = st.sidebar.number_input("Port", value=5005, step=1)

if st.sidebar.button("Connect" if not st.session_state.connected else "Disconnect"):
    if st.session_state.connected:
        st.session_state.stop_event.set()
        st.session_state.connected = False
    else:
        st.session_state.stop_event = threading.Event()
        st.session_state.queue      = Queue()
        st.session_state.events     = []
        client = TCPClient(host, int(port), st.session_state.queue, st.session_state.stop_event)
        client.start()
        st.session_state.client_thread = client

# Pause / Step controls
st.session_state.desired_pause = st.sidebar.toggle(
    "Pause pipeline", value=st.session_state.desired_pause,
)
step_clicked = st.sidebar.button("Step once", disabled=not st.session_state.connected)

# NEW: Clear view button
if st.sidebar.button("Clear view"):
    st.session_state.events = []
    with st.session_state.queue.mutex:          # flush any queued items
        st.session_state.queue.queue.clear()

if st.session_state.connected and st.session_state.client_thread:
    st.session_state.client_thread.send_control(
        "pause" if st.session_state.desired_pause else "continue"
    )
    if step_clicked:
        st.session_state.client_thread.send_control("step")

# ───────── Drain queue ─────────
while True:
    try:
        evt = st.session_state.queue.get_nowait()
    except Empty:
        break
    if evt.get("event") == "_status":
        st.session_state.connected = evt.get("state") == "connected"
    else:
        st.session_state.events.append(evt)
        if len(st.session_state.events) > 1000:
            st.session_state.events = st.session_state.events[-1000:]

# ───────── Display ─────────
st.title("Agent TCP Debugger")
st.markdown("**Status:** " + ("🟢 Connected" if st.session_state.connected else "🔴 Disconnected"))

last_step: int | None = None

for event in reversed(st.session_state.events):
    if event.get("step") is not None:
        last_step = int(event["step"])
    parity = (last_step or 0) % 2
    active = event.get("active", True)

    if not active:
        style = _STYLE_INACTIVE_EVEN if parity == 0 else _STYLE_INACTIVE_ODD
    else:
        style = _STYLE_GREEN if parity == 0 else _STYLE_BLUE

    _render_event(event, style)

# Auto-refresh every second
st_autorefresh(interval=1000, limit=None, key="auto-refresh")
