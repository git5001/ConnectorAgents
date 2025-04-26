#!/usr/bin/env python3
"""
Debug Console – lightweight CLI to view events from ``TCPDebugger``.

Changes (April 2025)
--------------------
* ``input`` / ``output`` events now include ``agent_type`` and ``agent_uuid``.
  This console shows every key/value pair automatically, so **no code change
  was needed**, but the documentation has been updated for clarity.

Usage
-----
$ python debug_console.py [HOST] [PORT]

Dependencies:
    pip install rich
"""
from __future__ import annotations

import json
import socket
import sys
from typing import Any, Dict

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console(highlight=False, emoji=False)

def _render_event(event: Dict[str, Any]) -> None:  # noqa: D401
    evt_type = event.get("event", "<unknown>")
    console.rule(Text(evt_type.upper(), style="bold cyan"))

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold white")
    table.add_column()

    for key, value in event.items():
        if key == "event":
            continue
        pretty = json.dumps(value, indent=2) if not isinstance(value, str) else value
        table.add_row(key, pretty)

    console.print(table)


def listen(host: str = "127.0.0.1", port: int = 5005) -> None:  # noqa: D401
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        console.print(f"[green]Connecting to {host}:{port} …[/green]")
        sock.connect((host, port))
        console.print("[green]Connected! Waiting for events …[/green]")
        with sock.makefile("r") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    console.print(f"[red]Malformed JSON:[/red] {line}")
                    continue
                _render_event(event)


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5005

    try:
        listen(host, port)
    except ConnectionRefusedError:
        console.print(f"[red]Connection refused: {host}:{port}. Is TCPDebugger running?[/red]")
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")


if __name__ == "__main__":
    main()
