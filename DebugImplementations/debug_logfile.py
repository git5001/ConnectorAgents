#!/usr/bin/env python3
"""
Debug Logger – lightweight CLI to record and filter events from TCPDebugger to a log file.

Usage:
    python debug_logger.py logfile.log [--host HOST] [--port PORT] [--filter EVENT1,EVENT2,...]

Dependencies:
    None (uses Python standard library)
"""

import json
import re
import socket
import sys
import argparse
from typing import List, Optional

def listen(host: str, port: int, logfile: str, filters: Optional[List[str]] = None) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, open(logfile, "w") as logf:
        print(f"Connecting to {host}:{port}...")
        sock.connect((host, port))
        print(f"Connected! Writing events to {logfile}...")
        try:
            with sock.makefile("r") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logf.write(f"Malformed JSON: {line}\n\n")
                        logf.flush()
                        continue

                    evt_type = event.get("event", "<unknown>")
                    if filters and evt_type not in filters:
                        continue

                    payload_raw = event.pop("payload", None)

                    # Log the rest of the event
                    logf.write("Event:\n")
                    logf.write(json.dumps(event, indent=2))
                    logf.write("\n")

                    # Decode & pretty-print the payload
                    if payload_raw is not None:
                        logf.write("Payload:\n")
                        # Try to parse if it’s a JSON-string
                        if isinstance(payload_raw, str):
                            try:
                                payload_obj = json.loads(payload_raw)
                            except json.JSONDecodeError:
                                payload_obj = payload_raw
                        else:
                            payload_obj = payload_raw

                        if isinstance(payload_obj, (dict, list)):
                            dumped = json.dumps(payload_obj, indent=2)
                            lines = []
                            for locl in dumped.splitlines():
                                if '\\n' in locl:
                                    # Find the amount of leading space to reuse
                                    leading = re.match(r'^(\s*)', locl).group(1)
                                    extra_indent = leading + ' '*12  # Cheat: add spaces
                                    # Replace \n with \n + same indent
                                    locl = locl.replace('\\n', '\n' + extra_indent)
                                lines.append(locl)
                            logf.write('\n'.join(lines) + "\n")
                        else:
                            # payload_obj is a plain string—write it directly
                            logf.write(str(payload_obj) + "\n")

                    logf.write("\n")
                    logf.flush()

        except ConnectionResetError:
            print("Connection closed by remote. Exiting.")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")


def parse_filters(filter_str: Optional[str]) -> Optional[List[str]]:
    """Parses a comma-separated filter string into a list, or returns None if empty."""
    if not filter_str:
        return None
    items = [f.strip() for f in filter_str.split(',') if f.strip()]
    return items if items else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record and filter TCPDebugger events to a log file."
    )
    parser.add_argument(
        'logfile', help="Path to the output log file"
    )
    parser.add_argument(
        '--host', '-H', default='127.0.0.1',
        help="IP address to connect to (default: 127.0.0.1)"
    )
    parser.add_argument(
        '--port', '-P', type=int, default=5005,
        help="Port to connect to (default: 5005)"
    )
    parser.add_argument(
        '--filter', '-f',
        help="Comma-separated list of event types to log (e.g. user_message,input,output)."
    )
    args = parser.parse_args()

    filters = parse_filters(args.filter)
    if filters:
        print(f"Filtering events: {', '.join(filters)}")

    try:
        listen(args.host, args.port, args.logfile, filters)
    except ConnectionRefusedError:
        print(
            f"Connection refused: {args.host}:{args.port}. Is TCPDebugger running?",
            file=sys.stderr
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)


if __name__ == '__main__':
    main()
