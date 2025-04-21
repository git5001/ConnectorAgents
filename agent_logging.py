import logging

from rich.console import Console

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnectedAgents")


rich_console = Console(width=200)