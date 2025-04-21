from typing import Optional
from pydantic import BaseModel, Field
from rich.console import Console

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from agent_logging import rich_console


class PrintAgentConfig(BaseToolConfig):
    """
    Configuration for Print Agent.

    Attributes:
        log_to_file (bool): If true, logs output to a file instead of just printing.
        log_file_path (Optional[str]): File path for logging output if enabled.
    """
    log_to_file: bool = Field(False, description="If true, logs output to a file instead of just printing")
    log_file_path: Optional[str] = Field(None, description="File path for logging output if enabled")


class PrintMessageInput(BaseModel):
    """
    Schema for a printable message.

    Attributes:
        subject (str): Message subject.
        body (str): Message body content.
    """
    subject: str = Field(..., description="Message subject")
    body: str = Field(..., description="Message body content")


class PrintMessageOutput(BaseModel):
    """
    Schema for printing results.

    Attributes:
        success (bool): Indicates whether the message was printed successfully.
        message (str): Success or error message.
    """
    success: bool = Field(..., description="Indicates whether the message was printed successfully")
    message: str = Field(..., description="Success or error message")


class PrintAgent(ConnectedAgent):
    """
    Print Agent that mimics an Email Agent but prints output to the console instead of sending emails.
    Primarily used for debugging purposes.

    Attributes:
        input_schema (Type[BaseModel]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Defines the expected output schema.
        log_to_file (bool): Whether to log output to a file.
        log_file_path (Optional[str]): Path to log file if logging is enabled.
    """
    input_schema = PrintMessageInput
    output_schema = PrintMessageOutput

    def __init__(self, config: PrintAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes the Print Agent.

        Args:
            config (PrintAgentConfig): Configuration settings for the agent.
        """
        super().__init__(config, uuid)
        self.log_to_file: bool = config.log_to_file
        self.log_file_path: Optional[str] = config.log_file_path

    def print_message(self, message: PrintMessageInput) -> PrintMessageOutput:
        """
        Prints the message details to the console or logs to a file if enabled.

        Args:
            message (PrintMessageInput): The message details including subject and body.

        Returns:
            PrintMessageOutput: Contains success status and message.
        """
        output_text = (f"[bold red]PRINT AGENT OUTPUT[/bold red]\n"
                       f"[bold blue]# {message.subject}[/bold blue]\n\n"
                       f"[green]{message.body}[/green]\n"
                       f"-----------------------------------------------------\n")

        try:
            rich_console.print(output_text)

            if self.log_to_file and self.log_file_path:
                with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(output_text + "\n")

            return PrintMessageOutput(success=True, message="Message printed successfully.")
        except Exception as e:
            return PrintMessageOutput(success=False, message=f"Error printing message: {e}")

    def run(self, params: PrintMessageInput) -> PrintMessageOutput:
        """
        Runs the print agent synchronously, printing the message.

        Args:
            params (PrintMessageInput): The input parameters for the message.

        Returns:
            PrintMessageOutput: The output containing success status and message.
        """
        return self.print_message(params)
