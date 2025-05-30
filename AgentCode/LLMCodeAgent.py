import os
import re
from typing import Type, List, Optional
import io
import contextlib

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from openai.types import CompletionUsage
from pydantic import Field, BaseModel

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from util.AgentMemory import AgentMemory
from agent_logging import logger
from util.ChatSupport import ChatSupport, CallStep, Prompt
from util.LLMSupport import LLMAgentConfig, LLMModel, ChatMessage, LLMReply, LLMRequest

import subprocess

_cached_pkgs = None

def get_top_level_packages(limit=None):
    global _cached_pkgs
    if _cached_pkgs is None:
        try:
            result = subprocess.run(
                ["pip", "list", "--not-required", "--format=freeze"],
                stdout=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to get pip package list: %s", str(e))
            return []
        packages = result.stdout.strip().split("\n")
        clean_pkgs = [pkg.split("==")[0] for pkg in packages if pkg]
        if limit is not None:
            _cached_pkgs = clean_pkgs[:limit]
        else:
            _cached_pkgs = clean_pkgs
    return _cached_pkgs


def sysprompt(path, use_packages):
    if path:
        pathdata =f"- Output files can be written to: '{path}' No other path must be written to."
    else:
        pathdata = "- You are not allowed to write any files."

    available_packages = get_top_level_packages(30)
    if available_packages:
        pkgs = ', '.join(available_packages)
        packagedata = f"- The following Python packages are explicitly installed too: {pkgs}\n"
        packagedata = f"""
## Packages
Standard packaes are installed like 
- pandas, requests, ...
{packagedata}
            """
    else:
        packagedata = ""
    if not use_packages:
        packagedata = ""

    system_promt = f"""
You are a Python code-writing assistant.

Your job is to help solve problems by:
1. Understanding the user's request
2. Writing clear, executable Python code
3. Reviewing the output after execution
4. Refining the code if needed, based on results or errors

## Capabilities
- Generate Python code to solve tasks or answer questions
- Use appropriate libraries and packages when needed
- Adjust the code based on execution results or errors
- Explain your reasoning clearly to the user (in python comments)

## Code Execution
- The Python code you generate will be executed in a Python environment
- Execute everyting in a headless environemt. Use only headless GUI output (e.g. matplotlib)
- You will receive back the execution output or error message
- You must interpret this and adjust your code or logic if necessary
- Put in debug prints if you are unsure about the code

## Code Style
- Include all necessary imports
- Make code readable and self-contained
- Add helpful comments if it aids understanding
- Avoid relying on previous state unless explicitly told it is preserved

## APIs and Libraries
- Prefer domain-specific libraries when appropriate
- Use built-in functionality when it’s enough

## Interaction Flow
1. User makes a request describing what they want done
2. Agent generates Python code to fulfill that request
3. Code is executed and results are returned
4. Agent interprets the results and refines the code if needed 

{packagedata}

## Program output
- STDOUT and STDERR of the program are fed back to the user. 
- So all important program output must go to STDOUT possible errors to STDERR.
{pathdata}


## Output Format
- Output only python which can directly run. 
- Never output any backticks or markers
- Provide a concise explanations only inside python comments
- The code will and must directly run
"""
    return system_promt

class LLMCodeAgentConfig(LLMAgentConfig):
    """
    Configuration class for LLMAgent, defining model parameters and API key.
    """
    code_dir: Optional[str] = Field(None, description="code dir")

class CodeMessage(BaseModel):
    """
    Chat code result. the field must be called '*text' for the memor to work.
    """
    text: str = Field(..., description="The result of the query.")
    stdout: str = Field(..., description="The output of the query.")
    stderr: str = Field(..., description="The error output of the query.")

    def to_text(self) -> str:
        """Return a nicely formatted summary of the message."""
        parts = []
        if self.text:
            parts.append(f"Code:\n{self.text}")
        if self.stdout:
            parts.append(f"Standard Output:\n{self.stdout}")
        if self.stderr:
            parts.append(f"Error Output:\n{self.stderr}")
        return "\n".join(parts)

class CodeAgentState(BaseModel):
    """
    State model for the CodeLLMAgent, including memory of interactions.
    """
    memory: AgentMemory = Field(..., description="Agent memory for history of interactions.")


class CodeLLMAgent(ConnectedAgent):
    """
    An agent specialized in writing, executing, and refining Python code.

    Loop:
        1. Generate code based on SYSTEM_PROMPT and user query
        2. Execute code
        3. On error: send error back for up to 3 loops
        4. Return final code or error after 3 attempts
    """
    input_schema: Type[BaseIOSchema] = LLMRequest
    output_schema: Type[BaseIOSchema] = LLMReply
    state_schema: Type[BaseModel] = CodeAgentState

    def __init__(self, config: LLMCodeAgentConfig) -> None:
        super().__init__(config)
        self._state = CodeAgentState(memory=AgentMemory())
        self.model = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()
        if config.code_dir:
            os.makedirs(config.code_dir, exist_ok=True)
        self._chat = ChatSupport({CallStep.DEFAULT: self.model})

    def getHistory(self, n: Optional[int] = None) -> List[dict]:
        """
        Obtain the state history of the agent
        :return: The history as list of dict
        """
        if not self._state:
            return None
        if not self._state.memory:
            return None
        return self._state.memory.get_history(n)

    def getFormattedHistory(self, n: Optional[int] = None):
        history = self.getHistory(n)
        if not history:
            return None
        return '\n'.join(f"{msg['role']}: {msg['content']}" for msg in history)

    def run(self, user_input: LLMRequest) -> LLMReply:
        """
        Main entry point: generates, executes, and refines Python code based on the user's request.

        Args:
            user_input (LLMRequest): Contains the `user` message describing the task.

        Returns:
            LLMReply: Contains the final code, captured stdout/stderr or an error after up to 3 attempts.
        """
        history: List[dict] = self._state.memory.get_history() if self.config.use_memory else None
        last_error: Optional[str] = None
        final_code: Optional[str] = None
        usage: Optional[CompletionUsage] = None
        stdout_capture: str = ""
        stderr_capture: str = ""

        # Loop up to 3 attempts to refine code
        for attempt in range(1, 4):
            if attempt == 1:
                prompt_text = f"YOUR TASK: {user_input.user}"
                use_packages = False
            else:
                prompt_text = (
                        f"Previous code:\n# -------- \n{final_code}\n# --------"
                        f"Previous stdout:\n{stdout_capture}\n# --------"
                        f"Previous stderr:\n{stderr_capture}\n# --------"
                        f"The previous execution resulted in an error:\n{last_error}\n# --------"
                        f"Analyse the error carefully. Put a step by step analsis at the beginning of the new file "
                        f"in python comments, like"
                        f"# The previous code failed because I did not open a file"
                        f"User TASK:{user_input.user}\n# --------"
                        f"Now Please fix the previous code accordingly to fulfill the original request:"
                )
                use_packages = True

            sys_p = sysprompt(self.config.code_dir, use_packages)

            prompt = Prompt(
                step=CallStep.DEFAULT,
                sysText=sys_p,
                prompt=prompt_text,
                title=f"Code generation attempt #{attempt}",
                userMessage=None,
                temperature=0.2
            )
            logger.info(f"CodeLLMAgent: Generating code (attempt {attempt})... code dir {self.config.code_dir}")
            code_text, usage = self._chat.generate_chat(prompt, history)
            # Strip triple-backtick fences if present
            fence_match = re.match(r"^\s*```(?:\w+)?\n([\s\S]*?)\n```\s*$", code_text)
            if fence_match:
                code_text = fence_match.group(1)
            final_code = code_text
            logger.info(f"CodeLLMAgent: Generated code\n# -------\n{final_code}\n# ------")

            if self.config.code_dir:
                file_path = os.path.join(self.config.code_dir, "generated_code.py")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(final_code)
                file_path = os.path.join(self.config.code_dir, "prompt.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"{sys_p}\n\n{prompt_text}")
                logger.info(f"CodeLLMAgent: Code written to {file_path}")

            # Capture stdout and stderr during execution
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            execution_env = {}
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code_text, execution_env, execution_env)
                stdout_capture = stdout_buf.getvalue()
                stderr_capture = stderr_buf.getvalue()
                logger.info("Code executed successfully.")
                logger.info(f"CodeLLMAgent: stdout_capture {stdout_capture}")
                logger.info(f"CodeLLMAgent: stderr_capture {stderr_capture}")
                last_error = None
                break
            except Exception as e:
                stdout_capture = stdout_buf.getvalue()
                stderr_capture = stderr_buf.getvalue() + str(e)
                last_error = str(e)
                logger.warning(f"Execution error on attempt {attempt}: {last_error}")
                logger.warning(f"CodeLLMAgent: stdout_capture {stdout_capture}")
                logger.warning(f"CodeLLMAgent: stderr_capture {stderr_capture}")
                if attempt == 3:
                    logger.error("Max attempts reached. Returning error and logs.")
                    return LLMReply(
                        usage=usage,
                        reply=CodeMessage(text=f"Error after 3 attempts: {last_error}", stdout=stdout_capture, stderr=stderr_capture),
                        error=last_error
                    )
                # Continue to next iteration to refine code

        # Store conversation in memory if enabled
        if self.config.use_memory:
            self._state.memory.add_message(role="user", content=ChatMessage(text=user_input.user))
            self._state.memory.add_message(role="assistant", content=CodeMessage(text=final_code or "", stdout=stdout_capture, stderr=stderr_capture))



        # Return the final code snippet with captured outputs
        return LLMReply(
            usage=usage,
            reply=CodeMessage(text=final_code, stdout=stdout_capture, stderr=stderr_capture),
            error=None
        )
