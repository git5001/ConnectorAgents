"""Real‑world‑ish scenario:
User gives a task → LLM decides which tool to invoke → orchestrator builds the
corresponding agent chain:

    Phase 1:  LLMDecisionAgent
              (stubbed here; returns ToolDecision pydantic model)
    Phase 2:  one of
              • FileReadAgent
              • WebDownloadAgent
              • PythonEvalAgent

Finally the result is sent to ConsolePrinterAgent that simply prints.
"""
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.BaseOrchestratorAgent import BaseOrchestratorAgent, NextPhase, \
    OrchestratorResult
from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.NullSchema import NullSchema


# ---- shared schemas --------------------------------------------------------
class ToolDecision(BaseIOSchema):
    """ desc """
    tool: str            # "file", "web", "python"
    param: str           # path, url or python expression
    description: str = "LLM decision about which tool to call"

class FileContent(BaseIOSchema):
    """ desc """
    content: str
    description: str = "Content of a text file"
    def to_text(self):
        return self.content

class WebContent(BaseIOSchema):
    """ desc """
    html: str
    description: str = "Downloaded html page"
    def to_text(self):
        return self.html

class PythonResult(BaseIOSchema):
    """ desc """
    result: str
    description: str = "Result of evaluated expression"
    def to_text(self):
        return self.result

class TaskRequest(BaseIOSchema):
    """ desc """
    prompt: str
    description: str = "User request for the orchestrator"

class TaskAnswer(BaseIOSchema):
    """ desc """
    answer: str
    description: str = "Final answer returned by orchestrator"

# ---- tool agents -----------------------------------------------------------
class LLMPlanningAgent(ConnectedAgent):
    """Stub LLM that maps keywords to ToolDecision."""
    input_schema  = TaskRequest
    output_schema = ToolDecision
    uuid = "llm"

    def __init__(self):
        super().__init__(BaseToolConfig())

    def run(self, params: TaskRequest):
        p = params.prompt.lower()
        if "file" in p:
            return ToolDecision(tool="file", param="/tmp/demo.txt")
        if "web" in p:
            return ToolDecision(tool="web", param="https://example.com")
        return ToolDecision(tool="python", param="len('hello')")

class FileReadAgent(ConnectedAgent):
    input_schema  = ToolDecision
    output_schema = FileContent
    uuid = "file"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: ToolDecision):
        return FileContent(content=f"<stubbed file>{params.param}</stubbed>")

class WebDownloadAgent(ConnectedAgent):
    input_schema  = ToolDecision
    output_schema = WebContent
    uuid = "web"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: ToolDecision):
        return WebContent(html=f"<html>stub {params.param}</html>")

class PythonEvalAgent(ConnectedAgent):
    input_schema  = ToolDecision
    output_schema = PythonResult
    uuid = "python"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: ToolDecision):
        return PythonResult(result="5  # stubbed")

class ConsolePrinterAgent(ConnectedAgent):
    """Sink that prints whatever it receives and passes nothing on."""
    input_schema  = TaskAnswer
    output_schema = NullSchema  # will never be used
    uuid = "printer"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: TaskAnswer):
        print("=== CONSOLE PRINTER ===\n", params)
        return NullSchema()

# ---- orchestrator ----------------------------------------------------------
class ToolChoosingOrchestrator(BaseOrchestratorAgent):
    input_schema  = TaskRequest
    output_schema = TaskAnswer

    def build_phase(self, task, ctx, phase):
        print("LLM PHASE ",phase)
        if phase == 1:
            llm = LLMPlanningAgent()
            llm.feed(task)
            ctx.scratch["sink"] = llm
            return [llm]
        elif phase == 2:
            decision: ToolDecision = ctx.scratch["decision"]
            if decision.tool == "file":
                ag = FileReadAgent()
            elif decision.tool == "web":
                ag = WebDownloadAgent()
            else:
                ag = PythonEvalAgent()
            ag.feed(decision)
            ctx.scratch["sink"] = ag
            return [ag]
        else:
            raise ValueError("Unexpected phase")

    def after_phase(self, outputs, ctx, phase):
        sink = ctx.scratch["sink"]
        msgs = outputs.get(sink, [])
        if not msgs:
            ctx.error = "sink produced no output"
            return NextPhase.DONE, phase
        payload = msgs[0]
        print("PAYÖOAD",payload)

        if phase == 1:
            ctx.scratch["decision"] = payload  # ToolDecision
            return NextPhase.CONTINUE, 2
        else:
            ctx.scratch["answer"] = payload.to_text()
            return NextPhase.DONE, phase

    def run(self, params: OrchestratorResult) -> TaskAnswer:
        print("Got run ",params)
        if not params.success:
            return TaskAnswer(answer=f"ERROR: {params.error}")
        return TaskAnswer(answer=params.ctx.scratch["answer"])
