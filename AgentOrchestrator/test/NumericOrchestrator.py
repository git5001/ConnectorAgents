# =============================================================================
# 2.  Demo: numeric pipeline (Double → maybe Increment)
# =============================================================================
from typing import Optional

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.BaseOrchestratorAgent import BaseOrchestratorAgent, NextPhase, \
    OrchestratorResult
from AgentFramework.core.ConnectedAgent import ConnectedAgent


class IntValue(BaseIOSchema):
    """ desc """
    value: int
    description: str = "An integer value"

class NumericRequest(BaseIOSchema):
    """ desc """
    value: int
    description: str = "Initial integer task"

class NumericResult(BaseIOSchema):
    """ desc """
    value: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    description: str = "Final integer result or error"

# --- child agents -----------------------------------------------------------
class Add100Agent(ConnectedAgent):
    input_schema = IntValue
    output_schema = IntValue
    uuid = "add100"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: IntValue):
        return IntValue(value=params.value + 100)

class DoubleAgent(ConnectedAgent):
    input_schema = IntValue
    output_schema = IntValue
    uuid = "double"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: IntValue):
        return IntValue(value=params.value * 2)

class Add200Agent(ConnectedAgent):
    input_schema = IntValue
    output_schema = IntValue
    uuid = "add200"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: IntValue):
        return IntValue(value=params.value + 200)

class IncrementAgent(ConnectedAgent):
    input_schema = IntValue
    output_schema = IntValue
    uuid = "increment"
    def __init__(self):
        super().__init__(BaseToolConfig())
    def run(self, params: IntValue):
        return IntValue(value=params.value + 1)

# --- orchestrator -----------------------------------------------------------
class NumericOrchestrator(BaseOrchestratorAgent):
    input_schema  = NumericRequest
    output_schema = NumericResult
    THRESHOLD = 300

    def build_phase(self, task, ctx, phase):
        if phase == 1:
            a1, b1 = Add100Agent(), DoubleAgent()
            a1.connectTo(b1)
            a1.feed(IntValue(value=task.value))
            ctx.scratch["sink"] = b1
            return [a1, b1]
        elif phase == 2:
            a2, b2 = Add200Agent(), IncrementAgent()
            a2.connectTo(b2)
            a2.feed(IntValue(value=ctx.scratch["phase1_result"]))
            ctx.scratch["sink"] = b2
            return [a2, b2]
        else:
            raise ValueError("No phase >2 defined")

    def after_phase(self, outputs, ctx, phase):
        sink = ctx.scratch["sink"]
        msgs = outputs.get(sink, [])
        if not msgs:
            ctx.error = "Sink produced no output"
            return NextPhase.DONE, phase
        value = msgs[0].value
        if phase == 1:
            ctx.scratch["phase1_result"] = value
            if value < self.THRESHOLD:
                return NextPhase.CONTINUE, phase + 1
            ctx.scratch["final"] = value
            return NextPhase.DONE, phase
        else:  # phase 2
            ctx.scratch["final"] = value
            return NextPhase.DONE, phase

    def run(self, params: OrchestratorResult):
        ctx = params.ctx
        if not params.success:
            return NumericResult(success=False, error=params.error)
        return NumericResult(value=ctx.scratch["final"], success=True)
