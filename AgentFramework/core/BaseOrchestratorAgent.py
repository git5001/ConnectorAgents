from __future__ import annotations
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Type, List, Tuple

from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_io_schema import BaseIOSchema


from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentFramework.core.ConnectedAgent import ConnectedAgent


# =============================================================================
# 0.  Context (black‑board)
# =============================================================================
class OrchestratorContext(BaseIOSchema):
    """
    Shared blackboard object passed between the orchestrator and all internal child agents.

    This context is automatically created fresh for each new task and attached to all agents
    via their `global_state` field.

    Fields:
        original_task (BaseIOSchema):
            The original task/request the orchestrator received from its upstream caller.
            Useful for child agents to refer back to the full prompt or metadata.

        scratch (Dict[str, object]):
            A mutable key–value store used to track intermediate state between phases.
            Subclasses may freely store results, flags, partial answers, etc.
    """
    original_task: BaseIOSchema
    scratch: Dict[str, object] = Field(default_factory=dict)

class OrchestratorResult(BaseModel):
    """
    The orchestrator result to run.
    """
    outputs: List[BaseModel]  # All agent output schema results
    ctx: OrchestratorContext  # Holds an instance of context
    success: bool = False # Success of run
    error: Optional[str] = None # if not success error message

# =============================================================================
# 1.  Next‑phase signalling enum
# =============================================================================
class NextPhase(Enum):
    """
    Signal returned by the `after_phase()` hook to control orchestration flow.

    Members:
        CONTINUE:
            Tells the base class to proceed to another phase.
            The next phase index must be specified alongside (e.g. phase+1 or any other int).

        DONE:
            Signals that the orchestrator is finished and `run()` should be called next.

    Usage:
        return (NextPhase.CONTINUE, phase + 1)     # Go to next phase
        return (NextPhase.DONE, phase)             # Stop and finalize
    """
    CONTINUE = "continue"  # run next phase
    DONE = "done"           # orchestration finished

# =============================================================================
# 2.  Generic BaseOrchestratorAgent (phase index managed internally)
# =============================================================================
class BaseOrchestratorAgent(ConnectedAgent):
    """
    A generic high-level agent that encapsulates multi-step orchestration logic using internal agents.

    This base class allows you to build and run multi-phase pipelines without manually managing
    agent scheduling, message passing, or task completion.

    Each task is handled in a blocking, step-by-step fashion:
      1. A new context (`OrchestratorContext`) is created and shared with all child agents.
      2. The `build_phase()` method is called to construct agents for the current phase.
      3. These agents are executed until completion via an internal `AgentScheduler`.
      4. The outputs of the phase are passed to `after_phase()` to determine what comes next.
      5. Once `NextPhase.DONE` is returned, the final result is produced via `run()`.

    Subclasses must implement:
        • build_phase(task, ctx, phase)
        • after_phase(outputs, ctx, phase)
        • run(ctx)

    Optional:
        • context_schema (default: OrchestratorContext) – can be subclassed for additional fields.

    Attributes:
        input_schema (Type[BaseIOSchema]):
            Required. The input schema expected by this orchestrator agent.

        output_schema (Type[BaseIOSchema]):
            Required. The output schema returned when orchestration is complete.

        context_schema (Type[BaseModel]):
            Optional. The data model used to track orchestration state.
            Default is `OrchestratorContext`.

        uuid (str):
            Identifier for the agent. Used for debug output and scheduling.
    """
    input_schema:  Type[BaseIOSchema]
    output_schema: Type[BaseIOSchema]

    context_schema: Type[BaseModel] = OrchestratorContext
    uuid = "base-orchestrator"

    # ----- helpers --------------------------------------------------------
    def filter_schema(self, outputs: Dict[ConnectedAgent, List[BaseIOSchema]],
                      schema: Type[BaseIOSchema]) -> List[BaseIOSchema]:
        """
        Extracts and returns all messages from the given output dictionary that match a specific schema type.

        Args:
            outputs (Dict[ConnectedAgent, List[BaseIOSchema]]):
                The scheduler output grouped by agent.

            schema (Type[BaseIOSchema]):
                The schema class to filter for.

        Returns:
            List[BaseIOSchema]:
                A flat list of messages whose type matches the given schema.
        """
        return [m for msgs in outputs.values() for m in msgs if isinstance(m, schema)]

    # ----- hooks ----------------------------------------------------------
    def build_phase(self, task: BaseIOSchema, ctx: BaseModel, phase: int) -> List[ConnectedAgent]:
        """
         Hook: Subclasses implement this to build a list of agents for the current orchestration phase.

         This method is called once per phase and is responsible for:
           • Creating the necessary ConnectedAgents
           • Feeding them if needed
           • Returning them as a list

         Args:
             task (BaseIOSchema):
                 The original task passed to the orchestrator.

             ctx (BaseModel):
                 The shared orchestration context (blackboard) for this run.

             phase (int):
                 The current phase index (starts at 1 and increases).

         Returns:
             List[ConnectedAgent]:
                 A list of agents to be scheduled and run in this phase.
         """

        raise NotImplementedError

    def after_phase(self, outputs: Dict[ConnectedAgent, List[BaseIOSchema]],
                    ctx: BaseModel, phase: int) -> Tuple[NextPhase, int]:
        """
        Hook: Subclasses use this to inspect the results of a phase and decide what happens next.

        This method is called after each phase has fully executed and can:
          • Analyze output messages
          • Store information in `ctx.scratch`
          • Decide whether the orchestration is complete or should continue

        Args:
            outputs (Dict[ConnectedAgent, List[BaseIOSchema]]):
                Final outputs of this phase, grouped by agent.

            ctx (BaseModel):
                The shared orchestration context (blackboard).

            phase (int):
                The current phase number.

        Returns:
            Tuple[NextPhase, int]:
                A (signal, next_phase) tuple:
                  • NextPhase.CONTINUE, new_phase → run another phase
                  • NextPhase.DONE, phase         → orchestration is complete
        """

        raise NotImplementedError

    def run(self, ctx: OrchestratorContext):
        """
        Hook: Called when orchestration is complete. Subclasses implement this to produce the final result.

        Args:
            ctx (OrchestratorResult):
                A structured result containing:
                  • all final outputs
                  • the shared context
                  • success flag and optional error message

        Returns:
            BaseIOSchema:
                The final output to return from the orchestrator agent.
        """
        raise NotImplementedError

    def flatten_outputs(self, outputs: Dict[ConnectedAgent, List[BaseIOSchema]]) -> List[BaseIOSchema]:
        """
        Flattens a scheduler-style output dictionary into a single flat list of all messages.

        Args:
            outputs (Dict[ConnectedAgent, List[BaseIOSchema]]):
                The output grouped by agent, as returned by AgentScheduler.get_final_outputs().

        Returns:
            List[BaseIOSchema]:
                A flattened list containing all output messages from all agents.
        """
        return [msg for msg_list in outputs.values() for msg in msg_list]

    # ----- ConnectedAgent interface --------------------------------------
    def process(self, params: BaseIOSchema, _parents, unique_id:str = None) -> OrchestratorResult:
        """
        Main entry point for orchestrator execution.

        This method is invoked by the outer scheduler and runs the full orchestration flow
        for one task. It drives the agent through one or more internal phases, each defined
        by `build_phase()` and evaluated via `after_phase()`.

        For each phase:
          1. It calls `build_phase(...)` to obtain a list of child agents to run.
          2. Runs those agents in a private `AgentScheduler` until all are idle.
          3. Passes their outputs to `after_phase(...)` to determine what to do next.
          4. When `NextPhase.DONE` is returned, it calls `run(...)` to produce the final result.

        If a phase returns no agents, orchestration halts and `run(...)` is called with
        an error-wrapped `OrchestratorResult`.

        Args:
            params (BaseIOSchema):
                The original input task message, validated against `input_schema`.

            _parents (List[str]):
                Message lineage (unused here but required by framework signature).

        Returns:
            BaseIOSchema:
                A message of type `output_schema`, produced by the final `run(...)` call.
        """

        ctx = self.context_schema(original_task=params)
        task = params  # keep original request available
        phase = 1
        while True:
            inner = AgentScheduler(uuid=f"{self.uuid}:phase-{phase}")
            agents = self.build_phase(task, ctx, phase)
            if not agents:
                error = f"build_phase returned no agents for phase {phase}"
                task_result = OrchestratorResult(success=False, error=error, outputs=[], ctx=ctx)
                return self.call_advanced_run(task_result, unique_id)
            for ag in agents:
                ag.global_state = ctx
                inner.add_agent(ag)

            inner.step_all()
            outputs = inner.get_final_outputs()

            decision, new_phase = self.after_phase(outputs, ctx, phase)
            if decision is NextPhase.DONE:
                all_outputs = self.flatten_outputs(outputs)
                task_result = OrchestratorResult(success=True, error=None, outputs=all_outputs, ctx=ctx)
                return self.call_advanced_run(task_result, unique_id)
            elif decision is NextPhase.CONTINUE:
                phase = new_phase
                continue
            else:
                raise ValueError("after_phase() must return NextPhase enum value")
