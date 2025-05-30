import logging
import os
import time
from typing import List, Dict

from pydantic import BaseModel

from AgentControlFlow.MyIfAgent import MyIfAgent, MyIfAgentInputSchema, MyIfAgentConfig
from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentFramework.support.DebugAgent import DebugAgent, DebugAgentConfig
from AgentFramework.core.PiplinePrinter import PipelinePrinter
from agent_logging import rich_console

# Set up root logger once
logging.basicConfig(level=logging.INFO)

def main():

    BASE_DIR = "t:/tmp/agents/"

    OUTPUT_DIR = f"{BASE_DIR}/agent_output"
    DEBUG_DIR = f"{BASE_DIR}/debug_save"
    SAVE_DIR = f"{BASE_DIR}/scheduler_save"
    ERROR_DIR = f"{BASE_DIR}/scheduler_error"
    LLM_LOG_DIR = f"{BASE_DIR}/log"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(LLM_LOG_DIR, exist_ok=True)

    ifAgent: MyIfAgent = MyIfAgent(MyIfAgentConfig())
    debugPositiveAgent: DebugAgent = DebugAgent(DebugAgentConfig())
    debugNeutralAgent: DebugAgent = DebugAgent(DebugAgentConfig())
    debugNegativeAgent: DebugAgent = DebugAgent(DebugAgentConfig())

    ifAgent.connectTo(debugPositiveAgent, from_output=ifAgent.schema("PositiveResponse"))
    ifAgent.connectTo(debugNeutralAgent, from_output=ifAgent.schema("NegativeResponse"))
    ifAgent.connectTo(debugNegativeAgent, from_output=ifAgent.schema("NeutralResponse"))

    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler(save_dir=SAVE_DIR, error_dir=ERROR_DIR, save_step=1)

    scheduler.add_agent(ifAgent)
    scheduler.add_agent(debugPositiveAgent)
    scheduler.add_agent(debugNeutralAgent)
    scheduler.add_agent(debugNegativeAgent)


    rich_console.print(f"[red]Feeding agents initially[/red]")
    ifAgent.feed(MyIfAgentInputSchema(condition="positive", message="Test message +++"))
    ifAgent.feed(MyIfAgentInputSchema(condition="negative", message="Test message ---"))


    # ------------------------------------------------------------------------------------------------
    # Loop all
    start_time = time.perf_counter()
    scheduler.step_all()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    outputs: Dict[str, List[BaseModel]] = scheduler.get_final_outputs()
    for outkey, outval in outputs.items():
        print(f"Result {outkey}: {outval}")

    # ------------------------------------------------------------------------------------------------
    # Print pipeline
    printer = PipelinePrinter(is_ortho=True,
                              direction='TB',
                              fillcolor='blue',
                              entry_exit_fillcolor = 'pink',
                                )
    printer.print_ascii(scheduler.agents)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_if.png')
    printer = PipelinePrinter(is_ortho=True,
                              direction='TB',
                              fillcolor='blue',
                              entry_exit_fillcolor='pink',
                              show_schemas = True,
                              schema_fillcolor = 'moccasin')
    printer.print_ascii(scheduler.agents)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_if_large.png')


if __name__ == '__main__':
    main()