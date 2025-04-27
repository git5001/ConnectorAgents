import logging
import logging
import os
import time

from AgentFramework.AgentScheduler import AgentScheduler
from AgentFramework.IdentityAgent import IdentityAgent, IdentityAgentConfig
from AgentFramework.PiplinePrinter import PipelinePrinter
from AgentFramework.TCPDebugger import TCPDebugger
from AgentMultiPort.MultiOutSimpleAgent2 import MultiOutSimpleAgent2, MultiOutSimpleAgentConfig, \
    MultiOutSimpleAgentSchema1, MultiOutSimpleAgentSchema2
from agent_logging import rich_console

# Set up root logger once
logging.basicConfig(level=logging.INFO)

def main():
    USE_DEBUGGER = False
    BASE_DIR = "t:/tmp/agents/"

    OUTPUT_DIR = f"{BASE_DIR}/agent_output"
    DEBUG_DIR = f"{BASE_DIR}/debug_save"
    SAVE_DIR = f"{BASE_DIR}/scheduler_save"
    ERROR_DIR = f"{BASE_DIR}/scheduler_error"
    LLM_LOG_DIR = f"{BASE_DIR}/log"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(LLM_LOG_DIR, exist_ok=True)

    multiAgent: MultiOutSimpleAgent2 = MultiOutSimpleAgent2(MultiOutSimpleAgentConfig())
    bufferAgent_1 = IdentityAgent(config=IdentityAgentConfig(), uuid='multi_1')
    bufferAgent_2 = IdentityAgent(config=IdentityAgentConfig(), uuid='multi_2')

    multiAgent.connectTo(bufferAgent_1, source_output_schema=MultiOutSimpleAgentSchema1)
    multiAgent.connectTo(bufferAgent_2, source_output_schema=MultiOutSimpleAgentSchema2)
    multiAgent.connectTo(multiAgent, source_output_schema=MultiOutSimpleAgentSchema1)




    # ------------------------------------------------------------------------------------------------
    # Debugger
    if USE_DEBUGGER:
        debugger = TCPDebugger()
    else:
        debugger = False
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler(save_dir=SAVE_DIR,
                                              error_dir=ERROR_DIR,
                                              save_step=1,
                                              debugger=debugger)

    scheduler.add_agent(multiAgent)
    scheduler.add_agent(bufferAgent_1)
    scheduler.add_agent(bufferAgent_2)

    # printer = PipelinePrinter(is_ortho=False, direction='LR', fillcolor='blue', show_schemas=True,schema_fillcolor='moccasin')
    # printer.print_ascii(scheduler.agents)
    # printer.to_dot(scheduler.agents,)
    # printer.to_png(scheduler.agents,  'r:/pipeline_condition_large.png')


    rich_console.print(f"[red]Feeding agents initially[/red]")
    multiAgent.feed(MultiOutSimpleAgentSchema1(number=4))
    #multiAgent.feed(NullSchema())
    #multiAgent.feed(NullSchema())


    # ------------------------------------------------------------------------------------------------
    # Loop all
    start_time = time.perf_counter()
    scheduler.step_all()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Expected outputs for each agent
    expected_agent_1_numbers = [4, 3, 2, 1]  # from first and third run
    expected_agent_2_ids = ["10"]  # from second and third run

    # Check Agent 1 outputs
    for i, output in enumerate(bufferAgent_1.get_final_outputs()):
        print("output #1 =", output.number)
        assert isinstance(output, MultiOutSimpleAgentSchema1), f"Unexpected type at Agent 1 index {i}"
        assert output.number == expected_agent_1_numbers[
            i], f"Agent 1 output #{i} expected {expected_agent_1_numbers[i]}, got {output.number}"

    # Check Agent 2 outputs
    for i, output in enumerate(bufferAgent_2.get_final_outputs()):
        print("output #2 =", output.id)
        assert isinstance(output, MultiOutSimpleAgentSchema2), f"Unexpected type at Agent 2 index {i}"
        assert output.id == expected_agent_2_ids[
            i], f"Agent 2 output #{i} expected {expected_agent_2_ids[i]}, got {output.id}"

    printer = PipelinePrinter(is_ortho=False, direction='LR', fillcolor='blue')
    printer.print_ascii(scheduler.agents)
    printer.to_dot(scheduler.agents,)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_multi.png')

    printer = PipelinePrinter(is_ortho=False, direction='LR', fillcolor='blue', show_schemas=True,schema_fillcolor='moccasin')
    printer.print_ascii(scheduler.agents)
    printer.to_dot(scheduler.agents,)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_multi_large.png')

    scheduler.close()

if __name__ == '__main__':
    main()