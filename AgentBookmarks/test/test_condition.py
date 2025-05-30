import logging
import os
import time

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentBookmarks.CountNumbersAgent import CountNumbersAgent, CountNumbersAgentConfig, CountNumbersAgentSchema
from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentFramework.support.CounterAgent import CounterAgent, CounterSchema, CounterAgentConfig
from AgentFramework.core.IdentityAgent import IdentityAgent, IdentityAgentConfig
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

    numbersAgent: CountNumbersAgent = CountNumbersAgent(CountNumbersAgentConfig(from_number=2, to_number=8))
    sinkAgent: IdentityAgent = IdentityAgent(uuid='sink_1', collect_input=True)
    counter1: CounterAgent = CounterAgent(CounterAgentConfig(counter_fields=['count']), collect_input=True)
    identityAgent1 = IdentityAgent(config=IdentityAgentConfig(), uuid='id_1')


    # ------------------------------------------------------------------------------------------------
    # Message transformation
    def numbers_condition(msg: CountNumbersAgentSchema) -> bool:
        """Check condition."""
        print("Check condition ",msg, msg.number % 2 == 0)
        if msg.number % 2 == 0:
            return False
        return True
    def transform_numbers_to_counter(data: CountNumbersAgentSchema) -> CounterSchema:
        """Converts result oto counter."""
        print("Got data ",data)
        return CounterSchema(counts={
            "count": 1
        })


    numbersAgent.connectTo(identityAgent1, condition=numbers_condition)
    identityAgent1.connectTo(sinkAgent)
    identityAgent1.connectTo(counter1, pre_transformer=transform_numbers_to_counter)



    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler(save_dir=SAVE_DIR, error_dir=ERROR_DIR, save_step=10)

    scheduler.add_agent(numbersAgent)
    scheduler.add_agent(identityAgent1)
    scheduler.add_agent(counter1)
    scheduler.add_agent(sinkAgent)

    printer = PipelinePrinter(is_ortho=False,
                              direction='LR',
                              fillcolor='blue',
                              entry_exit_fillcolor='yellow',
                              )
    printer.print_ascii(scheduler.agents)
    printer.to_dot(scheduler.agents,)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_condition.png')

    printer = PipelinePrinter(is_ortho=False,
                              direction='LR',
                              fillcolor='blue',
                              entry_exit_fillcolor='yellow',
                              show_schemas=True,
                              schema_fillcolor='moccasin')
    printer.print_ascii(scheduler.agents)
    printer.to_dot(scheduler.agents,)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_condition_large.png')
    printer.save_as_dot(scheduler.agents, 'r:/pipeline_condition_large.dot')


    printer = PipelinePrinter(direction="LR", show_schemas=True, schema_fillcolor="#FFD")
    printer.save_as_mermaid(scheduler.agents, 'r:/pipeline_condition_large.mmd')
    printer = PipelinePrinter(direction="LR", show_schemas=False, schema_fillcolor="#FFD")
    printer.save_as_mermaid(scheduler.agents, 'r:/pipeline_condition.mmd')

    # Optional render to SVG/PNG (requires the Mermaid CLI):
    #   $ mmdc -i pipeline.mmd -o pipeline.svg

    rich_console.print(f"[red]Feeding agenty initially[/red]")


    # ------------------------------------------------------------------------------------------------
    # Loop all
    start_time = time.perf_counter()
    goon = scheduler.step()
    scheduler.save_scheduler(SAVE_DIR)
    scheduler.save_scheduler(SAVE_DIR)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Check result
    for output in counter1.get_final_outputs():
        output: CounterSchema = output
        cnt = output.counts["count"]
        print("Counter=",cnt)
        assert cnt == 3, f"Expected count three got {cnt}"
    for sinkOutput in sinkAgent.get_final_outputs():
        numbers = [item.number for item in sinkOutput.data]
        print("Numbers",numbers)
        assert numbers == [3, 5, 7], f"Expected [3, 5, 7], got {numbers}"


if __name__ == '__main__':
    main()