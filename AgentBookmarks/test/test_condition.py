import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict

from atomic_agents.lib.base.base_tool import BaseToolConfig
from dotenv import load_dotenv
from openai import NOT_GIVEN
from pydantic import HttpUrl, TypeAdapter, BaseModel

from AgentBookmarks.CountNumbersAgent import CountNumbersAgent, CountNumbersAgentConfig, CountNumbersAgentSchema
from AgentBookmarks.WebpageToCategoryAgent import WebpageToCategoryAgent, WebpageToCategoryInput, BookmarkOutput
from AgentBookmarks.BookmarkMultiPortAggregatorAgent import BookmarkMultiPortAggregatorAgent, \
    BookmarkMultiPortAggregatorAgentConfig
from AgentBookmarks.CategoryGeneralizationLLMAgent import CategoryGeneralizationLLMAgent
from AgentBookmarks.CategoryMultiPortAggregatorAgent import CategoryMultiPortAggregatorAgentConfig, \
    CategoryMultiPortAggregatorAgent
from AgentBookmarks.DummyWebScraperAgent import DummyWebScraperAgent
from AgentBookmarks.FinalCollectPortAggregatorAgent import FinalCollectPortAggregatorAgent
from AgentBookmarks.FirefoxBookmarkAgent import FirefoxBookmarkAgentConfig, FirefoxBookmarkAgent, \
    FirefoxBookmarksOutput, Bookmark
from AgentBookmarks.FirefoxBookmarkStorageAgent import FirefoxBookmarkStorageAgent, FirefoxBookmarkStorageAgentConfig
from AgentBookmarks.GenerateCategoryForBookmarkAgent import GenerateCategoryForBookmarkAgent
from AgentBookmarks.WebScraperAgent import WebScraperAgent
from AgentFramework.AgentScheduler import AgentScheduler
from AgentFramework.CounterAgent import CounterAgent, CounterSchema, CounterAgentConfig
from AgentFramework.IdentityAgent import IdentityAgent, IdentityAgentConfig
from AgentFramework.ListCollectionAgent import ListCollectionAgent
from AgentFramework.LoadJsonAgent import LoadJsonAgentConfig, LoadJsonAgent
from AgentFramework.PiplinePrinter import PipelinePrinter
from AgentFramework.SaveJsonAgent import SaveJsonAgentConfig, SaveJsonAgent
from AtomicTools.browser_handling.BrowserManager import get_browser_manager
from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperToolConfig, WebpageScraperToolInputSchema, \
    WebpageScraperToolOutputSchema
from agent_config import DUMMY_WEB
from agent_logging import rich_console
from util.LLMSupport import LLMAgentConfig, Provider

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
    sinkAgent: ListCollectionAgent = ListCollectionAgent(BaseToolConfig(), uuid='collect_numbers')
    sinkCntAgent: ListCollectionAgent = ListCollectionAgent(BaseToolConfig(), uuid='collect_counter')
    counter1: CounterAgent = CounterAgent(CounterAgentConfig(counter_fields=['count']))
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
    identityAgent1.connectTo(sinkCntAgent, transformer=transform_numbers_to_counter)
    sinkCntAgent.connectTo(counter1)



    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler(save_dir=SAVE_DIR, error_dir=ERROR_DIR, save_step=10)

    scheduler.add_agent(numbersAgent)
    scheduler.add_agent(identityAgent1)
    scheduler.add_agent(counter1)
    scheduler.add_agent(sinkAgent)
    scheduler.add_agent(sinkCntAgent)

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