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
from AgentFramework.NullSchema import NullSchema
from AgentFramework.PiplinePrinter import PipelinePrinter
from AgentFramework.SaveJsonAgent import SaveJsonAgentConfig, SaveJsonAgent
from AgentMultiPort.MultiOutSimpleAgent import MultiOutSimpleAgentConfig, MultiOutSimpleAgent, \
    MultiOutSimpleAgentSchema1, MultiOutSimpleAgentSchema2
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

    multiAgent: MultiOutSimpleAgent = MultiOutSimpleAgent(MultiOutSimpleAgentConfig())
    bufferAgent_1 = IdentityAgent(config=IdentityAgentConfig(), uuid='multi_1')
    bufferAgent_2 = IdentityAgent(config=IdentityAgentConfig(), uuid='multi_2')

    multiAgent.connectTo(bufferAgent_1, output_schema=MultiOutSimpleAgentSchema1)
    multiAgent.connectTo(bufferAgent_2, output_schema=MultiOutSimpleAgentSchema2)




    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler(save_dir=SAVE_DIR, error_dir=ERROR_DIR, save_step=1)

    scheduler.add_agent(multiAgent)
    scheduler.add_agent(bufferAgent_1)
    scheduler.add_agent(bufferAgent_2)

    # printer = PipelinePrinter(is_ortho=False, direction='LR', fillcolor='blue', show_schemas=True,schema_fillcolor='moccasin')
    # printer.print_ascii(scheduler.agents)
    # printer.to_dot(scheduler.agents,)
    # printer.to_png(scheduler.agents,  'r:/pipeline_condition_large.png')


    rich_console.print(f"[red]Feeding agents initially[/red]")
    multiAgent.feed(NullSchema())
    multiAgent.feed(NullSchema())
    multiAgent.feed(NullSchema())


    # ------------------------------------------------------------------------------------------------
    # Loop all
    start_time = time.perf_counter()
    scheduler.step_all()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Expected outputs for each agent
    expected_agent_1_numbers = [0, 2]  # from first and third run
    expected_agent_2_ids = ["1", "2"]  # from second and third run

    # Check Agent 1 outputs
    for i, output in enumerate(bufferAgent_1.get_final_outputs()):
        assert isinstance(output, MultiOutSimpleAgentSchema1), f"Unexpected type at Agent 1 index {i}"
        assert output.number == expected_agent_1_numbers[
            i], f"Agent 1 output #{i} expected {expected_agent_1_numbers[i]}, got {output.number}"
        print("output #1 =", output.number)

    # Check Agent 2 outputs
    for i, output in enumerate(bufferAgent_2.get_final_outputs()):
        assert isinstance(output, MultiOutSimpleAgentSchema2), f"Unexpected type at Agent 2 index {i}"
        assert output.id == expected_agent_2_ids[
            i], f"Agent 2 output #{i} expected {expected_agent_2_ids[i]}, got {output.id}"
        print("output #2 =", output.id)

    printer = PipelinePrinter(is_ortho=False, direction='LR', fillcolor='blue')
    printer.print_ascii(scheduler.agents)
    printer.to_dot(scheduler.agents,)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_multi.png')

    printer = PipelinePrinter(is_ortho=False, direction='LR', fillcolor='blue', show_schemas=True,schema_fillcolor='moccasin')
    printer.print_ascii(scheduler.agents)
    printer.to_dot(scheduler.agents,)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_multi_large.png')


if __name__ == '__main__':
    main()