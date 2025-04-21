import json
import logging
import os
import time
from typing import List

from atomic_agents.lib.base.base_tool import BaseToolConfig
from dotenv import load_dotenv
from openai import NOT_GIVEN
from pydantic import HttpUrl, TypeAdapter

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

    BOOKMARKL_URL = "E:/GithubStuff/Git5001/data/bookmarks-2025-03-31.json"
    INPUT_DIR = "E:/GithubStuff/Git5001/data/input"

    OUTPUT_DIR = f"{BASE_DIR}/agent_output"
    DEBUG_DIR = f"{BASE_DIR}/debug_save"
    SAVE_DIR = f"{BASE_DIR}/scheduler_save"
    ERROR_DIR = f"{BASE_DIR}/scheduler_error"
    LLM_LOG_DIR = f"{BASE_DIR}/log"

    #RESTORE_DIR = "T:/tmp/agents/scheduler_save/step_59"
    #RESTORE_DIR = "T:/tmp/agents/scheduler_save/step_2660"
    RESTORE_DIR = None

    USE_LARGE_MODELS = True

    #OLLAMA_MODEL = "llama3.2:1b" # test only


    if not USE_LARGE_MODELS:
        OLLAMA_MODEL = "qwen2.5:7b" # fast, good
        OPENAI_MODEL = "gpt-4o-mini"
        OPENAI_MODEL = "gpt-4.1"
        #OPENAI_MODEL = "o4-mini"
    else:
        OLLAMA_MODEL = "qwen2.5:14b" # fits
        # OLLAMA_MODEL = "gemma3:12b"  # fits 90%  # 7141.59 @250
        OPENAI_MODEL = "gpt-4o"
        OPENAI_MODEL = "gpt-4.1"
        #OPENAI_MODEL = "o4-mini"
    OPENAI_MODEL_FINAL = "gpt-4o-mini"


    SHORT_LOOP_CNT = None  # -> 3510

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(LLM_LOG_DIR, exist_ok=True)


    ########################################################################
    # Secret configration
    ########################################################################
    # Set secrets
    # .env File must contain the following entries:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

	# Local ollama
    BASE_URL = "http://192.168.2.78:11434/v1"

    SKIP_FIREFOX_BOOKMARKS = True


    # LLM
    llmOllamaConfig = LLMAgentConfig(model=OLLAMA_MODEL,
                               provider=Provider.OLLAMA,
                               api_key=None,
                               base_url=BASE_URL,
                               log_dir=LLM_LOG_DIR,
                               max_token=32768,
                               timeout=440.0,
                               use_response=False)

    llmOpenAiConfig = LLMAgentConfig(model=OPENAI_MODEL,
                               provider=Provider.OPENAI,
                               api_key=OPENAI_API_KEY,
                               base_url=None,
                               log_dir=LLM_LOG_DIR,
                               max_token=16384,
                               timeout=3600,
                               use_response=True)

    finalLlmOpenAiConfig = LLMAgentConfig(model=OPENAI_MODEL_FINAL,
                               provider=Provider.OPENAI,
                               api_key=OPENAI_API_KEY,
                               base_url=None,
                               log_dir=LLM_LOG_DIR,
                               max_token=16384,
                               timeout=3600,
                               use_response=True)



    # Load Firefox
    firefoxLoadAgent = LoadJsonAgent(config=LoadJsonAgentConfig(filename=f"{INPUT_DIR}/load_bookmarks.json", model_class=FirefoxBookmarksOutput))

    # Firefox
    firefoxBookmarkAgent = FirefoxBookmarkAgent(config=FirefoxBookmarkAgentConfig())
    # firefoxDebugAgent = DebugBookmarkAgent(config=DebugBookmarkAgentConfig())

    # Save Firefox
    # firefoxSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{INPUT_DIR}/save_bookmarks.json"))

    # Web scraping
    webScraperConfig = WebpageScraperToolConfig(trim_excessive_content=True, max_content_length=100000)
    if DUMMY_WEB:
        webScraperAgent = DummyWebScraperAgent(config=webScraperConfig)
    else:
        webScraperAgent = WebScraperAgent(config=webScraperConfig)

    # Save web
    # webSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{OUTPUT_DIR}/save_web.json", use_uuid=True))

    webScraperSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_websrcape.json", use_uuid=True))


    llmAgent = WebpageToCategoryAgent(config=llmOllamaConfig)
    llmSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_llm.json", use_uuid=True))
    sinkLLMAgent: ListCollectionAgent = ListCollectionAgent(BaseToolConfig(), uuid='collect_llms')
    counterLLMAgent: CounterAgent = CounterAgent(CounterAgentConfig(counter_fields=['valid','overall']))
    counterSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_counter_llm_valid.json", use_uuid=False))


    mergingAgent = BookmarkMultiPortAggregatorAgent(BookmarkMultiPortAggregatorAgentConfig(), uuid='merge_initial')

    # Category creations
    categoryAgent: GenerateCategoryForBookmarkAgent = GenerateCategoryForBookmarkAgent(llmOllamaConfig)

    mergingAgentCats = CategoryMultiPortAggregatorAgent(CategoryMultiPortAggregatorAgentConfig(), uuid='merge_cats')
    # mergingSaveCatsAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{OUTPUT_DIR}/save_merger_cats.json", use_uuid=True))

    # Collection
    sinkAgent_CollectCats: ListCollectionAgent = ListCollectionAgent(BaseToolConfig(), uuid='collect_cats')
    categoriesSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_web_categories.json"))
    sinkAgent_Categories: ListCollectionAgent = ListCollectionAgent(BaseToolConfig(), uuid='categories')
    catGeneralizeAgent: CategoryGeneralizationLLMAgent = CategoryGeneralizationLLMAgent(llmOpenAiConfig)
    catGeneralizeSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_generalized_categories.json"))


    mergingFinalAgent = FinalCollectPortAggregatorAgent(finalLlmOpenAiConfig, uuid='merge_final')
    mergingSaveFinalAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_merger_final.json", use_uuid=False))

    firefoxStorageAgent = FirefoxBookmarkStorageAgent(config=FirefoxBookmarkStorageAgentConfig(filename=f"{OUTPUT_DIR}/firefox_out.json", use_uuid=False))

    # Buffer
    # bufferAgent_MergeCats = IdentityAgent(config=IdentityAgentConfig(), uuid='merge_cats')
    # bufferAgent_MergeCats.is_active = False
    # bufferAgent_LLMOut = IdentityAgent(config=IdentityAgentConfig(), uuid='llm_cats')
    # bufferAgent_LLMOut.is_active = False
    # bufferAgent_SinkCatOut = IdentityAgent(config=IdentityAgentConfig(), uuid='sink_cat')
    # bufferAgent_SinkCatOut.is_active = False


    # ------------------------------------------------------------------------------------------------
    # Message transformation
    def transform_bookmarks_to_webscraper(output_msg: FirefoxBookmarksOutput) -> List[WebpageScraperToolInputSchema]:
        """Converts each result into an individual WebpageScraperToolInputSchema instance."""
        results = []
        for i, result in enumerate(output_msg.bookmarks):
            if SHORT_LOOP_CNT and i > SHORT_LOOP_CNT: # DEBUG
                break
            try:
                validated_url = TypeAdapter(HttpUrl).validate_python(result.url)
                scraper_input = WebpageScraperToolInputSchema(url=validated_url, include_links=False)
            except Exception as e:
                print(f"[{i}] Validation failed for URL '{result.url}': {e} -> Igoring page")
                scraper_input = WebpageScraperToolInputSchema(url=None, include_links=False)
            results.append(scraper_input)
        return results

    def transform_bookmarks_to_aggregator(output_msg: FirefoxBookmarksOutput) -> List[Bookmark]:
        """Converts each result into an individual WebpageScraperToolInputSchema instance."""
        if SHORT_LOOP_CNT: # DEBUG
            result = output_msg.bookmarks[:SHORT_LOOP_CNT + 1]
            return result
        return output_msg.bookmarks

    def transform_webscraper_to_llm(web_page: WebpageScraperToolOutputSchema) -> WebpageToCategoryInput:
        """Converts result of webscrape to llm input."""
        return WebpageToCategoryInput(url="", content=web_page.content, metadata=web_page.metadata, webpage_error=web_page.error)

    def transform_llm_to_counter(bookmark: BookmarkOutput) -> CounterSchema:
        """Converts result of webscrape to llm input."""
        return CounterSchema(counts={
            "valid": 1 if bookmark.ist_gueltig else 0,
            "overall": 1
        })


    # ------------------------------------------------------------------------------------------------
    # Connect agents
    #firefoxLoadAgent.output_port.connect(firefoxSaveAgent.input_port)
    #firefoxBookarkAgent.output_port.connect(firefoxSaveAgent.input_port)

    firefoxLoadAgent.connectTo(webScraperAgent, transformer=transform_bookmarks_to_webscraper)
    # firefoxLoadAgent.connectTo(firefoxDebugAgent)
    #firefoxBookarkAgent.output_port.connect(webScraperAgent.input_port, transform_bookmarks_to_webscraper)

    # webScraperAgent.connectTo(webSaveAgent)



    webScraperAgent.connectTo(llmAgent, transformer=transform_webscraper_to_llm)
    webScraperAgent.connectTo(webScraperSaveAgent)

    webScraperAgent.connectTo(mergingAgent, target_port_name="web_scraping_result")
    llmAgent.connectTo(mergingAgent, target_port_name="llm_result")
    firefoxLoadAgent.connectTo(mergingAgent, target_port_name="bookmark", transformer=transform_bookmarks_to_aggregator)

    # mergingAgent.connectTo(mergingSaveAgent)
    mergingAgent.connectTo(categoryAgent)

    webScraperAgent.connectTo(mergingAgentCats, target_port_name="web_scraping_result")
    llmAgent.connectTo(mergingAgentCats, target_port_name="llm_result")
    firefoxLoadAgent.connectTo(mergingAgentCats, target_port_name="bookmark", transformer=transform_bookmarks_to_aggregator)
    categoryAgent.connectTo(mergingAgentCats, target_port_name="category")
    mergingAgentCats.connectTo(sinkAgent_CollectCats)

    llmAgent.connectTo(sinkLLMAgent, transformer=transform_llm_to_counter)
    sinkLLMAgent.connectTo(counterLLMAgent)
    counterLLMAgent.connectTo(counterSaveAgent)

    sinkAgent_CollectCats.connectTo(catGeneralizeAgent)
    sinkAgent_CollectCats.connectTo(categoriesSaveAgent)

    categoryAgent.connectTo(sinkAgent_Categories)

    catGeneralizeAgent.connectTo(catGeneralizeSaveAgent)

    sinkAgent_Categories.connectTo(mergingFinalAgent, target_port_name="rawCategories")
    catGeneralizeAgent.connectTo(mergingFinalAgent, target_port_name="finalCategories")
    firefoxLoadAgent.connectTo(mergingFinalAgent, target_port_name="bookmarks")
    mergingFinalAgent.connectTo(mergingSaveFinalAgent)
    mergingFinalAgent.connectTo(firefoxStorageAgent)



    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler(save_dir=SAVE_DIR, error_dir=ERROR_DIR, save_step=10)

    scheduler.add_agent(firefoxLoadAgent, not SKIP_FIREFOX_BOOKMARKS)
    scheduler.add_agent(firefoxBookmarkAgent, SKIP_FIREFOX_BOOKMARKS)
    scheduler.add_agent(webScraperAgent)
    scheduler.add_agent(webScraperSaveAgent, skipAgent=True)
    scheduler.add_agent(llmAgent)
    scheduler.add_agent(llmSaveAgent)
    scheduler.add_agent(sinkLLMAgent)
    scheduler.add_agent(counterLLMAgent)
    scheduler.add_agent(counterSaveAgent)

    scheduler.add_agent(mergingAgent)
    scheduler.add_agent(categoryAgent)
    scheduler.add_agent(categoriesSaveAgent)

    scheduler.add_agent(mergingAgentCats)
    scheduler.add_agent(sinkAgent_CollectCats)
    scheduler.add_agent(catGeneralizeAgent)
    scheduler.add_agent(catGeneralizeSaveAgent)
    scheduler.add_agent(sinkAgent_Categories)
    scheduler.add_agent(mergingFinalAgent)
    scheduler.add_agent(mergingSaveFinalAgent)
    scheduler.add_agent(firefoxStorageAgent)



    # ------------------------------------------------------------------------------------------------
    # Print pipeline
    printer = PipelinePrinter(is_ortho=False, direction='TB', fillcolor='blue', show_schemas=True,schema_fillcolor='moccasin')
    printer.print_ascii(scheduler.agents)
    printer.to_png(scheduler.agents, 'r:/pipeline_bookmarks_large.png')
    printer = PipelinePrinter(is_ortho=False, direction='TB', fillcolor='blue')
    printer.print_ascii(scheduler.agents)
    printer.to_png(scheduler.agents, 'r:/pipeline_bookmarks.png')
    #return


    # ------------------------------------------------------------------------------------------------
    # Start processing


    if RESTORE_DIR:
        rich_console.print(f"[red]Restoring scheduler from {RESTORE_DIR}[/red]")
        scheduler.load_agents(RESTORE_DIR)
        scheduler.load_state(RESTORE_DIR)
        print(json.dumps(scheduler.queque_sizes(), indent=4))
        rich_console.print(f"[red]Restoring scheduler done[/red]")

    else:
        rich_console.print(f"[red]Feeding agenty initially[/red]")
        firefoxLoadAgent.feed(None)
    #llmAgent.feed(None)
    # firefoxBookarkAgent.feed(FirefoxBookmarksInput(filepath=BOOKMARKL_URL))
    #webScraperAgent.feed(WebpageScraperToolInputSchema(url="https://arxiv.org/html/2408.01800v1"))



    # ------------------------------------------------------------------------------------------------
    # Loop all
    start_time = time.perf_counter()
    scheduler.step_all()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # at end of main
    get_browser_manager().close()


if __name__ == '__main__':
    main()