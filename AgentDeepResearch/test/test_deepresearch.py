import os
from typing import Dict, List

from atomic_agents.lib.base.base_tool import BaseToolConfig
from dotenv import load_dotenv
from pydantic import BaseModel

from AgentDeepResearch.CentralWorkerAgent import CentralWorkerAgent
from AgentDeepResearch.PageSummarizerAgent import PageSummarizerAgent
from AgentDeepResearch.QueryEnhancerAgent import QueryEnhancerAgent
from AgentDeepResearch.SearchAgent import SearchAgent
from AgentDeepResearch.SynthesisAgent import SynthesisAgent
from AgentDeepResearch.schemas import UserQueryInput, TavilySearchListModel, SynthezierInputModel, \
    SynthesisOutput, PageSummaryItemSchema
from AgentFramework.AgentScheduler import AgentScheduler
from AgentFramework.ListCollectionAgent import ListCollectionAgent, ListModel
from AgentFramework.PiplinePrinter import PipelinePrinter
from AgentFramework.PrintAgent import PrintAgent, PrintAgentConfig, PrintMessageInput
from AgentFramework.SaveJsonAgent import SaveJsonAgent, SaveJsonAgentConfig
from AgentFramework.TCPDebugger import TCPDebugger
from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolConfig, TavilySearchToolInputSchema, \
    TavilySearchResultItemSchema
from util.LLMSupport import LLMAgentConfig, Provider

BASE_DIR = "t:/tmp/agents_deep/"
OUTPUT_DIR = f"{BASE_DIR}/agent_output"
DEBUG_DIR = f"{BASE_DIR}/debug_save"
SAVE_DIR = f"{BASE_DIR}/scheduler_save"
ERROR_DIR = f"{BASE_DIR}/scheduler_error"
LLM_LOG_DIR = f"{BASE_DIR}/log"

USER_QUERY = "Explain the impact of quantum computing on cryptography."
USER_QUERY = "Research benefits and side effects of FSME vaccination in southern germany"
USER_QUERY = "Applications and Limitations of Diffusion Models in Generative Art and Scientific Visualization"
USER_QUERY = "Capabilities and Risks of Using Large Language Models for Autonomous Scientific Discovery"
USER_QUERY = "Capabilities and Risks of self-aware AI systems"
USER_QUERY = "Life on exo planets - knowledge and speculation"
USER_QUERY = "A report about the speed of all fast birds (maximum speed and usual travel speeds)"
USER_QUERY = "Explain the details of an electic car propulsion with focus on embedded DCDC boost converter, inverters etc"
REPORTFILENAME = f"{OUTPUT_DIR}/dcdc.md"
WEB_RESULTS = 20


def build_pipeline():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAi API KEY
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Taviliy API_KEY

    USE_DEBUGGER = False

    OLLAMA_MODEL = "qwen2.5:7b"
    OPENAI_MODEL = "gpt-4o-mini"
    OPENAI_MODEL_LARGE = "gpt-4.1-mini"
    OPENAI_MODEL_LARGE = "o4-mini"
    # Local ollama
    BASE_URL = "http://192.168.2.78:11434/v1"


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
                               timeout=3600,
                               use_response=True)
    llmOpenAiConfig2 = LLMAgentConfig(model=OPENAI_MODEL_LARGE,
                               provider=Provider.OPENAI_THINKING,
                               api_key=OPENAI_API_KEY,
                               base_url=None,
                               log_dir=LLM_LOG_DIR,
                               timeout=3600,
                               use_response=True)
    llmConfig = llmOpenAiConfig

    # Instantiate
    enhancer = QueryEnhancerAgent(config=llmConfig)
    worker = CentralWorkerAgent()

    tavily_config = TavilySearchToolConfig(api_key = TAVILY_API_KEY,
                                           max_results = WEB_RESULTS,
                                           days = 360,
                                           include_raw_content= True)
    searcher = SearchAgent(tavily_config)
    searcherSaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_searcher.json", use_uuid=True))
    page_summarizer = PageSummarizerAgent(config=llmConfig)
    page_summarizer_sink: ListCollectionAgent = ListCollectionAgent(BaseToolConfig())
    summarySaveAgent = SaveJsonAgent(config=SaveJsonAgentConfig(filename=f"{DEBUG_DIR}/save_summary.json", use_uuid=True))
    synthesizer = SynthesisAgent(config=llmOpenAiConfig2)

    printer = PrintAgent(PrintAgentConfig(log_to_file=True, log_file_path=f"{REPORTFILENAME}", log_console=False))


    # We use this transformer so that the worker gets a clearly defined schema class in case we later connect another list to it
    def transform_listmodel_2_tavily_listmodel(params: ListModel) -> TavilySearchListModel:
        """Converts result """
        return TavilySearchListModel(data=params.data)
    def transform_report_2_print(params: SynthesisOutput) -> PrintMessageInput:
        """Converts result """
        return PrintMessageInput(subject=None, body=params.text)


    # Wire
    enhancer.connectTo(worker, target_port_name="enhancer")
    worker.connectTo(searcher,output_schema=TavilySearchToolInputSchema)
    searcher.connectTo(worker, target_port_name="searcher")
    worker.connectTo(page_summarizer,output_schema=PageSummaryItemSchema)
    page_summarizer.connectTo(page_summarizer_sink)
    page_summarizer_sink.connectTo(worker, target_port_name="summarizer", transformer=transform_listmodel_2_tavily_listmodel)
    worker.connectTo(synthesizer, output_schema=SynthezierInputModel)
    synthesizer.connectTo(printer, transformer=transform_report_2_print)

    # Debugger
    if USE_DEBUGGER:
        debugger = TCPDebugger(start_paused=False)
    else:
        debugger = False

    # Scheduler
    scheduler = AgentScheduler(debugger=debugger)

    scheduler.add_agent(enhancer)
    scheduler.add_agent(worker)
    scheduler.add_agent(searcher)
    scheduler.add_agent(searcherSaveAgent)
    scheduler.add_agent(page_summarizer)
    scheduler.add_agent(page_summarizer_sink)
    scheduler.add_agent(summarySaveAgent)
    scheduler.add_agent(synthesizer)
    scheduler.add_agent(printer)


    return scheduler, enhancer



def demo_run():
    scheduler, entry = build_pipeline()
    # ------------------------------------------------------------------------------------------------
    # Print pipeline
    printer = PipelinePrinter(is_ortho=True,
                              direction='TB',
                              fillcolor='blue',
                              entry_exit_fillcolor = 'pink',
                                )
    printer.print_ascii(scheduler.agents)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_research.png')
    printer = PipelinePrinter(is_ortho=True,
                              direction='TB',
                              fillcolor='blue',
                              entry_exit_fillcolor='pink',
                              show_schemas = True,
                              schema_fillcolor = 'moccasin')
    printer.print_ascii(scheduler.agents)
    printer.save_as_png(scheduler.agents, 'r:/pipeline_research_large.png')
    #return

    entry.feed(UserQueryInput(query=USER_QUERY))
    scheduler.step_all()

    outputs: Dict[str, List[BaseModel]] = scheduler.get_final_outputs()
    for outkey, outval in outputs.items():
        print(f"Result {outkey}: {outval}")

if __name__ == "__main__":
    demo_run()
