
from typing import Dict, List

from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel, Field

from AgentFramework.MultiPortAgent import MultiPortAgent
from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolInputSchema, TavilySearchToolOutputSchema, \
    TavilySearchResultItemSchema
from .schemas import (
    EnhancedQueryOutput,
    TavilySearchListModel, SynthezierInputModel, PageSummaryItemSchema,
)


class CentralWorkerAgentConfig(BaseToolConfig):
    """Configuration of the central orchestrator."""
    max_iterations: int = 1  # single round for demo


class CentralWorkerState(BaseModel):
    """
    The local state of the worker.
    """
    original: str = Field(..., description="User original research question")
    search_1: str = Field(..., description="User research question for web search")
    search_2: str = Field(..., description="User research question for web search")
    search_3: str = Field(..., description="User research question for web search")
    enhanced: str = Field(..., description="User research question enhanced for research report generation")
    @staticmethod
    def empty() -> "CentralWorkerState":
        return CentralWorkerState(
            original="",
            search_1="",
            search_2="",
            search_3="",
            enhanced=""
        )

class CentralWorkerAgent(MultiPortAgent):
    """Orchestrates the research loop."""

    input_schema = EnhancedQueryOutput
    input_schemas = [EnhancedQueryOutput,
                     TavilySearchToolOutputSchema,
                     TavilySearchListModel]

    # Can output either a list of SearchTaskInput *or* ReportOutput
    output_schemas = [TavilySearchToolInputSchema,
                      PageSummaryItemSchema,
                      TavilySearchListModel,
                      SynthezierInputModel]
    _state: CentralWorkerState = None

    def __init__(self, config: CentralWorkerAgentConfig = CentralWorkerAgentConfig()):
        super().__init__(config)
        self.iterations = 0 # TODO Must be state to get saved
        self._state = CentralWorkerState.empty()

    def run(self, inputs: Dict[str, BaseModel]):
        print("Worker got inputs ", type(inputs), inputs.keys())
        # First input path – the enhanced query goes to a web search
        if EnhancedQueryOutput in inputs:
            eq: EnhancedQueryOutput = inputs[EnhancedQueryOutput]
            self.iterations += 1
            print(f"[Worker] Planning searches for #1: '{eq.search_1}'")
            print(f"[Worker] Planning searches for #2: '{eq.search_2}'")
            print(f"[Worker] Planning searches for #3: '{eq.search_3}'")
            # Store state
            self._state.original = eq.original
            self._state.search_1 = eq.search_1
            self._state.search_2 = eq.search_2
            self._state.search_3 = eq.search_3
            self._state.enhanced = eq.enhanced

            tavily_input_1 = TavilySearchToolInputSchema(queries = [eq.search_1])
            tavily_input_2 = TavilySearchToolInputSchema(queries = [eq.search_2])
            tavily_input_3 = TavilySearchToolInputSchema(queries = [eq.search_3])
            return [tavily_input_1, tavily_input_2, tavily_input_3]
        # Second input path – the web search shall be summarized as page summary
        if TavilySearchToolOutputSchema in inputs:
            ts: TavilySearchToolOutputSchema = inputs[TavilySearchToolOutputSchema]
            print(f"[Worker] Received search results #={len(ts.results)} for '{self._state.original}'")
            self.iterations += 1
            # Copy list over to enhancce it with research query, taht is we wrap the result and add an item
            pagesInputList:List[PageSummaryItemSchema] = []
            for item in ts.results:
                item:TavilySearchResultItemSchema = item
                pageItem = PageSummaryItemSchema(
                    title = item.title,
                    url = item.url,
                    content = item.content or "",
                    raw_content = item.raw_content or "",
                    research_query = self._state.enhanced)
                pagesInputList.append(pageItem)
            return pagesInputList
        # Third input path – the page summary goes to the synthezier
        if TavilySearchListModel in inputs:
            lm: TavilySearchListModel = inputs[TavilySearchListModel]
            print(f"[Worker] Received summary results {len(lm.data)} for '{self._state.original}'")

            self.iterations += 1
            input = EnhancedQueryOutput(original=self._state.original,
                                        search_1=self._state.search_1,
                                        search_2=self._state.search_2,
                                        search_3=self._state.search_3,
                                        enhanced=self._state.enhanced)
            out = SynthezierInputModel(data=lm.data, input=input)

            return out

        # Nothing to emit yet
        return None
