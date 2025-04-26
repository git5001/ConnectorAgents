
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
    search: str = Field(..., description="User research question for web search")
    enhanced: str = Field(..., description="User research question enhanced for research report generation")
    @staticmethod
    def empty() -> "CentralWorkerState":
        return CentralWorkerState(
            original="",
            search="",
            enhanced=""
        )

class CentralWorkerAgent(MultiPortAgent):
    """Orchestrates the research loop."""

    input_schema = EnhancedQueryOutput
    input_schemas = {
        "enhancer": EnhancedQueryOutput,
        "searcher": TavilySearchToolOutputSchema,
        "summarizer": TavilySearchListModel,
    }
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
        if "enhancer" in inputs:
            eq: EnhancedQueryOutput = inputs["enhancer"]
            self.iterations += 1
            print(f"[Worker] Planning searches for '{eq.search}'")
            # Store state
            self._state.original = eq.original
            self._state.search = eq.search
            self._state.enhanced = eq.enhanced

            tavily_input = TavilySearchToolInputSchema(
                 queries = [eq.search]
            )
            return tavily_input
        # Second input path – the web search shall be summarized as page summary
        if "searcher" in inputs:
            ts: TavilySearchToolOutputSchema = inputs["searcher"]
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
        if "summarizer" in inputs:
            lm: TavilySearchListModel = inputs["summarizer"]
            print(f"[Worker] Received summary results {len(lm.data)} for '{self._state.original}'")

            self.iterations += 1
            input = EnhancedQueryOutput(original=self._state.original, search=self._state.search, enhanced=self._state.enhanced)
            out = SynthezierInputModel(data=lm.data, input=input)

            return out

        # Nothing to emit yet
        return None
