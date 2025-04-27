import textwrap

from pydantic import BaseModel, Field

from AgentFramework.ConnectedAgent import ConnectedAgent
from AtomicTools.tavily_search.tool.tavily_search import TavilySearchResultItemSchema
from agent_config import DUMMY_LLM
from agent_logging import logger
from util.LLMSupport import LLMAgentConfig, LLMModel
from .schemas import PageSummary, DebugModel, PageSummaryItemSchema


class LLMSummaryOutput(BaseModel):
    """Schema representing the structured output of an AI output."""

    research: str = Field(..., description="Research extract of the content.")
    relevance:int = Field(..., description="Relevance of the content 0..100 for the reseach question.")

class PageSummarizerAgentState(BaseModel):
    """Page counter"""

    count: int = Field(..., description="Page counter for debug only")


class PageSummarizerAgent(ConnectedAgent):
    input_schema = PageSummaryItemSchema
    output_schema = PageSummary
    # --- State -----------------------------------------------------------
    state_schema = PageSummarizerAgentState
    _state: PageSummarizerAgentState

    def __init__(self, config: LLMAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config, uuid)
        self._state = PageSummarizerAgentState(count=0)

        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()

    def run(self, params: PageSummaryItemSchema) -> PageSummary:
        query = params.research_query
        print(f"[Summarizer] Summarizing #{self._state.count}: {params.url} for {query}")
        self._state.count += 1
        if DUMMY_LLM:
            dummy_research = (
                f"Based on analysis of the page titled '{params.title}', "
                f"key insights include: summarizing the main arguments, context, and implications."
            )
            return PageSummary(
                url=params.url,
                title=params.title,
                web_summary=params.content,
                research_summary=dummy_research,
            )
        content = f"{params.raw_content}"
        llm_result:LLMSummaryOutput = self._run_llm(content, query, params.content)
        return PageSummary(
            url=params.url,
            title=params.title,
            web_summary=params.content,
            research_summary=llm_result.research,
        )


    def _run_llm(self, content:str, question:str, sum_content:str) -> LLMSummaryOutput:
        user_prompt = textwrap.dedent(f"""
You are an research analytical assistant. You will receive a research question 
and input data as raw text. 
You must find all relevant information from the input data which is related
to the research question and present that in the output. Slightly related context
which is important to the research must be kept in a a bit more condensed form.
So ultimate goal: Collect research information for a later research study. 

The answer shall be detailed and exhaustive.
The answer will serve as input in a research report, so you must
be comprehensive and detailled. In particular you must maintain all 
technical or scientific detail. Everything which is important for
the report you must store. In doubt store more than less.

---
### Research Question:
{question}

---
Your mission:
1. Locate every passage in the Raw Text that answers or illuminates the Research Question. 
   Extract those passages and explain them in full detail 
   (include also brief quotes or precise paraphrases).
2. The goal is a research paper, so you must maintain and keep all technical details, all
   procedural details, all techical highlights, etc    
3. Identify also passages that provide broader context but don’t directly answer the question—condense 
  each of these into a summary to preserve context.
4. Remove any passages that are unrelated (e.g., advertisements, navigation menus, user comments).

Finally, evaluate how well the Raw Text addresses the Research Question on a scale from 0 
(no relevant content) to 100 (comprehensively covered).

Rules: 
- Do never refer to a webpage, article or similar, 
  the output must be suitable for a research paper, so write the technical detail
- All relevant context must be kept. 
- The output can be long and detailed 
- Only generate text in **English**.
- Output must be **strictly a valid JSON object** in the format shown.
- Do **not** include any extra explanations, comments, or text before or after the JSON.
- Make sure all types and structures in the JSON are correct (e.g., strings, integers).
- If any string contains quotes, they must be correctly escaped in the output.
---


### Output Format (Example):
{{
  "research": "Extensive research content with all details about the research query",
  "relevance": 0-100
}}
---
### Input Data raw text:
{content}

---  
### Output Format -  strict JSON:
{{
  "research": str
  "relevance": int 
}}          
Begin the JSON output now with {{
        """).strip()

        try:
            result_object, usage = self.model.execute_llm_schema(None,
                                                                 user_prompt,
                                                                 targetType=LLMSummaryOutput,
                                                                 fix_function=None,
                                                                 title='Step Page summary LLM')
            result_object:LLMSummaryOutput = result_object
        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed with E11 {e}")
            result_object = LLMSummaryOutput(summary="N/A", relevance=0)
        debug_data = DebugModel(prompt=user_prompt,
                                content=sum_content,
                                result=result_object.research,
                                relevance=result_object.relevance)
        if self.debugger:
            self.debugger.user_message(
                name="LLM prompt and result",
                agent=self,
                data=debug_data
            )

        return result_object