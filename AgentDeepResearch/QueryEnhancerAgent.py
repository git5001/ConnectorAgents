import textwrap

from pydantic import BaseModel

from AgentFramework.ConnectedAgent import ConnectedAgent
from agent_config import DUMMY_LLM
from agent_logging import logger
from util.LLMSupport import LLMAgentConfig, LLMModel
from .schemas import UserQueryInput, EnhancedQueryOutput, DebugModel


# ------------------------------------------------------------
# 1.  A tiny pydantic model just for the LLM’s JSON response
# ------------------------------------------------------------
class LLMEnhanceOutput(BaseModel):
    search_1: str
    search_2: str
    search_3: str
    enhanced: str


# ------------------------------------------------------------
# 2.  The QueryEnhancerAgent
# ------------------------------------------------------------
class QueryEnhancerAgent(ConnectedAgent):
    """
    Turns a raw user question into:
      - a terse web-search query
      - a polished ‘research-paper-ready’ formulation
    """
    input_schema = UserQueryInput                 # {"query": str}
    output_schema = EnhancedQueryOutput           # {"original", "search", "enhanced"}

    def __init__(self, config: LLMAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config, uuid)

        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()

    # ---------- public entry-point ----------
    def run(self, params: UserQueryInput) -> EnhancedQueryOutput:
        print(f"[Enhancer] Enhancing query: {params.query}")

        if DUMMY_LLM:
            dummy_search = f"Search results for '{params.query}'"
            dummy_enhanced = f"Enhanced search query based on intent and context of: '{params.query}'"
            return EnhancedQueryOutput(
                original=params.query,
                search_1=dummy_search,
                search_2=dummy_search,
                search_3=dummy_search,
                enhanced=dummy_enhanced,
            )

        llm_result: LLMEnhanceOutput = self._run_llm(params.query)
        print(f"[Enhancer] Enhanced query is: {llm_result.enhanced}")
        print(f"[Enhancer] Seqarch #1 is: {llm_result.search_1}")
        print(f"[Enhancer] Seqarch #2 is: {llm_result.search_2}")
        print(f"[Enhancer] Seqarch #3 is: {llm_result.search_3}")

        return EnhancedQueryOutput(
            original=params.query,
            search_1=llm_result.search_1,
            search_2=llm_result.search_2,
            search_3=llm_result.search_3,
            enhanced=llm_result.enhanced,
        )

    # ---------- single-responsibility helper ----------
    def _run_llm(self, user_query: str) -> LLMEnhanceOutput:
        """
        Delegates the heavy lifting to the LLM.
        The prompt enforces strict JSON with *only* `search` and `enhanced`.
        """
        user_prompt = textwrap.dedent(f"""
You are an expert research assistant.

**Task**  
You will receive a raw user question.  

Produce distinct output types which improve the user prompt into two different directions:

1. **search** – Produce three concise query string optimised for a modern web search (search_1, search_2, search_3)
   - You produce three different web search terms which analyse the problem from three different angles.
     The searches should be as disjunct as possible to retrieve a large search space (e.g. physical, psychological, historical aspects)
   - Strip filler words, keep core keywords, add essential qualifiers.  
   - ≤ 15 words, lower-case, no punctuation except quotes or operators that improve search.
   - The search terms are the query to a web search engine about this topic and must find all relevant and diverse pages
   

2. **enhanced** – a well-formed research question suitable for defining and  designing an academic report (enhanced).
   - Rewrite in formal English, clarify scope, add any obviously missing variables
     (e.g., time-frame, population, metric) inferred from context.
   - Cover side and slightly related topics to generate a fully runded research  
   - several full sentences, clear and precise.
   - The enhanced question is the definition question for a research report
   

**Rules**
- Return *only* valid JSON with exactly the keys shown below.  
- No additional keys, text, or comments.  
- Escape quotes properly.  
- Language: English.

---
### User Question
{user_query}

---
### Strict JSON schema
{{
  "search_1": str,
  "search_2": str,
  "search_3": str,
  "enhanced": str
}}
Begin JSON now:
        """).strip()

        try:
            result_obj, usage = self.model.execute_llm_schema(
                None,
                user_prompt,
                targetType=LLMEnhanceOutput,
                fix_function=None,
                title="Query-Enhancement LLM",
            )
        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed with {e}")
            # fall back to pass-through so downstream agents still work
            result_obj = LLMEnhanceOutput(search_1=user_query, search_2="", search_3="", enhanced=user_query)

        # optional: keep a debugging artefact, mirroring your existing pattern
        debug_data = DebugModel(
            prompt=user_prompt,
            content="",
            result=result_obj.model_dump_json(),
            relevance=100,
        )
        if self.debugger:
            self.debugger.user_message(
                name="Enhancer LLM prompt+result",
                agent=self,
                data=debug_data,
            )

        return result_obj
