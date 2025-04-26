from typing import List

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AtomicTools.tavily_search.tool.tavily_search import TavilySearchTool, TavilySearchToolInputSchema, \
    TavilySearchToolOutputSchema, TavilySearchResultItemSchema
from agent_config import DUMMY_WEB


class SearchAgent(ConnectedAgent, TavilySearchTool):
    input_schema = TavilySearchToolInputSchema
    output_schema = TavilySearchToolOutputSchema

    def __init__(self, config: BaseToolConfig = BaseToolConfig(), uuid='default') -> None:
        ConnectedAgent.__init__(self, config, uuid)  # Explicitly call ConnectedAgent
        TavilySearchTool.__init__(self, config)  # Explicitly call WebpageScraperTool


    def run(self, params: TavilySearchToolInputSchema) -> TavilySearchToolOutputSchema:

        print(f"[Search] Running search for '{params.queries}'")
        if DUMMY_WEB:
            dummy_results: List[TavilySearchResultItemSchema] = []

            for i in range(5):  # change 5 to however many you want
                dummy_result = TavilySearchResultItemSchema(
                    title=f"Dummy Search Result Title {i}",
                    url=f"https://example.com/dummy-result-{i}",
                    content=f"This is a dummy content snippet from search result {i}.",
                    raw_content="This is dumym raw conrtent",
                    score=round(0.9 - i * 0.1, 2),  # just some varied score logic
                    query=None,
                    answer=None,
                )
                dummy_results.append(dummy_result)

            result_list = TavilySearchToolOutputSchema(results=dummy_results)
            return result_list

        result = TavilySearchTool.run(self, params)
        print("Search erresults amout ",len(result.results))
        return result

