from typing import Dict, Tuple
from pydantic import BaseModel

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolOutputSchema
from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperToolOutputSchema

from AgentFramework.MultiPortAggregatorAgent import MultiPortAggregatorAgent

from AgentNews.NewsSchema import MergedOutput, LLMNewsOutput


class NewsMultiPortAggregatorAgentConfig(BaseToolConfig):
    """
    Configuration class for NewsMultiPortAggregatorAgent.
    """
    pass


class NewsMultiPortAggregatorAgent(MultiPortAggregatorAgent):
    """
    Agent that merges search results, web scraping results, and LLM-generated news into a unified data structure.

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Specifies expected input schemas.
        output_schema (Type[BaseModel]): Defines the expected output schema.
    """
    input_schemas = [
        TavilySearchToolOutputSchema,
        WebpageScraperToolOutputSchema,
        LLMNewsOutput
    ]
    output_schema = MergedOutput

    def run(self, inputs: Dict[str, Tuple[str, BaseModel]]) -> BaseModel:
        """
        Merges search results, web scraping data, and LLM-generated content into a single structured output.

        Args:
            inputs (Dict[str, Tuple[str, BaseModel]]): Dictionary containing categorized input data.

        Returns:
            MergedOutput: A structured output containing aggregated information.
        """
        search_result: TavilySearchToolOutputSchema = inputs[TavilySearchToolOutputSchema]
        web_scraping_result: WebpageScraperToolOutputSchema = inputs[WebpageScraperToolOutputSchema]
        llm_result: LLMNewsOutput = inputs[LLMNewsOutput]

        url = search_result.url
        title = search_result.title
        web_title = web_scraping_result.metadata.title
        content = web_scraping_result.content

        return MergedOutput(
            url=url,
            title=title,
            webtitle=web_title,
            content=content,
            news_title=llm_result.news_title,
            keywords=llm_result.keywords,
            news_abstract=llm_result.news_abstract,
            news_list=llm_result.news_list,
            news_content=llm_result.news_content,
        )
