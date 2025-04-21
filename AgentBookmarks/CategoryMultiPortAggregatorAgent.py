from typing import Dict, Tuple

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentBookmarks.WebpageToCategoryAgent import BookmarkOutput
from AgentBookmarks.FirefoxBookmarkAgent import FirefoxBookmarksOutput, Bookmark
from AgentBookmarks.GenerateCategoryForBookmarkAgent import GenerateCategoryForBookmarkOutput
from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolOutputSchema
from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperToolOutputSchema

from AgentFramework.MultiPortAggregatorAgent import MultiPortAggregatorAgent

from AgentNews.NewsSchema import MergedOutput, LLMNewsOutput


class CategoryMultiPortAggregatorAgentConfig(BaseToolConfig):
    """
    Configuration class for NewsMultiPortAggregatorAgent.
    """
    pass

class CategoryMultiPortAggregatorOutput(BaseIOSchema):
    """Schema for structured and merged news summary output."""

    webpage: WebpageScraperToolOutputSchema = Field(..., description="The webpage scraper data")
    llm: BookmarkOutput = Field(..., description="The LLM summary")
    bookmark: Bookmark = Field(..., description="The original bookmark")
    category:GenerateCategoryForBookmarkOutput = Field(..., description="The calcalted category")


class CategoryMultiPortAggregatorAgent(MultiPortAggregatorAgent):
    """
    Agent that merges search results, web scraping results, and LLM-generated news into a unified data structure.

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Specifies expected input schemas.
        output_schema (Type[BaseModel]): Defines the expected output schema.
    """
    input_schemas = {
        "web_scraping_result": WebpageScraperToolOutputSchema,
        "llm_result": BookmarkOutput,
        "bookmark": Bookmark,
        "category": GenerateCategoryForBookmarkOutput
    }
    output_schema = CategoryMultiPortAggregatorOutput

    def run(self, inputs: Dict[str, Tuple[str, BaseModel]]) -> BaseModel:
        """
        Merges search results, web scraping data, and LLM-generated content into a single structured output.

        Args:
            inputs (Dict[str, Tuple[str, BaseModel]]): Dictionary containing categorized input data.

        Returns:
            MergedOutput: A structured output containing aggregated information.
        """
        web_scraping_result: WebpageScraperToolOutputSchema = inputs["web_scraping_result"]
        llm_result: BookmarkOutput = inputs["llm_result"]
        bookmark: Bookmark = inputs["bookmark"]
        category: Bookmark = inputs["category"]

        return CategoryMultiPortAggregatorOutput(
            webpage=web_scraping_result,
            llm=llm_result,
            bookmark=bookmark,
            category=category
        )
