from typing import Dict, Tuple

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentBookmarks.WebpageToCategoryAgent import BookmarkOutput
from AgentBookmarks.FirefoxBookmarkAgent import FirefoxBookmarksOutput, Bookmark
from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperToolOutputSchema

from AgentFramework.MultiPortAggregatorAgent import MultiPortAggregatorAgent

from AgentNews.NewsSchema import MergedOutput, LLMNewsOutput


class BookmarkMultiPortAggregatorAgentConfig(BaseToolConfig):
    """
    Configuration class for BookmarkMultiPortAggregatorAgent.

    Inherits from:
        BaseToolConfig: Base configuration structure for a tool.
    """
    pass


class BookmarkMultiPortAggregatorOutput(BaseIOSchema):
    """
    Schema for the structured output produced by BookmarkMultiPortAggregatorAgent.

    Attributes:
        webpage (WebpageScraperToolOutputSchema): Data extracted from the webpage.
        llm (BookmarkOutput): LLM-generated summary or categorization of the bookmark.
        bookmark (Bookmark): Original bookmark data from the browser.
    """
    webpage: WebpageScraperToolOutputSchema = Field(..., description="The webpage scraper data")
    llm: BookmarkOutput = Field(..., description="The LLM summary")
    bookmark: Bookmark = Field(..., description="The original bookmark")


class BookmarkMultiPortAggregatorAgent(MultiPortAggregatorAgent):
    """
    Agent that merges web scraping results, LLM summaries, and bookmark metadata into a single structured output.

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Mapping of expected input types for each source.
        output_schema (Type[BaseModel]): The final merged schema produced by the agent.
    """
    input_schemas = {
        "web_scraping_result": WebpageScraperToolOutputSchema,
        "llm_result": BookmarkOutput,
        "bookmark": Bookmark,
    }
    output_schema = BookmarkMultiPortAggregatorOutput

    def run(self, inputs: Dict[str, BaseModel]) -> BaseModel:
        """
        Merges the outputs of multiple tools (web scraper, LLM, and bookmark metadata) into a unified schema.

        Args:
            inputs (Dict[str,  BaseModel]):
                A dictionary where keys identify the input type and values are (source_id, data) tuples.
                Expected keys are:
                    - "web_scraping_result": Tuple containing the webpage scraper output
                    - "llm_result": Tuple containing the LLM's output
                    - "bookmark": Tuple containing the raw bookmark data

        Returns:
            BookmarkMultiPortAggregatorOutput: Combined structured output of the agent.
        """
        web_scraping_result: WebpageScraperToolOutputSchema = inputs["web_scraping_result"]
        llm_result: BookmarkOutput = inputs["llm_result"]
        bookmark: Bookmark = inputs["bookmark"]

        # Return the unified output structure
        return BookmarkMultiPortAggregatorOutput(
            webpage=web_scraping_result,
            llm=llm_result,
            bookmark=bookmark,
        )
