from typing import List
from pydantic import Field

from atomic_agents.lib.base.base_io_schema import BaseIOSchema

from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolOutputSchema


# -----------------------------------------------------------------------------
class LLMNewsInput(BaseIOSchema):
    """Schema representing the input from the user to the AI agent."""

    news_title: str = Field(..., description="The title of the news article")
    news_content: str = Field(..., description="The raw content of the news article.")


class LLMNewsOutput(BaseIOSchema):
    """Schema representing the structured output of an AI-processed news article."""

    news_title: str = Field(..., description="The title of the news article")
    keywords: List[str] = Field(...,
                                description="A list of keywords to characterize the news article (like DPA news categories)")
    news_abstract: str = Field(...,
                               description="A short paragraph as an abstract of the news article. Displayed to the user for a quick overview.")
    news_list: List[str] = Field(...,
                                 description="An itemized list of all important information in the news article, serving as a quick overview.")
    news_content: str = Field(...,
                              description="An extensive summary of the contents of the news article, containing all important information.")


# -----------------------------------------------------------------------------
class TextualSummaryInput(BaseIOSchema):
    """Schema for a collection of processed news data."""

    result: TavilySearchToolOutputSchema = Field(..., description="Search result items")
    news: LLMNewsInput = Field(..., description="Search result web content")
    summary: LLMNewsOutput = Field(..., description="Summary and description of the news")


class TextualSummaryOutput(BaseIOSchema):
    """Schema for structured news summary output."""

    news_text: str = Field(..., description="The news as a summarized markdown text")

# -----------------------------------------------------------------------------

class SummaryOutput(BaseIOSchema):
    """Schema for structured news summary output in string format."""

    output: str = Field("", description="The news as a summary text")


# -----------------------------------------------------------------------------
class MergedOutput(BaseIOSchema):
    """Schema for structured and merged news summary output."""

    url: str = Field(..., description="The URL of the news article")
    title: str = Field(..., description="The title of the news article")
    webtitle: str = Field(..., description="The title of the news webpage")
    content: str = Field(..., description="The full content of the news article")
    news_title: str = Field(..., description="The title of the processed news article")
    keywords: List[str] = Field(...,
                                description="A list of keywords to characterize the news article (like DPA news categories)")
    news_abstract: str = Field(..., description="A short paragraph summarizing the news article for a quick overview.")
    news_list: List[str] = Field(...,
                                 description="An itemized list of key information extracted from the news article.")
    news_content: str = Field(...,
                              description="A comprehensive summary containing all important details from the news article.")

    def pretty_print(self) -> None:
        """Prints a formatted representation of the merged news output."""
        print("\n" + "=" * 50)
        print(f"URL: {self.url}\n")
        print(f"Title: {self.title}\n")
        print(f"Webtitle: {self.webtitle}\n")
        print(f"Content: {self.content}\n")
        print(f"News Title: {self.news_title}\n")
        print(f"Keywords: {', '.join(self.keywords)}\n")
        print("News Abstract:\n" + "-" * 50)
        print(f"{self.news_abstract}\n")
        print("News List:\n" + "-" * 50)
        for item in self.news_list:
            print(f"- {item}")
        print("\nNews Content:\n" + "-" * 50)
        print(f"{self.news_content}\n")
        print("=" * 50 + "\n")

