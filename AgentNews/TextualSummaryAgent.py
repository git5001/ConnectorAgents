from pydantic import BaseModel, Field

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentNews.NewsSchema import TextualSummaryOutput, TextualSummaryInput


class TextualSummaryAgentConfig(BaseToolConfig):
    """
    Configuration for the Textual Summary Agent.

    Attributes:
        output_format (str): Defines the format of the output, allowing 'text', 'html', or 'markdown'.
    """
    output_format: str = Field("markdown", description="Output format: 'text', 'html', or 'markdown'.")


class TextualSummaryAgent(ConnectedAgent):
    """
    An agent that generates structured textual summaries from news search results and webpage content.

    Attributes:
        input_schema (Type[MarkdownSummaryInput]): Defines the expected input schema.
        output_schema (Type[MarkdownSummaryOutput]): Defines the expected output schema.
        output_format (str): The format of the output ('text', 'html', or 'markdown').
    """
    input_schema = TextualSummaryInput
    output_schema = TextualSummaryOutput

    def __init__(self, config: TextualSummaryAgentConfig) -> None:
        """
        Initializes the Textual Summary Agent.

        Args:
            config (TextualSummaryAgentConfig): Configuration for the agent.
        """
        super().__init__(config)
        self.output_format: str = config.output_format

    def summarize_news(self, news_data: TextualSummaryInput) -> TextualSummaryOutput:
        """
        Processes news data and generates a formatted summary in the specified format.

        Args:
            news_data (TextualSummaryInput): Contains titles, keywords, abstracts, and summary information.

        Returns:
            TextualSummaryOutput: A structured textual summary in the chosen format.

        Raises:
            ValueError: If the specified output format is not 'text', 'html', or 'markdown'.
        """
        if self.output_format == "markdown":
            summary_text = f"""# {news_data.summary.news_title}
## Categories:
{', '.join(news_data.summary.keywords)}
## Short Abstract:
{news_data.summary.news_abstract}
## Content Summary:
{news_data.summary.news_content}
[Read more]({news_data.result.url})
"""
        elif self.output_format == "html":
            summary_text = f"""
                <html>
                    <head><title>{news_data.summary.news_title}</title></head>
                    <body>
                    <h2>{news_data.summary.news_title}</h2>
                    <p><strong>Categories:</strong> {', '.join(news_data.summary.keywords)}</p>
                    <p><em>{news_data.summary.news_abstract}</em></p>
                    <p>&nbsp;</p>
                    <p>{news_data.summary.news_content}</p>
                    <p><a href="{news_data.result.url}">Read more</a></p>
                </body>
                </html>
                """
        elif self.output_format == "text":
            summary_text = f"""Title: {news_data.summary.news_title}
Categories: {', '.join(news_data.summary.keywords)}
Short Abstract: {news_data.summary.news_abstract}
Content Summary: {news_data.summary.news_content}
Read more: {news_data.result.url}
"""
        else:
            raise ValueError("Invalid output format. Choose 'text', 'html', or 'markdown'.")

        return TextualSummaryOutput(news_text=summary_text)

    def run(self, params: TextualSummaryInput) -> TextualSummaryOutput:
        """
        Runs the agent synchronously to generate a structured news summary.

        Args:
            params (TextualSummaryInput): Input news data.

        Returns:
            TextualSummaryOutput: The formatted news summary in the specified format.
        """
        return self.summarize_news(params)
